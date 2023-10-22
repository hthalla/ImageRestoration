"""
    Train Cycle_GAN models.
"""

from csv import writer

from update_image import update_image_pool
from generate_fake import generate_fake_samples
from generate_real import generate_real_samples

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    count = 0
    # define properties of the training run
    n_epochs, n_batch, = 50, 1
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        count+=1
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch, count)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch, count)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA],
                                                           [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB],
                                                          [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        if count%250 == 0:
            # The data assigned to the list
            list_data=[i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2]
            # Writing into csv file
            with open('log/run.csv', 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(list_data)
                f_object.close()

            # serialize model to JSON
            model_json = g_model_AtoB.to_json()
            with open("model/g_model_AtoB.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            g_model_AtoB.save_weights("weights/g_model_AtoB.h5")

            model_json = g_model_BtoA.to_json()
            with open("model/g_model_BtoA.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            g_model_BtoA.save_weights("weights//g_model_BtoA.h5")
