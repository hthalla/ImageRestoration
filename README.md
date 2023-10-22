# Restoration of degraded images captured during adverse weather conditions with Generative Adversarial Networks

## Dataset
The model has been trained on the Radiate dataset. </br>

![Adverse](dataset/rain/rain1.jpg)

![Clear](dataset/city/city1.jpg)

## Dependencies
* Tensorflow
* Matplotlib

## Cycle-GAN
### Network layout
![cycle-gan](images/network_layout.png)

### Forward consistency loss
![forward](images/forward_consistency.png)

### Backward consistency loss
![forward](images/backward_consistency.png)

### Identity loss
![identity1](images/identity_loss1.png)

![identity1](images/identity_loss2.png)

## Results
![iter_13250](images/Iter_13250.png)

## Summary
Observed significant improvement in the results in the early iterations. But, after a while the GAN network trades-off diversity to fidelity and hence, tries to augment vehicles/generate trees inplace of rain drops etc. Concluding with this as the limitation of GAN/Cycle-GAN.

The report of this project can be found in the below link.</br>
[report](report/Report.pdf)
