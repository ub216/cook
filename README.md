## 2-Class prediction

This project trains a mobileNet (https://arxiv.org/pdf/1704.04861.pdf) inspired network for performing two class prediction.

## Dependencies
- Cuda-8.0, Cudnn-6

- Python, Tensorflow, Keras

## Running
Dowload the git repository and run the dockerfile to install all dependencies. The dockerfile also downloads the data and runs the training code.

## Training
The training data (sushi-vs-sandwich) was split into a ~90/10 % training/validation split giving a total of 724 training and 80 testing images.

## Performance

**Architecture**:
![Alt text](evaluations/cnn_vs_mNet_loss.png?raw=true "Title")
![Alt text](evaluations/cnn_vs_mNet_validation_loss.png?raw=true "Title")

Compared two different architectures (standard CNN vs mobileNet). Even with nearly 1/4th parameters MobileNet performs in the same range as a standard CNN but requires more iterations to converge. This could be the result of split of a single convolutional layer into two making it harder to optimize.

**Data augmentation & dropout:**
![Alt text](evaluations/mNet_loss.png?raw=true "Title")
![Alt text](evaluations/mNet_validation_loss.png?raw=true "Title")

Due to the small dataset size the networks quickly overfit on the data. This makes data augmentation and dropout regularization imperative to prevent overfitting at the cost of faster convergence.

**Class-wise performance:**
![Alt text](evaluations/mNet_precisionRecall.png?raw=true "Title")

Due to higher intra-class variation in the "class 1" (sushi) we observe a poorer performance on this class compared to "class 0" (sandwich).

**Losses:**
![Alt text](evaluations/ce_vs_se_acc.png?raw=true "Title")
![Alt text](evaluations/ce_vs_se_validation_acc.png?raw=true "Title")

Experimented with two different losses, cross entropy (CE) and the squared error (SE) loss. The CE loss performed better than SE. This is because of the characteristics of the two losses. The SE loss incentivize high confidence on easier examples at the cost of poor performance on the difficult ones.

## Deploy


## Acknoledgment

The depthwise convolution code is based on Keras seprable convolution layer code and modified as per needs.
Dockerfile is based on Craig Citro dockerfile for tensorflow (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker) and changed for this project.
