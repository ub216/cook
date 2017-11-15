## 2-Class prediction

This project trains a mobileNet (https://arxiv.org/pdf/1704.04861.pdf) inspired network for performing two class prediction.

## Dependencies
- Cuda-8.0, Cudnn-6

- Python, Tensorflow, Keras

## Running
Download the git repository and run the dockerfile to install all dependencies. The dockerfile also downloads the data and runs the training code.

## Training
The training data (sushi-vs-sandwich) was split into a ~90/10 % training/validation split giving a total of 724 training and 80 testing images.

## Performance

**Architecture**:
Compared two different architectures (standard CNN vs mobileNet).

![Alt text](evaluations/cnn_vs_mNet_loss.png?raw=true "Title")
![Alt text](evaluations/cnn_vs_mNet_validation_loss.png?raw=true "Title")

Even with nearly 1/4th parameters MobileNet performs in the same range as a standard CNN but requires more iterations to converge. This could be the result of split of a single convolutional layer into two making it harder to optimize.

**Data augmentation & dropout:**
Due to the small dataset size the networks quickly overfit on the data. This makes data augmentation and dropout regularization imperative to prevent overfitting at the cost of faster convergence.

![Alt text](evaluations/mNet_loss.png?raw=true "Title")
![Alt text](evaluations/mNet_validation_loss.png?raw=true "Title")

**Class-wise performance:**
Due to higher intra-class variation in the "class 1" (sushi) we observe a poorer performance on this class compared to "class 0" (sandwich).

![Alt text](evaluations/mNet_precisionRecall.png?raw=true "Title")

**Losses:**
Experimented with two different losses, cross entropy (CE) and the squared error (SE) loss.

![Alt text](evaluations/ce_vs_se_acc.png?raw=true "Title")
![Alt text](evaluations/ce_vs_se_validation_acc.png?raw=true "Title")

The CE loss performed better than SE. This is because of the characteristics of the two losses. The SE loss incentivize high confidence on easier examples at the cost of poor performance on the difficult ones.

## Deploying

As such the current performance of the classifier is poor but could be improved with more training data or using pre-trained layers on complimentary tasks.

For real world product, binary classification is very limiting. It might be more useful to predict the probability of the classes rather than binary outputs.

Furthermore when the input image does not belong to either of them the above solution is also limiting. Training with background label could help address this but is at the cost of requiring additional training data (to cover all possible "background class") and the associate training class imbalance problem.

Predicting network confidence rather than having an extra label might be a possible solution to address this as it allows the user to make a more informed decision.


## Acknowledgement

The depthwise convolution code is based on Keras separable convolution layer code and modified as per needs.
Dockerfile is based on Craig Citro dockerfile for tensorflow (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker) and changed for this project.

