## 2-Class prediction

This project trains a mobileNet (https://arxiv.org/pdf/1704.04861.pdf) inspired network for a two class prediction problem.

## Dependencies
- Tensorflow, Keras

- python, Cuda

## How-to


## Analysis

**Architecture**:
![Alt text](evaluation/cnn_vs_mNet_loss.png?raw=true "Title")
![Alt text](evaluation/cnn_vs_mNet_validation_loss.png?raw=true "Title")
Compared two different architectures (standard CNN vs mobileNet). Even with nearly 1/4th parameters MobileNet performs in the same range as a standard CNN but requires more iterations to converge. This could be the result of split of a single convolutional layer into two making it harder to optimize.

**Data augmentation & dropout:**
![Alt text](evaluation/mNet_loss.png?raw=true "Title")
![Alt text](evaluation/mNet_validation_loss.png?raw=true "Title")
Due to the small dataset size the networks quickly overfit on the data. This makes data augmentation and dropout regularization imperative to prevent overfitting at the cost of faster convergence.

**Class-wise performance:**
![Alt text](evaluation/mNet_precisionRecall.png?raw=true "Title")
Due to higher intra-class variation in the "class 1" (sushi) we observe a poorer performance on this class compared to the "class 0" (sandwich).

**Losses:**
![Alt text](evaluation/ce_vs_se_acc.png?raw=true "Title")
![Alt text](evaluation/ce_vs_se_validation_acc.png?raw=true "Title")
Experimented with two different losses, cross entropy (CE) and the squared error (SE) loss. The CE loss performed better than SE. This is because of the characteristics of the two losses. The SE loss incentives getting high confidence for easier examples at the cost of poor performance on the harder examples.
