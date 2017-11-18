## 2-Class prediction

This project trains a mobileNet (https://arxiv.org/pdf/1704.04861.pdf) inspired network for performing two class prediction.

## Dependencies
- Cuda-8.0, Cudnn-6

- Python3, Tensorflow, Keras

## Running
The simplest way to run is to fork the project and run in [docker cloud](https://cloud.docker.com/swarm/ub216/dashboard/onboarding/cloud-registry) (or run docker locally). Note that after training I plot a few figures which can be viewed only if you pass the approriate [flags](http://wiki.ros.org/docker/Tutorials/GUI). 

The above method would run in cpu. To use gpu please run with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Else download the git repository and make sure all the dependecies are met. Download the dataset (http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip) and unzip in the same folder. Run split_data.py followed by train.py.

```
git clone https://github.com/ub216/cook
cd cook
wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip
unzip sushi_or_sandwich_photos.zip
python3 split_data.py
python3 train.py
```

## Training
The training data (sushi-vs-sandwich) was split into a ~90%-10 % training/validation split giving a total of 724 training and 80 testing images.

## Performance

**Architecture**:
Compared two different architectures (standard CNN vs mobileNet).

![Alt text](evaluations/cnn_vs_mNet_loss.png?raw=true "Title")
![Alt text](evaluations/cnn_vs_mNet_validation_loss.png?raw=true "Title")

Even with nearly 1/4th parameters MobileNet performs in the same range as a standard CNN but requires more iterations to converge. This could be the result of a single convolutional layer been split into two making it harder to optimize.

**Data augmentation & dropout:**
Due to the small dataset size the network quickly overfits on the data. This makes data augmentation and dropout regularization imperative to prevent overfitting and is at the cost of slower convergence.

![Alt text](evaluations/mNet_loss.png?raw=true "Title")
![Alt text](evaluations/mNet_validation_loss.png?raw=true "Title")

**Class-wise performance:**
Due to higher intra-class variation in the "class 1" (sushi) we observe a poorer performance on this class compared to "class 0" (sandwich).

![Alt text](evaluations/mNet_precisionRecall.png?raw=true "Title")

**Losses:**
Experimented with two different losses, cross entropy (CE) and the squared error (SE) loss.

![Alt text](evaluations/ce_vs_se_acc.png?raw=true "Title")
![Alt text](evaluations/ce_vs_se_validation_acc.png?raw=true "Title")

The CE loss performed better than SE. This is because of the characteristics of the two losses. The SE loss incentivizes high confidence on easier examples at the cost of poor performance on the difficult ones.

## Deploying

As such the current performance of the classifier is poor but could be improved with more training data or pre-training layers on complimentary tasks.

For real world product, binary classification is very limiting. It might be more useful to predict the probability of the classes rather than binary outputs.

Furthermore when the input image does not belong to either of them the above solution is also limiting. Training with background label could help address this but is at the cost of requiring additional training data (to cover all possible "background class"). This methods also comes with the additional problem of training class imbalance.

Predicting network confidence rather than having an extra label might be a possible solution to address this as it allows the user to make a more informed decision.

## Acknowledgement

The depthwise convolution code is based on Keras separable convolution layer code and modified as per needs.
