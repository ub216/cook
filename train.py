import dltools
import os
import sys
import numpy as np
import keras
from keras import backend as K
from keras.callbacks import LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


sys.setrecursionlimit(10000)

config = {
    "num_classes": 1,
    "width": 1,
    "batch_size": 16,
    "image_size": (224,224),
    "dataset_folder": os.getcwd(),
    "augment_dataset": True,
    "dropout": False,
    "mobileNet" : True,
    "no_training": 724,
    "no_testing": 80
}

########################################################################################################################
# DEFINE THE NETWORK
########################################################################################################################

with dltools.utility.VerboseTimer("Define network"):
    # Define the Keras variables
    input_var = K.placeholder((config["batch_size"], config["image_size"][0], config["image_size"][1], 3),dtype=K.floatx())
    if config["mobileNet"]:
        builder = dltools.architectures_mobileNet.mobileNetBuilder(num_classes=config["num_classes"],alpha=config["width"])
    else:
        builder = dltools.architectures_simpleNet.simpleNetBuilder(num_classes=config["num_classes"],alpha=config["width"])
    network = builder.build(dropout=config["dropout"])#(input_tensor=input_var)

print("number of parameters in model: %d" % network.count_params())

########################################################################################################################
# DEFINE LOSS AND COMPILE TRAIN FUNCTIONS
########################################################################################################################
with dltools.utility.VerboseTimer("Define loss"):
    # Get the raw network outputs
    network.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])

########################################################################################################################
# SET UP DATA
########################################################################################################################

with dltools.utility.VerboseTimer("Setup data"):
    if config["augment_dataset"]:
        train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,
                                        rotation_range=20,width_shift_range=0.2,
                                        height_shift_range=0.2,horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            config["dataset_folder"] + '/sushi_or_sandwich/train',
            target_size=config["image_size"],
            batch_size=config["batch_size"],
            class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
            config["dataset_folder"] + '/sushi_or_sandwich/val',
            target_size=config["image_size"],
            batch_size=config["batch_size"],
            class_mode='binary',
            shuffle=False)

########################################################################################################################
# SET UP CALLBACKS
########################################################################################################################

lr = 1e-3
clsPre = []
clsRe = []
noPerClsEx = config["no_testing"]//2
network.optimizer.lr.assign(lr)
def scheduler(epoch):
    global lr
    if not np.mod(epoch,20):
        lr *= 0.5
        network.optimizer.lr.assign(lr)
        print('New learning rate %f' %(lr))
    return lr
update_lr = LearningRateScheduler(scheduler)
"""
class prLoss(Callback):
    def on_train_begin(self, logs={}):
        self.clsRe = []
        self.clsPr = []

    def on_epoch_end(self, batch, logs={}):
        tst = network.predict_generator(validation_generator,steps=config["no_testing"] // config["batch_size"])
        self.clsRe.append([sum(tst[:noPerClsEx] < 0.5)/noPerClsEx,sum(tst[noPerClsEx:] > 0.5)/noPerClsEx])
        self.clsPr.append([sum(tst[:noPerClsEx] < 0.5)/(sum(tst < 0.5) if sum(tst < 0.5) > 0 else 1) ,sum(tst[noPerClsEx:] > 0.5)/(sum(tst > 0.5) if sum(tst > 0.5) > 0 else 1)])
    
computeLosses = prLoss()
"""
########################################################################################################################
# TRAINING
########################################################################################################################

train = network.fit_generator(
            train_generator,
            steps_per_epoch=config["no_training"] // config["batch_size"],
            epochs=100,
            callbacks=[update_lr],
            validation_data=validation_generator,
            validation_steps=config["no_testing"] // config["batch_size"])
network.save_weights("model.h5")

########################################################################################################################
# PLOT RESULTS
########################################################################################################################
out_prob = network.predict_generator(validation_generator,steps=config["no_testing"] // config["batch_size"])
name = 'No augment'
if (config["dropout"] and config["augment_dataset"]):
    name = 'Augment and dropout'
elif(config["dropout"]):
    name = 'Dropout'
elif(config["augment_dataset"]):
    name = 'Augment'

filt = network.get_weights()
dltools.utility.plot(np.asarray(train.history['loss'])[:,np.newaxis],np.asarray(train.history['val_loss'])[:,np.newaxis],[name],out_prob,filt[0])

