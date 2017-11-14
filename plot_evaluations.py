import numpy as np
import dltools

########################################################################################################################
# Plot results of standard CNN vs mobileNet
########################################################################################################################

loss = np.load('evaluations/cnn_vs_mNet_loss.npy')
val_loss = np.load('evaluations/cnn_vs_mNet_validation_loss.npy')
name = ['CNN', 'mobileNet']
dltools.utility.plot(loss,val_loss,name)

########################################################################################################################
# Plot results of mobileNet with different training methods
########################################################################################################################

loss = np.load('evaluations/mNet_loss.npy')
val_loss = np.load('evaluations/mNet_validation_loss.npy')
out_prob = np.load('evaluations/mNet_validation_probability.npy')
name = ['w/o augmentation','augmentation','dropout+augemtnation']
dltools.utility.plot(loss,val_loss,name,out_prob)

########################################################################################################################
# Plot results of standard Cross Entropy-vs-Squared Error loss
########################################################################################################################

acc = np.load('evaluations/ce_vs_se_accuracy.npy')
val_acc = np.load('evaluations/ce_vs_se_validation_accuracy.npy')
name = ['CE', 'SE']
dltools.utility.plot(acc,val_acc,name,plot_loss=False)


