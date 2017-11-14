import os.path
import signal
import numpy as np
import time
import matplotlib.pyplot as plt

class Timer(object):
    """
    A simple timer class that let's you measure the execution time of a specific block of code.

    Use as:

    with Timer() as t:
        # do something
    t.interval
    """
    def __init__(self):
        self.interval = 0
        self.running = False

    def __enter__(self):
        """
        Enters the critical section whose runtime is measured.
        """
        self.start = time.clock()
        self.running = True
        return self

    def __exit__(self, *args):
        """
        Leaves the critical section.
        """
        self.end = time.clock()
        self.running = False
        self.interval = self.end - self.start

class VerboseTimer(Timer):
    """
    A timer class that prints the time to stdout.
    """

    def __init__(self, text):
        """
        Initializes a new instance of the VerboseTimer class.

        :param text: The title of the block.
        """
        super().__init__()
        self.text = text

    def __enter__(self):
        """
        Enters the critical section whose runtime is measured.
        """
        super().__enter__()
        print("%s..." % self.text, end="", flush=True)
        return self

    def __exit__(self, *args):
        """
        Leaves the critical section.
        """
        super().__exit__(*args)
        print(" [%.2fs]" % self.interval)

def plot(loss,val_loss,name,out_prob=None,filt=None,plot_loss=True):
    noPlts = len(name)
    epochs = loss.shape[0]

    plt.figure(1)
    plt.rcParams['axes.facecolor'] = 'gray'
    axs1 = []
    for x in range(noPlts):
        tmp, =plt.plot(np.arange(epochs),loss[:,x],label=name[x])
        axs1.append(tmp)    
    if(plot_loss):
        plt.legend(handles=axs1)
        plt.title('Training loss')
    else:
        plt.legend(handles=axs1,loc=2)
        plt.title('Training accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Loss')


    plt.figure(2)
    plt.rcParams['axes.facecolor'] = 'gray'
    axs2 = []
    for x in range(noPlts):
        tmp, = plt.plot(np.arange(epochs),val_loss[:,x],label=name[x])
        axs2.append(tmp)    
    if(plot_loss):
        plt.legend(handles=axs2)
        plt.title('Validation loss')
    else:
        plt.legend(handles=axs2,loc=2)
        plt.title('Validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Loss')


    if (filt is not None):
        plt.figure(3)
        noFilt = filt.shape[-1]
        noPltAxis = np.ceil(np.sqrt(noFilt)).astype('int')
        for x in range(noFilt):
            plt.subplot(noPltAxis,noPltAxis,x+1)
            filt[:,:,:,x] -= filt[:,:,:,x].min()
            filt[:,:,:,x] /= filt[:,:,:,x].max()
            plt.imshow(filt[:,:,:,x])
            plt.axis('off')
        #plt.rcParams['axes.facecolor'] = 'gray'
        #plt.title('First layer filters')

    if (out_prob is not None):
        plt.figure(4)
        plt.rcParams['axes.facecolor'] = 'gray'
        axs4=[]

        out_label = np.zeros(out_prob.shape[0])
        out_label[out_label.shape[0]//2:] = 1
        out = np.column_stack((out_prob,out_label))
        out = out[out[:,0].argsort(),:]
        tp = np.cumsum(1-out[:,1])
        pr = tp/(np.arange(len(tp))+1)
        re = tp/np.sum(out_label)

        tmp, =plt.plot(re,pr,label='class 0')
        axs4.append(tmp)

        out = out[::-1,:]
        tp = np.cumsum(out[:,1])
        pr = tp/(np.arange(len(tp))+1)
        re = tp/np.sum(out_label)

        tmp, =plt.plot(re,pr,label='class 1')
        axs4.append(tmp)

        plt.legend(handles=axs4)
        plt.title('Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        """
        if (type(prReLoss)!=list):
            pr = prReLoss.clsPr
            pr = np.concatenate(np.concatenate(pr))
            pr = pr.reshape((epochs,-1))

            re = prReLoss.clsRe
            re = np.concatenate(np.concatenate(re))
            re = re.reshape((epochs,-1))
        else:
            pr = prReLoss[0]
            re = prReLoss[1]

        plt.figure(4)
        axs4=[]
        for x in range(pr.shape[1]):
            tmp, =plt.plot(re[:,x],pr[:,x],label='class'+str(x))
            axs4.append(tmp)
        plt.legend(handles=axs4)
        plt.title('Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        """
    plt.show()


