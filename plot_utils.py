import matplotlib.pyplot as plt
import numpy as np
import time

def plot_losses(train_loss, test_loss, epoch):
    fig, ax = plt.subplots(1, 1)
    ax.plot([i for i in range(epoch+1)], train_loss, color='blue')
    ax.plot([i for i in range(epoch+1)], test_loss, color='red')
    plt.show(block=False)
    time.sleep(2)
    plt.close(fig)

def plot_img(imgs, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    im = np.transpose(imgs[0,...].cpu(), (1, 2, 0))
    #im = np.transpose(im.cpu(), (1, 2, 0))
    ax.imshow(im)
    plt.axis('off')
    plt.close()
    return ax
