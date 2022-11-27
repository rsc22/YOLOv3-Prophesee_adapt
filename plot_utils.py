import matplotlib.pyplot as plt
import time

def plot_losses(train_loss, test_loss, epoch):
    fig, ax = plt.subplots(1, 1)
    ax.plot([i for i in range(epoch+1)], train_loss, color='blue')
    ax.plot([i for i in range(epoch+1)], test_loss, color='red')
    plt.show(block=False)
    time.sleep(2)
    plt.close(fig)
