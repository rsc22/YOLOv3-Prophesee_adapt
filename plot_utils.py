import matplotlib.pyplot as plt


def plot_losses(train_loss, test_loss, epoch, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.plot([i for i in range(epoch+1)], train_loss, color='blue')
    ax.plot([i for i in range(epoch+1)], test_loss, color='red')
    return ax