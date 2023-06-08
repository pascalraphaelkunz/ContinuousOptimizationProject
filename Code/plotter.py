import matplotlib.pyplot as plt

def plot_accuracy(x, datasetName):
    data = x
    x = range(len(x))

    plt.plot(x, data)
    plt.xlabel('iteration')
    plt.ylabel('accuracy in percent')
    plt.title(f'Accuracy over iteration in {datasetName} dataset')
    plt.savefig(f'Accuracy over iteration in {datasetName} dataset')

    plt.show()
    
    plt.close()

def plot_logloss(x, datasetName):
    data = x
    x = range(len(x))

    plt.plot(x, data)
    plt.xlabel('iteration')
    plt.ylabel('log loss')
    plt.title(f'Logloss over iterations in {datasetName} dataset')
    plt.savefig(f'Logloss over iterations in {datasetName} dataset')

    plt.show()
    plt.close()

