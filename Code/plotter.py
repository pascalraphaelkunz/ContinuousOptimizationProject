import matplotlib.pyplot as plt

def plot_accuracy(x, datasetName, loss="normal", method="normal_newton"):
    data = x
    x = range(len(x))

    plt.plot(x, data)
    plt.xlabel('iteration')
    plt.ylabel('accuracy in percent')
    plt.title(f'Accuracy over iteration in {datasetName} dataset with method {method}')
    plt.savefig(f'Accuracy over iteration in {datasetName} dataset with method {method}')   
    plt.close()

def plot_logloss(x, datasetName, loss="non-regularized", method="normal_newton"):
    data = x
    x = range(len(x))

    plt.plot(x, data)
    plt.xlabel('iteration')
    plt.ylabel('log loss')
    plt.title(f'Logloss over iterations in {datasetName} dataset with method {method}')
    plt.savefig(f'Logloss {loss} over iterations in {datasetName} dataset with method {method}')
    plt.close()

