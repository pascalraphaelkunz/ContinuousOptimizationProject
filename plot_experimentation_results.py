import pandas as pd
import matplotlib.pyplot as plt
import os

plotspath = "Plots/"
def plot_data(path):
    # Read the CSV file
    data = pd.read_csv(path)
    filename = path.split('/')[-1].split('.csv')[0]

    # Group the data by method
    grouped_data = data.groupby('Method')


    # Iterate over each group
    for method, group in grouped_data:
        plt.plot(group['Iteration'], group['Log Loss'], label=method)

    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title(f'Log Loss for each Method in {filename}')
    plt.legend()
    plt.savefig(plotspath+f'Log Loss for each Method in {filename}')
    plt.show()
    plt.close()

    # Iterate over each group
    for method, group in grouped_data:
        plt.plot(group['Iteration'], group['Accuracy'], label=method)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy in percent')
    plt.title(f'Accuracy for each Method in {filename}')
    plt.legend()
    plt.savefig(plotspath+f'Accuracy for each Method in {filename}')
    plt.show()

resultDir = "Results/"
for file in os.listdir("Results"):
    filepath = resultDir + file
    plot_data(filepath)