import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('/home/fewaki31-adm/Documents/Lectures/Continuous Optimization/ContinuousOptimizationProject/experimentation_result_a9a1686397730.9361079.csv')

# Group the data by method
grouped_data = data.groupby('Method')


# Iterate over each group
for method, group in grouped_data:
    plt.plot(group['Iteration'], group['Log Loss'], label=method)

plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Log Loss for each Method in a9a dataset')
plt.legend()
plt.show()
plt.close()

# Iterate over each group
for method, group in grouped_data:
    plt.plot(group['Iteration'], group['Accuracy'], label=method)
plt.xlabel('Iteration')
plt.ylabel('Accuracy in percent')
plt.title('Accuracy for each Method in a9a dataset')
plt.legend()
plt.show()
