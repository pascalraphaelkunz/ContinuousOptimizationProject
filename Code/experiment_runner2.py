import csv
import newton
from time import time

methods = ["normal_newton", "regularized_newton", "linesearch_newton", "trust_region_newton"]

# Create a CSV file for storing the results
filename = f"experimentation_result_a9a{str(time())}.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "Iteration", "Accuracy", "Log Loss"])

    for method in methods:
        print(method)

        # Run Native Newton's Code for ijcnn dataset
        _, acc, log_loss = newton.run(40, 'Datasets/a9a/train.txt', 'Datasets/a9a/test.txt', type="a9a", method=method, step_size=0.2, H=1)

        # Write the results to the CSV file
        for i in range(len(acc)):
            writer.writerow([method, i+1, acc[i], log_loss[i]])

        print("done")
    

print("Results saved to experimentation_result.csv")