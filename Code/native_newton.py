import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient(y_true, y_pred, X):
    return np.dot(X.T, y_pred - y_true) / len(y_true)

def hessian(y_pred, X):
    diag = np.diag(y_pred * (1 - y_pred))
    return np.dot(np.dot(X.T, diag), X) / len(y_pred)

def newton_method(X, y_true, num_iterations=10, learning_rate=1):
    # Initialize weights
    num_features = X.shape[1]
    weights = np.zeros(num_features)

    for _ in range(num_iterations):
        # Calculate predicted probabilities
        y_pred = sigmoid(np.dot(X, weights))

        # Calculate gradient and Hessian
        grad = gradient(y_true, y_pred, X)
        hess = hessian(y_pred, X)

        # Update weights using Newton's method
        weights -= np.dot(np.linalg.inv(hess), grad) * learning_rate

    return weights

# Example usage
# Generate distinct random data with clear separation
np.random.seed(42)
X1 = np.random.rand(50, 3) * 0.4 + 0.3
X2 = np.random.rand(50, 3) * 0.4 + 0.7
X = np.vstack((X1, X2))
y_true = np.array([0] * 50 + [1] * 50)

# Add bias term to the feature matrix
X = np.c_[np.ones(X.shape[0]), X]

# Optimize using Newton's method
weights = newton_method(X, y_true)

# Plot the data points with their labels
plt.scatter(X[:, 1], X[:, 2], c=y_true, cmap='bwr', alpha=0.6)

# Plot the decision boundary
x_values = np.linspace(0, 1, 100)
y_values = -(weights[0] + weights[1] * x_values) / weights[2]
plt.plot(x_values, y_values, color='black')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification')
plt.show()