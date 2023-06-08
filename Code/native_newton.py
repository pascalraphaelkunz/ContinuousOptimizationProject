import numpy as np
import data_clean as data_clean
import plotter as plot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def log_loss_reg(y, y_pred, w, reg_param=0.001):  
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    reg_loss = reg_param * np.sum(w**2/(1+w**2))  # Non-convex regularizer    
    total_loss = loss + reg_loss    
    return total_loss

def gradient(y_true, y_pred, X):
    return np.dot(X.T, y_pred - y_true) / len(y_true)

def hessian(y_pred, X):
    diag = np.diag(y_pred * (1 - y_pred))
    return np.dot(np.dot(X.T, diag), X) / len(y_pred)

def hessian_reg(y_pred, X, regularization=1e-6):
    diag = np.diag(y_pred * (1 - y_pred))
    reg_term = regularization * np.eye(X.shape[1])
    return np.dot(np.dot(X.T, diag), X) / len(y_pred) + reg_term

def newton_method(X, y_true, X_test, y_true_test, num_iterations=100, learning_rate=1):
    # Initialize weights
    num_features = X.shape[1]
    weights = np.zeros(num_features)
    accuracy_list = []
    logloss_list = []

    for _ in range(num_iterations):

        # Calculate predicted probabilities
        y_pred = sigmoid(np.dot(X, weights))
        accuracy, logloss = calculate_accuracy(X_test, weights, y_true_test)
        accuracy_list.append(accuracy)
        logloss_list.append(logloss)

        # Calculate gradient and Hessian
        grad = gradient(y_true, y_pred, X)
        print(grad)
        hess = hessian(y_pred, X)
        # Update weights using Newton's method
        weights -= np.dot(np.linalg.inv(hess), grad) * learning_rate        
    return weights, accuracy_list, logloss_list

def calculate_accuracy(X, weights, labels_test):
   # Calculate predicted probabilities for the test dataset using the learned weights
    test_pred_probs = sigmoid(np.dot(X, weights))

    # Apply a threshold to obtain binary predictions (e.g., threshold = 0.5)
    threshold = 0.5
    test_predictions = (test_pred_probs >= threshold).astype(int)

    # Evaluate the predictions using appropriate evaluation metrics (e.g., accuracy, log loss)
    accuracy = np.mean(test_predictions == labels_test)
    logloss = log_loss_reg(labels_test, test_pred_probs, weights)
    return accuracy, logloss

def run(num_iterations, filepath_train, filepath_test, type):
    if type == "a9a":
        full_dataset = np.array(data_clean.process_data_a9a(file_path=filepath_train))
        full_dataset_test = np.array(data_clean.process_data_a9a(file_path=filepath_test))
    else:
        full_dataset = np.array(data_clean.process_data_ijcnn(file_path=filepath_train))
        full_dataset_test = np.array(data_clean.process_data_ijcnn(file_path=filepath_test))
    
    labels = full_dataset[:, 0]
    labels = np.where(labels == -1, 0, labels)
    X = full_dataset[:, 1:]

    # Add bias term to the feature matrix of the test dataset
    X= np.c_[np.ones(X.shape[0]), X]

    labels_test = full_dataset_test[:, 0]
    labels_test = np.where(labels_test == -1, 0, labels_test)
    X_test = full_dataset_test[:, 1:]

    # Add bias term to the feature matrix of the test dataset
    X_test= np.c_[np.ones(X_test.shape[0]), X_test]


    # Optimize using Newton's method

    weights, accuracy, log_loss = newton_method(X, labels, X_test, labels_test, num_iterations=num_iterations)
    plot.plot_accuracy(accuracy, type)
    plot.plot_logloss(log_loss, type)