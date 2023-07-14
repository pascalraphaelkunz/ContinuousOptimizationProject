import numpy as np
import data_clean as data_clean
import plotter as plot
import csv




def sigmoid(x):
    '''
    Applies sigmoid function to input
    '''
    return 1 / (1 + np.exp(-x))


def log_loss(y_true, y_pred):
    '''
    Returns Log-loss of y_true and y_pred
    '''
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def log_loss_reg(y, y_pred, w, reg_param=0.001):
    '''
    Returns regularized Log-loss of y_true and y_pred
    '''  
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    reg_loss = reg_param * np.sum(w**2/(1+w**2))  # Non-convex regularizer    
    total_loss = loss + reg_loss    
    return total_loss

def gradient(y_true, y_pred, X, w, reg_param=0.001, regularized=False):
    '''
    Computes gradient of log-loss or regularized log-loss
    '''  
    d_reg_loss = (2 * reg_param * w) * (1 / (1 + w**2)**2)  if regularized else 0
    return np.dot(X.T, y_pred - y_true) / len(y_true) + d_reg_loss

def hessian(y_pred, X):
    '''
    returns hessian matrix
    '''
    diag = np.diag(y_pred * (1 - y_pred))
    return np.dot(np.dot(X.T, diag), X) / len(y_pred)

def update_weights(method: str, hess, gradient, weights, H=1, step_size=0.01, y_true=None, X = None, loss = None):
    '''
    Given a specific newton method, we update the weights differently
    '''
    if method == 'normal_newton':
        weights -= np.dot(np.linalg.inv(hess), gradient)
        return weights
    if method == 'regularized_newton':
        # Method from https://arxiv.org/abs/2112.02089
        weights -= np.dot(np.linalg.inv(hess + np.sqrt(H*np.linalg.norm(gradient)) * np.eye(len(weights))), gradient)
        return weights
    if method == 'linesearch_newton':
        # Line search strategy
        direction = np.dot(np.linalg.inv(hess), gradient)
        for i in range(10):
            new_weights = weights - step_size * direction
            y_pred = sigmoid(np.dot(X, new_weights))
            new_loss = log_loss(y_true, y_pred)
            if new_loss < loss:
                weights = new_weights
                break
            step_size /= 2
        return weights
    elif method == 'trust_region_newton':
        # Solve the trust region subproblem and update weights
        p = solve_trust_region_subproblem(hess, gradient, step_size)
        weights += p
        return weights
    elif method == 'damped_newton':
        #Solve damped_newton method according to https://arxiv.org/pdf/2211.00140.pdf
        #Lipschitz constant of logistic loss = 1 therefore a_k = (-1 + sqrt(1+2))/1
        weights -= np.dot((-1 + np.sqrt(1+2))/1 *np.linalg.inv(hess), gradient)
        return weights       
    else:
        raise ValueError("Invalid method specified.")

def solve_trust_region_subproblem(hess, grad, delta):
    """
    Solve the trust region subproblem to obtain the step p.
    """
    # Compute the Newton step
    p = np.linalg.solve(hess + delta * np.eye(len(grad)), -grad)

    # Apply the trust region constraint
    norm_p = np.linalg.norm(p)
    if norm_p <= delta:
        return p
    else:
        return (delta / norm_p) * p

def newton_method(X, y_true, X_test, y_true_test, num_iterations=100, regularized=False, method='normal_newton', H=1, step_size=0.01):
    # Initialize weights
    num_features = X.shape[1]
    weights = np.zeros(num_features)
    accuracy_list = []
    logloss_list = []

    for _ in range(num_iterations):

        # Calculate predicted probabilities
        y_pred = sigmoid(np.dot(X, weights))
        accuracy, logloss = calculate_accuracy(X_test, weights, y_true_test, regularized=regularized)
        accuracy_list.append(accuracy)
        logloss_list.append(logloss)

        # Calculate gradient and Hessian
        grad = gradient(y_true, y_pred, X, w=weights, regularized=regularized)
        hess = hessian(y_pred, X)
        # Update weights using Newton's method
        weights = update_weights(method, hess, grad, weights, H=H, y_true=y_true, X=X, loss=logloss, step_size=step_size)    
    return weights, accuracy_list, logloss_list

def calculate_accuracy(X, weights, labels_test, regularized=False):
    # Calculate predicted probabilities for the test dataset using the learned weights
    test_pred_probs = sigmoid(np.dot(X, weights))

    # Apply a threshold to obtain binary predictions (e.g., threshold = 0.5)
    threshold = 0.5
    test_predictions = (test_pred_probs >= threshold).astype(int)

    # Evaluate the predictions using appropriate evaluation metrics (e.g., accuracy, log loss)
    accuracy = np.mean(test_predictions == labels_test)
    logloss = log_loss_reg(labels_test, test_pred_probs, weights) if regularized else log_loss(labels_test, test_pred_probs)
    
    return accuracy, logloss



def run(num_iterations, filepath_train, filepath_test, type, regularized=False, method="normal_newton", H=1, step_size=0.01):
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
    weights, accuracy, log_loss = newton_method(X, labels, X_test, labels_test, num_iterations=num_iterations, regularized=regularized, method=method, H=H, step_size=step_size)
 
    return weights, accuracy, log_loss