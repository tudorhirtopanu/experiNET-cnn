import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """
    Calculate the binary cross-entropy loss between true labels and predicted probabilities.

    :param y_true: numpy array
        The true binary labels. Each element should be either 0 or 1.
    :param y_pred: numpy array
        The predicted probabilities. Each element should be a probability value between 0 and 1.
    :return: float
        The mean binary cross-entropy loss calculated over all samples.
    """
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    """
    Calculate the derivative of the binary cross-entropy loss with respect to the predicted probabilities.

    :param y_true: numpy array
        The true binary labels. Each element should be either 0 or 1.
    :param y_pred: numpy array
        The predicted probabilities. Each element should be a probability value between 0 and 1.
    :return: numpy array
        The gradient of the loss with respect to y_pred. This is used for backpropagation during training.
    """
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error loss between true values and predicted values.

    :param y_true: numpy array
        The true values. These are the actual target values.
    :param y_pred: numpy array
        The predicted values from the model.
    :return: float
        The mean squared error loss calculated over all samples. This value represents
        the average of the squared differences between the true and predicted values.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """
    Calculate the derivative of the Mean Squared Error loss with respect to the predicted values.

    :param y_true: numpy array
        The true values. These are the actual target values.
    :param y_pred: numpy array
        The predicted values from the model.
    :return: numpy array
        The gradient of the loss with respect to y_pred. This is used for backpropagation
        during training to update the weights. The gradient indicates how the loss
        would change with a small change in the predicted values.
    """
    return 2 * (y_pred - y_true) / np.size(y_true)
