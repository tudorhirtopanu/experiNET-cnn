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


def categorical_cross_entropy(y_true, y_pred):
    """
    Calculate the categorical cross-entropy loss between true labels and predicted probabilities.

    :param y_true: numpy array
        The true labels, one-hot encoded (batch_size x num_classes).
    :param y_pred: numpy array
        The predicted probabilities (batch_size x num_classes).
    :return: float
        The mean categorical cross-entropy loss calculated over all samples in the batch.
    """
    epsilon = 1e-10

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute the categorical cross-entropy loss for each sample in the batch
    batch_losses = -np.sum(y_true * np.log(y_pred), axis=1)

    # Average the loss over the batch
    loss = np.mean(batch_losses)
    return loss


def categorical_cross_entropy_prime(y_true, y_pred):
    """
    Calculate the derivative of the categorical cross-entropy loss with respect to the predicted probabilities.

    :param y_true: numpy array
        The true labels, one-hot encoded. Each row should be a one-hot encoded vector of the true class.
    :param y_pred: numpy array
        The predicted probabilities. Each row should be a probability distribution (sum to 1) over classes.
    :return: numpy array
        The gradient of the loss with respect to y_pred. This is used for backpropagation during training.
    """
    epsilon = 1e-10

    # Clip predictions to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Mean gradient over the batch
    grad = y_pred - y_true
    return grad



