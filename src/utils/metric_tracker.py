import numpy as np


class MetricTracker:
    def __init__(self):
        """
        Initialize the metric tracker with empty lists for tracking various metrics.
        """
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def record_train_metrics(self, loss, accuracy):
        """
        Record training loss and accuracy for each epoch.

        :param loss: float
            Training loss for the epoch.
        :param accuracy: float
            Training accuracy for the epoch.
        """
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def record_val_metrics(self, loss, accuracy):
        """
        Record validation loss and accuracy for each epoch.

        :param loss: float
            Validation loss for the epoch.
        :param accuracy: float
            Validation accuracy for the epoch.
        """
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)

    def get_metrics(self):
        """
        Retrieve all tracked metrics.

        :return:
            Dictionary containing lists of training and validation losses and accuracies.
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

    def plot_metrics(self):
        """
        Plot the tracked metrics.
        """
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.train_losses) + 1)

        # Plot training and validation loss
        plt.figure()
        plt.plot(epochs, self.train_losses, 'r', label='Training loss')
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'b', label='Validation loss')
            plt.title('Training and Validation Loss')
        else:
            plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot training and validation accuracy
        plt.figure()
        plt.plot(epochs, self.train_accuracies, 'r', label='Training Accuracy')
        if self.val_accuracies:
            plt.plot(epochs, self.val_accuracies, 'b', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
        else:
            plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """
        Calculate accuracy by comparing predicted labels to true labels. This function works for both binary and multi-class classification.

        :param y_true: numpy array
            True labels (either binary labels (0 or 1) or one-hot encoded vectors).
        :param y_pred: numpy array
            Predicted probabilities (either a single probability for binary or a probability vector for multi-class).

        :return:
            Accuracy as a float.
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Check the shape of y_pred to determine if it is binary or multi-class
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            # Binary classification case
            # Convert probabilities to binary predictions (0 or 1) based on a threshold of 0.5
            predictions = (y_pred.flatten() > 0.5).astype(int)

            # If y_true is one-dimensional (just labels 0 or 1), ensure it's in the right shape
            if y_true.ndim > 1:
                y_true = y_true.flatten()
            true_labels = y_true  # For binary classification, true_labels is the same as y_true

        else:
            # Multi-class classification case
            # Convert predicted probabilities to class labels by taking the argmax
            predictions = y_pred.argmax(axis=1)

            # Convert one-hot encoded true labels to class labels
            true_labels = y_true.argmax(axis=1)

        # Calculate accuracy
        accuracy = np.mean(predictions == true_labels)
        return accuracy


