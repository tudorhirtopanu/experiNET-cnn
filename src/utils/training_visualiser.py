import matplotlib.pyplot as plt


class TrainingVisualiser:
    def __init__(self):
        """
        Initialize the visualizer with empty lists for batch errors, training epoch errors,
        and validation epoch errors.
        """
        self.train_epoch_errors = []
        self.val_epoch_errors = []

    def record_epoch_error(self, train_error, val_error=None):
        """
        Record the epoch error for both training and validation.

        :param train_error: The training error to record.
        :param val_error: The validation error to record (optional).
        """
        self.train_epoch_errors.append(train_error)
        if val_error is not None:
            self.val_epoch_errors.append(val_error)

    def plot_errors(self):
        """
        Plot the epoch errors on the x-axis as epochs.
        """
        plt.figure(figsize=(12, 6))

        # Plot epoch errors
        epochs = range(1, len(self.train_epoch_errors) + 1)
        plt.plot(epochs, self.train_epoch_errors, label='Epoch Error', linewidth=2, color='orange')

        # Plot validation epoch errors if available
        if len(self.val_epoch_errors)>0:
            plt.plot(epochs, self.val_epoch_errors, label='Validation Error', linewidth=2, color='blue')

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Epoch Error During Training')
        plt.legend()

        # Set x-axis ticks to be whole numbers only
        plt.xticks(epochs)

        plt.show()
