import matplotlib.pyplot as plt


class TrainingVisualiser:
    def __init__(self):
        """
        Initialize the visualizer with empty lists for batch and epoch errors.
        """
        self.batch_errors = []
        self.epoch_errors = []

    def record_batch_error(self, error):
        """
        Record the batch error.

        :param error: The error to record.
        """
        self.batch_errors.append(error)

    def record_epoch_error(self, error):
        """
        Record the epoch error.

        :param error: The error to record.
        """
        self.epoch_errors.append(error)

    def plot_errors(self):
        """
        Plot the epoch errors on the x-axis as epochs.
        """
        plt.figure(figsize=(12, 6))

        # Plot epoch errors
        epochs = range(1, len(self.epoch_errors) + 1)
        plt.plot(epochs, self.epoch_errors, label='Epoch Error', linewidth=2, color='orange')

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Epoch Error During Training')
        plt.legend()

        # Set x-axis ticks to be whole numbers only
        plt.xticks(epochs)

        plt.show()
