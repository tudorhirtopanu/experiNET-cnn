from src.utils.metric_tracker import MetricTracker


class Model:
    def __init__(self, loss_function, loss_function_prime):
        """
        Initialize the model with a specific loss function.

        :param loss_function: Callable
            A function that computes the loss.
        :param loss_function_prime: Callable
            The derivative of the loss function for backpropagation.
        """
        self.layers = []
        self.loss_function = loss_function
        self.loss_function_prime = loss_function_prime
        self.metric_tracker = MetricTracker()

    def add(self, layer):
        """
        Add a layer to the model.

        :param layer: Layer
            A layer object (e.g. Dense, Tanh).
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Perform a forward pass through all layers in the model.

        :param x: numpy array
            Input data.
        :return: numpy array
            Output after passing through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        """
        Perform a backward pass through all layers in the model (in reverse order).

        :param grad: numpy array
            Gradient of the loss with respect to the output.
        :param learning_rate: float
            Learning rate for updating the layers' parameters.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, data_gen, epochs, learning_rate):
        """
        Train the model using the provided data.

        :param data_gen: ImageDataGenerator
             Instance with training and validation(optional) data.
        :param epochs: Int
            Number of training epochs.
        :param learning_rate: float
            Learning rate for updating the layers' parameters.
        """

        has_validation_data = data_gen.val_directory is not None and data_gen.num_val_images > 0

        for e in range(epochs):

            total_error = 0  # Accumulated error over the epoch
            total_accuracy = 0  # Accumulated accuracy over the epoch
            batch_number = 0  # Current batch being iterated
            total_batches = len(data_gen)  # Total number of batches in the data generator

            for batch_images, batch_labels in data_gen.__iter__(mode='train'):
                batch_number += 1

                # Forward pass through the model
                output = self.forward(batch_images)

                # Compute the error for the current batch
                batch_error = self.loss_function(batch_labels, output)

                # Calculate the gradient of the loss function with respect to the output
                grad = self.loss_function_prime(batch_labels, output)

                # Perform the backward pass using the accumulated gradients
                self.backward(grad, learning_rate)

                # Update total error across all batches
                total_error += batch_error

                # Compute batch accuracy and accumulate it
                batch_accuracy = MetricTracker.calculate_accuracy(batch_labels, output)
                total_accuracy += batch_accuracy

                # Print progress for the current batch
                print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches} --- Batch Err={batch_error:.6f}, Batch Accuracy={batch_accuracy:.2%}', end='')

            # Calculate and print average error and accuracy for the epoch
            average_error = total_error / len(data_gen)
            average_accuracy = total_accuracy / len(data_gen)

            # Record the training metrics
            self.metric_tracker.record_train_metrics(average_error, average_accuracy)

            print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches} --- Avg Err={average_error:.6f}, Avg Accuracy={average_accuracy:.2%}', end='')

            # Check if validation data is available
            if has_validation_data:
                # Perform validation on the validation data
                val_average_error, val_average_accuracy = self.validate(data_gen)

                # Record the validation metrics
                self.metric_tracker.record_val_metrics(val_average_error, val_average_accuracy)

                print(f', Val Err={val_average_error:.6f}, Val Accuracy={val_average_accuracy:.2%}')
            else:
                print()

        # Plot the metrics after training
        self.metric_tracker.plot_metrics()

    def validate(self, data_gen):
        """
        Perform validation on the validation data.

        :param data_gen: ImageDataGenerator
            Instance with validation data
        :return: tuple (validation_error, validation_accuracy)
            Average validation error and accuracy
        """
        val_total_error = 0  # Accumulated validation error
        val_total_accuracy = 0  # Accumulated validation accuracy
        val_total_batches = 0  # Total number of validation samples

        # Iterate through the validation data generator
        for val_batch_images, val_batch_labels in data_gen.__iter__(mode='val'):

            # Forward pass through the model
            output = self.forward(val_batch_images)

            # Calculate batch validation loss
            val_batch_error = self.loss_function(val_batch_labels, output)

            # Calculate batch validation accuracy
            val_batch_accuracy = MetricTracker.calculate_accuracy(val_batch_labels, output)

            # Accumulate the total validation error and accuracy
            val_total_error += val_batch_error
            val_total_accuracy += val_batch_accuracy

            val_total_batches += 1

        # Return average validation error and average validation accuracy
        return val_total_error / val_total_batches, val_total_accuracy / val_total_batches

