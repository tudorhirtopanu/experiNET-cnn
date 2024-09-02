from src.utils.training_visualiser import TrainingVisualiser


class Model:
    def __init__(self, loss_function, loss_function_prime):
        """
        Initialize the model with a specific loss function.

        :param loss_function: A function that computes the loss.
        :param loss_function_prime: The derivative of the loss function for backpropagation.
        """
        self.layers = []
        self.loss_function = loss_function
        self.loss_function_prime = loss_function_prime
        self.visualiser = TrainingVisualiser()

    def add(self, layer):
        """
        Add a layer to the model.

        :param layer: A layer object (e.g. Dense, Tanh).
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Perform a forward pass through all layers in the model.

        :param x: Input data.
        :return: Output after passing through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        """
        Perform a backward pass through all layers in the model (in reverse order).

        :param grad: Gradient of the loss with respect to the output.
        :param learning_rate: Learning rate for updating the layers' parameters.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, data_gen, epochs, learning_rate):
        """
        Train the model using the provided data.

        :param data_gen: ImageDataGenerator instance with training and validation(optional) data.
        :param epochs: Number of training epochs.
        :param learning_rate: Learning rate for updating the layers' parameters.
        """

        has_validation_data = data_gen.val_directory is not None and data_gen.num_val_images > 0

        for e in range(epochs):

            total_error = 0  # Accumulated error over the epoch
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

                # Print progress for the current batch
                print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches}, Batch Error={batch_error:.6f}', end='')

            # Calculate and print average error for the epoch
            average_error = total_error / len(data_gen)

            print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches}, Average Error={average_error:.6f}', end='')

            # Check if validation data is available
            if has_validation_data:

                # Perform validation on the validation data and calculate the average validation error
                val_average_error = self.validate(data_gen)

                self.visualiser.record_epoch_error(average_error, val_average_error)

                print(f', Validation Error={val_average_error:.6f}')
            else:
                self.visualiser.record_epoch_error(average_error)
                print()

        self.visualiser.plot_errors()

    def validate(self, data_gen):
        """
        Perform validation on the validation data.

        :param data_gen: ImageDataGenerator instance with validation data.
        :return: Average validation error.
        """
        val_total_error = 0  # Accumulated validation error
        val_total_batches = 0  # Total number of validation samples

        # Iterate through the validation data generator
        for val_batch_images, val_batch_labels in data_gen.__iter__(mode='val'):

            # Forward pass through the model
            output = self.forward(val_batch_images)

            # Calculate batch validation loss
            val_batch_error = self.loss_function(val_batch_labels, output)

            # Accumulate the total validation error
            val_total_error += val_batch_error

            # Accumulate the total validation error
            val_total_batches += 1

        # Return average validation error
        return val_total_error / val_total_batches

