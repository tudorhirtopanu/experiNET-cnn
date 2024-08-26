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
            num_samples = 0  # Total number of samples in the epoch
            batch_number = 0  # Current batch being iterated
            total_batches = len(data_gen)  # Total number of batches in the data generator

            for batch_images, batch_labels in data_gen.__iter__(mode='train'):
                batch_number += 1  # Increase batch number
                batch_error = 0  # Accumulated error for the current batch

                for image, label in zip(batch_images, batch_labels):

                    output = self.forward(image)  # Forward pass through the model
                    batch_error += self.loss_function(label, output)  # Calculate loss for the current image
                    grad = self.loss_function_prime(label, output)  # Compute gradient of loss with respect to output
                    self.backward(grad, learning_rate)  # Backward pass and parameter update

                total_error += batch_error  # Update total error across all batches
                num_samples += len(batch_images)  # Update the total number of samples
                batch_average_error = batch_error / len(batch_images)  # Compute average error for the current batch

                # Print progress for the current batch
                print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches}, Batch Error={batch_average_error:.6f}', end='')

            # Calculate and print average error for the epoch
            average_error = total_error / num_samples
            print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches}, Average Error={average_error:.6f}', end='')

            # Validation loop
            if has_validation_data:
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
        val_num_samples = 0  # Total number of validation samples

        for val_batch_images, val_batch_labels in data_gen.__iter__(mode='val'):
            for image, label in zip(val_batch_images, val_batch_labels):
                output = self.forward(image)  # Forward pass through the model
                val_total_error += self.loss_function(label, output)  # Calculate validation loss
                val_num_samples += 1  # Update the count of validation samples

        return val_total_error / val_num_samples  # Return average validation error
