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

    def train(self, train_gen, epochs, learning_rate):
        """
        Train the model using the provided data.

        :param train_gen: An iterable that provides batches of training data.
        :param epochs: Number of training epochs.
        :param learning_rate: Learning rate for updating the layers' parameters.
        """

        for e in range(epochs):

            total_error = 0
            num_samples = 0
            batch_number = 0
            total_batches = len(train_gen)

            for batch_images, batch_labels in train_gen:
                batch_number += 1
                batch_error = 0

                for image, label in zip(batch_images, batch_labels):
                    output = self.forward(image)
                    batch_error += self.loss_function(label, output)
                    grad = self.loss_function_prime(label, output)
                    self.backward(grad, learning_rate)

                total_error += batch_error
                num_samples += len(batch_images)
                batch_average_error = batch_error / len(batch_images)

                # Record batch error
                self.visualiser.record_batch_error(batch_average_error)

                print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches}, Batch Error={batch_average_error:.6f}', end='')

            average_error = total_error / num_samples
            self.visualiser.record_epoch_error(average_error)
            print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches}, Average Error={average_error:.6f}')
        self.visualiser.plot_errors()
