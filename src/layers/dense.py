from src.layers.layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        Initializes a fully connected (dense) layer.

        :param input_size: int
            Number of input neurons (from the previous layer).
        :param output_size: int
            Number of output neurons for this layer.

        Weights are initialized with a normal distribution, and biases are
        initialized randomly. These parameters are adjusted during training.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(output_size)

    def forward(self, input_data):
        """
        Computes the forward pass for the dense layer.

        :param input_data: numpy array
            Input data to the layer (shape: batch_size x input_size).
        :return: numpy array
            Output of the layer (shape: batch_size x output_size).
        """
        self.input = input_data

        # Perform matrix multiplication and add bias
        output = np.dot(self.input, self.weights.T) + self.bias

        return output

    def backward(self, output_gradient, learning_rate):
        """
        Computes the backward pass and updates layer parameters.

        :param output_gradient: numpy array
            Gradient of the loss with respect to the layer's output.
        :param learning_rate: float
            Learning rate for updating the weights and biases.
        :return: numpy array
            Gradient of the loss with respect to the layer's input,
            to be propagated to the previous layer.
        """
        batch_size = self.input.shape[0]

        # Compute gradients for weights and input
        weights_gradient = np.dot(output_gradient.T, self.input) / batch_size
        input_gradient = np.dot(output_gradient, self.weights)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0) / batch_size

        return input_gradient
