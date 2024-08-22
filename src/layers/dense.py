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
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input_data):
        """
        Computes the forward pass for the dense layer.

        :param input_data: numpy array
            Input data to the layer (shape: input_size x batch_size).
        :return: numpy array
            Output of the layer (shape: output_size x batch_size).
        """
        self.input = input_data
        return np.dot(self.weights, self.input) + self.bias

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
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
