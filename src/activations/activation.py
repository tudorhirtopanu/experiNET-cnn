import numpy as np
from src.layers.layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        """
        Initialize an activation layer.

        :param activation: function
            The activation function to apply during the forward pass.
            This function should take a numpy array as input and return a numpy array.
        :param activation_prime: function
            The derivative of the activation function to use during the backward pass.
            This function should take a numpy array as input and return the gradient of the activation function.
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        """
        Perform the forward pass by applying the activation function.

        :param input_data: numpy array
            The input data to which the activation function will be applied.
            This should be the output from the previous layer.
        :return: numpy array
            The output data after applying the activation function.
            This output will be used as input for the next layer in the network.
        """
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass by computing the gradient of the loss with respect to the input.

        :param output_gradient: numpy array
            The gradient of the loss with respect to the output of this layer.
            This is provided by the subsequent layer during backpropagation.
        :param learning_rate: float
            The learning rate used for parameter updates.
            Note: This parameter is not used in the current implementation but is included for consistency with other layer types.
        :return: numpy array
            The gradient of the loss with respect to the input of this layer.
            This gradient will be passed to the previous layer during backpropagation.
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))
