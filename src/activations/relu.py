import numpy as np
from src.activations.activation import Activation


class ReLU(Activation):
    def __init__(self):
        """
        Initialize the ReLU activation layer.

        The ReLU function sets all negative input values to 0 and keeps positive input values unchanged.
        The derivative is used during backpropagation and is 1 for positive input values and 0 otherwise.
        """
        def relu(x):
            """
            Performs the forward pass using the ReLU activation function.

            :param x: numpy array
                Input data to the layer (shape: batch_size x input_size).
            :return: numpy array
                Output of the layer after applying the ReLU activation (shape: batch_size x input_size).
            """
            self.input = x
            # Apply ReLU activation (element-wise max operation with 0)
            return np.maximum(0, self.input)

        def relu_prime(x):
            """
            Performs the backward pass through the ReLU activation function.

            :param x: numpy array
                Gradient of the loss with respect to the output of this layer (shape: batch_size x input_size).
            :return: numpy array
                Gradient of the loss with respect to the input of this layer (shape: batch_size x input_size).
            """
            # Gradient is 0 for input values <= 0, and 1 for input values > 0
            return x > 0

        super().__init__(relu, relu_prime)
