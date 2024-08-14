import numpy as np
from src.activations.activation import Activation


class Tanh(Activation):
    def __init__(self):
        """
        Initialize the Tanh activation layer.

        The Tanh function squashes input values to a range between -1 and 1.
        The derivative is used during backpropagation.
        """
        def tanh(x):
            """
            Tanh activation function.

            :param x: numpy array or scalar
                The input data to which the Tanh function will be applied.
            :return: numpy array or scalar
                The output data after applying the Tanh function. Values are in the range (-1, 1).
            """
            return np.tanh(x)

        def tanh_prime(x):
            """
            Derivative of the Tanh activation function.

            :param x: numpy array or scalar
                The input data for which the derivative of the Tanh function is computed.
            :return: numpy array or scalar
                The gradient of the Tanh function with respect to the input. Used during backpropagation.
            """
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
