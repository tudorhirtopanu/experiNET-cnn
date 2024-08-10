import numpy as np
from activation import Activation


class Sigmoid(Activation):
    def __init__(self):
        """
        Initialize the Sigmoid activation layer.

        The Sigmoid function squashes input values to a range between 0 and 1.
        The derivative is used during backpropagation.
        """

        def sigmoid(x):
            """
            Sigmoid activation function.

            :param x: numpy array or scalar
                The input data to which the Sigmoid function will be applied.
            :return: numpy array or scalar
                The output data after applying the Sigmoid function. Values are in the range (0, 1).
            """
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            """
            Derivative of the Sigmoid activation function.

            :param x: numpy array or scalar
                The input data for which the derivative of the Sigmoid function is computed.
            :return: numpy array or scalar
                The gradient of the Sigmoid function with respect to the input. Used during backpropagation.
            """
            s = sigmoid(x)  # Compute Sigmoid output
            return s * (1 - s)  # Compute the derivative using the Sigmoid output

        super().__init__(sigmoid, sigmoid_prime)
