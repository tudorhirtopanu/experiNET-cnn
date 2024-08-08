import numpy as np
from layer import Layer


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        """
        Initialize a Reshape layer.

        :param input_shape: tuple
            The shape of the input data that this layer will reshape.
            This should match the shape of the data that is passed to this layer during the forward pass.
        :param output_shape: tuple
            The desired shape to which the input data will be reshaped.
            The total number of elements must be the same as in the input_shape.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        """
        Perform the forward pass by reshaping the input data.

        :param input: numpy array
            The input data to be reshaped. Its shape must match self.input_shape.
        :return: numpy array
            The reshaped output data with shape self.output_shape.
        """
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass by reshaping the output gradient to the shape of the input.

        :param output_gradient: numpy.ndarray
            The gradient of the loss with respect to the output of this layer.
            This is reshaped to match the input shape during the backward pass.
        :param learning_rate: float
            The learning rate used for parameter updates. Not used in this layer, but included for consistency.
        :return: numpy.ndarray
            The gradient of the loss with respect to the input of this layer, reshaped to self.input_shape.
        """
        return np.reshape(output_gradient, self.input_shape)
