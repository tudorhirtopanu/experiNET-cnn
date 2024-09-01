import numpy as np
from src.layers.layer import Layer


class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        """
        Initialize a Flatten layer.

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

    def forward(self, input_data):
        """
        Perform the forward pass by reshaping the input data.

        :param input_data: numpy array
            The input data to be reshaped. Its shape must match self.input_shape.
        :return: numpy array
            The reshaped output data with shape self.output_shape.
        """
        self.input = input_data
        batch_size = input_data.shape[0]

        # Flatten to (batch_size, flattened_size)
        reshaped_data = np.reshape(input_data, (batch_size, *self.output_shape))
        return reshaped_data

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
        batch_size = output_gradient.shape[0]
        reshaped_gradient = np.reshape(output_gradient, (batch_size, *self.input_shape))
        return reshaped_gradient
