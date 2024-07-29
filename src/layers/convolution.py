import numpy as np
from src.utils.cross_correlation import correlate2D


class Convolution:
    def __init__(self, input_shape, kernel_size, depth):
        """
        Initialise the convolutional layer

        :param input_shape: tuple containing depth x height x width of input
        :param kernel_size: int representing height x width of kernel
        :param depth: int representing how many kernels we want
        """

        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size+1, input_width - kernel_size+1)

        # Number of kernels, depth of each kernel, size of matrices in each kernel
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        # initialise kernels and biases randomly
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.kernels_shape)

    def forward(self, input_array):
        """
        Perform the forward pass of the convolutional layer.

        :param input_array: 3D numpy array with shape (input_depth, input_height, input_width)
                            representing the input to the convolutional layer.
        :return: 3D numpy array with shape (depth, output_height, output_width)
                 representing the output of the convolutional layer.
        """
        self.input = input_array
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output += correlate2D(self.input[j], self.kernels[i, j])
        return self.output
        
