import numpy as np
from src.utils.convolution_utils import correlate2D, convolve2D
from layer import Layer


class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
        Initialize the convolutional layer.

        :param input_shape: tuple of (depth, height, width)
            The shape of the input data where depth is the number of input channels,
            height is the height of the input, and width is the width of the input.
        :param kernel_size: int
            The size (height and width) of the convolutional kernels. Assumes square kernels.
        :param depth: int
            The number of filters (kernels) in the convolutional layer. This determines the number of output channels.
        """

        super().__init__()
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
        Perform the forward pass through the convolutional layer.

        :param input_array: numpy array
            A 3D array of shape (input_depth, input_height, input_width) representing the input data.
        :return: numpy array
            A 3D array of shape (depth, output_height, output_width) representing the output of the convolutional layer.
        """

        # Store the input array for use in backpropagation
        self.input = input_array

        # Initialize the output by copying the biases (one per output channel)
        self.output = np.copy(self.biases)

        # Iterate over each output channel
        for i in range(self.depth):

            # Iterate over each input channel
            for j in range(self.input_depth):

                # Perform 2D correlation between the input and the current kernel
                # Add the result to the output array
                self.output += correlate2D(self.input[j], self.kernels[i, j])
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Compute gradients for backpropagation and update the kernels and biases.

        :param output_gradient: numpy array
            The gradient of the loss with respect to the output of this layer, of shape (depth, output_height,
            output_width).
        :param learning_rate: float
            The learning rate used to scale the updates for kernels and biases.
        :return: numpy array
            The gradient of the loss with respect to the input of this layer, of shape (input_depth, input_height,
            input_width).
        """

        # Initialize gradients for kernels and inputs
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        # Loop over each kernel and input channel
        for i in range(self.depth):  # Iterate over each output channel
            for j in range(self.input_depth):  # Iterate over each input channel

                # Compute the gradient with respect to the kernel
                kernels_gradient[i, j] = correlate2D(self.input[j], output_gradient[i])

                # Compute the gradient with respect to the input
                input_gradient[j] += convolve2D(output_gradient[i], self.kernels[i, j], "full")

        # Update the kernels using the computed gradients
        self.kernels -= learning_rate * kernels_gradient

        # Update the biases using the output gradient
        self.biases -= learning_rate * output_gradient

        # Return the gradient with respect to the input
        return input_gradient
