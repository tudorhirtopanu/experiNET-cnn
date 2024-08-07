import numpy as np
from src.utils.convolution_utils import correlate2D, convolve2D


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
        Perform the backpropagation step to compute gradients and update the kernels and biases.

        :param output_gradient: Gradient of the loss with respect to the output of this layer
        :param learning_rate: The learning rate used to update the kernels and biases
        :return: Gradient of the loss with respect to the input of this layer
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
