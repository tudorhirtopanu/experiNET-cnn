import numpy as np
from src.utils.convolution_utils import correlate2D, convolve2D
from src.utils.activation_utils import get_activation_function
from src.layers.layer import Layer


class Convolution(Layer):
    def __init__(self, kernel_size, depth, activation=None):
        """
        Initialize the convolutional layer.

        :param kernel_size: int
            The size (height and width) of the convolutional kernels. Assumes square kernels.
        :param depth: int
            The number of filters (kernels) in the convolutional layer. This determines the number of output channels.
        """

        super().__init__()

        self.kernel_size = kernel_size
        self.depth = depth
        self.kernels_shape = None
        self.output_shape = None
        self.bias = None
        self.kernels = None
        self.input_shape = None

        self.activation_name = activation
        self.activation = get_activation_function(activation)

    def initialize_weights(self, input_depth, input_height, input_width):
        """
        Initialize the kernels and bias for the convolutional layer.

        This method sets up the necessary weights and biases for the convolutional layer
        based on the input dimensions. The weight initialization follows He initialization.

        :param input_depth: int
            The number of input channels/depth of the input data.
        :param input_height: int
            The height of the input data.
        :param input_width: int
            The width of the input data.

        This method performs the following tasks:
        - Sets the `input_shape` attribute using the provided input dimensions.
        - Calculates the `output_shape` based on the depth of the filters, the kernel size, and the input dimensions.
        - Initializes the `kernels` (weights) using a random normal distribution scaled according to the He initialization method.
        - Initializes the `bias` as a zero vector with the length equal to the number of filters (`depth`).
        """
        self.input_shape = (input_depth, input_height, input_width)
        self.output_shape = (self.depth, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1)
        self.kernels_shape = (self.depth, input_depth, self.kernel_size, self.kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2 / (input_depth * self.kernel_size * self.kernel_size))
        self.bias = np.zeros(self.depth)

    def forward(self, input_array):
        """
        Perform the forward pass through the convolutional layer.

        :param input_array: numpy array
            A 4D array of shape (batch_size, input_depth, input_height, input_width) representing the input data.
        :return: numpy array
            A 4D array of shape (batch_size, depth, output_height, output_width) representing the output of the convolutional layer.
        """

        if self.kernels is None or self.bias is None:
            # Initialize weights if they haven't been initialized (happens during training)
            input_depth, input_height, input_width = input_array.shape[1:]
            self.initialize_weights(input_depth, input_height, input_width)

        self.input = input_array  # Store the input for use in backpropagation
        batch_size = input_array.shape[0]

        # Initialize the output array
        self.output = np.zeros((batch_size, *self.output_shape))

        # Iterate over each item in the batch
        for b in range(batch_size):
            # Iterate over each output channel
            for i in range(self.depth):
                # Iterate over each input channel
                for j in range(self.input_shape[0]):
                    # Perform 2D correlation between the input and the current kernel
                    self.output[b, i] += correlate2D(self.input[b, j], self.kernels[i, j])

                # Add the bias for the current filter across the entire feature map
                self.output[b, i] += self.bias[i]

        self.output = self.activation.forward(self.output)

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Compute gradients for backpropagation and update the kernels and bias.

        :param output_gradient: numpy array
            The gradient of the loss with respect to the output of this layer, of shape
            (batch_size, depth, output_height, output_width).
        :param learning_rate: float
            The learning rate used to scale the updates for kernels and bias.
        :return: numpy array
            The gradient of the loss with respect to the input of this layer, of shape
            (batch_size, input_depth, input_height, input_width).
        """
        batch_size = output_gradient.shape[0]

        output_gradient = self.activation.backward(self.output, learning_rate) * output_gradient

        # Initialize gradients for kernels and inputs
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros((batch_size, *self.input_shape))

        # Iterate over each item in the batch
        for b in range(batch_size):
            for i in range(self.depth):  # Iterate over each output channel
                for j in range(self.input_shape[0]):  # Iterate over each input channel
                    # Compute the gradient with respect to the kernel
                    kernels_gradient[i, j] += correlate2D(self.input[b, j], output_gradient[b, i])

                    # Compute the gradient with respect to the input
                    input_gradient[b, j] += convolve2D(output_gradient[b, i], self.kernels[i, j], "full")

        # Update the kernels using the computed gradients
        self.kernels -= learning_rate * kernels_gradient / batch_size

        # Update the bias using the output gradient: sum over height and width, then average over batch
        bias_gradient = np.mean(output_gradient, axis=(0, 2, 3))  # Shape (3,)
        self.bias -= learning_rate * bias_gradient

        # Return the gradient with respect to the input
        return input_gradient

