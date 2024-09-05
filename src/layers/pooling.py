import numpy as np
from src.layers.layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size=(2, 2), stride=2, mode='max'):
        """
        Initialise the Pooling layer.

        :param pool_size: tuple of (pool_height, pool_width)
            Height and width of the pooling window, assumes square pooling.
        :param stride: int
            The stride of the pooling window.
        :param mode: str
            The pooling operation to apply: 'max', 'min' or 'average'.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, input_array):
        """
        Perform the forward through the Pooling layer.

        :param input_array: numpy array
            A 4D array of shape (batch_size, depth, input_height, input_width) representing the input data.
        :return: numpy array
            A 4D array of shape (batch_size, depth, output_height, output_width) representing the output of the Pooling layer.
        """
        self.input = input_array
        batch_size, depth, input_height, input_width = input_array.shape
        pool_height, pool_width = self.pool_size

        # Calculate output dimensions
        output_height = (input_height - pool_height) // self.stride + 1
        output_width = (input_width - pool_width) // self.stride + 1

        # Initialize output
        self.output = np.zeros((batch_size, depth, output_height, output_width))

        # Perform pooling
        for b in range(batch_size):
            for d in range(depth):
                for i in range(0, input_height - pool_height + 1, self.stride):
                    for j in range(0, input_width - pool_width + 1, self.stride):
                        # Define the current pooling window
                        pool_region = input_array[b, d, i:i + pool_height, j:j + pool_width]

                        # Apply pooling operation based on mode
                        if self.mode == 'max':
                            self.output[b, d, i // self.stride, j // self.stride] = np.max(pool_region)
                        elif self.mode == 'min':
                            self.output[b, d, i // self.stride, j // self.stride] = np.min(pool_region)
                        elif self.mode == 'average':
                            self.output[b, d, i // self.stride, j // self.stride] = np.mean(pool_region)
                        else:
                            raise ValueError("Invalid pooling mode. Choose 'max', 'min', or 'average'.")

        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """
        Perform the backward pass through the Pooling layer.

        :param output_gradient: numpy array
            The gradient of the loss with respect to the output of this layer.
        :param learning_rate: float
            The learning rate used for parameter updates. Not used in this layer, but included for consistency.
        :return: numpy array
            The gradient of the loss with respect to the input of this layer
        """
        batch_size, depth, input_height, input_width = self.input.shape
        pool_height, pool_width = self.pool_size
        input_gradient = np.zeros_like(self.input)

        for b in range(batch_size):
            for d in range(depth):
                for i in range(0, input_height - pool_height + 1, self.stride):
                    for j in range(0, input_width - pool_width + 1, self.stride):

                        # Define the current pooling window
                        pool_region = self.input[b, d, i:i + pool_height, j:j + pool_width]

                        if self.mode == 'max':
                            max_value = np.max(pool_region)
                            mask = (pool_region == max_value)
                            input_gradient[b, d, i:i + pool_height, j:j + pool_width] += mask * output_gradient[b, d, i // self.stride, j // self.stride]
                        elif self.mode == 'min':
                            min_value = np.min(pool_region)
                            mask = (pool_region == min_value)
                            input_gradient[b, d, i:i + pool_height, j:j + pool_width] += mask * output_gradient[b, d, i // self.stride, j // self.stride]
                        elif self.mode == 'average':
                            average_gradient = output_gradient[b, d, i // self.stride, j // self.stride] / (pool_height * pool_width)
                            input_gradient[b, d, i:i + pool_height, j:j + pool_width] += average_gradient

        return input_gradient
