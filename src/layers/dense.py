from src.layers.layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, output_size):
        """
        Initializes a fully connected (dense) layer.

        :param output_size: int
            Number of output neurons for this layer.

        Weights are initialized with a normal distribution, and bias are
        initialized randomly. These parameters are adjusted during training.
        """
        super().__init__()

        self.input_size = None
        self.output_size = output_size

        # Weights and bias will be initialized in the first forward pass
        self.weights = None
        self.bias = None

    def initialize_weights(self, input_size):
        """
        Initializes weights and bias based on the input size using He initialization.
        This is only called during training, not during prediction.

        :param input_size: int
            The number of input features (or neurons) that this dense layer will receive.
        """
        self.input_size = input_size
        self.weights = np.random.randn(self.output_size, self.input_size) * np.sqrt(2 / self.input_size)
        self.bias = np.zeros(self.output_size)

    def forward(self, input_data):
        """
        Computes the forward pass for the dense layer.

        :param input_data: numpy array
            Input data to the layer (shape: batch_size x input_size).
        :return: numpy array
            Output of the layer (shape: batch_size x output_size).
        """
        if self.weights is None or self.bias is None:
            # Only initialize during training
            self.initialize_weights(input_data.shape[1])

        self.input = input_data

        # Perform matrix multiplication and add bias
        output = np.dot(self.input, self.weights.T) + self.bias

        return output

    def backward(self, output_gradient, learning_rate):
        """
        Computes the backward pass and updates layer parameters.

        :param output_gradient: numpy array
            Gradient of the loss with respect to the layer's output.
        :param learning_rate: float
            Learning rate for updating the weights and bias.
        :return: numpy array
            Gradient of the loss with respect to the layer's input,
            to be propagated to the previous layer.
        """
        batch_size = self.input.shape[0]

        # Compute gradients for weights and input
        weights_gradient = np.dot(output_gradient.T, self.input) / batch_size
        input_gradient = np.dot(output_gradient, self.weights)

        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0) / batch_size

        return input_gradient
