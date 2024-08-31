import numpy as np
from src.layers.layer import Layer


class Softmax(Layer):
    def forward(self, input_data):
        """
        Forward pass for the softmax layer.

        The softmax function converts logits or scores into probabilities that sum to 1,
        which is often used in the output layer of a classification network.

        :param input_data: numpy array
            Input data, typically the logits or scores for each class.

        :return : numpy array
            Output probabilities after applying the softmax function.
        """
        # Subtracting max for numerical stability
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Store the output to use in backpropagation
        self.output = probabilities
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """
        Backward pass for the softmax layer that works with any loss function.

        This method computes the gradient of the loss with respect to the input of the softmax layer.
        It utilizes the output of the softmax layer and the Jacobian matrix for the softmax function.

        :param output_gradient: numpy array
            Gradient of the loss with respect to the output.
        :param learning_rate: float, optional
            Learning rate (not used in this layer but included for compatibility).

        :return : numpy array
            Gradient of the loss with respect to the input of the softmax layer.
        """
        # Initialize the gradient tensor for the inputs to softmax
        input_gradient = np.zeros_like(self.output)

        # Iterate over each sample in the batch to compute the gradient for each sample
        for i, (softmax_output, grad_output) in enumerate(zip(self.output, output_gradient)):
            # Flatten the softmax output to use it in Jacobian computation
            s = softmax_output.reshape(-1, 1)

            # Compute the Jacobian matrix for softmax output
            # Jacobian is a matrix of partial derivatives, here it adjusts for the fact softmax is element-wise
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)

            # Calculate the gradient of the loss with respect to the softmax input
            input_gradient[i] = np.dot(jacobian_matrix, grad_output)

        return input_gradient

