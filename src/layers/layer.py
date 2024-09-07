from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        """
        Initialize a base Layer object.

        Attributes:
            input   Holds the input data fed into this layer during the forward pass.
            output  Holds the output data produced by this layer during the forward pass.
        """
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data):
        """
        Perform the forward pass through the layer.

        This method should be overridden by subclasses to implement specific layer functionality.

        :param input_data: numpy array
            The input data to the layer.
        :return: numpy array
            The output data produced by the layer after applying the layer-specific transformation.

        Note:
            This method should set the `self.input` and `self.output` attributes.
            Subclasses must implement this method to define the layer's behavior.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass to compute gradients and update parameters.

        This method should be overridden by subclasses to implement layer-specific backpropagation logic.

        :param output_gradient: numpy.ndarray
            The gradient of the loss function with respect to the output of this layer.
        :param learning_rate: float
            The learning rate used to update the layer's parameters.
        :return: numpy array
            The gradient of the loss function with respect to the input of this layer.
            This gradient is passed to the previous layer during backpropagation.
        Note:
            This method should compute the gradients with respect to the layer's parameters and update those parameters.
            It should also compute and return the gradient with respect to the input, which is required for
            backpropagation through the network.
        """
        pass
