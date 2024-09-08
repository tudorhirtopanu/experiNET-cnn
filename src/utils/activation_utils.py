from src.activations.softmax import Softmax
from src.activations.tanh import Tanh
from src.activations.relu import ReLU
from src.activations.sigmoid import Sigmoid


def get_activation_function(name):
    """
    Returns the activation_name class corresponding to the given string name.

    :param name: str
        The name of the activation_name function
    :return: Activation
        The activation_name class corresponding to the provided name.
    """

    activations = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax
    }

    if name in activations:
        return activations[name]()
    else:
        raise ValueError(f"Activation function '{name}' is not recognized.")
