import h5py
import os

from src.layers.dense import Dense
from src.layers.convolution import Convolution
from src.layers.flatten import Flatten
from src.activations.relu import ReLU
from src.activations.sigmoid import Sigmoid
from src.activations.softmax import Softmax
from src.activations.tanh import Tanh
from src.model.model import Model

from src.utils.losses import binary_cross_entropy, binary_cross_entropy_prime, mse, mse_prime, categorical_cross_entropy, categorical_cross_entropy_prime


def load_model(name):
    """
    Load a model from an HDF5 file.

    :param name: str
        The name of the saved model.
    :return: Model
        The reconstructed model object.
    """

    model = Model(None, None)  # Initialize without loss functions

    # Get file path for loading the model
    current_dir = os.path.dirname(__file__)
    content_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    save_dir = os.path.join(content_root_dir, 'saved_models')
    file_path = os.path.join(save_dir, name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The model file at {file_path} does not exist.")

    try:
        with h5py.File(file_path, 'r') as f:
            # Load the architecture information
            architecture_group = f['architecture']

            # Load class mappings from datasets
            if 'class_indices_keys' in f and 'class_indices_values' in f:
                # Keys are class names (strings), decode them from bytes to strings
                class_indices_keys = [key.decode('utf8') for key in f['class_indices_keys'][:]]
                # Values are class indices (integers), no need to decode
                class_indices_values = list(f['class_indices_values'][:])
                model.class_indices = dict(zip(class_indices_keys, class_indices_values))

            if 'index_to_class_keys' in f and 'index_to_class_values' in f:
                # Keys are class indices (integers), no need to decode
                index_to_class_keys = list(f['index_to_class_keys'][:])
                # Values are class names (strings), decode them from bytes to strings
                index_to_class_values = [val.decode('utf8') for val in f['index_to_class_values'][:]]
                model.index_to_class = dict(zip(index_to_class_keys, index_to_class_values))

            # Assign loss functions to the model
            assign_loss_functions(model, architecture_group)

            # Iterate through all layers in the saved model
            for _, layer_group in architecture_group.items():
                layer = reconstruct_layer(layer_group)
                model.add(layer)

    except OSError as e:
        raise ValueError(f"Error opening the HDF5 file: {e}")

    return model


def assign_loss_functions(model, architecture_group):
    """
    Assign loss functions to the model based on saved attributes.

    :param model: Model
        The model to which loss functions will be assigned.
    :param architecture_group: h5py Group
        The HDF5 group containing model architecture information.
    """
    loss_function_name = architecture_group.attrs['loss_function']
    loss_function_prime_name = architecture_group.attrs['loss_function_prime']

    # Map strings back to the actual loss function using a predefined dictionary
    loss_function_mapping = {
        'binary_cross_entropy': binary_cross_entropy,
        'binary_cross_entropy_prime': binary_cross_entropy_prime,
        'mse': mse,
        'mse_prime': mse_prime,
        'categorical_cross_entropy': categorical_cross_entropy,
        'categorical_cross_entropy_prime': categorical_cross_entropy_prime
    }

    # Retrieve loss functions from the mapping
    loss_function = loss_function_mapping.get(loss_function_name)
    loss_function_prime = loss_function_mapping.get(loss_function_prime_name)

    # Assign the loaded loss function to the model
    model.loss_function = loss_function
    model.loss_function_prime = loss_function_prime


def reconstruct_layer(layer_group):
    """
    Reconstruct a layer from its saved attributes in an HDF5 file.

    :param layer_group: h5py Group
        The HDF5 group containing layer information.
    :return: Layer
        The reconstructed layer object.
    """
    class_name = layer_group.attrs['class_name']

    # Reconstruct each layer based on its class name
    if class_name == 'Flatten':
        input_shape = tuple(layer_group.attrs['input_shape'])
        output_shape = tuple(layer_group.attrs['output_shape'])
        layer = Flatten(input_shape, output_shape)

    elif class_name == 'Dense':
        input_size = layer_group.attrs['input_size']
        output_size = layer_group.attrs['output_size']
        layer = Dense(input_size, output_size)
        layer.weights = layer_group['weights'][:]
        layer.biases = layer_group['biases'][:]

    elif class_name == 'Convolution':
        input_shape_con = tuple(layer_group.attrs['input_shape_conv'])
        kernel_size = layer_group.attrs['kernel_size']
        depth = layer_group.attrs['depth']
        layer = Convolution(input_shape_con, kernel_size, depth)
        layer.kernels = layer_group['kernels'][:]
        layer.biases = layer_group['biases'][:]

    elif class_name == 'ReLU':
        layer = ReLU()
    elif class_name == 'Sigmoid':
        layer = Sigmoid()
    elif class_name == 'Softmax':
        layer = Softmax()
    elif class_name == 'Tanh':
        layer = Tanh()
    else:
        raise ValueError(f"Unsupported layer type: {class_name}")

    return layer