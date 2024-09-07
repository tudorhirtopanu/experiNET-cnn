import h5py
import os
import numpy as np

from src.utils.metric_tracker import MetricTracker
from src.layers.convolution import Convolution
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.layers.pooling import Pooling

from src.activations.relu import ReLU
from src.activations.softmax import Softmax
from src.activations.sigmoid import Sigmoid
from src.activations.tanh import Tanh


class Model:
    def __init__(self, loss_function, loss_function_prime):
        """
        Initialize the model with a specific loss function.

        :param loss_function: Callable
            A function that computes the loss.
        :param loss_function_prime: Callable
            The derivative of the loss function for backpropagation.
        """
        self.layers = []
        self.loss_function = loss_function
        self.loss_function_prime = loss_function_prime
        self.metric_tracker = MetricTracker()

        # To be set during training with ImageDataGenerator
        self.index_to_class = None
        self.class_indices = None
        self.image_dimensions = ()

    def print_information(self):
        """
        Print detailed information about the model structure, including each layer's class name.

        Note: This function will be updated to contain more information about the model structure.
        """

        # Print a header for the model structure information
        print("Model Structure:")

        # Loop through each layer in the model's list of layers
        for layer in self.layers:
            # Print the class name of each layer
            print(layer.__class__.__name__)

    def get_input_dimensions(self):
        """

        :return: The expected dimensions of the input image
        """
        return self.image_dimensions

    def predict(self, x):
        """
        Perform a forward pass through the model to make a prediction and return both
        the predicted class index and the corresponding class name.

        :param x: numpy array
            Input data representing an image
        :return: tuple
            The confidence score and the class name.

        NOTE: This method currently only supports predictions for one image, as opposed to batches of images.
        """

        # Remove depth from x.shape
        input_shape = x.shape[1:]

        # Check if the input image dimensions match the model's expected dimensions
        if input_shape != self.image_dimensions:
            raise ValueError(f"Input image dimensions {input_shape} do not match the model's expected dimensions {self.image_dimensions}")

        # Forward pass to get raw predictions
        confidence_batch = self.forward(x)

        confidence_arr = confidence_batch[0]

        if len(confidence_arr) == 1:
            confidence_score = confidence_arr[0]
            predicted_index = int(np.round(confidence_arr))
        else:
            predicted_index = np.argmax(confidence_arr, axis=-1)
            confidence_score = confidence_arr[predicted_index]

        # Retrieve the class name using index_to_class mapping
        class_name = self.index_to_class[predicted_index] if self.index_to_class is not None else None

        return confidence_score, class_name

    def add(self, layer):
        """
        Add a layer to the model.

        :param layer: Layer
            A layer object (e.g. Dense, Tanh).
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Perform a forward pass through all layers in the model.

        :param x: numpy array
            Input data.
        :return: numpy array
            Output after passing through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        """
        Perform a backward pass through all layers in the model (in reverse order).

        :param grad: numpy array
            Gradient of the loss with respect to the output.
        :param learning_rate: float
            Learning rate for updating the layers' parameters.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, data_gen, epochs, learning_rate):
        """
        Train the model using the provided data.

        :param data_gen: ImageDataGenerator
             Instance with training and validation(optional) data.
        :param epochs: Int
            Number of training epochs.
        :param learning_rate: float
            Learning rate for updating the layers' parameters.
        """
        self.image_dimensions = data_gen.target_size

        # Adjust image dimensions based on image mode
        if data_gen.image_mode == 'RGB':
            # Add a dimension of 3 at the start for RGB images
            self.image_dimensions = (3,) + self.image_dimensions
        elif data_gen.image_mode == 'L':
            # Add a dimension of 1 at the start for grayscale images
            self.image_dimensions = (1,) + self.image_dimensions

        self.index_to_class = data_gen.index_to_class
        self.class_indices = data_gen.class_indices
        has_validation_data = data_gen.val_directory is not None and data_gen.num_val_images > 0

        for e in range(epochs):

            total_error = 0  # Accumulated error over the epoch
            total_accuracy = 0  # Accumulated accuracy over the epoch
            batch_number = 0  # Current batch being iterated
            total_batches = len(data_gen)  # Total number of batches in the data generator

            for batch_images, batch_labels in data_gen.__iter__(mode='train'):
                batch_number += 1

                # Forward pass through the model
                output = self.forward(batch_images)

                # Compute the error for the current batch
                batch_error = self.loss_function(batch_labels, output)

                # Calculate the gradient of the loss function with respect to the output
                grad = self.loss_function_prime(batch_labels, output)

                # Perform the backward pass using the accumulated gradients
                self.backward(grad, learning_rate)

                # Update total error across all batches
                total_error += batch_error

                # Compute batch accuracy and accumulate it
                batch_accuracy = MetricTracker.calculate_accuracy(batch_labels, output)
                total_accuracy += batch_accuracy

                # Print progress for the current batch
                print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches} --- Batch Err={batch_error:.6f}, Batch Accuracy={batch_accuracy:.2%}', end='')

            # Calculate and print average error and accuracy for the epoch
            average_error = total_error / len(data_gen)
            average_accuracy = total_accuracy / len(data_gen)

            # Record the training metrics
            self.metric_tracker.record_train_metrics(average_error, average_accuracy)

            print(f'\rEpoch {e + 1}/{epochs}, Batch {batch_number}/{total_batches} --- Train Err={average_error:.6f}, Train Accuracy={average_accuracy:.2%}', end='')

            # Check if validation data is available
            if has_validation_data:
                # Perform validation on the validation data
                val_average_error, val_average_accuracy = self.validate(data_gen)

                # Record the validation metrics
                self.metric_tracker.record_val_metrics(val_average_error, val_average_accuracy)

                print(f', Val Err={val_average_error:.6f}, Val Accuracy={val_average_accuracy:.2%}')
            else:
                print()

        # Plot the metrics after training
        self.metric_tracker.plot_metrics()

    def validate(self, data_gen):
        """
        Perform validation on the validation data.

        :param data_gen: ImageDataGenerator
            Instance with validation data
        :return: tuple (validation_error, validation_accuracy)
            Average validation error and accuracy
        """
        val_total_error = 0  # Accumulated validation error
        val_total_accuracy = 0  # Accumulated validation accuracy
        val_total_batches = 0  # Total number of validation samples

        # Iterate through the validation data generator
        for val_batch_images, val_batch_labels in data_gen.__iter__(mode='val'):

            # Forward pass through the model
            output = self.forward(val_batch_images)

            # Calculate batch validation loss
            val_batch_error = self.loss_function(val_batch_labels, output)

            # Calculate batch validation accuracy
            val_batch_accuracy = MetricTracker.calculate_accuracy(val_batch_labels, output)

            # Accumulate the total validation error and accuracy
            val_total_error += val_batch_error
            val_total_accuracy += val_batch_accuracy

            val_total_batches += 1

        # Return average validation error and average validation accuracy
        return val_total_error / val_total_batches, val_total_accuracy / val_total_batches

    def save(self, name):
        """
        :param name: str
            The path where the model should be saved
        """

        # Get file path for saving the model
        current_dir = os.path.dirname(__file__)
        content_root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        save_dir = os.path.join(content_root_dir, 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, name)

        # Open a file for writing in HDF5 format
        with h5py.File(file_path, 'w') as f:

            # Save the model architecture
            architecture_group = f.create_group('architecture')

            # Get names of loss function and its prime
            architecture_group.attrs['loss_function'] = self.loss_function.__name__
            architecture_group.attrs['loss_function_prime'] = self.loss_function_prime.__name__

            # Save image dimensions if available
            if self.image_dimensions is not None:
                architecture_group.attrs['image_dimensions'] = self.image_dimensions

            # Convert dictionaries to lists of tuples and save as attributes
            if self.class_indices is not None:
                # Keys as strings (class names)
                f.create_dataset('class_indices_keys', data=np.array(list(self.class_indices.keys()), dtype='S'))
                # Values as integers (class indices)
                f.create_dataset('class_indices_values', data=np.array(list(self.class_indices.values()), dtype=int))

            if self.index_to_class is not None:
                # Keys as integers (class indices)
                f.create_dataset('index_to_class_keys', data=np.array(list(self.index_to_class.keys()), dtype=int))
                # Values as strings (class names)
                f.create_dataset('index_to_class_values', data=np.array(list(self.index_to_class.values()), dtype='S'))

            # Save each layer's configuration and weights
            for i, layer in enumerate(self.layers):

                # Create a group for each layer
                layer_group = architecture_group.create_group(f'layer_{i}')

                # Set an attribute for the layer group
                layer_group.attrs['class_name'] = layer.__class__.__name__

                # Save parameters for Dense layer
                if isinstance(layer, Dense):
                    # Save input and output sizes for Dense layer
                    layer_group.attrs['output_size'] = layer.output_size

                    # Save weights and bias
                    layer_group.create_dataset('weights', data=layer.weights)
                    layer_group.create_dataset('bias', data=layer.bias)


                # Save parameters for Convolution layer
                elif isinstance(layer, Convolution):

                    if layer.input_shape is not None:
                        layer_group.attrs['input_shape'] = layer.input_shape
                    if layer.output_shape is not None:
                        layer_group.attrs['output_shape'] = layer.output_shape

                    # Save kernel size (assumed to be square) and depth
                    layer_group.attrs['kernel_size'] = layer.kernels.shape[2]
                    layer_group.attrs['depth'] = layer.depth

                    # Save kernels and bias as datasets for convolutional layers
                    layer_group.create_dataset('kernels', data=layer.kernels)
                    layer_group.create_dataset('bias', data=layer.bias)

                # Save parameters for Pooling layer
                elif isinstance(layer, Pooling):
                    # Save pooling parameters
                    layer_group.attrs['pool_size'] = layer.pool_size
                    layer_group.attrs['stride'] = layer.stride
                    layer_group.attrs['mode'] = layer.mode

                # Check for specific activation layers
                if isinstance(layer, (ReLU, Sigmoid, Softmax, Tanh)):
                    layer_group.attrs['activation_function'] = layer.activation.__name__
                    layer_group.attrs['activation_function_prime'] = layer.activation_prime.__name__

            print(f'Model saved to {file_path}')
