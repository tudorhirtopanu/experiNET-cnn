# experiNET-CNN

A convolutional neural network library built from scratch using only **numpy**, without any deep learning frameworks. This library implements core components like convolutional, dense, pooling layers, and supports training and inference on datasets with batch processing. The models can also be saved and loaded using **h5py**-based custom functions. Graphs are generated with accuracy and error per epoch for training data as well as validation data (if provided).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Predicting with the Model](#predicting-with-the-model)
- [Future Features](#future-features)
- [License](#license)

## Features

- **Custom CNN Layers**: Convolutional, Dense, Pooling, Activation (e.g., ReLU, Sigmoid, etc.).
- **Batch Processing**: Dynamically loads images from directories based on batches, keeping memory usage low.
- **Data Handling**: Handles data input from a directory structure where each subdirectory represents a class.
- **Training and Prediction**: Train models on datasets and use them for inference.
- **Model Saving & Loading**: Save and load models using custom h5py functions.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/tudorhirtopanu/experiNET-cnn.git
cd experiNET-cnn
pip install h5py numpy matplotlib
```

## Usage

### Training the Model

To train your model, follow these steps:

1. **Specify Dataset Paths**:
   - You must provide the path to the directory containing the training images, structured with subdirectories as classes. Optionally, you can also provide a validation dataset path for evaluation during training.

2. **Image Data Generator Parameters**:
   - The `ImageDataGenerator` is used to preprocess and load data in batches, keeping memory usage low. You need to specify parameters like:
     - `batch_size`: Number of samples per gradient update.
     - `target_size`: Tuple specifying the target height and width of the images.
     - `image_mode`: 'RGB' for color images or 'L' for black-and-white images.
     - `shuffle`: Whether to shuffle the dataset.
     - `rescale`: Rescale factor to apply to the pixel values.
   - **Optional**: You can pass a validation directory for validation data.
   - **Note**: Augmentation functionality (e.g., rotation, flipping) is not yet fully supported, but will be added in future versions.

3. **Loss Function**:
   - When initializing the model, you must specify an appropriate loss function and its corresponding prime (derivative) function. The options include:
     - `binary_cross_entropy` and `binary_cross_entropy_prime` for binary classification.
     - `categorical_cross_entropy` and `categorical_cross_entropy_prime` for multi-class classification.
     - `mse` (mean squared error) and `mse_prime` for regression tasks.

4. **Adding Layers**:

   - **Convolutional Layer**  
     A convolutional layer applies filters to the input to extract spatial features.  
     **Params:**
     - `kernel_size`: Size of the filter to apply (e.g. `3` for a 3x3 filter).
     - `depth`: Number of kernels/filters.
     - `activation`: Activation function to apply (`relu`, `sigmoid`, `tanh`, `softmax`).

   - **Dense Layer**  
     A fully connected layer that operates on flattened input data, typically used for classification or regression tasks.  
     **Params:**
     - `output_size`: Number of neurons in the dense layer.
     - `activation`: Activation function to apply (`relu`, `sigmoid`, `tanh`, `softmax`).

   - **Pooling Layer**  
     A pooling layer down-samples the feature maps, reducing the spatial dimensions of the data.  
     **Params:**
     - `pool_size`: Tuple specifying the height and width of the pooling window (default is `(2, 2)`).
     - `stride`: The stride of the pooling window (default is `2`).
     - `mode`: The pooling operation to apply (`'max'`, `'min'`, or `'average'`). By default, it is `'max'`.

   - **Flatten Layer**  
     A layer that reshapes the multi-dimensional input into a 2D array, preparing the data for fully connected layers.  
     **Params:**  
     *(None)* – this layer does not require any parameters.

5. **Training Configuration**:
   - Specify the number of `EPOCHS` and the `LEARNING_RATE`.
   - After setting the model layers and loss function, call the `train` method to start training.
  
6. **Saving the Model**:
   - Call the `save` method and specify the model name.

### Predicting with the Model

To use the model for prediction on a new image, follow these steps:

1. **Specify Image Path**:  
   You need to provide the path to a singular image file that the model will use for prediction.
   
2. **Specify Model Name**:  
   Ensure you specify the name of the trained model file (e.g. model.h5) that you want to load.

## Future Features

- Add support for data augmentation during training.
- Implement additional loss functions and optimizers.
- Include more activation functions (e.g., Leaky ReLU, ELU).
- Add more metrics for performance evaluation and enable saving graphs
- Provide an in depth model evaluation

## License

This project is licensed under the [MIT License](./LICENSE). You are free to use, modify, and distribute this software as long as you include the original license. See the `LICENSE` file for more details.


