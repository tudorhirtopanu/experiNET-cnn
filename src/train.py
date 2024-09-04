# Layers
from src.layers.convolution import Convolution
from src.layers.dense import Dense
from src.layers.flatten import Flatten

# Activation Functions
from activations.softmax import Softmax
from activations.tanh import Tanh
from activations.relu import ReLU
from activations.sigmoid import Sigmoid

# Loss Functions
from utils.losses import (
    binary_cross_entropy,
    binary_cross_entropy_prime,
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    mse,
    mse_prime
)

from src.model.model import Model
from src.utils.image_data_generator import ImageDataGenerator

# Path to directory containing training images and validation images (optional)
train_dataset_path = "path/to/train/dataset"
val_dataset_path = "path/to/val/dataset"

# Create an image data generator
data_gen = ImageDataGenerator(
    train_directory=train_dataset_path,
    val_directory=val_dataset_path,
    batch_size=32,
    target_size=(28, 28),
    image_mode='L',
    shuffle=True,
    rescale=1 / 255,
    augmentation=None
)

# Create a model with a loss function and its prime
model = Model(binary_cross_entropy, binary_cross_entropy_prime)

# Add layers to the model
model.add(Convolution((1, 28, 28), 3, 3))
model.add(ReLU())
model.add(Flatten((3, 26, 26), (3 * 26 * 26,)))
model.add(Dense(3 * 26 * 26, 100))
model.add(ReLU())
model.add(Dense(100, 1))
model.add(Sigmoid())

# Train the model
EPOCHS = 30
LEARNING_RATE = 0.01
model.train(data_gen, EPOCHS, LEARNING_RATE)

# Save the model
model.save('model.h5')
