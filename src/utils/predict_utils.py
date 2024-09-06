import numpy as np
from PIL import Image


def preprocess_and_normalize(image, image_dimension=(28, 28)):
    """
    Normalize and prepare an image (grayscale or RGB) to be passed through the model.

    :param image: numpy array
        A 2D numpy array representing a grayscale image or a 3D numpy array representing an RGB image.
    :param image_dimension: tuple of (height, width)
        Tuple representing height and width that the image will be resized to
    :return: numpy array
        A 4D numpy array with shape (1, depth, height, width) ready to be passed into the model.
        Depth is 1 for grayscale images and 3 for RGB images.
    """
    # Check if the image is 2D (grayscale) or 3D (RGB)
    if image.ndim == 2:
        # Grayscale image
        is_grayscale = True
        depth = 1
    elif image.ndim == 3 and image.shape[2] == 3:
        # RGB image
        is_grayscale = False
        depth = 3
    else:
        raise ValueError("Input image must be either a 2D array (grayscale) or a 3D array (RGB).")

    # Resize the image
    dimensions = image_dimension
    if is_grayscale:
        # Resize grayscale image
        image = Image.fromarray(image).resize(dimensions)
        image = np.array(image)
    else:
        # Resize RGB image
        image = Image.fromarray(image).resize(dimensions)
        image = np.array(image)

    # Normalize pixel values
    image = image.astype('float32') / 255.0

    # Add batch dimension and depth dimension to make it 4D: (1, depth, height, width)
    if is_grayscale:
        image = np.expand_dims(image, axis=(0, 1))  # (1, 1, height, width)
    else:
        image = np.expand_dims(image, axis=0)  # (1, height, width, depth)
        image = np.moveaxis(image, -1, 1)  # (1, depth, height, width)

    return image


def load_image(image_path):
    """
    Load an image from a file path and convert it to a numpy array.

    :param image_path: str
        Path to the image file.
    :return: numpy array
        Loaded image as a numpy array in grayscale.
    """
    # Load the image using PIL
    img = Image.open(image_path)

    # Convert the image to a numpy array
    img_array = np.array(img)

    return img_array

