import os
import numpy as np
from PIL import Image
import random


class ImageDataGenerator:
    """
    Custom image data generator class to load and preprocess images from a directory.
    The directory structure is assumed to represent different classes, where each class
    is represented by a subdirectory containing images.

    The generator yields batches of images and corresponding labels, and supports
    options for shuffling, rescaling, and image augmentation.
    """

    # TODO: Allow users to specify whether augmentation or rescaling should occur during training or inference
    def __init__(self, directory, batch_size=32, target_size=(150, 150), image_mode='RGB', shuffle=True, rescale=None, augmentation=None):
        """
        Custom image data generator.

        :param directory: string
            Path to the dataset directory.
        :param batch_size: int
            Number of images per batch.
        :param target_size: tuple
            Size to resize images (width, height).
        :param image_mode: str
            'RGB' for color images, 'L' for grayscale.
        :param shuffle: bool
            Whether to shuffle the dataset after each epoch.
        :param rescale: float
            Rescaling factor for image normalization (e.g., 1/255 for [0,1] range).
        :param augmentation: function
            Applies augmentation to images.
        """
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_mode = image_mode
        self.shuffle = shuffle
        self.rescale = rescale
        self.augmentation = augmentation

        if not os.path.isdir(self.directory):
            raise ValueError(f"Directory {self.directory} does not exist.")

        # Collect all image paths and labels
        self.image_paths, self.labels = self._collect_image_paths_and_labels()

        # Number of images
        self.num_images = len(self.image_paths)

        if self.num_images == 0:
            raise ValueError("No images found in the directory.")

        # Shuffle the data initially
        if self.shuffle:
            self._shuffle_data()

    def _collect_image_paths_and_labels(self):
        """
        Collect image file paths and corresponding labels from the directory.
        Assumes that subdirectories represent classes.

        :return image_paths: list
            Image file paths.
        :return labels: list
            Labels corresponding to image paths.
        """
        image_paths = []
        labels = []

        # Only consider subdirectories as class names
        class_names = [d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, d))]

        # Handle case with exactly two classes
        if len(class_names) == 2:
            positive_class = 'positive' in class_names
            negative_class = 'negative' in class_names

            # Assign labels based on 'positive' and 'negative' names, or randomly otherwise
            if positive_class and negative_class:
                # Assign labels based on class names
                label_map = {'positive': 1, 'negative': 0}
            else:
                random.shuffle(class_names)
                label_map = {class_names[0]: 0, class_names[1]: 1}

            for class_name in class_names:
                class_dir = os.path.join(self.directory, class_name)
                if os.path.isdir(class_dir):
                    label = label_map[class_name.lower()]
                    for filename in os.listdir(class_dir):
                        if filename.endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(class_dir, filename))
                            labels.append(label)

        # Handle the case with more than 2 classes
        # TODO: add support for one hot encoding
        else:
            self.class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
            for class_name in class_names:
                class_dir = os.path.join(self.directory, class_name)
                if os.path.isdir(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(class_dir, filename))
                            labels.append(self.class_indices[class_name])

        return image_paths, labels

    def _shuffle_data(self):
        """
        Shuffle the image paths and corresponding labels to randomize the order of the data.
        """
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def _load_image(self, image_path):
        """
        Load an image and preprocess it.

        :param image_path: string
            Path to the image file.

        :return image: np array
            Preprocessed image array.
        """
        image = Image.open(image_path).convert(self.image_mode)
        image = image.resize(self.target_size)
        image = np.array(image)

        if self.rescale:
            image = image * self.rescale

        # Add a batch dimension (if grayscale, add channel dimension)
        if self.image_mode == 'L':
            image = np.expand_dims(image, axis=-1)

        # Rearrange the axes to have the depth/channels first: (channels, height, width)
        image = np.transpose(image, (2, 0, 1))

        return image

    def __iter__(self):
        """
        Iterator method to make the class iterable.

        :return self: iterator
            Returns the iterator object itself.
        """
        self.index = 0
        if self.shuffle:
            self._shuffle_data()
        return self

    # TODO: reset iterator at end of epoch & use lazy loading
    def __next__(self):
        """
        Fetch the next batch of images and labels.

        :return batch_images: np.array
            Batch of images.
        :return batch_labels: np.array
            Batch of corresponding labels.

        :raises StopIteration: If there are no more batches to fetch.
        """
        if self.index >= self.num_images:
            raise StopIteration

        end_index = min(self.index + self.batch_size, self.num_images)

        # Get the batch data
        batch_image_paths = self.image_paths[self.index:end_index]
        batch_labels = self.labels[self.index:end_index]

        batch_images = [self._load_image(image_path) for image_path in batch_image_paths]

        # Convert lists to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        # Apply augmentation if available
        if self.augmentation:
            batch_images = np.array([self.augmentation(image) for image in batch_images])

        self.index += self.batch_size
        return batch_images, batch_labels

    def __len__(self):
        """
        Get the total number of batches per epoch.

        :return: int
            Total number of batches.
        """
        return int(np.ceil(self.num_images / float(self.batch_size)))
