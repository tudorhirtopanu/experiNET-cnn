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

    def __init__(self, train_directory, val_directory=None, batch_size=32, target_size=(150, 150), image_mode='RGB', shuffle=True, rescale=None, augmentation=None):
        """
        Custom image data generator.

        :param train_directory: string
            Path to the train dataset directory.
        :param val_directory: string
            Path to the validation dataset directory.
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
        self.train_directory = train_directory
        self.val_directory = val_directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_mode = image_mode
        self.shuffle = shuffle
        self.rescale = rescale
        self.augmentation = augmentation
        self.mode = 'train'

        if not os.path.isdir(self.train_directory):
            raise ValueError(f"Directory {self.train_directory} does not exist.")

        # Collect all image paths and labels
        self.train_image_paths, self.train_labels = self._collect_image_paths_and_labels(self.train_directory)

        # Number of images
        self.num_train_images = len(self.train_image_paths)

        # Load validation data if provided
        if self.val_directory:

            if not os.path.isdir(self.val_directory):
                raise ValueError(f"Directory {self.val_directory} does not exist.")

            self.val_image_paths, self.val_labels = self._collect_image_paths_and_labels(self.val_directory)
            self.num_val_images = len(self.val_image_paths)

        else:
            self.val_image_paths, self.val_labels = None, None
            self.num_val_images = 0

        if self.num_train_images == 0:
            raise ValueError("No images found in the directory.")

        if self.val_directory and self.num_val_images == 0:
            raise ValueError("No images found in the validation directory.")

        # Shuffle the data initially
        if self.shuffle:
            self._shuffle_data()

    def _collect_image_paths_and_labels(self, directory):
        image_paths = []
        labels = []

        # Only consider subdirectories as class names
        class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        self.class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.index_to_class = {idx: class_name for class_name, idx in self.class_indices.items()}
        num_classes = len(class_names)

        for class_name in class_names:
            class_dir = os.path.join(directory, class_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, filename))

                    # Condition to one-hot encode based on number of classes
                    if num_classes > 2:
                        # One-hot encode the label for multi-class
                        label = np.zeros(num_classes)
                        label[self.class_indices[class_name]] = 1
                    else:
                        # Binary classification (use 0 or 1)
                        label = np.array([self.class_indices[class_name]])

                    labels.append(label)

        return image_paths, labels

    def _shuffle_data(self):
        """
        Shuffle the image paths and corresponding labels to randomize the order of the data.
        """
        combined = list(zip(self.train_image_paths, self.train_labels))
        random.shuffle(combined)
        self.train_image_paths, self.train_labels = zip(*combined)

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

    def __iter__(self, mode='train'):
        """
        Iterator method to make the class iterable.

        :return self: iterator
            Returns the iterator object itself.
        """
        self.index = 0
        self.mode = mode
        if self.shuffle and self.mode == 'train':
            self._shuffle_data()
        return self

    # TODO: use lazy loading
    def __next__(self):
        """
        Fetch the next batch of images and labels.

        :return batch_images: np.array
            Batch of images.
        :return batch_labels: np.array
            Batch of corresponding labels.

        :raises StopIteration: If there are no more batches to fetch.
        """

        if self.mode == 'train':
            image_paths, labels, num_images = self.train_image_paths, self.train_labels, self.num_train_images
        else:
            image_paths, labels, num_images = self.val_image_paths, self.val_labels, self.num_val_images

        if self.index >= self.num_train_images:
            self.index = 0
            raise StopIteration

        end_index = min(self.index + self.batch_size, num_images)

        # Get the batch data
        batch_image_paths = image_paths[self.index:end_index]
        batch_labels = labels[self.index:end_index]

        batch_images = [self._load_image(image_path) for image_path in batch_image_paths]

        # Convert lists to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        # Apply augmentation if available
        if self.mode == 'train' and self.augmentation:
            batch_images = np.array([self.augmentation(image) for image in batch_images])

        self.index += self.batch_size
        return batch_images, batch_labels

    def __len__(self):
        """
        Get the total number of batches per epoch.

        :return: int
            Total number of batches.
        """
        #TODO: Make it context aware for when to use train or val
        if self.mode == 'train':
            return int(np.ceil(self.num_train_images / float(self.batch_size)))
        elif self.mode == 'val':
            return int(np.ceil(self.num_val_images / float(self.batch_size)))
        else:
            raise ValueError("Invalid mode. Use 'train' or 'val'.")

