import numpy as np


def pad_input(input_array, kernel_shape, mode="valid"):
    """
    Pad the input array based on the specified mode.

    :param input_array: 2D numpy array
    :param kernel_shape: Tuple (height, width) of the kernel
    :param mode: String, 'valid', or 'full'
    :return: Padded 2D numpy array
    """

    kernel_height, kernel_width = kernel_shape

    if mode == "full":
        pad_height = kernel_height - 1
        pad_width = kernel_width - 1
        padded_input = np.pad(input_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                              constant_values=0)
    elif mode == "valid":
        padded_input = input_array
    else:
        raise ValueError(f"Invalid mode '{mode}'. Mode should be 'valid', or 'full'.")

    return padded_input

def correlate2D(input_array, kernel, mode="valid"):
    """
    Perform 2D cross-correlation between a 2d input array and a kernel.

    :param input_array: 2D numpy array
    :param kernel: 2D numpy array
    :param mode: String, 'valid', 'full', or 'same'
    :return: 2D numpy array (result of the correlation)
    """

    # Get kernel dimensions
    kernel_height, kernel_width = kernel.shape

    # Pad the input and get the dimensions
    padded_input = pad_input(input_array, (kernel_height, kernel_width), mode)
    padded_height, padded_width = padded_input.shape

    # Get output dimensions
    output_height = padded_height - kernel_height + 1
    output_width = padded_width - kernel_width + 1

    # Initialise output array
    output = np.zeros((output_height, output_width))

    # Perform 2D correlation
    for i in range(output_height):
        for j in range(output_width):
            sub_matrix = padded_input[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(sub_matrix * kernel)

    return output

def convolve2D(input_array, kernel, mode="valid"):
    """
    Perform 2D convolution between a 2D input array and a kernel by flipping the kernel and calling correlate2D.

    :param input_array: 2D numpy array
    :param kernel: 2D numpy array
    :param mode: String, 'valid' or 'full'
    :return: 2D numpy array (result of the convolution)
    """

    flipped_kernel = np.flip(np.flip(kernel, 0), 1)
    return correlate2D(input_array, flipped_kernel, mode)
