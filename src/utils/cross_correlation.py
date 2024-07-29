import numpy as np

# TODO: Add padding options


def correlate2D(input_array, kernel):
    """
    Perform 2D cross-correlation between a 2d input array and a kernel.

    :param input_array: 2D numpy array
    :param kernel: 2D numpy array
    :return: 2D numpy array (result of the correlation)
    """

    # Get input array and kernel dimensions
    input_height, input_width = input_array.shape
    kernel_height, kernel_width = kernel.shape

    # Get output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    # Initialise output array
    output = np.zeros((output_height, output_width))

    # Perform 2D correlation
    for i in range(output_height):
        for j in range(output_width):
            sub_matrix = input_array[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(sub_matrix * kernel)

    return output
