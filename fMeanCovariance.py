import numpy as np


def calcMeanCovariance(pixel_input, weight_input):
    A = weight_input.shape
    W = np.sum(weight_input)
    avg = np.array([
        [np.sum(np.multiply(pixel_input[:, :, 0], weight_input)) / W,
         np.sum(np.multiply(pixel_input[:, :, 1], weight_input)) / W,
         np.sum(np.multiply(pixel_input[:, :, 2], weight_input)) / W]])
    covar = np.zeros((3, 3))
    for i in range(A[0]):
        for j in range(A[1]):
            if weight_input[i, j] != 0:
                pixel_single = np.array([pixel_input[i, j, :]])
                a = pixel_single - avg
                b = np.transpose(a)
                Mean = np.dot(b, a)
                Y = weight_input[i, j] * Mean
                covar = covar + Y
    covar = covar / W
    return avg, covar
