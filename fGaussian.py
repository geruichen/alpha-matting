import cv2
import numpy as np


def getSquareGaussian(dimension, sigma):
    gaussTemp = cv2.getGaussianKernel(dimension, sigma)
    kernel = np.dot(gaussTemp, gaussTemp.T)
    return kernel


def calcGaussFall(r_3, c_3, gaus, full_image, wind):
    s1 = gaus.shape
    s2 = full_image.shape
    s3 = wind.shape
    gaus1 = gaus
    z1 = s1[0] - s3[0]
    z2 = s1[1] - s3[1]
    if r_3 <= np.floor(s1[0] / 2):
        gaus1 = gaus[z1:, :]

    if r_3 >= s2[0] - np.floor(s1[0] / 2):
        gaus1 = gaus[:-z1, :]

    s1 = gaus1.shape
    gaus2 = gaus1
    if c_3 <= np.floor(s1[1] / 2):
        gaus2 = gaus1[:, z2:]

    if c_3 >= s2[1] - np.floor(s1[1] / 2):
        gaus2 = gaus1[:, :-z2]
    return gaus2
