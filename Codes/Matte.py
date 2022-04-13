import numpy
import numpy as np
import cv2
import time
from skimage.metrics import structural_similarity as ssim
from fProcess import calcProcessing


def getFixed(rgb_image, mask):
    masked1 = rgb_image[:, :, 0] * mask
    masked2 = rgb_image[:, :, 1] * mask
    masked3 = rgb_image[:, :, 2] * mask
    masked = np.dstack((masked1, masked2, masked3))
    return masked


def main():
    input_image = cv2.imread("C19.png")   # Input Composite Image
    input_trimap = cv2.imread("T19.png")  # Input Trimap Image
    input_gt = cv2.imread("GT19.png")     # Input Ground Truth Image
    cv2.imshow('Alpha groundtruth', input_gt)
    input_trimap = input_trimap[:, :, 0] # Get single channel

    composite_image = np.asarray(input_image) / 255 # scalling the inputs
    trimap = np.asarray(input_trimap) / 255
    gt = np.asarray(input_gt) / 255
    dimensions = composite_image.shape

    f_trimap = (trimap == 1)
    b_trimap = (trimap == 0)
    u_trimap = ~f_trimap & ~b_trimap

    FG = getFixed(composite_image, f_trimap) # Get fixed Foreground and Background
    BG = getFixed(composite_image, b_trimap)
    UG = np.zeros((dimensions[0], dimensions[1]))
    UG[u_trimap] = 1

    alpha_init = np.zeros((dimensions[0], dimensions[1])) # Initiate Alpha
    alpha_init[f_trimap] = 1
    alpha_init[u_trimap] = numpy.NAN

    alpha_scan = np.argwhere(u_trimap == 1) # Unknown pixel positions
    r = alpha_scan[:, 0]
    c = alpha_scan[:, 1]

    window_size_init = 31 # GIVE ALL USER INPUTS HERE
    sigma_c = 0.05
    sigma_g = 10
    max_iteration = 200
    maxlikelihood_thresh = 0.01
    parameters = [window_size_init, sigma_c, sigma_g, max_iteration, maxlikelihood_thresh]

    start_time = time.time() # Pass all te parametrs and input for processing
    alpha_temp = calcProcessing(composite_image, r, c, FG, BG, UG, alpha_init, parameters)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    alpha_fin = np.dstack((alpha_temp, alpha_temp, alpha_temp)) # Convert alha into 3 channels
    cv2.imshow('Alpha from our own Algorithm',alpha_fin)
    extracted_output = composite_image * alpha_fin
    gt_output = composite_image * gt

    calc_PSNR = cv2.PSNR(extracted_output, gt_output)
    print(calc_PSNR)
    ssim_noise = ssim(gt[:, :, 0], alpha_fin[:, :, 0])
    print(ssim_noise)
    print("Computation Done")


if __name__ == '__main__':
    main()
    # IMAE INPUTS TO BE GIVEN AT LINE 18, 19, 20. PARAMETERS TO BE GIVEN AT LINE 46
