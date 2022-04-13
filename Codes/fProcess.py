import numpy as np
from fWindow import checkWindow
import fGaussian
from fMeanCovariance import calcMeanCovariance
from fFBALikelihood import getmaxLike
from tqdm import tqdm


def calcProcessing(comp_image, r_1, c_1, fg, bg, unknown, alpha, input_parameters):
    N_init = input_parameters[0]
    sig_c = input_parameters[1]
    sig_g = input_parameters[2]
    max_ite = input_parameters[3]
    mll_thr = input_parameters[4]

    for i in tqdm(range(r_1.size),desc='progress',leave=True):
        main_pixel = [comp_image[r_1[i], c_1[i]]]
        main_pixel = np.transpose(main_pixel)
        wind_fg, wind_bg, wind_unknown, wind_alpha, N = checkWindow(fg, bg, unknown, alpha, r_1[i], c_1[i], N_init)

        g_init = fGaussian.getSquareGaussian(N, sig_g)
        g_init = g_init / np.max(g_init)
        g = fGaussian.calcGaussFall(r_1[i], c_1[i], g_init, alpha, wind_alpha)
        wi_fg = np.multiply(np.square(wind_alpha), g)
        wi_fg[np.isnan(wi_fg)] = 0
        wi_bg = np.multiply(np.square(1 - wind_alpha), g)
        wi_bg[np.isnan(wi_bg)] = 0
        mean_f, cov_f = calcMeanCovariance(wind_fg, wi_fg)
        mean_b, cov_b = calcMeanCovariance(wind_bg, wi_bg)
        f, b, a = getmaxLike(max_ite, mll_thr, mean_f, cov_f, mean_b, cov_b, main_pixel, wind_alpha, wind_unknown,
                             sig_c)
        # if a < 0.2:
        #    a = 0
        # if a > 0.8:
        #    a = 1
        alpha[r_1[i], c_1[i]] = a
    return alpha
