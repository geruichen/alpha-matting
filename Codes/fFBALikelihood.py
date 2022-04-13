import numpy as np


def calAlpha(f_in, b_in, c_in_2):
    z1 = (c_in_2 - b_in)
    z2 = (f_in - b_in)
    z = (z1[0] * z2[0]) + (z1[1] * z2[1]) + (z1[2] * z2[2])
    a_temp = z / np.linalg.norm((f_in - b_in)) ** 2
    alpha = max(0, min(1, a_temp))
    return alpha


def calcFB(fcv, bcv, fm, bm, a_pixel, c_pixel, s_c):
    if np.linalg.det(fcv) == 0:
        f_cv_inv = np.linalg.pinv(fcv)
    else:
        f_cv_inv = np.linalg.inv(fcv)

    if np.linalg.det(bcv) == 0:
        b_cv_inv = np.linalg.pinv(bcv)
    else:
        b_cv_inv = np.linalg.inv(bcv)

    i = np.eye(3)
    t1 = f_cv_inv + (i * (a_pixel ** 2 / s_c ** 2))
    t2 = (i * a_pixel * (1 - a_pixel)) / s_c ** 2
    t3 = (i * a_pixel * (1 - a_pixel)) / s_c ** 2
    t4 = b_cv_inv + i * (((1 - a_pixel) / s_c) ** 2)

    t5 = np.dot(f_cv_inv, fm) + ((c_pixel * a_pixel) / s_c ** 2)
    t6 = np.dot(b_cv_inv, bm) + ((c_pixel * (1 - a_pixel)) / s_c ** 2)

    rhs = np.concatenate((t5, t6), axis=0)
    lhs_1 = np.concatenate((t1, t3), axis=0)
    lhs_2 = np.concatenate((t2, t4), axis=0)
    lhs = np.concatenate((lhs_1, lhs_2), axis=1)

    if np.linalg.det(lhs) == 0:
        F_B = np.dot(np.linalg.pinv(lhs), rhs)
    else:
        F_B = np.dot(np.linalg.inv(lhs), rhs)

    return F_B


def getmaxLike(max_it, mll_thresh, f_m, f_cv, b_m, b_cv, c_rgb, a_wind, u_wind, sig_c):
    iteration = 1
    mll_dif = 10
    b_m = np.transpose(b_m)
    f_m = np.transpose(f_m)
    while mll_dif > mll_thresh and iteration < max_it:
        if iteration == 1:
            count = np.count_nonzero(u_wind)
            a_temp = np.nansum(a_wind) / count

            fb_fin = calcFB(f_cv, b_cv, f_m, b_m, a_temp, c_rgb, sig_c)
            f_fin = fb_fin[0:3]
            b_fin = fb_fin[3:6]
        else:
            a_fin = calAlpha(f_fin, b_fin, c_rgb)
            fb_fin = calcFB(f_cv, b_cv, f_m, b_m, a_fin, c_rgb, sig_c)
            f_fin = fb_fin[0:3]
            b_fin = fb_fin[3:6]

        if iteration > 1:
            if np.linalg.det(f_cv) == 0:
                f_cv_inv = np.linalg.pinv(f_cv)
            else:
                f_cv_inv = np.linalg.inv(f_cv)

            if np.linalg.det(b_cv) == 0:
                b_cv_inv = np.linalg.pinv(b_cv)
            else:
                b_cv_inv = np.linalg.inv(b_cv)

            llf = np.matmul(np.matmul(-np.transpose(f_fin - f_m), f_cv_inv), (f_fin - f_m)) / 2
            llb = np.matmul(np.matmul(-np.transpose(b_fin - b_m), b_cv_inv), (b_fin - b_m)) / 2
            llc_fba = - (np.linalg.norm(c_rgb - (a_fin * f_fin) - (1 - a_fin) * b_fin)) ** 2 / (sig_c ** 2)
            mll = llc_fba + llb + llf

        if iteration == 2:
            mll_prev = mll

        if iteration > 2:
            mll_dif = abs(mll - mll_prev)
            mll_prev = mll

        iteration = iteration + 1

    return f_fin, b_fin, a_fin
