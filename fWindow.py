import math
import numpy as np


def checkWindow(fg_1, bg_1, ug_1, ag_1, r_2, c_2, wind_size):
    flag_w = 0
    while flag_w == 0:
        w_fg_init = calcWindow(fg_1, r_2, c_2, wind_size)
        w_bg_init = calcWindow(bg_1, r_2, c_2, wind_size)
        w_ug_init = calcWindow(ug_1, r_2, c_2, wind_size)
        w_ag_init = calcWindow(ag_1, r_2, c_2, wind_size)

        nz_fg_pos = (w_fg_init[:, :, 0] == 0) & (w_fg_init[:, :, 1] == 0) & (w_fg_init[:, :, 2] == 0)
        nz_fg = np.size(nz_fg_pos) - np.count_nonzero(nz_fg_pos)
        nz_bg_pos = (w_bg_init[:, :, 0] == 0) & (w_bg_init[:, :, 1] == 0) & (w_bg_init[:, :, 2] == 0)
        nz_bg = np.size(nz_bg_pos) - np.count_nonzero(nz_bg_pos)

        if nz_fg < 40 or nz_bg < 40:
            wind_size = wind_size + 2
        else:
            flag_w = 1
    return w_fg_init, w_bg_init, w_ug_init, w_ag_init, wind_size


def calcWindow(og, w_r, w_c, siz):
    M = og.shape
    window_half = math.floor(siz / 2)

    rmin = max(0, w_r - window_half - 1)
    rmax = min(M[0] - 1, w_r + window_half)
    cmin = max(0, w_c - window_half - 1)
    cmax = min(M[1] - 1, w_c + window_half)

    if len(M) == 3:
        window_output = og[rmin:rmax, cmin:cmax, :]
    else:
        window_output = og[rmin:rmax, cmin:cmax]
    return window_output
