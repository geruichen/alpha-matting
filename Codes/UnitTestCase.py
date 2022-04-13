import unittest
from Matte import calcMeanCovariance
from Matte import calcFB
from Matte import calAlpha
import numpy as np


class TestCalc(unittest.TestCase):

    def test_mean(self):
        p_m_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        p_m_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        p_m_3 = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        p_m = np.dstack((p_m_1, p_m_2, p_m_3))

        w_m = np.array([[1, 3, 5], [2, 4, 1], [1, 2, 2]])

        gt_m = np.array([[4.619, 5.3809, 5.38009]])

        m = calcMeanCovariance(p_m, w_m)

        np.testing.assert_allclose(gt_m, m[0], rtol=1e-03)
        print("Mean Test Case Passed")

    def test_cov(self):
        p_c_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        p_c_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        p_c_3 = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        p_c = np.dstack((p_c_1, p_c_2, p_c_3))

        w_c = np.array([[1, 3, 5], [2, 4, 1], [1, 2, 2]])

        gt_c = np.array([[5.5692, -5.5692, 2.4308],
                         [-5.5692, 5.5692, -2.4308],
                         [2.4308, -2.4308, 4.8073]])

        c = calcMeanCovariance(p_c, w_c)

        np.testing.assert_allclose(gt_c, c[1], rtol=1e-03)
        print("Covariance Test Case Passed")

    def test_fb(self):
        fcv = np.array([[1, 2, 3], [7, 5, 6], [4, 8, 9]])
        bcv = np.array([[1, 2, 3], [7, 5, 6], [4, 8, 9]])
        fm = np.array([[0.1], [0.2], [0.3]])
        bm = np.array([[0.1], [0.2], [0.3]])
        c = np.array([[0.2], [0.5], [0.3]])
        a = 0.5
        sc = 0.05

        fb = calcFB(fcv, bcv, fm, bm, a, c, sc)

        gt_fb = np.array([[0.19], [0.50], [0.29], [0.19], [0.50], [0.29]])

        np.testing.assert_allclose(gt_fb, fb, rtol=1e-01)
        print("F and B Test Case Passed")

    def test_a(self):
        f = np.array([[0.8], [0.6], [0.1]])
        b = np.array([[0.2], [0.5], [0.3]])
        c = np.array([[0.7], [0.8], [0.2]])

        a = calAlpha(f, b, c)

        gt_a = 0.85

        np.testing.assert_allclose(gt_a, a, rtol=1e-01)
        print("Alpha Test Case Passed")


if __name__ == '__main__':
    unittest.main()
