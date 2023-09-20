from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.exact_computation.utils import set_global_precision, mp_cross


def fma_cross_exp():
    # Generate 2 vectors in Cartesian coordinates that are closed to others in mpfr
    set_global_precision(128)
    w1 = np.array([mpfr('1.0'), mpfr('1.0000000000000010'), mpfr('1.0')])
    w2 = np.array([mpfr('0.5'), mpfr('1.0000000000000001'), mpfr('1.0000000000000001')])
    x1, y1, z1 = w1
    x2, y2, z2 = w2

    # Exact result of cross product in mpfr format
    y1z2_y2z1 = fmms(y1, z2, y2, z1)

    # Add disturbance to the w1 and w2
    u = mpfr(np.finfo(np.float64).eps)
    w1_disturbed = np.array([x1 + mpfr('3.0')*u, y1 + mpfr('3.0')*u, z1 + mpfr('3.0')*u])
    w2_disturbed = np.array([x2 + mpfr('3.0')*u, y2 + mpfr('3.0')*u, z2 + mpfr('3.0')*u])
    y1_disturbed, z1_disturbed = w1_disturbed[1], w1_disturbed[2]
    y2_disturbed, z2_disturbed = w2_disturbed[1], w2_disturbed[2]

    # Now calculate the cross product using FMA
    # If beta is bounded by (1+3u)^2, then the relative error is bounded by （1+3u）^2 - 1
    w1w2_disturbed = fmms(y1_disturbed, z2_disturbed, y2_disturbed, z1_disturbed)
    beta_rel_err = mpfr('6.0') * u + mpfr('9.0') * u**2

    # Calulate the relative error
    rel_err = abs((w1w2_disturbed - y1z2_y2z1) / y1z2_y2z1)
    abs_err = abs(w1w2_disturbed - y1z2_y2z1)
    print("Relative error: ", rel_err)
    print("Beta Relative error bound: ", beta_rel_err)
    print("Absolute error: ", abs_err)





if __name__ == "__main__":
    fma_cross_exp()