from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.exact_computation.utils import set_global_precision, mp_cross
from uxarray.grid.coordinates import normalize_in_place
from uxarray.grid.utils import cross_fma, angle_of_2_vectors


def fma_cross_exp():
    # Generate 2 vectors in Cartesian coordinates that are closed to others in mpfr
    set_global_precision(53)
    v1 = np.array([mpfr('1.0'), mpfr('1.0000000000000010'), mpfr('1.0')])
    v2 = np.array([mpfr('1.0'), mpfr('1.0000000000000001'), mpfr('1.0000000000000001')])
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    # Exact result of cross product in mpfr format
    y1z2_y2z1 = fmms(y1, z2, y2, z1)

    # Add disturbance to the v1 and v2
    u = mpfr(np.finfo(np.float64).eps)
    w1_disturbed = np.array([x1 + mpfr('3.0')*u, y1 + mpfr('3.0')*u, z1 + mpfr('3.0')*u])
    w2_disturbed = np.array([x2 + mpfr('3.0')*u, y2 + mpfr('3.0')*u, z2 + mpfr('3.0')*u])
    y1_disturbed, z1_disturbed = w1_disturbed[1], w1_disturbed[2]
    y2_disturbed, z2_disturbed = w2_disturbed[1], w2_disturbed[2]

    # Now calculate the cross product using FMA
    # If beta is bounded by (1+3u)^2, then the relative error is bounded by （1+3u）^2 - 1
    w1w2_disturbed = fmms(y1_disturbed, z2_disturbed, y2_disturbed, z1_disturbed)
    rel_err_bound = mpfr('6.0') * u + mpfr('9.0') * u ** 2
    abs_err_bound = y1z2_y2z1 * (mpfr('6.0') * u + mpfr('9.0') * u ** 2)

    # Calulate the relative error
    rel_err = abs((w1w2_disturbed - y1z2_y2z1) / y1z2_y2z1)
    abs_err = abs(w1w2_disturbed - y1z2_y2z1)
    print("Relative error: ", rel_err)
    print("Beta Relative error bound: ", rel_err_bound)
    print("Absolute error: ", abs_err)
    print("Absolute error bound: ", abs_err_bound)

def naive_cross_exp():
    set_global_precision(53)
    gca_pt1_1 = normalize_in_place(np.array([mpfr('0.5'), mpfr('0.5'), mpfr('0.5')]))
    gca_pt1_2 = normalize_in_place(
        np.array([mpfr('0.500000000000001'), mpfr('0.499999999999999'), mpfr('0.499999999999999')]))
    gca_pt2_1 = normalize_in_place(
        np.array([mpfr('0.500000000000002'), mpfr('0.500000000000002'), mpfr('0.499999999999999')]))
    gca_pt2_2 = normalize_in_place(
        np.array([mpfr('0.499999999999998'), mpfr('0.500000000000001'), mpfr('0.499999999999999')]))
    v1 = mp_cross(gca_pt1_1, gca_pt1_2)
    v2 = mp_cross(gca_pt2_1, gca_pt2_2)
    x = mp_cross(v1, v2)

    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    v1_float = np.cross(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = np.cross(gca_pt2_1_float, gca_pt2_2_float)

    x_float = np.cross(v1_float, v2_float)
    np.set_printoptions(precision=16)
    x_float_str = [str(x_float[0]), str(x_float[1]), str(x_float[2])]
    x_float_mpfr = [mpfr(x_float_str[0]), mpfr(x_float_str[1]), mpfr(x_float_str[2])]
    # Now calculate the relative error

    rel_err_x = abs((mpfr(x_float_str[0]) - x[0]) / x[0])
    rel_err_y = abs((mpfr(x_float_str[1]) - x[1]) / x[1])
    rel_err_z = abs((mpfr(x_float_str[2]) - x[2]) / x[2])

    # Get the max relative error
    rel_err = max(rel_err_x, rel_err_y, rel_err_z)

    print("naive real live example relative error: ", rel_err)

def closed_gca_gca_intersection():
    set_global_precision(53)
    gca_pt1_1 = normalize_in_place(np.array([mpfr('0.5'), mpfr('0.5'), mpfr('0.5')]))
    gca_pt1_2 = normalize_in_place(np.array([mpfr('0.500000000000001'), mpfr('0.499999999999999'), mpfr('0.499999999999999')]))
    gca_pt2_1 = normalize_in_place(np.array([mpfr('0.500000000000002'), mpfr('0.500000000000002'), mpfr('0.499999999999999')]))
    gca_pt2_2 = normalize_in_place(np.array([mpfr('0.499999999999998'), mpfr('0.500000000000001'), mpfr('0.499999999999999')]))
    v1 = mp_cross(gca_pt1_1, gca_pt1_2)
    v2 = mp_cross(gca_pt2_1, gca_pt2_2)
    x = mp_cross(v1, v2)

    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    v1_float = cross_fma(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = cross_fma(gca_pt2_1_float, gca_pt2_2_float)

    x_float = cross_fma(v1_float, v2_float)
    np.set_printoptions(precision=16)
    x_float_str = [str(x_float[0]), str(x_float[1]), str(x_float[2])]
    x_float_mpfr = [mpfr(x_float_str[0]), mpfr(x_float_str[1]), mpfr(x_float_str[2])]
    # Now calculate the relative error

    rel_err_x = abs((mpfr(x_float_str[0]) - x[0]) / x[0])
    rel_err_y = abs((mpfr(x_float_str[1]) - x[1]) / x[1])
    rel_err_z = abs((mpfr(x_float_str[2]) - x[2]) / x[2])

    # Get the max relative error
    rel_err = max(rel_err_x, rel_err_y, rel_err_z)

    print("real live example relative error: ", rel_err)

def long_gca_gca_intersection():
    set_global_precision(53)
    gca_pt1_1 = normalize_in_place(np.array([mpfr('0.5'), mpfr('0.5'), mpfr('0.5')]))
    gca_pt1_2 = normalize_in_place(np.array([mpfr('-0.45'), mpfr('-0.5'), mpfr('-0.5')]))
    angle = angle_of_2_vectors(gca_pt1_1, gca_pt1_2)
    u = mpfr(np.finfo(np.float64).eps)
    disturbance = 5 * u
    gca_pt2_1 = normalize_in_place(np.array([mpfr('0.5'), mpfr('0.5'), mpfr('0.5') - disturbance]))
    gca_pt2_2 = normalize_in_place(np.array([mpfr('0.5'), mpfr('0.5'), mpfr('0.5') + disturbance]))

    v1 = mp_cross(gca_pt1_1, gca_pt1_2)
    v2 = mp_cross(gca_pt2_1, gca_pt2_2)
    x = mp_cross(v1, v2)

    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    v1_float = cross_fma(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = cross_fma(gca_pt2_1_float, gca_pt2_2_float)

    x_float = cross_fma(v1_float, v2_float)
    np.set_printoptions(precision=16)
    x_float_str = [str(x_float[0]), str(x_float[1]), str(x_float[2])]
    x_float_mpfr = [mpfr(x_float_str[0]), mpfr(x_float_str[1]), mpfr(x_float_str[2])]
    # Now calculate the relative error

    rel_err_x = abs((mpfr(x_float_str[0]) - x[0]) / x[0])
    rel_err_y = abs((mpfr(x_float_str[1]) - x[1]) / x[1])
    rel_err_z = abs((mpfr(x_float_str[2]) - x[2]) / x[2])

    # Get the max relative error
    rel_err = max(rel_err_x, rel_err_y, rel_err_z)

    print("Long GCA example relative error: ", rel_err)


def calculate_rel_error():
    u = mpfr(np.finfo(np.float64).eps)
    term_1 = (mpfr('1.0') + mpfr('9.0') * u) / (mpfr('1.0') + mpfr('2.5') * u)
    term_2 = (mpfr('1.0') + u - 2 * u ** 2)
    rel_err = abs(term_1 * term_2 - mpfr('1.0'))
    print("Final relative error bound: ", rel_err)


if __name__ == "__main__":
    # fma_cross_exp()
    naive_cross_exp()
    # calculate_rel_error()
    # closed_gca_gca_intersection()
    long_gca_gca_intersection()