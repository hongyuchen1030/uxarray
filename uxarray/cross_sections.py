
import gmpy2
import numpy as np
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, ERROR_TOLERANCE
from uxarray.multi_precision_helpers import mp_dot, mp_cross, mp_norm, is_mpfr_array, precision_bits_to_decimal_digits
from uxarray.helpers import fma_cross, gram_schmidt, node_xyz_to_lonlat_rad, point_within_GCR, normalize_in_place


def get_GCA_GCA_intersections(gcr1_cart, gcr2_cart):
    # TODO: Reformat and cleanup
    """Get the intersection point(s) of two Great Circle Arcs. Overloaded with the mpfr version.

    Parameters
    ----------
    gcr1_cart : np.ndarray
        Cartesian coordinates of the first GCR
    gcr2_cart : np.ndarray
        Cartesian coordinates of the second GCR

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s)
    """
    # Check if the two GCRs are in the cartesian format (size of three)
    if gcr1_cart.shape[1] != 3 or gcr2_cart.shape[1] != 3:
        raise ValueError(
            "The two GCRs must be in the cartesian[x, y, z] format")
    w0, w1 = gcr1_cart
    v0, v1 = gcr2_cart
    v1_lonlat = node_xyz_to_lonlat_rad(v1)

    # Check if the two GCRs are in the mpfr format (contains type of mpfr)
    if is_mpfr_array(gcr1_cart) or is_mpfr_array(gcr2_cart):
        # The two GCRs are in the mpfr format
        w0w1_norm = mp_cross(w0, w1)
        v0v1_norm = mp_cross(v0, v1)

        cross_norms = mp_cross(w0w1_norm, v0v1_norm)

        if all(gmpy2.cmp(arr_val, gmpy2.mpfr('0')) == 0 for arr_val in cross_norms):
            return np.array([gmpy2.mpfr('0'), gmpy2.mpfr('0'), gmpy2.mpfr('0')])

        x1 = cross_norms
        x2 = -x1

        x1_latlon = node_xyz_to_lonlat_rad(x1)
        x2_latlon = node_xyz_to_lonlat_rad(x2)

        # Find out whether X1 or X2 is within the interval [w0, w1]
        if point_within_GCR(x1, [w0, w1]) and point_within_GCR(x1, [v0, v1]):
            return x1
        elif point_within_GCR(x2, [w0, w1]) and point_within_GCR(x2, [v0, v1]):
            return x2
        elif all(gmpy2.cmp(arr_val, gmpy2.mpfr('0')) == 0 for arr_val in x1):
            return [gmpy2.mpfr('0'), gmpy2.mpfr('0'), gmpy2.mpfr('0')]  # two vectors are parallel to each other
        else:
            return [gmpy2.mpfr('-1'), gmpy2.mpfr('-1'), gmpy2.mpfr('-1')]  # Intersection out of the interval or
    else:
        w0w1_norm = np.cross(w0, w1)

        v0v1_norm = np.cross(v0, v1)

        cross_norms = np.cross(w0w1_norm, v0v1_norm)

        # Then check if the cross_norms's L2 norm is larger than 1.0 or not
        if np.linalg.norm(cross_norms)- 1.0 == ERROR_TOLERANCE:
            pass
        elif np.linalg.norm(cross_norms) - 1.0 < 1.0:
            # Make sure the cross_norms's L2 norm is larger than 1.0
            cross_norms = cross_norms * 10

        if np.allclose(cross_norms, 0, atol=ERROR_TOLERANCE):
            return np.array([0, 0, 0])

        # Now use the bisection root finding method to find the intersection point on the sphere
        max_point = cross_norms
        min_point = np.array([0, 0, 0])
        mid_point = (max_point + min_point) / 2
        max_iter = 1000
        iter_count = 0
        while np.abs(np.linalg.norm(mid_point) - 1.0) > ERROR_TOLERANCE and iter_count < max_iter:
            if np.linalg.norm(max_point - min_point) < ERROR_TOLERANCE:
                break
            mid_point = (max_point + min_point) / 2
            if np.linalg.norm(mid_point) - 1.0 > ERROR_TOLERANCE:
                max_point = mid_point
            elif np.linalg.norm(mid_point) - 1.0 < ERROR_TOLERANCE:
                min_point = mid_point
            else:
                break
            iter_count += 1

        x1 = mid_point
        x2 = -x1

        if point_within_GCR(x1, [w0, w1]) and point_within_GCR(x1, [v0, v1]):
            return x1
        elif point_within_GCR(x2, [w0, w1]) and point_within_GCR(x2, [v0, v1]):
            return x2
        elif np.all(x1 == 0):
            return np.array([0, 0, 0])  # two vectors are parallel to each other
        else:
            return np.array([-1, -1, -1])  # Intersection out of the interval or


def get_GCA_constLat_intersections(gcr1_cart, gcr2_cart):
    pass