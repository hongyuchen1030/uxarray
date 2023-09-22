from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.exact_computation.utils import set_global_precision, multi__newton_raphson_solver_for_gca_constLat
from uxarray.grid.coordinates import node_lonlat_rad_to_xyz


if __name__ == "__main__":
    # Define the lon-lat values in radians
    set_global_precision(128)
    lonlat_deg_1 = [mpfr(np.deg2rad(170.0)), mpfr(np.deg2rad(89.99))]
    lonlat_deg_2 = [mpfr(np.deg2rad(170.0)), mpfr(np.deg2rad(10.0))]

    # Convert lon-lat values to xyz using a function (assuming node_lonlat_rad_to_xyz is defined)
    GCR1_cart = np.array([node_lonlat_rad_to_xyz(lonlat_deg_1), node_lonlat_rad_to_xyz(lonlat_deg_2)])
    initial_guess = np.array([mpfr('-0.4924038765061042'), mpfr('0.08682408883346515'), mpfr('0.8660254037844386')])
    res = multi__newton_raphson_solver_for_gca_constLat(initial_guess, GCR1_cart, error_tol=mpfr('1.0e-15'), verbose= False, max_iter=100)

    # Add disturbance to the initial guess
    u = mpfr(np.finfo(np.float64).eps)

    # A rough relative error bound for normalization times division
    disturbance = 2 * u
    initial_guess_disturbed = initial_guess + np.array([disturbance, disturbance, disturbance])
    res_disturbed = multi__newton_raphson_solver_for_gca_constLat(initial_guess_disturbed, GCR1_cart, error_tol=mpfr('1.0e-15'), verbose= True, max_iter=100)