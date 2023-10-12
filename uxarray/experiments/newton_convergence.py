import gmpy2
from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.exact_computation.utils import set_global_precision, multi__newton_raphson_solver_for_gca_constLat, \
    multi_one_var_newton_raphson_solver_for_gca_constLat, convert_to_multiprecision, mp_cross, mp_norm
from uxarray.grid.coordinates import node_lonlat_rad_to_xyz, normalize_in_place
from uxarray.grid.lines import point_within_gca
from uxarray.grid.utils import cross_fma

def newton_convergence():
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
    set_global_precision()

def one_var_newton_convergence():
    set_global_precision(1000)
    # The GCA whose end point is extremely closed to each other
    lonlat_deg_1 = [np.deg2rad(43.0), np.deg2rad(45.0)]
    lonlat_cart_1 = node_lonlat_rad_to_xyz(lonlat_deg_1)
    lonlat_cart_mpfr_1 = node_lonlat_rad_to_xyz(np.array([gmpy2.radians(mpfr('43.0')), gmpy2.radians(mpfr('45.0'))]))



    lonlat_deg_2 = [np.deg2rad(44.0), np.deg2rad(44.0)]
    lonlat_cart_2 = node_lonlat_rad_to_xyz(lonlat_deg_2)
    lonlat_cart_mpfr_2 = node_lonlat_rad_to_xyz(np.array([gmpy2.radians(mpfr('44.0')), gmpy2.radians(mpfr('44.0'))]))

    GCA_cart = np.array([lonlat_cart_1, lonlat_cart_2])
    GCA_cart_mpfr = np.array([lonlat_cart_mpfr_1, lonlat_cart_mpfr_2])

    constLat_rad = np.deg2rad(44.5)

    initial_guess = gca_constLat_numerical_cal(GCA_cart, constLat_rad, fma_disabled=False, verbose=False)

    # Convert the initial guess to mpfr array
    initial_guess_mpfr = convert_to_multiprecision(initial_guess, str_mode=False)
    exact_result_mpfr = gca_constLat_numerical_cal(GCA_cart_mpfr, gmpy2.radians(44.5), fma_disabled=False, verbose=False)

    # Make sure the initial guess is on the opposite side of the sphere
    random_point_on_sphere = np.array([-initial_guess_mpfr[0], -initial_guess_mpfr[1], -initial_guess_mpfr[2]])

    res = multi_one_var_newton_raphson_solver_for_gca_constLat(random_point_on_sphere, GCA_cart_mpfr, error_tol=mpfr('0.0'), verbose= False, max_iter=100, test_mode=True, ref_result=exact_result_mpfr[0])
    set_global_precision()

def gca_constLat_numerical_cal(gca_cart, constLat, fma_disabled=False, verbose=False):
    """Calculate the intersection point(s) of a Great Circle Arc (GCA) and a constant latitude line in a
    Cartesian coordinate system.

    To reduce relative errors, the Fused Multiply-Add (FMA) operation is utilized.
    A warning is raised if the given coordinates are not in the cartesian coordinates, or
    they cannot be accurately handled using floating-point arithmetic.

    Parameters
    ----------
    gca_cart : [2, 3] np.ndarray Cartesian coordinates of the two end points GCA.
    constLat : float
        The constant latitude of the latitude line.
    fma_disabled : bool, optional (default=False)
        If True, the FMA operation is disabled. And a naive `np.cross` is used instead.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s).

    Raises
    ------
    ValueError
        If the input GCA is not in the cartesian [x, y, z] format.
    """
    if np.any(
            np.vectorize(lambda x: isinstance(x, (gmpy2.mpfr, gmpy2.mpz)))(
                gca_cart)):
        constZ = gmpy2.sin(constLat)
        x1, x2 = gca_cart
        n = mp_cross(x1, x2)
        nx, ny, nz = n

        s_tilde = gmpy2.sqrt(nx ** mpfr('2.0') + ny ** mpfr('2.0') - mp_norm(n) ** mpfr('2.0') * constZ ** mpfr('2.0'))

        p1_x = -(mpfr('1.0') / (nx ** mpfr('2.0') + ny ** mpfr('2.0'))) * (constZ * nx * nz + s_tilde * ny)
        p2_x = -(mpfr('1.0') / (nx ** mpfr('2.0') + ny ** mpfr('2.0'))) * (constZ * nx * nz - s_tilde * ny)
        p1_y = -(mpfr('1.0') / (nx ** mpfr('2.0') + ny ** mpfr('2.0'))) * (constZ * ny * nz - s_tilde * nx)
        p2_y = -(mpfr('1.0') / (nx ** mpfr('2.0') + ny ** mpfr('2.0'))) * (constZ * ny * nz + s_tilde * nx)

        p1 = np.array([p1_x, p1_y, constZ])
        p2 = np.array([p2_x, p2_y, constZ])
    else:
        constZ = np.sin(constLat)
        x1, x2 = gca_cart
        n = cross_fma(x1, x2)
        nx, ny, nz = n

        s_tilde = np.sqrt(nx ** 2 + ny ** 2 - np.linalg.norm(n) ** 2 * constZ ** 2)
        p1_x = -(1.0 / (nx ** 2 + ny ** 2)) * (constZ * nx * nz + s_tilde * ny)
        p2_x = -(1.0 / (nx ** 2 + ny ** 2)) * (constZ * nx * nz - s_tilde * ny)
        p1_y = -(1.0 / (nx ** 2 + ny ** 2)) * (constZ * ny * nz - s_tilde * nx)
        p2_y = -(1.0 / (nx ** 2 + ny ** 2)) * (constZ * ny * nz + s_tilde * nx)

        p1 = np.array([p1_x, p1_y, constZ])
        p2 = np.array([p2_x, p2_y, constZ])

    # Now test which intersection point is within the GCA range
    res = np.array([])
    if point_within_gca(p1, gca_cart):
        return p1
    elif point_within_gca(p2, gca_cart):
        return p2

    return res

if __name__ == "__main__":
    # newton_convergence()
    one_var_newton_convergence()
