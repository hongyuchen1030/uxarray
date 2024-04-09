import gmpy2
from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.grid.coordinates import normalize_in_place, node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad
from uxarray.grid.intersections import gca_constLat_intersection, gca_constLat_intersection_accurate
from uxarray.utils.computing import cross_fma, _acc_sqrt, _two_prod_fma, _two_sum, _two_square, _comp_prod_FMA, \
    _sum_of_squares_re
from uxarray.exact_computation.utils import set_global_precision, mp_cross, mp_norm, mp_dot
import numpy as np
import matplotlib.pyplot as plt
import time


# Define the function delta_hat_g_minus
def delta_hat_g_minus(u, n_x, n_y, n_z, z_0):
    temp = n_x ** 2 + n_y ** 2 - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2
    tilda_s = np.sqrt(n_x ** 2 + n_y ** 2 - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2)
    delta_hat_s_long_term_numerator = (n_x ** 2 + n_y ** 2) * 2 * u - np.linalg.norm(
        [n_x, n_y, n_z]) ** 2 * z_0 ** 2 * 5 * u
    delta_hat_s_long_term_denominator = (n_x ** 2 + n_y ** 2) - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2
    delta_hat_s_long_term = delta_hat_s_long_term_numerator / delta_hat_s_long_term_denominator
    delta_hat_s = np.abs(np.sqrt(1 + u / (1 + u) + delta_hat_s_long_term * (1 + u / (1 + u))) - 1)
    numerator = np.abs(z_0 * n_x * n_z * (1 + u / (1 + u)) ** 2 - 1) - tilda_s * n_y * abs(
        (1 + delta_hat_s) * u / (1 + u) - 1) + u / (1 + u)
    denominator = np.abs(z_0 * n_x * n_z - tilda_s * n_y)
    return numerator / denominator


def calculate_delta_s_tilde():
    set_global_precision()
    u = mpfr(np.finfo(np.float64).eps)
    res = (mpfr('21') * u) / (mpfr('16') * u + mpfr('2'))
    ratio = res / u

    pass


def accurate_constLat_GCA_Intersection():
    import pyfma
    GCR1_cart = np.array([
        node_lonlat_rad_to_xyz([np.deg2rad(44.9998),
                                np.deg2rad(44.9998)]),
        node_lonlat_rad_to_xyz([np.deg2rad(45.0002),
                                np.deg2rad(45.0002)])
    ])

    z0 = np.arcsin(np.deg2rad(45.0))

    # Convert them to the gmpy2 mpfr format
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [
        mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
    z0_mpfr = mpfr(str(z0))
    # Obtain the normal vector of the GCA
    n = mp_cross(GCR1_cart[0], GCR1_cart[1])
    nx_sqr_ny_sqr_mpfr = n[0] * n[0] + n[1] * n[1]
    norm_n_square_z0_mpfr = mp_norm(n) * mp_norm(n) * z0_mpfr * z0_mpfr
    s_tilde = gmpy2.sqrt(n[0] * n[0] + n[1] * n[1] - mp_norm(n) * mp_norm(n) * z0_mpfr * z0_mpfr)
    p1_x = -(mpfr('1.0') / (n[0] * n[0] + n[1] * n[1])) * (z0_mpfr * n[0] * n[2] + s_tilde * n[1])

    # Now the accurate algo
    p1_float = gca_constLat_intersection_accurate(GCR1_cart, z0)

    np.set_printoptions(precision=16)
    p1_x_float_str = str(p1_float)
    rel_err_x = abs((mpfr(p1_x_float_str) - p1_x) / p1_x)

    print("Accurate Algo ConstLat GCA  relative error: ", rel_err_x)


def naive_constLat_GCA_Intersection():
    GCR1_cart = np.array([
        node_lonlat_rad_to_xyz([np.deg2rad(44.9998),
                                np.deg2rad(44.9998)]),
        node_lonlat_rad_to_xyz([np.deg2rad(45.0002),
                                np.deg2rad(45.0002)])
    ])

    constZ = np.arcsin(np.deg2rad(45.0))

    # Convert them to the gmpy2 mpfr format
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [
        mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
    z0_mpfr = mpfr(str(constZ))
    # Obtain the normal vector of the GCA
    n = mp_cross(GCR1_cart[0], GCR1_cart[1])
    s_tilde = gmpy2.sqrt(n[0] * n[0] + n[1] * n[1] - mp_norm(n) * mp_norm(n) * z0_mpfr * z0_mpfr)
    p1_x = -(mpfr('1.0') / (n[0] * n[0] + n[1] * n[1])) * (z0_mpfr * n[0] * n[2] + s_tilde * n[1])

    # The naive approach
    n = np.cross(GCR1_cart[0], GCR1_cart[1])
    nx, ny, nz = n

    s_tilde = np.sqrt(nx ** 2 + ny ** 2 - np.linalg.norm(n) ** 2 * constZ ** 2)
    p1_x_float = -(1.0 / (nx ** 2 + ny ** 2)) * (constZ * nx * nz + s_tilde * ny)

    np.set_printoptions(precision=16)
    p1_x_float_str = str(p1_x_float)
    rel_err_x = abs((mpfr(p1_x_float_str) - p1_x) / p1_x)

    print("naive Algo ConstLat GCA  relative error: ", rel_err_x)


def multi_precision_constLat_GCA_Intersection(GCR1_cart, constZ):
    # Convert them to the gmpy2 mpfr format
    set_global_precision(53)
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [
        mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
    z0_mpfr = mpfr(str(constZ))
    # Obtain the normal vector of the GCA
    n = mp_cross(GCR1_cart[0], GCR1_cart[1])

    # Assume n, z0_mpfr are already defined

    # Calculate the squares of the components of n
    n0_squared = n[0] * n[0]
    n1_squared = n[1] * n[1]

    # Calculate the square of the norm of n
    norm_n_squared = mp_norm(n) * mp_norm(n)

    # Calculate the square of z0_mpfr
    z0_squared = z0_mpfr * z0_mpfr

    # Subtract the square of the product of norm_n and z0 from the sum of the squares of n's components
    difference = n0_squared + n1_squared - norm_n_squared * z0_squared

    # Calculate the square root of the difference
    s_tilde = gmpy2.sqrt(difference)

    # Calculate the product of z0_mpfr and the components of n
    z0_n0_product = z0_mpfr * n[0]
    z0_n0_n2_product = z0_n0_product * n[2]

    # Multiply s_tilde with the second component of n
    s_tilde_n1_product = s_tilde * n[1]

    # Add the two products together
    sum_of_products = z0_n0_n2_product + s_tilde_n1_product

    # Calculate the denominator which is the sum of the squares of the first two components of n
    denominator = n0_squared + n1_squared

    # Invert the denominator
    denominator_inverted = mpfr('1.0') / denominator

    # Multiply the inverted denominator with the sum of the products
    quotient = denominator_inverted * sum_of_products

    # Negate the quotient
    negated_quotient = -quotient

    # Convert the result to a float
    p1_x = float(negated_quotient)

    # p1_x now holds the final result

    return p1_x


def accute_intersection(GCR1_cart, constZ):
    # Convert them to the gmpy2 mpfr format
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [
        mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
    z0_mpfr = mpfr(str(constZ))
    # Obtain the normal vector of the GCA
    n = mp_cross(GCR1_cart[0], GCR1_cart[1])
    s_tilde = gmpy2.sqrt(n[0] * n[0] + n[1] * n[1] - mp_norm(n) * mp_norm(n) * z0_mpfr * z0_mpfr)
    p1_x = -(mpfr('1.0') / (n[0] * n[0] + n[1] * n[1])) * (z0_mpfr * n[0] * n[2] + s_tilde * n[1])

    # The accurate approach
    # Now the accurate algo
    p1_float = gca_constLat_intersection_accurate(GCR1_cart, constZ)

    np.set_printoptions(precision=16)
    p1_x_float_str = str(p1_float)
    rel_err_x = abs((mpfr(p1_x_float_str) - p1_x) / p1_x)
    return rel_err_x


def get_naive_intersection(GCR1_cart, constZ):
    n = np.cross(GCR1_cart[0], GCR1_cart[1])
    # Calculate the squares of nx and ny
    nx = n[0]
    ny = n[1]
    nz = n[2]

    nx_squared = nx ** 2
    ny_squared = ny ** 2

    # Calculate the norm of n, then square it
    norm_n = np.linalg.norm(n)
    norm_n_squared = norm_n ** 2

    # Calculate the square of constZ
    constZ_squared = constZ ** 2

    # Calculate the expression inside the square root
    sqrt_expression = nx_squared + ny_squared - norm_n_squared * constZ_squared

    # Calculate the square root (s_tilde)
    s_tilde = np.sqrt(sqrt_expression)

    # Calculate the denominator
    denominator = nx_squared + ny_squared

    # Invert the denominator
    denominator_inverted = 1.0 / denominator

    # Calculate the terms for the numerator
    constZ_nx_nz = constZ * nx * nz
    s_tilde_ny = s_tilde * ny

    # Sum the terms to get the numerator
    numerator = constZ_nx_nz + s_tilde_ny

    # Calculate the final value for p1_x_float
    p1_x_float = -denominator_inverted * numerator
    return p1_x_float


def naive_intersection(GCR1_cart, constZ):
    # Convert them to the gmpy2 mpfr format
    # The naive approach
    n = np.cross(GCR1_cart[0], GCR1_cart[1])
    nx, ny, nz = n

    s_tilde = np.sqrt(nx ** 2 + ny ** 2 - np.linalg.norm(n) ** 2 * constZ ** 2)
    p1_x_float = -(1.0 / (nx ** 2 + ny ** 2)) * (constZ * nx * nz + s_tilde * ny)

    np.set_printoptions(precision=16)
    p1_x_float_str = str(p1_x_float)
    rel_err_x = abs((mpfr(p1_x_float_str) - p1_x) / p1_x)
    return rel_err_x


def GCA_length(GCA):
    GCA_cross = mp_cross(GCA[0], GCA[1])
    GCA_cross_norm = mp_norm(GCA_cross)
    GCA_dot = mp_dot(GCA[0], GCA[1])
    GCA_length = gmpy2.atan2(GCA_cross_norm, GCA_dot)
    return GCA_length


def plot_comparison():
    v1_lon = np.arange(44, 45, 0.0001)
    v1_lat = np.arange(46, 45, -0.0001)
    v2_lon = np.arange(46, 45, -0.0001)
    v2_lat = np.arange(44, 45, 0.0001)

    GCAs = []
    constZs = []
    gca_lengths = []
    rel_errors_naive = []
    rel_errors_accute = []

    for i in range(len(v1_lon)):
        point1 = node_lonlat_rad_to_xyz([np.deg2rad(v1_lon[i]), np.deg2rad(v1_lat[i])])
        point2 = node_lonlat_rad_to_xyz([np.deg2rad(v2_lon[i]), np.deg2rad(v2_lat[i])])
        constZ = np.arcsin(
            np.deg2rad((v1_lat[i] + v2_lat[i]) / 2))  # Assuming you are using latitude for constZ calculation
        GCA = np.array([point1, point2])

        GCAs.append(GCA)
        constZs.append(constZ)

        # Calculate GCA length
        gca_length = GCA_length(GCA)
        gca_lengths.append(gca_length)

        # Calculate relative errors for both methods
        rel_error_naive = naive_intersection(GCA, constZ)
        rel_errors_naive.append(rel_error_naive)

        rel_error_accute = accute_intersection(GCA, constZ)
        rel_errors_accute.append(rel_error_accute)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(gca_lengths, rel_errors_naive, marker='o', linestyle='-', label='Naive Method', alpha=0.3)
    plt.scatter(gca_lengths, rel_errors_accute, marker='.', linestyle='-', label='Our Method', alpha=0.3)

    # Customizing x-axis ticks
    # Customizing x-axis ticks
    ax = plt.gca()  # Get current axis
    # Increase font size for labels and title

    # Plotting maximum relative error lines
    max_naive_rel_err = max(rel_errors_naive)
    max_accute_rel_err = max(rel_errors_accute)

    plt.axhline(y=max_naive_rel_err, color='#1f77b4', linestyle='--')
    plt.axhline(y=max_accute_rel_err, color='#ff7f0e', linestyle='--')

    # Draw horizontal lines for maximum relative errors
    plt.axhline(y=max_naive_rel_err, color='#1f77b4', linestyle='--', linewidth=5)
    plt.axhline(y=max_accute_rel_err, color='#ff7f0e', linestyle='--', linewidth=5)

    # Calculate the offset needed to place the text just above the orange dashed line
    y_offset = (max_accute_rel_err - min(rel_errors_accute)) * 0.05  # small percentage above the max error line

    tick_size = 18
    label_fontsize = 18  # Adjust as needed
    title_fontsize = 20  # Adjust as needed

    # Adding text for maximum naive relative error
    plt.text(x=max(gca_lengths) * 0.5, y=max_naive_rel_err,
             s=f"Naive Max Rel_Err: {max_naive_rel_err:.2e}",
             color='#1f77b4', ha='center', va='bottom', fontsize=tick_size, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    # Adding text for maximum accurate relative error, adjusted to be above the orange line
    plt.text(x=max(gca_lengths) * 0.5, y=max_accute_rel_err + y_offset,
             s=f"Our Max Rel_Err: {max_accute_rel_err:.2e}",
             color='#ff7f0e', ha='center', va='bottom', fontsize=tick_size, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    ax = plt.gca()  # Get current axis
    ax.set_yscale('log')

    plt.xlabel('GCA Length in radians', fontsize=label_fontsize)
    plt.ylabel('Relative Error (log scale)', fontsize=label_fontsize)
    plt.title('Comparison of Different Constant Latitude GCA Intersection Methods \n (10000 Arcs Exp)',
              fontsize=title_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.grid(True)
    plt.show()


def compare_run_time():
    v1_lon = np.arange(44, 45, 0.0001)
    v1_lat = np.arange(46, 45, -0.0001)
    v2_lon = np.arange(46, 45, -0.0001)
    v2_lat = np.arange(44, 45, 0.0001)

    GCAs = []
    constZs = []
    gca_lengths = []
    total_time_naive = 0
    total_time_accute = 0
    total_time_multi = 0


    for i in range(len(v1_lon)):
        point1 = node_lonlat_rad_to_xyz([np.deg2rad(v1_lon[i]), np.deg2rad(v1_lat[i])])
        point2 = node_lonlat_rad_to_xyz([np.deg2rad(v2_lon[i]), np.deg2rad(v2_lat[i])])
        constZ = np.arcsin(
            np.deg2rad((v1_lat[i] + v2_lat[i]) / 2))  # Assuming you are using latitude for constZ calculation
        GCA = np.array([point1, point2])

        # Calculate relative errors for both methods
        start_time = time.time()# Run 1000 times to get the average run time
        p1_accurate = gca_constLat_intersection_accurate(GCA, constZ)
        total_time_accute += time.time() - start_time

        start_time = time.time()
        p1_naive = get_naive_intersection(GCA, constZ)
        total_time_naive += time.time() - start_time

        start_time = time.time()
        p1_multi = multi_precision_constLat_GCA_Intersection(GCA, constZ)
        total_time_multi += time.time() - start_time


    # Compute average run times
    average_time_naive = total_time_naive / len(v1_lon)
    average_time_accute = total_time_accute / len(v1_lon)
    average_time_multi = total_time_multi / len(v1_lon)

    print(f"Average run time for naive_intersection: {average_time_naive} seconds")
    print(f"Average run time for accute_intersection: {average_time_accute} seconds")
    print(f"Average run time for multi_precision_constLat_GCA_Intersection: {average_time_multi} seconds")



def calculate_s_tile_sqe():
    u = mpfr(np.finfo(np.float64).eps)
    two = mpfr('2.0')
    six = mpfr('6.0')
    one = mpfr('1.0')
    eight = mpfr('8.0')
    gamma_26 = ((two * u) / (one - two * u)) * ((six * u) / (one - six * u))
    res = gamma_26 + u / eight + u / eight * gamma_26 + two * u + two * u * gamma_26
    u_cnt = res / u
    pass


if __name__ == "__main__":
    # u = mpfr(np.finfo(np.float64).eps)
    # four = mpfr('4.0')
    # three = mpfr('3.0')
    # eight = mpfr('8.0')
    # two = mpfr('2.0')
    # forty_one = mpfr('41')
    # five = mpfr('5')
    # eighteen = mpfr('18')
    # eighty_one = mpfr('81')
    # nine = mpfr('9')
    # sixty_nine = mpfr('69')
    # two_ninety_seven = mpfr('297')
    # one = mpfr('1.0')
    #
    #
    # # Calculate each term using mpfr
    # term1 = four * u
    # term2 = u / eight
    # term3 = four * u ** 2
    # term4 = (u * u) / two
    # term5 = (u * u * u) / two
    #
    # term6 = (forty_one * u) / eight
    # term7 = (five * u * u * u) / eight
    # term8 = (eighteen * u * u) / (one - (nine * u) + (eighteen * u * u))
    # term9 = ((eighty_one * u * u * u * u) + (nine * u * u * u * u * u)) / eight
    # term10 = (sixty_nine * u * u) / (four * (one - (nine * u) + (eighteen * u * u)) * (one - (nine * u) + (eighteen * u * u)))
    # term11 = (two_ninety_seven * u * u * u) / (four * (one - (nine * u) + (eighteen * u * u)))
    # term12 = (u * u * u * u) / two
    #
    # delta_nxny = term1 + term2 + term3 + term4 + term5
    # delta_nsz = term6 + term7 + term8 + term9 + term10 + term11 + term12
    #
    # res= gmpy2.cmp(delta_nxny, delta_nsz)
    # pass

    # # Sum all terms to get the final result
    # result = term1 + term2 + term3 + term4 + term5
    # calculate_delta_s_tilde()
    # accurate_constLat_GCA_Intersection()
    # naive_constLat_GCA_Intersection()
    # plot_comparison()
    compare_run_time()
