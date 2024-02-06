import gmpy2
from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.grid.coordinates import normalize_in_place, node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad
from uxarray.grid.intersections import gca_constLat_intersection, gca_constLat_intersection_accurate
from uxarray.utils.computing import cross_fma, _acc_sqrt, _two_prod_fma, _two_sum, _two_square, _comp_prod_FMA, _sum_of_squares_re
from uxarray.exact_computation.utils import set_global_precision, mp_cross, mp_norm, mp_dot
import numpy as np
import matplotlib.pyplot as plt



# Define the function delta_hat_g_minus
def delta_hat_g_minus(u, n_x, n_y, n_z, z_0):
    temp = n_x ** 2 + n_y ** 2 - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2
    tilda_s = np.sqrt(n_x ** 2 + n_y ** 2 - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2)
    delta_hat_s_long_term_numerator = (n_x **2 + n_y ** 2) * 2 * u - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2 * 5 * u
    delta_hat_s_long_term_denominator = (n_x **2 + n_y ** 2) - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2
    delta_hat_s_long_term = delta_hat_s_long_term_numerator / delta_hat_s_long_term_denominator
    delta_hat_s = np.abs(np.sqrt(1 + u/(1+u) + delta_hat_s_long_term * (1 + u/(1+u))) - 1)
    numerator = np.abs(z_0 * n_x * n_z * (1 + u/(1+u))**2 - 1) - tilda_s * n_y * abs((1 + delta_hat_s) * u/(1+u) - 1) + u/(1+u)
    denominator = np.abs(z_0 * n_x * n_z - tilda_s * n_y)
    return numerator / denominator

def calculate_delta_s_tilde():
    set_global_precision()
    u = mpfr(np.finfo(np.float64).eps)
    res = (mpfr('21')*u)/(mpfr('16')*u+mpfr('2'))
    ratio = res/u

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
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
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
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
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


def accute_intersection(GCR1_cart, constZ):
    # Convert them to the gmpy2 mpfr format
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
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

def naive_intersection(GCR1_cart, constZ):
    # Convert them to the gmpy2 mpfr format
    GCR1_cart_mpfr = [mpfr(str(GCR1_cart[0][0])), mpfr(str(GCR1_cart[0][1])), mpfr(str(GCR1_cart[0][2]))], [mpfr(str(GCR1_cart[1][0])), mpfr(str(GCR1_cart[1][1])), mpfr(str(GCR1_cart[1][2]))]
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
        constZ = np.arcsin(np.deg2rad( (v1_lat[i] + v2_lat[i]) / 2  )) # Assuming you are using latitude for constZ calculation
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
    plt.scatter(gca_lengths, rel_errors_naive, marker='o', linestyle='-', label='Naive Method',alpha=0.3)
    plt.scatter(gca_lengths, rel_errors_accute, marker='.', linestyle='-', label='Our Method',alpha=0.3)

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
             color='#1f77b4', ha='center', va='bottom', fontsize=tick_size, fontweight='bold',bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    # Adding text for maximum accurate relative error, adjusted to be above the orange line
    plt.text(x=max(gca_lengths) * 0.5, y=max_accute_rel_err + y_offset,
             s=f"Our Max Rel_Err: {max_accute_rel_err:.2e}",
             color='#ff7f0e', ha='center', va='bottom', fontsize=tick_size, fontweight='bold',bbox=dict(facecolor='white', edgecolor='none', pad=3.0))


    ax = plt.gca()  # Get current axis
    ax.set_yscale('log')



    plt.xlabel('GCA Length in radians', fontsize=label_fontsize)
    plt.ylabel('Relative Error (log scale)', fontsize=label_fontsize)
    plt.title('Comparison of Different Constant Latitude GCA Intersection Methods \n (10000 Arcs Exp)', fontsize=title_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.grid(True)
    plt.show()


def calculate_s_tile_sqe():
    u = mpfr(np.finfo(np.float64).eps)
    two = mpfr('2.0')
    six = mpfr('6.0')
    one = mpfr('1.0')
    eight = mpfr('8.0')
    gamma_26 = ((two * u) / (one - two * u)) * ((six * u)/(one - six * u))
    res = gamma_26 + u/eight + u/eight * gamma_26 + two * u  + two * u * gamma_26
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
    u = mpfr(np.finfo(np.float64).eps)
    two = mpfr('2.0')
    six = mpfr('6.0')
    one = mpfr('1.0')
    eight = mpfr('8.0')
    four = mpfr('4.0')
    five = mpfr('5.0')
    rel_err = (one + u/(one + u))
    ans = (mpfr('360.0') * u * u+ mpfr('57.5')*u + mpfr('12.0')) * (one + two * u/(one + u))
    u_cnt = ans / u
    pass




