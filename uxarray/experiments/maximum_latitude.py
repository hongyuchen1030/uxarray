from gmpy2 import mpfr, fmms
import gmpy2
import numpy as np
from uxarray.exact_computation.utils import set_global_precision, mp_cross, convert_to_multiprecision, mp_norm, mp_dot
from uxarray.grid.coordinates import normalize_in_place, node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad
from uxarray.utils.computing import cross_fma, dot_fma, norm_faithful, _comp_prod_FMA, _vec_sum
from uxarray.grid.utils import angle_of_2_vectors
import matplotlib.pyplot as plt

def check_mpfr_conversion(np_array, mpfr_array):
    for i in range(len(np_array)):
        temp = abs(mpfr(np_array[i]) - mpfr_array[i])
        if gmpy2.cmp(abs(mpfr(np_array[i]) - mpfr_array[i]), mpfr('0')) > 0:
            return -1


def cond_num_dot_check(vec_1, vec_2):
    v1_abs = [abs(x) for x in vec_1]
    v2_abs = [abs(x) for x in vec_2]

    nom = mp_dot(v1_abs, v2_abs)
    denom = mp_dot(vec_1, vec_2)

    cond = mpfr('2.0') * nom / denom
    u = mpfr(np.finfo(np.float64).eps)

    if gmpy2.cmp(cond, mpfr('1.0') / u) > 0:
        return -1
    else:
        return 1

def accurate_lat(GCA_Cart, GCA_based = None):
    set_global_precision(53)
    import pyfma
    x1_vec, x2_vec = GCA_Cart

    a = get_a(x1_vec, x2_vec)

    if a <= 0 or a >= 1:
        return np.nan



    x1, y1, z1 = x1_vec
    x2, y2, z2 = x2_vec
    x1_float = float(mpfr(x1))
    y1_float = float(mpfr(y1))
    z1_float = float(mpfr(z1))

    x2_float = float(mpfr(x2))
    y2_float = float(mpfr(y2))
    z2_float = float(mpfr(z2))

    x1_vec_float = [x1_float, y1_float, z1_float]
    x2_vec_float = [x2_float, y2_float, z2_float]

    comprofma = _comp_prod_FMA([2.0, z1_float, z2_float, dot_fma(x1_vec_float, x2_vec_float)])
    nom_fl = _vec_sum([comprofma, -z1_float * z1_float, -z2_float * z2_float])

    if z1 + z2 >= 0:
        denom = -pyfma.fma(-(z1_float+z2_float), dot_fma(x1_vec_float, x2_vec_float), (z1_float+z2_float))
    else:
        denom = pyfma.fma(z1_float+z2_float, dot_fma(x1_vec_float, x2_vec_float), -(z1_float+z2_float))

    accurate = nom_fl / denom

    # The multiprecision version of the naive method
    x1_vec_mp = [mpfr(x1_float), mpfr(y1_float), mpfr(z1_float)]
    x2_vec_mp = [mpfr(x2_float), mpfr(y2_float), mpfr(z2_float)]


    if cond_num_dot_check(x1_vec_float, x2_vec_float) == -1:
        return np.nan

    if check_mpfr_conversion(x1_vec_float, x1_vec_mp) == -1 or check_mpfr_conversion(x2_vec_float, x2_vec_mp) == -1:
        return np.nan

    if is_same_plane(GCA_based, [x1_vec_mp, x2_vec_mp]) == -1:
        return np.nan

    x1_mp, y1_mp, z1_mp = x1_vec_mp
    x2_mp, y2_mp, z2_mp = x2_vec_mp
    comp_prod_mp = mpfr('2.0') * z1_mp * z2_mp * mp_dot(x1_vec_mp, x2_vec_mp)
    comp_prod_rel_err = abs((comp_prod_mp - mpfr(comprofma)) / comp_prod_mp)

    nom_mp = mpfr('2.0') * z1_mp * z2_mp * mp_dot(x1_vec_mp, x2_vec_mp) - z1_mp * z1_mp - z2_mp * z2_mp
    nom_rel_err = abs((nom_mp - mpfr(nom_fl)) / nom_mp)


    denom_mp = (z1_mp + z2_mp) * (mp_dot(x1_vec_mp, x2_vec_mp) - mpfr('1.0'))

    denom_mp_rel_err = abs((denom_mp - mpfr(denom)) / denom_mp)

    res_multi = nom_mp / denom_mp

    np.set_printoptions(precision=16)
    p1_x_float_str = str(accurate)
    rel_err_x = abs((mpfr(p1_x_float_str) - res_multi) / res_multi)

    return rel_err_x


def naive_lat(GCA_Cart):
    x1_vec, x2_vec = GCA_Cart
    x1, y1, z1 = x1_vec
    x2, y2, z2 = x2_vec

    nom = 2.0 * z1 * z2 * np.dot(x1_vec, x2_vec) - z1 * z1 - z2 * z2
    denom = (z1 + z2) * (np.dot(x1_vec, x2_vec) - 1.0)

    naive = nom / denom

    # The multiprecision version of the naive method
    x1_vec_mp = convert_to_multiprecision(x1_vec, str_mode=False)
    x2_vec_mp = convert_to_multiprecision(x2_vec, str_mode=False)
    x1_mp, y1_mp, z1_mp = x1_vec_mp
    x2_mp, y2_mp, z2_mp = x2_vec_mp
    nom_mp = 2.0 * z1_mp * z2_mp * mp_dot(x1_vec_mp, x2_vec_mp) - z1_mp * z1_mp - z2_mp * z2_mp
    denom_mp = (z1_mp + z2_mp) * (mp_dot(x1_vec_mp, x2_vec_mp) - 1.0)

    res_multi = nom_mp / denom_mp

    nom_rel_err = abs((nom_mp - mpfr(nom)) / nom_mp)
    denom_rel_err = abs((denom_mp - mpfr(denom)) / denom_mp)


    np.set_printoptions(precision=16)
    p1_x_float_str = str(naive)
    rel_err_x = abs((mpfr(p1_x_float_str) - res_multi) / res_multi)

    return rel_err_x


def GCA_length(GCA):
    GCA_cross = mp_cross(GCA[0], GCA[1])
    GCA_cross_norm = mp_norm(GCA_cross)
    GCA_dot = mp_dot(GCA[0], GCA[1])
    GCA_length = gmpy2.atan2(GCA_cross_norm, GCA_dot)
    return GCA_length



# Function to get a pair of points for the Great Circle Arc
def create_folded_pairs(points):
    """
    Create pairs of points by folding the array.

    Args:
    points: Array of great circle points

    Returns:
    List of pairs of points
    """
    num_points = len(points)
    return [(points[i], points[num_points - 1 - i]) for i in range(num_points // 2)]





def great_circle_points(lon1, lat1, lon2, lat2, num_elements):
    """
    Generate points on a great circle arc given two endpoints.

    Args:
    lon1, lat1: Longitude and Latitude of the first point in degrees
    lon2, lat2: Longitude and Latitude of the second point in degrees
    num_elements: Number of points in the generated arc

    Returns:
    Array of points (longitude, latitude) on the great circle
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Convert to Cartesian coordinates
    x_vec_1 = node_lonlat_rad_to_xyz([lon1, lat1])
    x_vec_2 = node_lonlat_rad_to_xyz([lon2, lat2])

    # Interpolate a from 0 to 1
    a_1 = np.linspace(0, 0.4, num_elements //2)
    a_2 = np.linspace(0.41, 1, num_elements //2)
    # Now combine the two arrays
    a = np.concatenate((a_1, a_2))

    x_new_points = np.outer(1-a, x_vec_1) + np.outer(a, x_vec_2)

    # Apply normalization
    x_points_nom = [normalize_in_place(x) for x in x_new_points]
    return x_points_nom




def get_a(vec_1, vec_2):
    x1, y1, z1 = vec_1
    x2, y2, z2 = vec_2
    nom = z1 * dot_fma(vec_1, vec_2) - z2
    denom = (z1 +z2)* dot_fma(vec_1, vec_2) - (z1 + z2)

    return nom / denom

def is_same_plane(based_GCA, check_GCAec_2):
    # See if their normal vectors are parallel
    based_GCA_cross = mp_cross(based_GCA[0], based_GCA[1])
    check_GCAec_2_cross = mp_cross(check_GCAec_2[0], check_GCAec_2[1])
    res = mp_cross(based_GCA_cross, check_GCAec_2_cross)
    max_res = max(abs(res[0]), abs(res[1]), abs(res[2]))
    if gmpy2.cmp(max_res, mpfr(np.finfo(np.float64).eps))<=0:
        return 1
    else:
        return -1


def plot_comparison():
    # Example usage with num_elements = 1000
    num_elements = 10000
    lon1, lat1 = -120, 30  # Point 1
    lon2, lat2 = 50, -10  # Point 2

    # Convert to Cartesian coordinates
    GCA_based = [node_lonlat_rad_to_xyz([np.deg2rad(lon1), np.deg2rad(lat1)]), node_lonlat_rad_to_xyz([np.deg2rad(lon2), np.deg2rad(lat2)])]

    great_circle_arc_1000 = great_circle_points(lon1, lat1, lon2, lat2, num_elements)
    # Generate a few Great Circle Arcs using this method
    gca_pairs = create_folded_pairs(great_circle_arc_1000)

    GCAs = []
    gca_lengths = []
    rel_errors_naive = []
    rel_errors_accute = []
    a_s = []

    # Problematic GCA
    gca_prob = []
    gca_prob_lonlat = []
    a_prob = []
    accu_err_prob = []


    for i in range(len(gca_pairs)):
        point1 = gca_pairs[i][0]
        point2 = gca_pairs[i][1]
        GCA = np.array([point1, point2])

        GCAs.append(GCA)
        # Calculate GCA length
        gca_length = GCA_length(GCA)
        gca_lengths.append(gca_length)

        # Calculate relative errors for both methods
        rel_error_naive = naive_lat(GCA)
        rel_errors_naive.append(rel_error_naive)

        rel_error_accute = accurate_lat(GCA, GCA_based = GCA_based)
        if rel_error_accute >= 1.0e-15:
            gca_prob.append(gca_pairs[i])
            a_prob.append(get_a(point1, point2))
            accu_err_prob.append(rel_error_accute)
        rel_errors_accute.append(rel_error_accute)

        a = get_a(point1, point2)
        a_s.append(a)

    # Remove the 'nan' values from naive_rel_err and accute_rel_err, and gca_lengths
    # Filtering out 'nan' values
    filtered_naive_rel_err = []
    filtered_accute_rel_err = []
    filtered_gca_lengths = []
    filtered_a_s = []
    for i in range(len(rel_errors_accute)):
        if not gmpy2.is_nan(rel_errors_accute[i]):
            filtered_naive_rel_err.append(rel_errors_naive[i])
            filtered_accute_rel_err.append(rel_errors_accute[i])
            filtered_gca_lengths.append(gca_lengths[i])
            filtered_a_s.append(a_s[i])

    # Plotting
    plt.figure(figsize=(10, 6))

    tick_size = 18
    label_fontsize = 18  # Adjust as needed
    title_fontsize = 20  # Adjust as needed

    plt.scatter(filtered_a_s, filtered_naive_rel_err, marker='o', linestyle='-', label='Naive Algorithm',alpha=0.3)
    plt.scatter(filtered_a_s, filtered_accute_rel_err, marker='.', linestyle='-', label='Accurate Algorithm',alpha=0.3)

    # Customizing x-axis ticks
    # Customizing x-axis ticks
    ax = plt.gca()  # Get current axis
    ax.set_yscale('log')

    # Plotting maximum relative error lines
    max_naive_rel_err = max(filtered_naive_rel_err)
    max_accute_rel_err = max(filtered_accute_rel_err)

    plt.axhline(y=max_naive_rel_err, color='#1f77b4', linestyle='--', linewidth=5)
    plt.axhline(y=max_accute_rel_err, color='#ff7f0e', linestyle='--', linewidth=5)

    # Adding text for maximum values
    # Ensure the y value for the text is within the visible range of the plot
    plt.text(x=0.5, y=max_naive_rel_err * 1.01, s=f"Naive Max Rel_Err: {max_naive_rel_err:.2e}", color='#1f77b4',
             verticalalignment='bottom', horizontalalignment='left',             fontsize=tick_size,
             fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', pad=3.0))
    plt.text(x=0.5, y=max_accute_rel_err * 1.01, s=f"Our Max Rel_Err: {max_accute_rel_err:.2e}", color='#ff7f0e',
             verticalalignment='bottom', horizontalalignment='left'  ,           fontsize=tick_size,
             fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    plt.xlabel(
        r'$a = \frac{z_1 (\mathbf{x}_1 \cdot \mathbf{x}_2) - z_2}{(z_1 + z_2) (\mathbf{x}_1 \cdot \mathbf{x}_2 - 1)}$',
        fontsize=14)

    plt.ylabel('Relative Error', fontsize=label_fontsize)
    plt.title('Comparison of Different Maximum Latitude Methods \n (10000 Arcs Exp)', fontsize=title_fontsize)
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    plot_comparison()
