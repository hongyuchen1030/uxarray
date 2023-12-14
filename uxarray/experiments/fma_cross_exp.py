from gmpy2 import mpfr, fmms
import gmpy2
import numpy as np
from uxarray.exact_computation.utils import set_global_precision, mp_cross, convert_to_multiprecision, mp_norm, mp_dot
from uxarray.grid.coordinates import normalize_in_place, node_lonlat_rad_to_xyz
from uxarray.utils.computing import cross_fma, dot_fma, norm_faithful
from uxarray.grid.utils import angle_of_2_vectors
import matplotlib.pyplot as plt


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
    set_global_precision()
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
    x = normalize_in_place(x)
    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    v1_float = np.cross(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = np.cross(gca_pt2_1_float, gca_pt2_2_float)

    x_float = np.cross(v1_float, v2_float)
    x_float = x_float / np.linalg.norm(x_float)
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
    set_global_precision()
    gca_pt1_1 = normalize_in_place(np.array([mpfr('0.5'), mpfr('0.5'), mpfr('0.5')]))
    gca_pt1_2 = normalize_in_place(np.array([mpfr('0.500000000000001'), mpfr('0.499999999999999'), mpfr('0.499999999999999')]))
    gca_pt2_1 = normalize_in_place(np.array([mpfr('0.500000000000002'), mpfr('0.500000000000002'), mpfr('0.499999999999999')]))
    gca_pt2_2 = normalize_in_place(np.array([mpfr('0.499999999999998'), mpfr('0.500000000000001'), mpfr('0.499999999999999')]))

    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    # Convert it back to the mpfr format to enure the input are the same
    gca_pt1_1 = [mpfr(str(gca_pt1_1_float[0])), mpfr(str(gca_pt1_1_float[1])), mpfr(str(gca_pt1_1_float[2]))]
    gca_pt1_2 = [mpfr(str(gca_pt1_2_float[0])), mpfr(str(gca_pt1_2_float[1])), mpfr(str(gca_pt1_2_float[2]))]
    gca_pt2_1 = [mpfr(str(gca_pt2_1_float[0])), mpfr(str(gca_pt2_1_float[1])), mpfr(str(gca_pt2_1_float[2]))]
    gca_pt2_2 = [mpfr(str(gca_pt2_2_float[0])), mpfr(str(gca_pt2_2_float[1])), mpfr(str(gca_pt2_2_float[2]))]
    v1 = mp_cross(gca_pt1_1, gca_pt1_2)
    v2 = mp_cross(gca_pt2_1, gca_pt2_2)
    x = mp_cross(v1, v2)
    # normalize the x
    x = normalize_in_place(x)


    v1_float = cross_fma(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = cross_fma(gca_pt2_1_float, gca_pt2_2_float)
    x_float = cross_fma(v1_float, v2_float)

    #Normalize the x_float using our norm_faithful
    x_float = x_float / norm_faithful(x_float)


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

    # normalize the x
    x = normalize_in_place(x)

    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    v1_float = cross_fma(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = cross_fma(gca_pt2_1_float, gca_pt2_2_float)
    x_float = cross_fma(v1_float, v2_float)

    #Normalize the x_float using our norm_faithful
    x_float = x_float / norm_faithful(x_float)
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
    set_global_precision(53)
    u = mpfr(np.finfo(np.float64).eps)
    v_nonorm = mpfr('1.0') + mpfr('9.0') * u
    v_norm = mpfr('1.0') + (mpfr('10.0') * u/(mpfr('1.0') + u))
    term_1 = v_nonorm/v_norm
    term_2 = (mpfr('1.0') + u - mpfr('2.0') * u ** mpfr('2.0'))
    rel_err = abs(term_1 * term_2 - mpfr('1.0'))
    ratio = rel_err / u
    print("Final relative error bound: ", rel_err)

def check_mpfr_conversion(np_array, mpfr_array):
    for i in range(len(np_array)):
        if gmpy2.cmp(abs(np_array[i] - float(mpfr_array[i])), mpfr('0')) != 0:
            return -1

def accute_intersection(gca_pt1_1, gca_pt1_2, gca_pt2_1, gca_pt2_2):
    # Now convert the input gca as float64
    set_global_precision(53)
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    # Convert it back to the mpfr format to enure the input are the same
    gca_pt1_1 = [mpfr(str(gca_pt1_1_float[0])), mpfr(str(gca_pt1_1_float[1])), mpfr(str(gca_pt1_1_float[2]))]
    gca_pt1_2 = [mpfr(str(gca_pt1_2_float[0])), mpfr(str(gca_pt1_2_float[1])), mpfr(str(gca_pt1_2_float[2]))]
    gca_pt2_1 = [mpfr(str(gca_pt2_1_float[0])), mpfr(str(gca_pt2_1_float[1])), mpfr(str(gca_pt2_1_float[2]))]
    gca_pt2_2 = [mpfr(str(gca_pt2_2_float[0])), mpfr(str(gca_pt2_2_float[1])), mpfr(str(gca_pt2_2_float[2]))]

    # If gca_pt1_1_float and gca_pt1_2_float are different more than u, then we don't want to proceed
    if check_mpfr_conversion(gca_pt1_1_float, gca_pt1_1) == -1 or check_mpfr_conversion(gca_pt1_2_float, gca_pt1_2) == -1:
        return np.nan
    if check_mpfr_conversion(gca_pt2_1_float, gca_pt2_1) == -1 or check_mpfr_conversion(gca_pt2_2_float, gca_pt2_2) == -1:
        return np.nan


    v1 = mp_cross(gca_pt1_1, gca_pt1_2)
    v2 = mp_cross(gca_pt2_1, gca_pt2_2)
    x_prenorm = mp_cross(v1, v2)
    norm_mp = mp_norm(x_prenorm)
    # normalize the x
    x = normalize_in_place(x_prenorm)


    v1_float = cross_fma(gca_pt1_1_float, gca_pt1_2_float)
    v1_float_err_x = abs((mpfr(v1_float[0]) - v1[0]) / v1[0])
    v1_float_err_y = abs((mpfr(v1_float[1]) - v1[1]) / v1[1])
    v1_float_err_z = abs((mpfr(v1_float[2]) - v1[2]) / v1[2])

    v2_float = cross_fma(gca_pt2_1_float, gca_pt2_2_float)
    v2_float_err_x = abs((mpfr(v2_float[0]) - v2[0]) / v2[0])
    v2_float_err_y = abs((mpfr(v2_float[1]) - v2[1]) / v2[1])
    v2_float_err_z = abs((mpfr(v2_float[2]) - v2[2]) / v2[2])

    x_float_prenorm = cross_fma(v1_float, v2_float)

    # Find the relative error of the prenorm
    rel_err_prenorm_x = abs((mpfr(x_float_prenorm[0]) - x_prenorm[0]) / x_prenorm[0])
    rel_err_prenorm_y = abs((mpfr(x_float_prenorm[1]) - x_prenorm[1]) / x_prenorm[1])
    rel_err_prenorm_z = abs((mpfr(x_float_prenorm[2]) - x_prenorm[2]) / x_prenorm[2])


    norm_faith = norm_faithful(x_float_prenorm)

    #Normalize the x_float using our norm_faithful
    x_float = x_float_prenorm / norm_faithful(x_float_prenorm)


    np.set_printoptions(precision=16)
    x_float_str = [str(x_float[0]), str(x_float[1]), str(x_float[2])]
    x_float_mpfr = [mpfr(x_float_str[0]), mpfr(x_float_str[1]), mpfr(x_float_str[2])]

    # Now calculate the relative error

    rel_err_x = abs((mpfr(x_float_str[0]) - x[0]) / x[0])
    rel_err_y = abs((mpfr(x_float_str[1]) - x[1]) / x[1])
    rel_err_z = abs((mpfr(x_float_str[2]) - x[2]) / x[2])

    # Get the max relative error
    rel_err = max(rel_err_x, rel_err_y, rel_err_z)
    return rel_err

def naive_intersection(gca_pt1_1, gca_pt1_2, gca_pt2_1, gca_pt2_2):
    v1 = mp_cross(gca_pt1_1, gca_pt1_2)
    v2 = mp_cross(gca_pt2_1, gca_pt2_2)
    x = mp_cross(v1, v2)
    x = normalize_in_place(x)
    # Now convert the input gca as float64
    gca_pt1_1_float = [np.float64(gca_pt1_1[0]), np.float64(gca_pt1_1[1]), np.float64(gca_pt1_1[2])]
    gca_pt1_2_float = [np.float64(gca_pt1_2[0]), np.float64(gca_pt1_2[1]), np.float64(gca_pt1_2[2])]
    gca_pt2_1_float = [np.float64(gca_pt2_1[0]), np.float64(gca_pt2_1[1]), np.float64(gca_pt2_1[2])]
    gca_pt2_2_float = [np.float64(gca_pt2_2[0]), np.float64(gca_pt2_2[1]), np.float64(gca_pt2_2[2])]

    v1_float = np.cross(gca_pt1_1_float, gca_pt1_2_float)
    v2_float = np.cross(gca_pt2_1_float, gca_pt2_2_float)

    x_float = np.cross(v1_float, v2_float)
    x_float = x_float / np.linalg.norm(x_float)
    np.set_printoptions(precision=16)
    x_float_str = [str(x_float[0]), str(x_float[1]), str(x_float[2])]
    x_float_mpfr = [mpfr(x_float_str[0]), mpfr(x_float_str[1]), mpfr(x_float_str[2])]
    # Now calculate the relative error

    rel_err_x = abs((mpfr(x_float_str[0]) - x[0]) / x[0])
    rel_err_y = abs((mpfr(x_float_str[1]) - x[1]) / x[1])
    rel_err_z = abs((mpfr(x_float_str[2]) - x[2]) / x[2])

    # Get the max relative error
    rel_err = max(rel_err_x, rel_err_y, rel_err_z)
    return rel_err

def GCA_length(GCA):
    GCA_cross = mp_cross(GCA[0], GCA[1])
    GCA_cross_norm = mp_norm(GCA_cross)
    GCA_dot = mp_dot(GCA[0], GCA[1])
    GCA_length = gmpy2.atan2(GCA_cross_norm, GCA_dot)
    return GCA_length

# Function to concatenate arrays
def concatenate_gca(lon1, lat1, lon2, lat2):
    GCA = []
    for i in range(len(lon1)):
        point1 = node_lonlat_rad_to_xyz(np.array([np.deg2rad(lon1[i]), np.deg2rad(lat1[i])]))
        point2 = node_lonlat_rad_to_xyz(np.array([np.deg2rad(lon2[i]), np.deg2rad(lat2[i])]))
        GCA.append([point1, point2])
    return GCA
def plot_comparison():
    num_elements = 10000

    # For GCA_1_v1_lon and GCA_2_v1_lon
    GCA_1_v1_lon = np.linspace(0.1, 89.9, num_elements)
    GCA_2_v1_lon = np.linspace(0.1, 89.9, num_elements)

    # For GCA_1_v1_lat and GCA_2_v2_lat
    GCA_1_v1_lat = np.linspace(44.9, 0, num_elements)
    GCA_2_v2_lat = np.linspace(44.9, 0, num_elements)

    # For GCA_1_v2_lon and GCA_2_v2_lon
    GCA_1_v2_lon = np.linspace(179.1, 89.9, num_elements)
    GCA_2_v2_lon = np.linspace(179.1, 89.9, num_elements)

    # For GCA_1_v2_lat and GCA_2_v1_lat
    GCA_1_v2_lat = np.linspace(-44.9, 0, num_elements)
    GCA_2_v1_lat = np.linspace(-44.9, 0, num_elements)

    # Concatenating the data arrays
    GCA_1 = concatenate_gca(GCA_1_v1_lon, GCA_1_v1_lat, GCA_1_v2_lon, GCA_1_v2_lat)
    GCA_2 = concatenate_gca(GCA_2_v1_lon, GCA_2_v1_lat, GCA_2_v2_lon, GCA_2_v2_lat)

    GCA_1_mpfr = convert_to_multiprecision(np.array(GCA_1), str_mode=False)
    GCA_2_mpfr = convert_to_multiprecision(np.array(GCA_2), str_mode=False)

    naive_rel_err = []
    accute_rel_err = []
    gca_lengths = []

    for i in range(len(GCA_1)):
        if i == 2:
            pass
        naive_rel_err.append(naive_intersection(GCA_1[i][0], GCA_1[i][1], GCA_2[i][0], GCA_2[i][1]))
        accute_rel_err.append(accute_intersection(GCA_1_mpfr[i][0], GCA_1_mpfr[i][1], GCA_2_mpfr[i][0], GCA_2_mpfr[i][1]))
        gca_lengths.append(max(abs(GCA_length(GCA_1[i])), abs(GCA_length(GCA_2[i]))))

    # Now we plot the results

    # Remove the 'nan' values from naive_rel_err and accute_rel_err, and gca_lengths
    # Filtering out 'nan' values
    filtered_naive_rel_err = []
    filtered_accute_rel_err = []
    filtered_gca_lengths = []
    for i in range(len(accute_rel_err)):
        if not gmpy2.is_nan(accute_rel_err[i]):
            filtered_naive_rel_err.append(naive_rel_err[i])
            filtered_accute_rel_err.append(accute_rel_err[i])
            filtered_gca_lengths.append(gca_lengths[i])

    # Now we plot the results
    plt.figure(figsize=(12, 6))
    plt.scatter(filtered_gca_lengths, filtered_naive_rel_err, marker='o', label='Naive Method', linestyle='-',alpha=0.3)
    plt.scatter(filtered_gca_lengths, filtered_accute_rel_err, marker='.', label='Our Method', linestyle='-',alpha=0.3)

    # Plotting maximum relative error lines
    max_naive_rel_err = max(filtered_naive_rel_err)
    max_accute_rel_err = max(filtered_accute_rel_err)

    plt.axhline(y=max_naive_rel_err, color='#1f77b4', linestyle='--', linewidth=5)
    plt.axhline(y=max_accute_rel_err, color='#ff7f0e', linestyle='--', linewidth=5)

    tick_size = 18
    label_fontsize = 18  # Adjust as needed
    title_fontsize = 20  # Adjust as needed

    # Adding text for maximum values with a white background
    plt.text(x=2, y=max_naive_rel_err ,
             s=f"Naive Max Rel_Err: {max_naive_rel_err:.2e}",
             color='#1f77b4',
             verticalalignment='bottom',
             horizontalalignment='center',
             fontsize=tick_size,
             fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    plt.text(x=2, y=max_accute_rel_err,
             s=f"Our Max Rel_Err: {max_accute_rel_err:.2e}",
             color='#ff7f0e',
             verticalalignment='bottom',
             horizontalalignment='center',
             fontsize=tick_size,
             fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='none', pad=3.0))

    ax = plt.gca()  # Get current axis
    ax.set_yscale('log')

    plt.title('Comparison of Different Two-GCAs Intersection Methods (10000 Arcs Exp)', fontsize=title_fontsize)
    plt.xlabel('GCA Length in radians', fontsize=label_fontsize)
    plt.ylabel('Relative Error (log scale)', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # fma_cross_exp()
    # naive_cross_exp()
    # calculate_rel_error()
    # closed_gca_gca_intersection()
    # long_gca_gca_intersection()
    plot_comparison()
