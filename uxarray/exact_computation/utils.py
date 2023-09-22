import gmpy2
from gmpy2 import mpfr, mpz
import numpy as np
import math
from uxarray.constants import FLOAT_PRECISION_BITS, INT_FILL_VALUE_MPZ, ERROR_TOLERANCE


def set_global_precision(global_precision=FLOAT_PRECISION_BITS):
    """Set the global precision of the mpfr numbers.
    Important Note:
    1. To avoid arithmetic overflow, the global precision should always be higher than any other precision speicified
    in the code.
    2. Modifying the precision by calling this function will modify all following codes running context until
    another call to this function.

    Parameters
    ----------
    global_precision : int, optional
        The global precision of the expected multiprecision float.
        The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.

    Returns
    -------
    None
    """

    gmpy2.get_context().precision = global_precision


def convert_to_multiprecision(input_array,
                              str_mode=True,
                              precision=FLOAT_PRECISION_BITS):
    """Convert a numpy array to a list of mpfr numbers.

    The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.
    https://gmpy2.readthedocs.io/en/latest/mpfr.html
    If the input array contains fill values INT_FILL_VALUE, the fill values will be converted to INT_FILL_VALUE_MPZ,
    which is the multi-precision integer representation of INT_FILL_VALUE.

    Parameters
    ----------
    input_array : numpy array, float/string, shape is arbitrary
        The input array to be converted to mpfr. The input array should be float or string. If the input array is float,
        str_mode should be False. If the input array is string, str_mode should be True.

    str_mode : bool, optional
        If True, the input array should be string when passing into the function.
        If False, the input array should be float when passing into the function.
        str_mode is True by default and is recommended. Because to take advantage of the higher precision provided by
        the mpfr type, always pass constants as strings.
    precision : int, optional
        The precision of the mpfr numbers. The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.

    Returns
    ----------
    mpfr_array : numpy array, mpfr type, shape will be same as the input_array
        The output array with mpfr type, which supports correct
        rounding, selectable rounding modes, and many trigonometric, exponential, and special functions. A context
        manager is used to control precision, rounding modes, and the behavior of exceptions.

    Raises
    ----------
    ValueError
        The input array should be string when str_mode is True, if not, raise
        ValueError('The input array should be string when str_mode is True.')
    """

    # To take advantage of the higher precision provided by the mpfr type, always pass constants as strings.
    # https://gmpy2.readthedocs.io/en/latest/mpfr.html
    flattened_array = np.ravel(input_array)
    mpfr_array = np.array(flattened_array, dtype=object)
    if not str_mode:
        # Cast the input 2D array to string array
        for idx, val in enumerate(flattened_array):
            if gmpy2.cmp(mpz(val), INT_FILL_VALUE_MPZ) == 0:
                mpfr_array[idx] = INT_FILL_VALUE_MPZ
            else:
                decimal_digit = precision_bits_to_decimal_digits(precision)
                format_str = "{0:+." + str(decimal_digit) + "f}"
                val_str = format_str.format(val)
                mpfr_array[idx] = mpfr(val_str, precision)

    else:

        if ~np.all([
                np.issubdtype(type(element), np.str_)
                for element in flattened_array
        ]):
            raise ValueError(
                'The input array should be string when str_mode is True.')
        # Then convert the input array to mpfr array
        for idx, val in enumerate(flattened_array):
            if val == "INT_FILL_VALUE":
                mpfr_array[idx] = INT_FILL_VALUE_MPZ
            else:
                mpfr_array[idx] = mpfr(val, precision)

    mpfr_array = mpfr_array.reshape(input_array.shape)

    return mpfr_array


def unique_coordinates_multiprecision(input_array_mpfr,
                                      precision=FLOAT_PRECISION_BITS):
    """Find the unique coordinates in the input array with mpfr numbers.

    The default precision of an mpfr is 53 bits - the same precision as Python’s `float` type.
    It can recognize the fill values INT_FILL_VALUE_MPZ, which is the multi-precision integer representation of
    INT_FILL_VALUE.

    Parameters:
    ----------
    input_array_mpfr : numpy.ndarray, gmpy2.mpfr type
        The input array containing mpfr numbers.

    precision : int, optional
        The precision in bits used for the mpfr calculations. Default is FLOAT_PRECISION_BITS.

    Returns:
    -------
    unique_arr ： numpy.ndarray, gmpy2.mpfr
        Array of unique coordinates in the input array.

    inverse_indices: numpy.ndarray, int
        The indices to reconstruct the original array from the unique array. Only provided if return_inverse is True.

    Raises
    ----------
    ValueError
        The input array should be string when str_mode is True, if not, raise
        ValueError('The input array should be string when str_mode is True.')
    """

    # Check if the input_array is in th mpfr type
    try:
        # Flatten the input_array_mpfr to a 1D array so that we can check the type of each element
        input_array_mpfr_copy = np.ravel(input_array_mpfr)
        for i in range(len(input_array_mpfr_copy)):
            if type(input_array_mpfr_copy[i]) != gmpy2.mpfr and type(
                    input_array_mpfr_copy[i]) != gmpy2.mpz:
                raise ValueError(
                    'The input array should be in the mpfr type. You can use convert_to_mpfr() to '
                    'convert the input array to mpfr.')
    except Exception as e:
        raise e

    unique_arr = []
    inverse_indices = []
    m, n = input_array_mpfr.shape
    unique_dict = {}
    current_index = 0

    # The decimal digits of the precision
    precision_width = precision_bits_to_decimal_digits(precision)

    for i in range(m):
        # We only need to check the first element of each row since the elements in the same row are the same type
        # (Either mpfr for valid coordinates or INT_FILL_VALUE_MPZ for fill values)
        if type(input_array_mpfr[i][0]) == gmpy2.mpfr:
            format_string = "{0:+." + str(precision_width) + "Uf}"
        elif type(input_array_mpfr[i][0]) == gmpy2.mpz:
            format_string = "{:<+" + str(precision_width) + "d}"
        else:
            raise ValueError(
                'The input array should be in the mpfr/mpz type. You can use convert_to_multiprecision() '
                'to convert the input array to multiprecision format.')
        hashable_row = tuple(
            format_string.format(x) for x in input_array_mpfr[i])

        if hashable_row not in unique_dict:
            unique_dict[hashable_row] = current_index
            unique_arr.append(input_array_mpfr[i])
            inverse_indices.append(current_index)
            current_index += 1
        else:
            inverse_indices.append(unique_dict[hashable_row])

    unique_arr = np.array(unique_arr)
    inverse_indices = np.array(inverse_indices)

    return unique_arr, inverse_indices


def decimal_digits_to_precision_bits(decimal_digits):
    """Convert the number of decimal digits to the number of bits of precision.

    Parameters
    ----------
    decimal_digits : int
        The number of decimal digits of precision

    Returns
    -------
    bits : int
        The number of bits of precision
    """
    bits = math.ceil(decimal_digits * math.log2(10))
    return bits


def precision_bits_to_decimal_digits(precision):
    """Convert the number of bits of precision to the number of decimal digits.

    Parameters
    ----------
    precision : int
        The number of bits of precision

    Returns
    -------
    decimal_digits : int
        The number of decimal digits of precision
    """
    # Compute the decimal digit count using gmpy2.log10()
    log10_2 = gmpy2.log10(gmpy2.mpfr(2))  # Logarithm base 10 of 2
    log10_precision = gmpy2.div(1,
                                log10_2)  # Logarithm base 10 of the precision
    decimal_digits = gmpy2.div(precision, log10_precision)

    # Convert the result to an integer
    decimal_digits = int(math.floor(decimal_digits))

    return decimal_digits


def mp_cross(v1, v2):
    """Compute the cross product of two vectors in multiprecision. Already utilized the FMA operation
    Parameters
    ----------
    v1 : list/np.ndarray
        The first vector
    v2 : list/np.ndarray
        The second vector

    Returns
    -------
    cross_product : np.ndarray
        The cross product of the two vectors"""
    # Calculate the cross product of two vectors
    x = gmpy2.fmms(v1[1], v2[2], v1[2], v2[1])
    y = gmpy2.fmms(v1[2], v2[0], v1[0], v2[2])
    z = gmpy2.fmms(v1[0], v2[1], v1[1], v2[0])
    return np.array([x, y, z])

def mp_dot(v1, v2):
    """Compute the dot product of two vectors in multiprecision. Already utilized the FMA operation
    Parameters
    ----------
    v1 : list/np.ndarray
        The first vector
    v2 : list/np.ndarray
        The second vector

    Returns
    -------
    dot_product : mpfr
        The dot product of the two vectors
    """
    # Calculate the dot product of two vectors
    v1xv2x_v1yv2y = gmpy2.fmma(v1[0], v2[0], v1[1], v2[1])
    dot_product = gmpy2.fma(v1[2], v2[2], v1xv2x_v1yv2y)
    return dot_product

def mp_norm(vector):
    """Compute the norm of a vector in multiprecision.
    Parameters
    ----------
    vector : list/np.ndarray
        The vector

    Returns
    -------
    norm : mpfr
        The norm of the vector
    """
    if vector.ndim == 1:
        # Handle 1D array case
        return gmpy2.sqrt(mp_dot(vector, vector))
    else:
        norm_squared = gmpy2.fsum(mp_dot(v, v) for v in vector)
        return gmpy2.sqrt(norm_squared)

def is_mpfr_array(arr):
    """
    Check if the input array contains elements of mpfr datatype.

    Parameters
    ----------
    arr : numpy.array
        The input array.

    Returns
    -------
    bool
        True if the array contains elements of mpfr datatype, False otherwise.
    """
    is_mpfr = np.vectorize(lambda x: isinstance(x, (gmpy2.mpfr, gmpy2.mpz)))
    return np.any(is_mpfr(arr))

def multi__inv_jacobian(x0, x1, y0, y1, z0, z1, x_i_old, y_i_old):
    # d_dx = (x0 * x_i_old - x1 * x_i_old * z0 + y0 * y_i_old * z1 - y1 * y_i_old * z0 - y1 * y_i_old * z0)
    # d_dy = 2 * (x0 * x_i_old * z1 - x1 * x_i_old * z0 + y0 * y_i_old * z1 - y1 * y_i_old * z0)
    #
    # # row 1
    # J[0, 0] = y_i_old / d_dx
    # J[0, 1] = (x0 * z1 - z0 * x1) / d_dy
    # # row 2
    # J[1, 0] = x_i_old / d_dx
    # J[1, 1] = (y0 * z1 - z0 * y1) / d_dy

    # The Jacobian Matrix
    #jacobian = [[gmpy2.fmms(y0, z1, z0, y1), gmpy2.fmms(z0, x1, x0, z1)], [mpfr('2') * x_i_old, mpfr('2') * y_i_old]]
    x0z1_z0x1 = gmpy2.fmms(x0, z1, z0, x1)
    y0z1_y1z0 = gmpy2.fmms(y0, z1, z0, y1)
    denominator = gmpy2.fmma(x0z1_z0x1, x_i_old, y0z1_y1z0, y_i_old)
    if denominator == 0:
        raise ValueError('The denominator of the inverse Jacobian is zero. The inverse Jacobian is not defined.')
    inverse_jacobian = [[y_i_old/denominator, x0z1_z0x1/(mpfr('2.0') * denominator)],
                        [-x_i_old/denominator, y0z1_y1z0/(mpfr('2.0') * denominator)]]


    return inverse_jacobian


def multi__newton_raphson_solver_for_gca_constLat(init_cart, gca_cart, max_iter=1000, verbose=False, error_tol=mpfr(str(ERROR_TOLERANCE))):
    """
    Multiprecision Solver for the intersection point between a great circle arc and a constant latitude.

    Args:
        init_cart (np.ndarray): Initial guess for the intersection point in mpfr format.
        w0_cart (np.ndarray): First vector defining the great circle arc in mpfr format.
        w1_cart (np.ndarray): Second vector defining the great circle arc in mpfr format.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        error_tol (mpfr, optional): The error tolerance. Defaults to ERROR_TOLERANCE.

    Returns:
        np.ndarray or None: The intersection point or None if the solver fails to converge.
    """
    w0_cart, w1_cart = gca_cart
    y_guess = [init_cart[0], init_cart[1]]
    y_new = y_guess
    constZ = init_cart[2]
    error = gmpy2.inf()

    _iter = 1

    while (gmpy2.cmp(error, error_tol) == 1) and _iter < max_iter:
        f_vector = np.array([
            mp_dot(mp_cross(w0_cart, w1_cart), np.array([y_guess[0], y_guess[1], constZ])),
            y_guess[0] * y_guess[0] + y_guess[1] * y_guess[1] + constZ * constZ - mpfr('1.0')
        ])

        j_inv = multi__inv_jacobian(w0_cart[0], w1_cart[0], w0_cart[1], w1_cart[1], w0_cart[2], w1_cart[2], y_guess[0],
                                 y_guess[1])

        # Calculate y_new using gmpy2.fmma for matrix multiplication and gmpy2.fmms for subtraction
        j_inv_mul_f_vector = [gmpy2.fmma(j_inv[0][0], f_vector[0], j_inv[0][1], f_vector[1]),
                              gmpy2.fmma(j_inv[1][0], f_vector[0], j_inv[1][1], f_vector[1])]
        y_new = [y_guess[0] - j_inv_mul_f_vector[0], y_guess[1] - j_inv_mul_f_vector[1]]

        # Calculate the absolute differences between y_guess and y_new
        abs_diff_0 = abs(y_guess[0] - y_new[0])
        abs_diff_1 = abs(y_guess[1] - y_new[1])

        # Find the maximum absolute difference using gmpy2.maxnum
        error = gmpy2.maxnum(abs_diff_0, abs_diff_1)

        y_guess = y_new

        if verbose:
            # Print out the error using the mpfr format

            print(f"Newton method iter: {_iter}, error: ")
            print("{0:.20Df}".format(error))
        _iter += 1

    return np.append(y_new, constZ)


