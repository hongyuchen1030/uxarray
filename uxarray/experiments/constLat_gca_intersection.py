import gmpy2
from gmpy2 import mpfr, fmms
import numpy as np
from uxarray.grid.utils import cross_fma
from uxarray.grid.coordinates import normalize_in_place
from uxarray.exact_computation.utils import set_global_precision, mp_cross, mp_norm

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
    set_global_precision(128)
    u = mpfr(np.finfo(np.float64).eps)
    delta_s_tilde_1= gmpy2.sqrt((mpfr('1.0') + (mpfr('3.0') * u/(mpfr('1.0') +u))))
    delta_s_tilde_2 = mpfr('1.0') + (mpfr('25')/mpfr('8')  ) * u * u
    delta_s_tilde = abs(delta_s_tilde_1 * delta_s_tilde_2 - mpfr('1.0'))
    res = gmpy2.cmp(delta_s_tilde, u)
    delta_long_term = abs((mpfr('1.0') + delta_s_tilde) * (mpfr('1.0') + u)- mpfr('1.0'))
    delta_inv_nxny_1 = mpfr('1.0') + (u / (mpfr('1.0') + u))
    delta_inv_nxny_2 = mpfr('1.0') + (u / (mpfr('2.0') - u))
    delta_inv_nxny = abs((delta_inv_nxny_1 / delta_inv_nxny_2) * (mpfr('1.0') + u - 2 * u * u) - mpfr('1.0'))
    delta_final = delta_long_term + delta_inv_nxny + (u / (mpfr('1.0') + u))
    ratio = delta_final / u
    pass

if __name__ == "__main__":
    # # Values for n_x, n_z, and n_y
    # # Define the input values
    # x_1_values = np.linspace(-1, 1, 10)
    # y_1_values = np.linspace(-1, 1, 10)
    # z_1_values = np.linspace(-1, 1, 10)
    #
    # # Create an array of 3D vectors
    # vectors = np.column_stack((x_1_values, y_1_values, z_1_values))
    #
    # # Normalize each vector using the normalize_in_place function
    # normalized_vectors = np.apply_along_axis(normalize_in_place, 1, vectors)
    #
    # # Extract the normalized x, y, and z values
    # x_1_normalized = normalized_vectors[:, 0]
    # y_1_normalized = normalized_vectors[:, 1]
    # z_1_normalized = normalized_vectors[:, 2]
    #
    # x_2_values = x_1_normalized + 10 * np.finfo(np.float64).eps
    # y_2_values = y_1_normalized + 10 * np.finfo(np.float64).eps
    # z_2_values = z_1_normalized - 10 * np.finfo(np.float64).eps
    #
    # # n_x, n_y, n_z are the cross product of the above values of x, y, z
    # # We are using the cross_fm function defined in uxarray.grid.utils
    # # Create a vectorized version of the cross_fma function
    # vectorized_cross_fma = np.vectorize(cross_fma, signature='(n),(n)->(n)')
    #
    # # Create arrays for x1 and x2 using the input values
    # x1 = np.array([x_1_values, y_1_values, z_1_values]).T
    # x2 = np.array([x_2_values, y_2_values, z_2_values]).T
    #
    # # Calculate n_x, n_y, n_z for all pairs using vectorized function
    # n_values = vectorized_cross_fma(x1, x2)
    #
    # # Extract n_x, n_y, n_z from the resulting array
    # n_x_values, n_y_values, n_z_values = n_values.T
    # n_x_values = np.linspace(-1, 1, 1000)
    # n_y_values = np.linspace(-1, 1, 1000)
    # n_z_values = np.linspace(-1, 1, 1000)
    #
    #
    # z0_values = np.linspace(-1, 1, 1000)
    #
    # # Constant value for u
    # u = np.finfo(np.float64).eps
    #
    # # Iterate over the values of n_x, n_y, n_z and z_0 to get the maximum value of delta_hat_g_minus
    # delta_hat_g_minus_max = 0.0
    # for n_x in n_x_values:
    #     for n_y in n_y_values:
    #         for n_z in n_z_values:
    #             for z_0 in z0_values:
    #                 tilde_s_squared = n_x ** 2 + n_y ** 2 - np.linalg.norm([n_x, n_y, n_z]) ** 2 * z_0 ** 2
    #                 if tilde_s_squared < 0:
    #                     continue
    #                 delta_hat_g_minus_max = max(delta_hat_g_minus_max, delta_hat_g_minus(u, n_x, n_y, n_z, z_0))
    #
    #                 # If the current value of delta_hat_g_minus is the maximum, print the values of n_x, n_y, n_z and z_0 that produced it
    #                 # So as the current value of delta_hat_g_minus is the maximum, the values of n_x, n_y, n_z and z_0 that produced it are the ones that we are looking for
    #                 if delta_hat_g_minus_max == delta_hat_g_minus(u, n_x, n_y, n_z, z_0):
    #                     # Print them in the same line
    #                     print("n_x = ", n_x, ", n_y = ", n_y, ", n_z = ", n_z, ", z_0 = ", z_0,  ", delta_hat_g_minus_max = ", delta_hat_g_minus_max)
    #
    #
    calculate_delta_s_tilde()




