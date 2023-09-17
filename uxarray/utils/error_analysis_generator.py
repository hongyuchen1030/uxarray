
from gmpy2 import mpfr
ULP_64BITS = 6 * 10**-16
ULP_32BITS = 6 * 10**-8

def multiply_relative_error(x, y):
    # The input x and y are in the mpfr format
    return (x * y) * ULP_64BITS
