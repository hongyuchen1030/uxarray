import gmpy2
import numpy as np
import uxarray.multi_precision_helpers as mph


def incircle_predicate(p1, p2, p3, p4):
    # Calculate the necessary expressions and perform the predicate check
    # Example calculation based on the provided Λ expression:
    term1 = (p1[0] - p4[0]) / (p2[0] - p4[0])
    term2 = (p1[1] - p4[1]) / (p2[1] - p4[1])
    term3 = ((p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2) / ((p2[0] - p4[0]) ** 2 + (p2[1] - p4[1]) ** 2)
    term4 = (p3[1] - p4[1]) / (p2[1] - p4[1])
    term5 = ((p3[0] - p4[0]) ** 2 + (p3[1] - p4[1]) ** 2) / ((p2[0] - p4[0]) ** 2 + (p2[1] - p4[1]) ** 2)
    lambda_prime = term1 * term2 * term3 + term4 * term5

    # Calculate the threshold δ(1) for the predicate check, According to the
    # "https://inria.hal.science/inria-00344297/document" paper, the threshold can be slightly overestimated
    # With a precomputed error for a bounded problem. And we compute the forward error analysis
    # using the fact that all coordinates must be in range [-1, 1]

    return abs(lambda_prime) > threshold
