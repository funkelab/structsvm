import numpy as np
from .linear_costs import LinearCosts


class HammingCosts(LinearCosts):

    def __init__(self, ground_truth):

        # H(y',y) = Σ|y'_i - y_i|
        #         = Σ_{i:y'_i = 1} (1-y_i) + Σ_{i:y'_i = 0} y_i
        #         = |y'|² + Σ_{i:y'_i = 1} -y_i + Σ_{i:y'_i = 0} y_i
        #         =  c    + <l,y>
        #
        #   with l_i := -1 if y'_i = 1
        #                1 else

        coefficients = np.ones_like(ground_truth, dtype=np.float32)
        coefficients -= 2*ground_truth
        offset = np.sum(ground_truth)

        self.set_coefficients(coefficients)
        self.set_offset(offset)
