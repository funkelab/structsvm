import logging
import numpy as np
import ilpy
from .hamming_costs import HammingCosts

logger = logging.getLogger(__name__)


class SoftMarginLoss:
    '''Implements the soft margin loss, i.e.,

       L(w) = max_y <w,φ(x')y' - φ(x')y> + Δ(y',y)

    for a ground truth y', features φ(x'), and a linear cost function Δ(y', y).
    The set of valid ys is given by linear constraints.

    Args:

        constraints (ilpy.LinearConstraints):

             Constraints on y.

        features (ndarray):

             Features φ(x'), one column per component of y.

        ground_truth (ndarray, binary):

             The ground truth y'.

        costs (class, optional):

             The cost function Δ(y',y) to use. Defaults to Hamming costs.
    '''

    def __init__(
            self,
            constraints,
            features,
            ground_truth,
            costs=None):

        self._num_variables = ground_truth.size

        self._features = features
        self._ground_truth = ground_truth

        # the linear and constant term of the cost function Δ(y',y):
        # Δ(y',y) = <g,y> + b
        if costs is None:
            costs = HammingCosts(ground_truth)
        self._costs = costs
        self._b = self._costs.get_offset()
        self._g = self._costs.get_coefficients()

        # combined features of the ground truth and current y*
        self._d = features@ground_truth

        # setup solver
        self._solver = ilpy.LinearSolver(
            self._num_variables,
            ilpy.VariableType.Binary)
        self._solver.set_constraints(constraints)

        # setup objective
        self._objective = ilpy.LinearObjective(self._num_variables)
        self._objective.set_sense(ilpy.Sense.Maximize)

    def value_and_gradient(self, w):

        # L(w) = max_y <w,φ(x')y' - φ(x')y>     + Δ(y',y)
        #      = max_y <wφ(x'),y'-y>            + Δ(y',y)
        #      = max_y <wφ(x'),y'> - <wφ(x'),y> + Δ(y',y)
        #
        #   f := wφ(x')

        f = w@self._features

        logger.debug("wφ(x') = %s", f)

        #
        #      = max_y <f,y'>  - <f,y> +  Δ(y',y)
        #      = max_y    a    - <f,y> +  b + <g, y>

        a = np.dot(f, self._ground_truth)

        #      = max_y (a + b) + <(g - f),y>

        # update objective
        self._objective.set_constant(a + self._b)

        for i in range(self._num_variables):
            self._objective.set_coefficient(i, self._g[i] - f[i])

        logger.debug("objective is %s", self._objective)

        # solve
        self._solver.set_objective(self._objective)
        solution, _ = self._solver.solve()

        # read optimal value L(w)
        value = solution.get_value()

        # ∂L(w)/∂w = φ(x')y' - φ(x')y*
        #          = d       - e

        # compute gradient
        e = self._features@np.array(solution)
        gradient = self._d - e

        return value, gradient
