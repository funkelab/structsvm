from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ilpy
import numpy as np

from .hamming_costs import HammingCosts

if TYPE_CHECKING:
    from .linear_costs import LinearCosts

logger = logging.getLogger("structsvm")


class SoftMarginLoss:
    """Implements the soft margin loss, i.e.,

       L(w) = max_y <w,φ(x')y' - φ(x')y> + Δ(y',y)

    for a ground truth y', features φ(x'), and a linear cost function Δ(y', y).
    The set of valid ys is given by linear constraints.

    Args:

        constraints (ilpy.Constraints):

             Constraints on y.

        features (ndarray):

             Features φ(x'), one column per component of y.

        ground_truth (ndarray, binary):

             The ground truth y'.

        costs (class, optional):

             The cost function Δ(y',y) to use. Defaults to Hamming costs.
    """

    def __init__(
        self,
        constraints: ilpy.Constraints,
        features: np.ndarray,
        ground_truth: np.ndarray,
        costs: LinearCosts | None = None,
    ):
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
        self._scale = self._costs.get_scaling_factor()

        self._b = self._b / self._scale
        self._g = self._g / self._scale
        self._features = self._features / self._scale

        # combined features of the ground truth and current y*
        self._d = self._features @ ground_truth

        # setup solver
        self._solver = ilpy.Solver(self._num_variables, ilpy.VariableType.Binary)
        self._solver.set_constraints(constraints)

        # setup objective
        self._objective = ilpy.Objective(self._num_variables)
        self._objective.set_sense(ilpy.Sense.Maximize)

    def value_and_gradient(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        """Computes the value and gradient of the soft margin loss."""
        # L(w) = max_y <w,φ(x')y' - φ(x')y>     + Δ(y',y)
        #      = max_y <wφ(x'),y'-y>            + Δ(y',y)
        #      = max_y <wφ(x'),y'> - <wφ(x'),y> + Δ(y',y)
        #
        #   f := wφ(x')

        f = w @ self._features

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
        # solve the QP
        solution = self._solver.solve()

        # read optimal value L(w)
        # get_value() excludes the constant term, so add it back
        value = solution.get_value() + a + self._b

        # ∂L(w)/∂w = φ(x')y' - φ(x')y*
        #          = d       - e

        # compute gradient
        e = self._features @ np.array(solution)
        gradient = self._d - e

        return value, gradient
