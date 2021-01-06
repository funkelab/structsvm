import logging
import numpy as np
import pylp

logger = logging.getLogger(__name__)


class BundleMethod:
    '''Create a new bundle method for the given value and gradient callback.

    Args:

        value_gradient_callback:

           A function returning the value and gradient at a given position w:
           ``value, gradient = value_gradient_callback(w)``

        dims:

           The size of the vector w.

        regularizer_weight:

           The weight of the quadratic regularizer.

        eps:

           Convergence threshold.
    '''

    def __init__(
            self,
            value_gradient_callback,
            dims,
            regularizer_weight,
            eps):

        self._value_gradient_callback = value_gradient_callback
        self._dims = dims
        self._lambda = regularizer_weight
        self._eps = eps

        self._solver = pylp.QuadraticSolver(
            dims + 1,
            pylp.VariableType.Continuous)
        # one variable for each component of w and for ξ
        self._objective = pylp.QuadraticObjective(dims + 1)

        self._setup_qp()

    def optimize(self, max_iterations=None):
        '''Find ``w`` that minimizes the function given by
        ``value_gradient_callback``.
        '''

        # 1. w_0 = 0, t = 0
        # 2. t++
        # 3. compute a_t = ∂L(w_t-1)/∂w
        # 4. compute b_t =  L(w_t-1) - <w_t-1,a_t>
        # 5. ℒ_t(w) = max_i <w,a_i> + b_i
        # 6. w_t = argmin λ½|w|² + ℒ_t(w)
        # 7. ε_t = min_i [ λ½|w_i|² + L(w_i) ] - [ λ½|w_t|² + ℒ_t(w_t) ]
        #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
        #            smallest L(w) ever seen    current min of lower bound
        # 8. if ε_t > ε, goto 2
        # 9. return w_t

        # initial w and minimal value
        w = np.zeros((self._dims,), dtype=np.float64)
        min_value = np.inf

        t = 0

        while max_iterations is None or t < max_iterations:

            t += 1

            logger.info("----------------- iteration %d", t)

            w_tm1 = w

            logger.debug("current w is %s", w_tm1)

            # get current value and gradient
            L_w_tm1, a_t = self._value_gradient_callback(w_tm1)

            logger.debug("       L(w)              is: %f", L_w_tm1)
            logger.debug("      ∂L(w)/∂            is: %s", a_t)

            # update smallest observed value of regularized L
            min_value = min(
                min_value,
                L_w_tm1 + 0.5*self._lambda*np.dot(w_tm1, w_tm1))

            logger.debug(" min_i L(w_i) + ½λ|w_i|² is: %f", min_value)

            # compute hyperplane offset
            b_t = L_w_tm1 - np.dot(w_tm1, a_t)

            logger.debug("adding hyperplane %s*w + %f", a_t, b_t)

            # update lower bound
            self._add_hyperplane(a_t, b_t)

            # update w and get minimal value
            w, min_lower = self._find_min_lower_bound()

            logger.debug(" min_w ℒ(w)   + ½λ|w|²   is: %f", min_lower)
            logger.debug(" w* of ℒ(w)   + ½λ|w|²   is: %s", w)

            # compute gap
            eps_t = min_value - min_lower

            logger.info("          ε   is: %f", eps_t)

            # converged?
            if eps_t <= self._eps:

                if eps_t >= 0:

                    logger.info("converged!")

                else:

                    logger.warning("ε < 0 -- something went wrong")
                    logger.warning(
                        "(if |ε| is very small this might still be fine)")

                break

        return w

    def _setup_qp(self):

        # w* = argmin λ½|w|² + ξ, s.t. <w,a_i> + b_i ≤ ξ ∀i

        # regularizer
        for i in range(self._dims):
            self._objective.set_quadratic_coefficient(i, i, 0.5*self._lambda)

        # ξ
        self._objective.set_coefficient(self._dims, 1.0)

        # we minimize
        self._objective.set_sense(pylp.Sense.Minimize)

        # set objective (does not change)
        self._solver.set_objective(self._objective)

    def _add_hyperplane(self, a, b):
        '''Add a hyperplane to the bundle. The hyperplane is parameterized by a
        vector ``a`` and an offset ``b``.
        '''

        # <w,a> + b ≤  ξ
        #       <=>
        # <w,a> - ξ ≤ -b

        constraint = pylp.LinearConstraint()

        for i in range(self._dims):
            constraint.set_coefficient(i, a[i])

        constraint.set_coefficient(self._dims, -1.0)
        constraint.set_relation(pylp.Relation.LessEqual)
        constraint.set_value(-b)

        self._solver.add_constraint(constraint)

    def _find_min_lower_bound(self):

        # solve the QP
        solution, _ = self._solver.solve()

        # read the solution
        w = np.array([solution[i] for i in range(self._dims)])
        value = solution.get_value()

        return w, value
