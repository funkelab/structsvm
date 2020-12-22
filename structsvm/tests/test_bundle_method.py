import structsvm as ssvm
import unittest
import logging

logging.basicConfig(level=logging.INFO)


class TestBundleMethod(unittest.TestCase):

    def test_quadratic(self):

        # f(x) = (x - 1)**2
        def value_gradient(x):
            return (x[0] - 1.0)**2, 2*(x - 1)

        bundle_method = ssvm.BundleMethod(
            value_gradient,
            dims=1,
            regularizer_weight=0.0,
            eps=1e-5)

        w = bundle_method.optimize()
        self.assertAlmostEqual(w[0], 1.0)
