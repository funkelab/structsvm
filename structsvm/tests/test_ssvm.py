import structsvm as ssvm
import pylp
import unittest
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class TestSsvm(unittest.TestCase):

    def test_chose_one(self):

        # simple ILP to chose exactly one indiciator variable

        num_variables = 10
        num_features = 12
        ground_truth = np.zeros((num_variables,))
        ground_truth[0] = 1

        constraints = pylp.LinearConstraints()
        chose_one = pylp.LinearConstraint()
        for i in range(num_variables):
            chose_one.set_coefficient(i, 1.0)
        chose_one.set_relation(pylp.Relation.Equal)
        chose_one.set_value(1.0)
        constraints.add(chose_one)

        features = np.random.random((num_features, num_variables))

        loss = ssvm.SoftMarginLoss(constraints, features, ground_truth)
        bundle_method = ssvm.BundleMethod(
            loss.value_and_gradient,
            dims=num_features,
            regularizer_weight=0.1,
            eps=1e-6)

        w = bundle_method.optimize(1000)

        costs = features.T@w

        # costs for first variable should be minimal
        for i in range(1, num_variables):
            self.assertGreaterEqual(costs[i], costs[0])
