import unittest
import numpy as np
from ensemble_learning.local_weights import local_weights

class TestWeightedSumOptimization(unittest.TestCase):

    def test_scaling(self):
        predictions = [1, 3, 5, 7, 9]
        target = 5
        lambda_val = 0.1
        c = 2

        weights = local_weights(predictions, target, lambda_val)
        scaled_weights = local_weights([x * c for x in predictions], target * c, lambda_val)

        np.testing.assert_allclose(weights, scaled_weights, rtol=1e-5, atol=1e-8)

    def test_sum_to_one(self):
        predictions = [2, 4, 6, 8, 10]
        target = 7
        lambda_val = 0.1

        weights = local_weights(predictions, target, lambda_val)
        self.assertAlmostEqual(np.sum(weights), 1, delta=1e-8)

    def test_uniform_weights(self):
        predictions = [1, 2, 3]
        target = 2

        weights = local_weights(predictions, target, alpha=1, beta=1)
        expected_weights = np.full(3, 1/3)
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-5, atol=1e-8)

    def test_best_weight(self):
        predictions = [1, 2, 3]
        target = 2

        weights = local_weights(predictions, target, alpha=0, beta=1)
        expected_weights = np.array([0, 1, 0])
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
