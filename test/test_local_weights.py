import unittest
import numpy as np

class TestWeightedSumOptimization(unittest.TestCase):

    def test_scaling(self):
        predictions = [1, 3, 5, 7, 9]
        y = 5
        lambda_val = 0.1
        c = 2
        
        weights = weighted_sum_optimization(predictions, y, lambda_val)
        scaled_weights = weighted_sum_optimization([x * c for x in predictions], y * c, lambda_val)
        
        np.testing.assert_allclose(weights, scaled_weights, rtol=1e-5, atol=1e-8)

    def test_sum_to_one(self):
        predictions = [2, 4, 6, 8, 10]
        y = 7
        lambda_val = 0.1

        weights = weighted_sum_optimization(predictions, y, lambda_val)
        self.assertAlmostEqual(np.sum(weights), 1, delta=1e-8)

    def test_known_solution(self):
        predictions = [2, 3, 4]
        y = 3
        lambda_val = 0.1

        weights = weighted_sum_optimization(predictions, y, lambda_val)
        expected_weights = np.array([0.33333333, 0.33333333, 0.33333333])
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
