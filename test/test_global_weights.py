import unittest
import numpy as np
from global_weights import global_weights

class TestGlobalWeights(unittest.TestCase):
    def test_global_weights(self):
        predictions = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([2, 4, 6])

        expected_weights = np.array([0.4, 0.8])

        actual_weights = global_weights(predictions, y)

        np.testing.assert_almost_equal(actual_weights, expected_weights, decimal=7)

    def test_invalid_input_shapes(self):
        predictions = np.array([[1, 2], [3, 4]])
        y = np.array([2, 4, 6])

        with self.assertRaises(ValueError):
            global_weights(predictions, y)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
