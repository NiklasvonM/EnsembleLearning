"""
Test `global_weights`
"""

import numpy as np
import pytest
from ensemblelearning.global_weights import global_weights


def test_perfect_model():
    """
    Test that a perfect model receives all the weight.
    """
    random_values1 = np.random.rand(10)
    random_values2 = np.random.rand(10)
    predictions = np.column_stack((random_values1, random_values2))
    targets = random_values1

    expected_weights = np.array([1, 0])

    actual_weights = global_weights(predictions, targets)

    np.testing.assert_almost_equal(actual_weights, expected_weights, decimal=7)


def test_invalid_input_shapes():
    """
    Check that an error is thrown when the length of inputs doesn't match.
    """
    predictions = np.array([[1, 2], [3, 4]])
    targets = np.array([2, 4, 6])

    with pytest.raises(ValueError):
        global_weights(predictions, targets)


def test_negative_coefficients_handling():
    """
    Test that the safety check works correctly:
    If the linear regression would return all negative weights,
    they replaced with uniform weights.
    """
    predictions = np.array([[1, 2]])
    targets = [-1]

    expected_weights = np.full(2, 0.5)

    actual_weights = global_weights(predictions, targets)
    np.testing.assert_almost_equal(actual_weights, expected_weights, decimal=7)
