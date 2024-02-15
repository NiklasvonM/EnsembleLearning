"""
Test `global_weights`
"""

import numpy as np
import numpy.typing as npt
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

    expected_weights: npt.NDArray[np.float64] = np.array([1.0, 0.0])

    actual_weights = global_weights(predictions=predictions, targets=targets)

    np.testing.assert_almost_equal(actual_weights, expected_weights, decimal=7)


def test_invalid_input_shapes():
    """
    Check that an error is thrown when the length of inputs doesn't match.
    """
    predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
    targets = np.array([2.0, 4.0, 6.0])

    with pytest.raises(ValueError):
        global_weights(predictions=predictions, targets=targets)


def test_negative_coefficients_handling():
    """
    Test that the safety check works correctly:
    If the linear regression would return all negative weights,
    they replaced with uniform weights.
    """
    predictions: npt.NDArray[np.float64] = np.array([[1.0, 2.0]])
    targets: npt.NDArray[np.float64] = np.array([-1.0])

    expected_weights: npt.NDArray[np.float64] = np.full(shape=2, fill_value=0.5)

    actual_weights = global_weights(predictions=predictions, targets=targets)
    np.testing.assert_almost_equal(actual_weights, expected_weights, decimal=7)
