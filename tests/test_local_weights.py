"""
Test `local_weights`
"""
import numpy as np
import numpy.typing as npt
import pytest
from ensemblelearning.local_weights import local_weights


def test_scaling():
    """
    Test that the local weights are invariant under scaling.
    """
    predictions: list[float] = [1.0, 3.0, 5.0, 7.0, 9.0]
    target = 5.0
    scaling_factor = 2.0

    weights = local_weights(predictions=predictions, target=target)
    scaled_weights = local_weights(
        [x * scaling_factor for x in predictions], target * scaling_factor
    )

    np.testing.assert_allclose(weights, scaled_weights, rtol=1e-5, atol=1e-8)


def test_sum_to_one():
    """
    Make sure that the local weights sum up to one.
    """
    predictions: list[float] = [2.0, 4.0, 6.0, 8.0, 10.0]
    target = 7.0
    alpha = 0.1

    weights = local_weights(predictions=predictions, target=target, alpha=alpha)
    assert np.sum(weights) == pytest.approx(1, abs=1e-8)


def test_uniform_weights():
    """
    Check that `alpha`=1 leads to uniform weights.
    """
    predictions: list[float] = [1.0, 2.0, 3.0]
    target = 2.0

    weights = local_weights(predictions=predictions, target=target, alpha=1, beta=1)
    expected_weights: npt.NDArray[np.float64] = np.full(3, 1 / 3)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-5, atol=1e-8)


def test_best_weight():
    """
    Check that `alpha`=0 leads to the best prediction receiving all the weight.
    """
    predictions: list[float] = [1.0, 2.0, 3.0]
    target = 2.0

    weights = local_weights(predictions, target, alpha=0.0, beta=1.0)
    expected_weights: npt.NDArray[np.float64] = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-5, atol=1e-8)
