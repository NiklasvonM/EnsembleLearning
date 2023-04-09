import cvxpy as cp
import numpy as np

def local_weights(predictions, y, regularization_parameter):
    """
    Computes the optimal weights for combining the predictions to match the target value y,
    using linear optimization with a regularization term.

    The optimization problem is formulated as follows:

    Let x_i' = (x_i - min(x_1,...,x_k,y)) / (max(x_1,...,x_k,y) - min(x_1,...,x_k,y))
    Let y' = (y - min(x_1,...,x_k,y)) / (max(x_1,...,x_k,y) - min(x_1,...,x_k,y))

    min  | w_1 * x_1' + ... + w_k * x_k' - y' |
         + lambda * (sum_i=1^k w_i * |x_i' - y'|) / (max(x_1,...,x_k,y) - min(x_1,...,x_k,y))

    subject to w_i >= 0, for all i = 1, ..., k
               sum(w_i) = 1

    where x_i are the predictions, y is the target value, and lambda is the regularization parameter.

    Args:
        predictions (list[float]): A list of k predictions.
        y (float): The target value.
        regularization_parameter (float): The regularization parameter (lambda).

    Returns:
        np.ndarray: A numpy array of the optimal weights.
    """

    k = len(predictions)

    # Return uniform weights if all predictions and y coincide
    if len(set(predictions + [y])) == 1:
        return np.full(k, 1/k)

    min_value = min(predictions + [y])
    max_value = max(predictions + [y])
    range_values = max_value - min_value
    scaled_predictions = [(x - min_value) / range_values for x in predictions]

    weights = cp.Variable(k)
    # Set the initial value of weights to the uniform distribution for a warm start.
    weights.value = np.full(k, 1/k)

    absolute_error = cp.abs(cp.sum(weights * scaled_predictions) - 1)
    penalty_term = cp.sum(cp.multiply(weights, cp.abs(scaled_predictions - 1)))
    objective = cp.Minimize(absolute_error + regularization_parameter * penalty_term / range_values)

    constraints = [weights >= 0, cp.sum(weights) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return weights.value
