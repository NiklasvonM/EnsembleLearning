'''
`local_weights`
'''
import cvxpy as cp
import numpy as np

def local_weights(predictions, target, alpha = 0.2, beta = 1):
    """
    Computes the optimal weights for combining the predictions to match the target value target,
    using linear optimization with a regularization term.

    The optimization problem is formulated as follows:

    Let x_i' = (x_i - min(x_1,...,x_k,y)) / (max(x_1,...,x_k,y) - min(x_1,...,x_k,y))
    Let y' = (y - min(x_1,...,x_k,y)) / (max(x_1,...,x_k,y) - min(x_1,...,x_k,y))

    min  | w_1 * x_1' + ... + w_k * x_k' - y' |
         + lambda * (sum_i=1^k w_i * |x_i' - y'|) / (max(x_1,...,x_k,y) - min(x_1,...,x_k,y))

    subject to w_i >= 0, for all i = 1, ..., k
               sum(w_i) = 1

    where x_i are the predictions, y is the target value
    and lambda is the regularization parameter.

    Args:
        predictions (list[float]): A list of k predictions.
        target (float): The target value.
        alpha (float in [0, 1]): First regularization parameter,
        specifying how much deviation from uniform weights
        is penalized.
        alpha=0 means that a perfect prediction
        receives all the weight and
        alpha=1 means that the weights are (roughly) uniform.

        beta (float]): Second regularization parameter,
        specifying how much a deviation from the target the weighted
        sum of observations is penalized.


    Returns:
        np.ndarray: A numpy array of the optimal weights.
    """

    k = len(predictions)

    # Return uniform weights if all predictions and target coincide
    if len(set(predictions + [target])) == 1:
        return np.full(k, 1/k)

    min_value = min(predictions + [target])
    max_value = max(predictions + [target])
    range_values = max_value - min_value
    scaled_predictions = np.array([(x - min_value) / range_values for x in predictions])
    scaled_target = (target - min_value) / range_values

    weights = cp.Variable(k)
    # Set the initial value of weights to the uniform distribution for a warm start.
    weights.value = np.full(k, 1/k)

    penalty_deviation_from_uniform = cp.sum((weights - np.full(k, 1/k))**2)
    penalty_poor_prediction = cp.sum(
        cp.multiply(weights, cp.abs(scaled_predictions - np.full(k, scaled_target)))) / k
    absolute_error = cp.abs(cp.sum(cp.multiply(weights, scaled_predictions)) - scaled_target)

    objective = cp.Minimize(
        alpha * penalty_deviation_from_uniform +
        (1-alpha) * penalty_poor_prediction +
        beta * absolute_error)

    cp.Problem(
        objective,
        constraints=[weights >= 0, cp.sum(weights) == 1]
    ).solve()

    return weights.value

if __name__ == "__main__":
    w = local_weights([1, 2, 3], 2, alpha=0, beta=1)
    print(f"Weights: {w.round(4)}")
