import numpy as np
from scipy.optimize import linprog

def local_weights(predictions, y, lambda_val=0.1, gamma=1.0):
    assert y > 0, "The target value y must be positive."
    k = len(predictions)
    normalized_predictions = predictions / y

    # Objective function coefficients
    c = np.concatenate((np.zeros(k), np.ones(k), [1])) + gamma * np.abs(normalized_predictions - 1)

    # Constraint matrix
    A_eq = np.concatenate((np.ones((1, k)), np.eye(k)), axis=1)
    A_ub = np.concatenate((np.eye(k), np.eye(k), np.zeros((k, 1))), axis=1)

    # Constraint vectors
    b_eq = np.array([1])
    b_ub = np.zeros(k)

    # Bounds for variables
    bounds = [(0, None) for _ in range(2 * k)] + [(None, None)]

    # Linear programming solution
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    return res.x[:k]
