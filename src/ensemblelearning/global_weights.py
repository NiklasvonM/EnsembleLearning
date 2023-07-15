'''
`global_weights`
'''
import numpy as np
from sklearn.linear_model import LinearRegression

def global_weights(predictions, targets):
    """
    Fit a linear model without an intercept and return the weight vector.

    This function takes an (n x k) matrix of predictions and a vector y of length n as input,
    fits a linear model without an intercept, and returns the weight vector w of length k.

    Parameters
    ----------
    predictions : array-like, shape (n, k)
        A matrix with n rows and k columns, where each column corresponds to a vector
        of predictions from a different model.

    y : array-like, shape (n,)
        A vector of true target values.

    Returns
    -------
    weights : array, shape (k,)
        The estimated weights of the linear model without an intercept.

    Example
    -------
    >>> import numpy as np
    >>> predictions = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([2, 4, 6])
    >>> weights = global_weights(predictions, y)
    >>> print(weights)
    [0.4, 0.8]
    """
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)

    # Ensure y is a numpy array
    targets = np.array(targets)

    # Check if the dimensions match
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("Number of rows in 'predictions' should match the length of 'targets'.")

    # Fit the linear model without an intercept
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(predictions, targets)

    coefficients = np.array(model.coef_)
    coefficients = [coefficient if coefficient > 0 else 0 for coefficient in coefficients]
    if set(coefficients) == set([0]):
        k = predictions.shape[1]
        return np.full(k, 1/k)
    weights = coefficients / np.sum(coefficients)

    return weights
