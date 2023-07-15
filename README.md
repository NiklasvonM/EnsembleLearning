[![Pylint](https://github.com/NiklasvonM/EnsembleLearning/actions/workflows/pylint.yml/badge.svg)](https://github.com/NiklasvonM/EnsembleLearning/actions/workflows/pylint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

# EnsembleLearning

## Usage

```python
from ensemble_learning.local_weights import local_weights

predictions = [1, 3, 5, 7, 9]
target = 5

weights = local_weights(predictions, target)
print("Weights:", weights)
```

## Examples

There is an example demonstrating how to use the module with time series forecasting models, such as ARIMA, Exponential Smoothing, and Prophet. To run the example, you need to install additional dependencies:

```bash
pip install ensemble-learning[examples]
python examples/time_series_local_weights.py
```
