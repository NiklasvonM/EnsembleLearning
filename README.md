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
