# EnsembleLearning

## Usage

```python
from ensemble-learning import local_weights

predictions = [1, 3, 5, 7, 9]
y = 5
lambda_val = 0.1

weights = local_weights(predictions, y, lambda_val)
print("Weights:", weights)
```



## Examples

There is an example demonstrating how to use the module with time series forecasting models, such as ARIMA, Exponential Smoothing, and Prophet. To run the example, you need to install additional dependencies:

```bash
pip install ensemble-learning[examples]
python examples/time_series_forecast_example.py
```

