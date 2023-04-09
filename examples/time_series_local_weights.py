import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from ensemble_learning.local_weights import local_weights

# Load Air Passengers dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
data.index.freq = "MS"

# Train-test split
train = data[:int(0.75 * len(data))]
test = data[int(0.75 * len(data)):]

# Fit the ARIMA model
arima_model = ARIMA(train, order=(5,1,0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(len(test))

# Train Exponential Smoothing model
ets_model = ExponentialSmoothing(train, seasonal="multiplicative", seasonal_periods=12).fit()
ets_forecast = ets_model.forecast(len(test))

# Fit the Prophet model
prophet_model = Prophet()
prophet_df = pd.DataFrame({'ds': train.index, 'y': [y[0] for y in train.values]})
prophet_model_fit = prophet_model.fit(prophet_df)
prophet_forecast = prophet_model_fit.predict(prophet_model_fit.make_future_dataframe(periods=len(test), freq='MS'))['yhat'][len(train):]

# Apply weighted sum algorithm
weights = np.array(
    [local_weights([arima, ets, prophet], y) for arima, ets, prophet, y in zip(
    arima_forecast,
    ets_forecast,
    prophet_forecast,
    test["Passengers"].values)])

# Visualize weights
plt.figure(figsize=(10, 6))
plt.plot(test.index, weights[:, 0], label="ARIMA")
plt.plot(test.index, weights[:, 1], label="Exponential Smoothing")
plt.plot(test.index, weights[:, 2], label="Prophet")
plt.xlabel("Date")
plt.ylabel("Weights")
plt.title("Weights for Different Models")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot([0, 1], [1, 0], 'r-')
plt.plot(weights[:, 0], weights[:, 1], "o")
plt.xlabel("weight ARIMA")
plt.ylabel("weight Exponential Smoothing")
plt.title("Weights for Different Models")
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.show()
