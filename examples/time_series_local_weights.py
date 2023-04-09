import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet

from weighted_sum_opt import weighted_sum_optimization

# Load Air Passengers dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
data.index.freq = "MS"

# Train-test split
train = data[:int(0.75 * len(data))]
test = data[int(0.75 * len(data)):]

# Train ARIMA model
arima_model = auto_arima(train, seasonal=True, m=12)
arima_forecast = arima_model.predict(n_periods=len(test))

# Train Exponential Smoothing model
ets_model = ExponentialSmoothing(train, seasonal="multiplicative", seasonal_periods=12).fit()
ets_forecast = ets_model.forecast(len(test))

# Train Prophet model
prophet_data = train.reset_index().rename(columns={"Month": "ds", "Passengers": "y"})
prophet_model = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True).fit(prophet_data)
prophet_forecast = prophet_model.predict(pd.DataFrame(test.index, columns=["ds"]))["yhat"].values

# Apply weighted sum algorithm
weights = np.array([weighted_sum_optimization([arima, ets, prophet], y) for arima, ets, prophet, y in zip(arima_forecast, ets_forecast, prophet_forecast, test["Passengers"].values)])

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
