import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import matplotlib.pyplot as plt
from ensemble_learning.global_weights import global_weights

# Load the airline-passenger dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)

# Split the dataset into training and test sets
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

# Fit the ARIMA model
arima_model = ARIMA(train, order=(5,1,0))
arima_model_fit = arima_model.fit()

# Fit the exponential smoothing model
exp_smoothing_model = ExponentialSmoothing(train)
exp_smoothing_model_fit = exp_smoothing_model.fit()

# Fit the Prophet model
prophet_model = Prophet()
prophet_df = pd.DataFrame({'ds': train.index, 'y': [y[0] for y in train.values]})
prophet_model_fit = prophet_model.fit(prophet_df)

# Make predictions for each model
arima_predictions = arima_model_fit.forecast(len(test))
exp_smoothing_predictions = exp_smoothing_model_fit.predict(start=len(train), end=len(train)+len(test)-1)
prophet_predictions = prophet_model_fit.predict(prophet_model_fit.make_future_dataframe(periods=len(test), freq='MS'))['yhat'][len(train):]

# Compute the weights for the models
predictions = np.array([arima_predictions, exp_smoothing_predictions, prophet_predictions])
weights = global_weights(np.transpose(predictions), [y[0] for y in test.values])

# Print the weights for each model
print('Weights for ARIMA, exponential smoothing, and Prophet:', weights.round(4))

# Make a plot of the actual and predicted values for each model
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, arima_predictions, label='ARIMA')
plt.plot(test.index, exp_smoothing_predictions, label='Exponential smoothing')
plt.plot(test.index, prophet_predictions, label='Prophet')
plt.legend()
plt.show()
