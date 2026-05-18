# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18/05/2025



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
#Developed By: Vignesh M
#Reg No: 21222340176

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset,perform data exploration
# Replaced '/content/AirPassengers.csv' with '/content/supermarket_10years.xlsx'
# The columns in 'supermarket_10years.xlsx' are likely different from 'AirPassengers.csv'.
# You will need to identify the column that contains dates/months and the column that contains the numerical values you want to forecast.

data = pd.read_excel('/content/supermarket_10years.xlsx')

# --- Start of user modification required ---
# The original data has 'Year' and 'Month' columns, and sales are in 'Sales_Amount'.
# We need to combine 'Year' and 'Month' to create a date column.
# 1. Create a combined date string and convert to datetime objects.
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'], format='%Y-%b')

# 2. Set the newly created 'Date' column as the index of the DataFrame.
data = data.set_index('Date')

# 3. Aggregate the 'Sales_Amount' data monthly.
data_monthly = data['Sales_Amount'].resample('MS').sum()
# --- End of user modification required ---

data_monthly.head()
data_monthly.plot()
plt.show()

# Scale the data and check for seasonality
scaler = MinMaxScaler()
# Ensure data_monthly is a Series when flattening and converting back to Series with index
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)
scaled_data.plot()
plt.show()

# The data seems to have additive trend and multiplicative seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(data_monthly, model="additive")
# Added seasonal_periods for robustness, assuming yearly seasonality (12 months)
decomposition = seasonal_decompose(data_monthly, model="additive", period=12)
decomposition.plot()
plt.show()

# Holt-Winters Exponential Smoothing (multiplicative seasonality requires positive values)
scaled_data_for_holtwinters = scaled_data.copy()
# This warning and adjustment is no longer strictly necessary if using seasonal='add' below,
# but keeping it for informational purposes or if 'mul' is reconsidered.
if (scaled_data_for_holtwinters <= 0).any():
    print("Warning: Scaled data contains non-positive values. Adjusting by adding a small constant, though 'add' seasonality is now used.")
    scaled_data_for_holtwinters = scaled_data_for_holtwinters + abs(scaled_data_for_holtwinters.min()) + 0.001

train_data = scaled_data_for_holtwinters[:int(len(scaled_data_for_holtwinters) * 0.8)]
test_data = scaled_data_for_holtwinters[int(len(scaled_data_for_holtwinters) * 0.8):]

# Ensure enough data for seasonal component
if len(train_data) < 2 * 12: # At least two seasonal cycles for good fitting
    print(f"Warning: Train data has only {len(train_data)} data points, which might be insufficient for seasonal_periods=12.")

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot(label="train_data")
test_predictions_add.plot(ax=ax, label="test_predictions_add")
test_data.plot(ax=ax, label="test_data")
ax.legend()
ax.set_title('Visual evaluation')
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"RMSE for test predictions: {rmse}")
print(f"Scaled data variance: {np.sqrt(scaled_data.var())}, mean: {scaled_data.mean()}")

# Final model on the full data for future forecasting
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast for the next 12 months (one year ahead)
forecast_steps = 12
final_predictions = final_model.forecast(steps=forecast_steps)

ax=data_monthly.plot(label="data_monthly")
final_predictions.plot(ax=ax, label="final_predictions")
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Prediction')
plt.show()
```

### OUTPUT:
TEST_PREDICTION
<img width="1185" height="540" alt="image" src="https://github.com/user-attachments/assets/ad4c1dd2-1155-4296-817b-12f599627696" />

<img width="1086" height="556" alt="image" src="https://github.com/user-attachments/assets/2332449e-a268-4b6b-ac95-d601a4fa1033" />

FINAL_PREDICTION

<img width="1484" height="618" alt="image" src="https://github.com/user-attachments/assets/3ef143e6-f24a-4599-8e8a-db21da9fc3d6" />

<img width="1069" height="630" alt="image" src="https://github.com/user-attachments/assets/00964b7a-7173-49f4-868e-f45f878bb6e1" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
