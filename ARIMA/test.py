
# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm # For auto_arima
import plotly.graph_objects as go

# ## 1. Data Loading and Preparation

# Download Bitcoin data
df_BTC = yf.download(
    tickers=["BTC-USD"],
    start="2020-01-01",
    end="2025-01-01" # Adjust end date if needed, yf downloads up to the day before
)

# Basic Cleaning and Selection
df_BTC.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Rename for clarity
print(f"Original shape: {df_BTC.shape}")
print('Null Values Before:', df_BTC.isnull().values.sum())
# Forward fill common for financial data if few NaNs exist, or drop
df_BTC.ffill(inplace=True) # Or df_BTC.dropna(inplace=True) if preferred
print('Null Values After FFill:', df_BTC.isnull().values.sum())

# Select Target and Set Index
df_BTC.reset_index(inplace=True)
df_BTC['Date'] = pd.to_datetime(df_BTC['Date'], format='%Y-%m-%d')
df_BTC = df_BTC[['Date', 'Close']]
df_BTC.set_index('Date', inplace=True)

# Ensure Daily Frequency
df_BTC = df_BTC.asfreq('D')
# Re-check for NaNs introduced by asfreq (if download missed days)
print('Null Values After AsFreq:', df_BTC.isnull().values.sum())
df_BTC.ffill(inplace=True) # Fill again if necessary
print(f"Shape after preproc: {df_BTC.shape}")
print(f"Frequency of the index: {df_BTC.index.freq}")
print("\nData Head:")
print(df_BTC.head())

# ## 2. Data Splitting (Train/Test)

# Define split point (e.g., 80% train, 20% test)
train_size = int(len(df_BTC) * 0.80)
train_data = df_BTC[:train_size]
test_data = df_BTC[train_size:]

print(f"\nTraining Data: {len(train_data)} points ({train_data.index.min()} to {train_data.index.max()})")
print(f"Test Data:     {len(test_data)} points ({test_data.index.min()} to {test_data.index.max()})")

# Define Forecast Horizon
fh = len(test_data) # Forecast horizon matches the length of the test set


# ## 3. Stationarity Check (Optional but Recommended)
# ARIMA models assume stationarity (constant mean, variance, autocorrelation). `auto_arima` can handle differencing automatically, but checking helps understanding.


def check_stationarity(timeseries):
    print("\nResults of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("=> Conclusion: Data is likely Stationary (reject H0)")
    else:
        print("=> Conclusion: Data is likely Non-Stationary (fail to reject H0)")

print("\n--- Stationarity Check on Training Data ---")
check_stationarity(train_data['Close'])
# We expect BTC prices to be non-stationary, requiring differencing (d > 0).


# ## 4. ARIMA Model Building (using auto_arima)
# `auto_arima` searches for the best ARIMA(p,d,q)(P,D,Q)m model based on AIC/BIC.


print("\n--- Running auto_arima ---")
# You might need to adjust 'm' based on suspected seasonality
# m=1 for non-seasonal ARIMA
# m=7 for daily data with weekly seasonality
# m=365 for daily data with yearly seasonality (can be very slow)
auto_model = pm.auto_arima(train_data['Close'],
                           start_p=1, start_q=1,
                           test='adf',       # Use ADF test to find optimal 'd'
                           max_p=3, max_q=3, # Max non-seasonal orders to test
                           m=365,              # Check for weekly seasonality
                           start_P=0, seasonal=True, # Allow seasonal components
                           d=None,           # Let ADF test determine 'd'
                           D=None,           # Let seasonal test determine 'D'
                           trace=True,       # Print models tried
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)    # Use stepwise algorithm for speed

print("\n--- Best Model Found ---")
print(auto_model.summary())

# Extract the best model order found
print(f"\nBest ARIMA Order: {auto_model.order}")
print(f"Best Seasonal Order: {auto_model.seasonal_order}")


# ## 5. Forecasting on Test Set


# Generate predictions for the forecast horizon (length of test set)
# Use the fitted auto_arima model
predictions_arima = auto_model.predict(n_periods=fh)

# Create a pandas Series for the predictions with the correct index
predictions_arima = pd.Series(predictions_arima, index=test_data.index)

print("\n--- ARIMA Predictions (first 5) ---")
print(predictions_arima.head())


# ## 6. Model Evaluation


# Calculate metrics
y_true = test_data['Close']
y_pred = predictions_arima

rmse_arima = np.sqrt(mean_squared_error(y_true, y_pred))
mae_arima = mean_absolute_error(y_true, y_pred)
mape_arima = mean_absolute_percentage_error(y_true, y_pred)
r2_arima = r2_score(y_true, y_pred) # R-squared can be negative for poor models

# Print Evaluation Metrics
print("\n--- ARIMA Model Evaluation Metrics (Hold-out Set) ---")
print(f"Root Mean Squared Error (RMSE): {rmse_arima:.4f}")
print(f"Mean Absolute Error (MAE):   {mae_arima:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_arima:.4%}")
print(f"R-squared (R²):              {r2_arima:.4f}") # R² interpretation depends on context

# Print Metrics for Comparison Table in Thesis
print("\n--- ARIMA Hold-out Metrics (for Thesis Table) ---")
print(f"ARIMA Hold-out RMSE:  {rmse_arima:.4f}")
print(f"ARIMA Hold-out MAE:   {mae_arima:.4f}")
print(f"ARIMA Hold-out MAPE:  {mape_arima:.4%}")
print(f"ARIMA Hold-out R²:    {r2_arima:.4f}")


# ## 7. Visualization


# Prepare data for plotting
plot_train = train_data.reset_index()
plot_test = test_data.reset_index()
plot_pred = pd.DataFrame({'Date': predictions_arima.index, 'Predictions': predictions_arima.values})

# Create Plotly figure
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=plot_train['Date'], y=plot_train['Close'],
                         mode='lines', name='Actual Price (Train)',
                         line=dict(color='blue')))
fig.add_trace(go.Scatter(x=plot_test['Date'], y=plot_test['Close'],
                         mode='lines', name='Actual Price (Test)',
                         line=dict(color='green')))
fig.add_trace(go.Scatter(x=plot_pred['Date'], y=plot_pred['Predictions'],
                         mode='lines', name='ARIMA Predictions',
                         line=dict(color='red', dash='dash')))

# Update layout
fig.update_layout(
    title="Bitcoin Price Forecasting using ARIMA",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Legend",
    template="plotly_white" # Optional: Set a clean template
)

# Show plot
fig.show()


# ## 8. Data Split Summary (for reference)


print("\n--- Data Split Summary ---")
train_start_date = train_data.index.min().strftime('%Y-%m-%d')
train_end_date = train_data.index.max().strftime('%Y-%m-%d')
test_start_date = test_data.index.min().strftime('%Y-%m-%d')
test_end_date = test_data.index.max().strftime('%Y-%m-%d')

print(f"Training Set: {len(train_data)} data points from {train_start_date} to {train_end_date}")
print(f"Test Set:     {len(test_data)} data points from {test_start_date} to {test_end_date}")




# %%
