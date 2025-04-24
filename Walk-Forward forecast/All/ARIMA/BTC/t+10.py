# %% [markdown]
# # # ARIMA Model for Bitcoin; horizon = 10
# # # Python version 3.11+

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time # To time execution

# Data and Preprocessing
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

# ARIMA
# Ensure pmdarima is installed: pip install pmdarima
import pmdarima as pm # For auto_arima

# Visualization
import plotly.graph_objects as go

# Plotting Style Preferences (Optional)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ## 2. Configuration

# %%
# --- User Defined Parameters ---
ticker = "BTC-USD"
start_date = "2017-11-09"
end_date = "2025-01-01"

# Define Train/Test Split Ratio for the *initial* training phase
train_split_ratio = 0.80

# >>> New Parameter: Forecast Horizon <<<
h = 10
print(f"Setting Walk-Forward Horizon to: h = {h}")

# auto_arima parameters
ARIMA_SEASONAL_PERIOD = 7 # Weekly seasonality (or set to 1 if no seasonality expected)

# Retraining configuration (Optional)
RETRAIN_FREQUENCY = 0 # Set > 0 to enable periodic refitting

# %% [markdown]
# ## 3. Data Loading and Preparation

# %%
print(f"--- Loading Data for {ticker} ---")
try:
    df_full = yf.download(tickers=[ticker], start=start_date, end=end_date, progress=False)
    if df_full.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    # Select 'Close' column directly
    if 'Close' not in df_full.columns:
         raise ValueError(f"'Close' column not found in downloaded data for {ticker}.")
    df_full = df_full[['Close']].copy()

    # Ensure daily frequency and forward fill missing values
    df_full = df_full.asfreq('D')
    df_full.ffill(inplace=True)
    df_full.dropna(inplace=True) # Drop any initial NaNs if ffill wasn't enough
    if df_full.empty:
        raise ValueError(f"Data for {ticker} became empty after processing.")
    print(f"Loaded {len(df_full)} data points for {ticker} from {df_full.index.min().strftime('%Y-%m-%d')} to {df_full.index.max().strftime('%Y-%m-%d')}.")
except Exception as e:
    print(f"Original Error: {e}")
    raise ValueError(f"Failed to load or process data for {ticker}. Check symbol and data source.")

# %% [markdown]
# ## 4. Data Splitting (Adjusted for h-step Evaluation)

# %%
# Split Data into initial training and test sets
n_total = len(df_full)
n_train = int(train_split_ratio * n_total)
# Adjust n_test: number of times we initiate an h-step forecast
n_test = n_total - n_train - h + 1

if n_test <= 0:
     raise ValueError(f"Not enough data for walk-forward with h={h}. Need at least {n_train + h} total points.")

train_data_df = df_full[:n_train]
test_data_full = df_full[n_train:] # Holds all data from the start of the test period

print(f"\nInitial Training Data: {n_train} points ({train_data_df.index.min().strftime('%Y-%m-%d')} to {train_data_df.index.max().strftime('%Y-%m-%d')})")
print(f"Test Data Available (for updates & targets): {len(test_data_full)} points")
print(f"Number of walk-forward steps (predictions to generate & evaluate): {n_test}")
# Determine the actual evaluation period dates
if h > 0 and len(test_data_full) >= h:
    eval_start_date = test_data_full.index[h-1].strftime('%Y-%m-%d')
    eval_end_date = test_data_full.index[-1].strftime('%Y-%m-%d')
    print(f"Evaluation Period Start (target t+{h}): {eval_start_date}")
    print(f"Evaluation Period End (target t+{h}): {eval_end_date}")
else:
    print("Evaluation Period cannot be determined due to insufficient test data for horizon.")


# %% [markdown]
# ## 5. Stationarity Check (on Initial Training Data)

# %%
def check_stationarity(timeseries):
    """Performs and prints ADF test results."""
    print("\nResults of Dickey-Fuller Test:")
    # Ensure input is pandas Series or numpy array
    ts_values = timeseries.values if isinstance(timeseries, pd.Series) else timeseries
    dftest = adfuller(ts_values, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=["Test Statistic", "p-value", "#Lags Used", "# Observations Used"],
    )
    for key, value in dftest[4].items():
        dfoutput[f"Critical Value ({key})"] = value
    print(dfoutput.to_string()) # Use to_string for clean print
    if dftest[1] <= 0.05:
        print("=> Conclusion: Data is likely Stationary (reject H0)")
    else:
        print("=> Conclusion: Data is likely Non-Stationary (fail to reject H0)")

print("\n--- Stationarity Check on Initial Training Data ---")
check_stationarity(train_data_df['Close'])

# %% [markdown]
# ## 6. Initial ARIMA Model Fit (using auto_arima)

# %%
print("\n--- Fitting Initial ARIMA Model on Training Data ---")
start_time_initial_train = time.time()

# Use auto_arima to find the best model once on the initial training data
# Note: Consider adjusting max_p, max_q etc. if needed based on ACF/PACF or domain knowledge
arima_model = pm.auto_arima(train_data_df['Close'],
                           start_p=1, start_q=1,
                           test='adf',        # Use ADF test to find 'd'
                           max_p=3, max_q=3,  # Maximum p and q to test
                           m=ARIMA_SEASONAL_PERIOD, # Frequency of the series (e.g., 7 for daily weekly seasonality)
                           start_P=0, seasonal=(ARIMA_SEASONAL_PERIOD > 1), # Test seasonal if m > 1
                           d=None,           # Let ADF test determine 'd'
                           D=None,           # Let seasonal test determine 'D'
                           trace=False,       # Don't print fits for every order
                           error_action='ignore', # Don't stop if an order fails
                           suppress_warnings=True, # Don't print warnings
                           stepwise=True)      # Use stepwise algorithm for faster search

end_time_initial_train = time.time()
print(f"Initial ARIMA training finished in {end_time_initial_train - start_time_initial_train:.2f} seconds.")
print("\n--- Initial Best Model Found ---")
print(arima_model.summary())
print(f"\nBest ARIMA Order (p,d,q): {arima_model.order}")
if ARIMA_SEASONAL_PERIOD > 1:
    print(f"Best Seasonal Order (P,D,Q,m): {arima_model.seasonal_order}")

# %% [markdown]
# ## 7. Walk-Forward Validation (Rolling Forecast) Loop - t+h Steps Ahead

# %%
print(f"\n--- Starting ARIMA Walk-Forward Validation for {n_test} steps (Predicting {h} steps ahead) ---")
start_time_walk_forward = time.time()

arima_walk_forward_predictions_h_step = [] # List to store the target h-step ahead predictions

# The arima_model object will be updated iteratively

for t in range(n_test):
    # 1. Predict h steps ahead from the current state of the model
    try:
        # n_periods=h asks for forecasts from t+1 to t+h
        yhat_h_steps = arima_model.predict(n_periods=h)
        # Check if prediction returned enough steps (can fail in rare cases)
        if len(yhat_h_steps) < h:
            raise ValueError(f"Predict returned only {len(yhat_h_steps)} steps, expected {h}")
        # 2. Store only the prediction for the target step 'h' (index h-1)
        prediction_target_h = yhat_h_steps[h-1]
    except Exception as e_pred:
         print(f"Warning: ARIMA predict failed at step {t+1}. Error: {e_pred}. Using last known value as fallback forecast.")
         # Fallback: Use the last observation known to the model before prediction
         # Accessing internal state requires care, using last data point is safer
         last_known_actual = df_full['Close'].iloc[n_train + t - 1] # Data point used for the *previous* update
         prediction_target_h = last_known_actual # Naive forecast as fallback

    arima_walk_forward_predictions_h_step.append(prediction_target_h)

    # 3. Get the ACTUAL value for the current step 't' which just occurred
    # This corresponds to index n_train + t in the full dataframe
    actual_index_t = n_train + t
    actual_value_t = df_full['Close'].iloc[actual_index_t]

    # 4. Update the model with the actual observation.
    # This incorporates the new information for the *next* iteration's prediction
    try:
        # Update using the actual value at index n_train + t
        arima_model.update(actual_value_t)
    except Exception as e_update:
        print(f"Warning: ARIMA update failed at step {t+1} with error: {e_update}. Model state might be stale for next prediction.")
        # Consider more robust error handling if this occurs frequently (e.g., force refit)

    # Log progress periodically
    if (t + 1) % 100 == 0:
        print(f"ARIMA Walk-Forward (h={h}) Step {t+1}/{n_test} complete.")

    # --- Optional Full Refitting Point ---
    # if RETRAIN_FREQUENCY > 0 and (t + 1) % RETRAIN_FREQUENCY == 0 and (t + 1) < n_test:
    #     print(f"\n--- Refitting ARIMA at step {t+1}/{n_test} ---")
    #     refit_start_time = time.time()
    #     current_history = df_full['Close'].iloc[:n_train + t + 1] # Data up to the point just observed
    #     # Re-run auto_arima on the updated history
    #     arima_model = pm.auto_arima(current_history,
    #                                start_p=1, start_q=1, test='adf',
    #                                max_p=3, max_q=3, m=ARIMA_SEASONAL_PERIOD,
    #                                start_P=0, seasonal=(ARIMA_SEASONAL_PERIOD > 1),
    #                                d=None, D=None, trace=False,
    #                                error_action='ignore', suppress_warnings=True,
    #                                stepwise=True)
    #     refit_end_time = time.time()
    #     print(f"ARIMA Refitting complete in {refit_end_time - refit_start_time:.2f} seconds. New order: {arima_model.order}")
    # --- End Optional Refitting ---


end_time_walk_forward = time.time()
total_walk_forward_time = end_time_walk_forward - start_time_walk_forward
print(f"\nARIMA Walk-Forward (h={h}) finished in {total_walk_forward_time:.2f} seconds.")

# Ensure predictions list is numpy array for metrics
arima_walk_forward_predictions_h_step = np.array(arima_walk_forward_predictions_h_step)

# --- FINAL LENGTH CHECK ---
print(f"Length of final predictions generated: {len(arima_walk_forward_predictions_h_step)}")
print(f"Number of walk-forward steps performed: {n_test}")
# They should match if the loop completed correctly

# %% [markdown]
# ## 8. Evaluate Walk-Forward Performance (t+h)

# %%
# Define the evaluation metrics function (reusable)
def evaluate_forecast(y_true, y_pred, model_name, horizon):
    """Calculates and prints standard evaluation metrics."""
    # Ensure inputs are flat numpy arrays
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    # Basic check for NaNs/Infs which can cause issues
    if np.any(np.isnan(y_true_flat)) or np.any(np.isnan(y_pred_flat)) or \
       np.any(np.isinf(y_true_flat)) or np.any(np.isinf(y_pred_flat)):
        print(f"Warning: NaNs or Infs detected in evaluation data for {model_name} (t+{horizon}). Metrics might be unreliable.")
        # Optional: clean or return NaNs
        valid_indices = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat) & ~np.isinf(y_true_flat) & ~np.isinf(y_pred_flat)
        y_true_flat = y_true_flat[valid_indices]
        y_pred_flat = y_pred_flat[valid_indices]
        if len(y_true_flat) == 0:
             print("Evaluation skipped: No valid points after cleaning.")
             return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    # Handle potential zeros in actuals for MAPE
    mask = y_true_flat != 0
    if np.all(mask):
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat))
    else:
        print(f"Warning: Zeros found in actual values. Calculating MAPE on non-zero values only.")
        if np.any(mask):
             mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]))
        else: # All actuals are zero
             mape = np.nan # Or define differently

    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    try:
        # Check if y_true is constant, which makes R2 undefined or misleading
        if np.var(y_true_flat) < 1e-9: # Check for near-zero variance
             r2 = np.nan
             print("Warning: Constant actual values detected. R² score is not meaningful (NaN).")
        else:
             r2 = r2_score(y_true_flat, y_pred_flat)
    except ValueError:
        r2 = np.nan

    print(f"\n--- {model_name} Walk-Forward (t+{horizon}) Evaluation Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.4%}")
    print(f"R²:   {r2:.4f}")
    print(f"Number of evaluation points: {len(y_true_flat)}")
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

# --- Correct Alignment for Evaluation ---
# The prediction made at step `t` (using data up to `n_train + t - 1`)
# forecasts the value for the target date `df_full.index[n_train + t + h - 1]`

# Determine the indices of the actual values corresponding to the predictions
start_actual_idx = n_train + h - 1
# The last prediction corresponds to t = n_test - 1
# Target index for last prediction = n_train + (n_test - 1) + h - 1
# = n_train + (n_total - n_train - h + 1) - 1 + h - 1
# = n_total - 1
# So the end index (exclusive) for slicing is n_total
end_actual_idx = n_total # Exclusive index

# Extract the actual values for comparison
y_test_actual_h_step = df_full['Close'].values[start_actual_idx : end_actual_idx]

# Check lengths before evaluation
if len(y_test_actual_h_step) != len(arima_walk_forward_predictions_h_step):
     # This check should fail only if n_test calculation or loop logic was wrong
     raise ValueError(f"Length mismatch after loop: Actual evaluation data ({len(y_test_actual_h_step)}) vs Predictions ({len(arima_walk_forward_predictions_h_step)})")

arima_wf_h_results = evaluate_forecast(y_test_actual_h_step, arima_walk_forward_predictions_h_step, f"ARIMA ({ticker})", horizon=h)

# %% [markdown]
# ## 9. Visualize Walk-Forward Results (t+h)

# %%
print("\n--- Plotting Walk-Forward Forecasts ---")

# Get the corresponding dates for the actual values being evaluated
prediction_dates = df_full.index[start_actual_idx : end_actual_idx]

# Final check before plotting
if len(prediction_dates) != len(arima_walk_forward_predictions_h_step):
     min_plot_len = min(len(prediction_dates), len(arima_walk_forward_predictions_h_step))
     print(f"Warning: Aligning plot data lengths to {min_plot_len}")
     prediction_dates = prediction_dates[:min_plot_len]
     plot_predictions = arima_walk_forward_predictions_h_step[:min_plot_len]
     plot_actuals = y_test_actual_h_step[:min_plot_len]
else:
     plot_predictions = arima_walk_forward_predictions_h_step
     plot_actuals = y_test_actual_h_step

results_df_wf = pd.DataFrame({
    'Actual': plot_actuals.flatten(),
    f'ARIMA (t+{h})': plot_predictions.flatten()
}, index=prediction_dates)

fig = go.Figure()
# Plot actuals for the evaluation period
fig.add_trace(go.Scatter(x=results_df_wf.index, y=results_df_wf['Actual'], mode='lines', name='Actual Price (Evaluation Period)', line=dict(color='black')))
# Plot the h-step ahead forecasts
fig.add_trace(go.Scatter(x=results_df_wf.index, y=results_df_wf[f'ARIMA (t+{h})'], mode='lines', name=f'ARIMA Walk-Forward (t+{h})', line=dict(color='blue', dash='dash'))) # Changed color

fig.update_layout(
    title=f'ARIMA Walk-Forward (t+{h}) Forecast Comparison for {ticker}',
    xaxis_title="Date (Target Date of Forecast)",
    yaxis_title="Price (USD)",
    legend_title="Data/Model",
    template="plotly_white"
)
fig.show()

# %% [markdown]
# ## 10. Walk-Forward Evaluation Period Summary

# %%
print(f"\n--- Walk-Forward Evaluation Summary ---")
print(f"Initial Training Data End Date: {train_data_df.index.max().strftime('%Y-%m-%d')}")
print(f"Walk-Forward Evaluation Period (Test Set Dates for Updates): {test_data_full.index.min().strftime('%Y-%m-%d')} to {test_data_full.index.max().strftime('%Y-%m-%d')}")
print(f"Number of Walk-Forward Steps Performed: {n_test}")
print(f"Forecast Horizon Evaluated at each Step: h = {h}")
# Ensure indices exist before formatting dates for eval period
if h > 0 and len(test_data_full) >= h:
    print(f"Evaluation Period (Target Dates): {test_data_full.index[h-1].strftime('%Y-%m-%d')} to {test_data_full.index[-1].strftime('%Y-%m-%d')}")
else:
    print("Evaluation Period cannot be determined due to insufficient test data for horizon.")