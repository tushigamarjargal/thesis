# %% [markdown]
# # LSTM Model v2.0 - Walk-Forward Validation with Hyperparameter Tuning
# # **t+180** Step-Ahead Predictions - Based on Commentator Feedback
# # Python Version: 3.9+

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time # To time execution

# Data and Preprocessing
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split # Needed for tuning split

# LSTM / ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import keras_tuner as kt

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
start_date = "2017-11-09" # Using your specified start date
end_date = "2025-01-01"

# Define Train/Test Split Ratio for the *initial* training phase
train_split_ratio = 0.80
# Define Validation Split Ratio *within* the initial training data for tuning
validation_split_ratio_for_tuning = 0.20

# LSTM Network Parameters
look_back = 60 # Input sequence length

# --- Walk-Forward Forecast Horizon ---
h = 180 # <<--- CHANGE: Set horizon to 180 steps ahead ---<<
print(f"Setting Walk-Forward Horizon to: h = {h}")

# Keras Tuner Configuration
MAX_TRIALS = 10
EXECUTIONS_PER_TRIAL = 2
TUNER_EPOCHS = 50
TUNER_PATIENCE = 5

# Final Training Configuration
FINAL_TRAINING_EPOCHS = 100
FINAL_TRAINING_PATIENCE = 10

# Optional: Retraining configuration
RETRAIN_FREQUENCY = 0
RETRAIN_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32 # Used if batch size not tuned or tuner fails

# %% [markdown]
# ## 3. Data Loading and Preparation

# %%
print(f"--- Loading Data for {ticker} ---")
try:
    df_full = yf.download(tickers=[ticker], start=start_date, end=end_date, progress=False)
    if df_full.empty: raise ValueError(f"No data downloaded for {ticker}.")
    if 'Close' not in df_full.columns: raise ValueError(f"'Close' column not found.")
    df_full = df_full[['Close']].copy()
    df_full = df_full.asfreq('D')
    df_full.ffill(inplace=True); df_full.dropna(inplace=True)
    if df_full.empty: raise ValueError(f"Data became empty.")
    print(f"Loaded {len(df_full)} data points for {ticker} from {df_full.index.min()} to {df_full.index.max()}.")
except Exception as e:
    raise ValueError(f"Failed to load data for {ticker}: {e}")

# %% [markdown]
# ## 4. Data Splitting (Adjusted for h-step Evaluation)

# %%
# Split Data into initial train+val and test sets
n_total = len(df_full)
n_train_val = int(train_split_ratio * n_total) # Train+Val split point
# Adjust n_test: number of times we initiate an h-step forecast
n_test = n_total - n_train_val - h + 1 # Can evaluate prediction for t+h up to this point

if n_test <= 0:
     raise ValueError(f"Not enough data for walk-forward with h={h}. Need at least {n_train_val + h} total points.")

train_val_data_df = df_full[:n_train_val] # Data for initial training and tuning
test_data_full = df_full[n_train_val:] # Full available test period data (used for actuals)

print(f"\nInitial Train+Validation Data: {n_train_val} points ({train_val_data_df.index.min().strftime('%Y-%m-%d')} to {train_val_data_df.index.max().strftime('%Y-%m-%d')})")
print(f"Test Data Available: {len(test_data_full)} points")
print(f"Number of walk-forward steps (predictions to generate & evaluate): {n_test}")
# Ensure indices exist before formatting dates
if h > 0 and len(test_data_full) >= h:
    print(f"Evaluation Period Start (target t+{h}): {test_data_full.index[h-1].strftime('%Y-%m-%d')}")
    print(f"Evaluation Period End (target t+{h}): {test_data_full.index[-1].strftime('%Y-%m-%d')}")
else:
    print("Evaluation Period cannot be determined due to insufficient test data for horizon.")


# %% [markdown]
# ## 5. Scaling (Fit on Initial Train Portion Only)

# %%
print("\n--- Scaling Data ---")
# Further split train_val_data into train and validation for tuning scaler fit
n_val_tune = int(validation_split_ratio_for_tuning * n_train_val)
n_train_tune = n_train_val - n_val_tune

train_tune_values_for_scaler = train_val_data_df['Close'].values[:n_train_tune].reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_tune_values_for_scaler)
print("Scaler fitted on initial training portion (excluding validation).")

# Transform the entire dataset
scaled_data = scaler.transform(df_full['Close'].values.reshape(-1, 1))

# Separate scaled train/validation portions needed for tuning
scaled_train_tune_data = scaled_data[:n_train_tune]
# scaled_val_tune_data = scaled_data[n_train_tune:n_train_val] # Not needed directly after sequences are made

# %% [markdown]
# ## 6. Sequence Generation Function

# %%
def create_sequences(data, look_back):
    X, Y = [], []
    if data.ndim == 1: data = data.reshape(-1, 1)
    if len(data) <= look_back: return np.array(X), np.array(Y) # Handle short data
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

# %% [markdown]
# ## 7. Prepare Data for Tuning

# %%
print("\n--- Preparing Sequences for Tuning ---")
x_train_tune, y_train_tune = create_sequences(scaled_train_tune_data, look_back)
val_tune_data_for_seq = scaled_data[n_train_tune - look_back : n_train_val, :]
x_val_tune, y_val_tune = create_sequences(val_tune_data_for_seq, look_back)

if x_train_tune.size > 0: x_train_tune = np.reshape(x_train_tune, (x_train_tune.shape[0], x_train_tune.shape[1], 1))
if x_val_tune.size > 0: x_val_tune = np.reshape(x_val_tune, (x_val_tune.shape[0], x_val_tune.shape[1], 1))

print('x_train_tune shape:', x_train_tune.shape); print('y_train_tune shape:', y_train_tune.shape)
print('x_val_tune shape:', x_val_tune.shape); print('y_val_tune shape:', y_val_tune.shape)
if not (x_train_tune.size > 0 and x_val_tune.size > 0): raise ValueError("Insufficient data for tuning sequences.")

# %% [markdown]
# ## 8. LSTM Model Building Function for Keras Tuner

# %%
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32), return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.3, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.3, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(1))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_batch_size = hp.Choice('batch_size', values=[16, 32, 64]) # Optional: tune batch size
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mean_squared_error')
    return model

# %% [markdown]
# ## 9. Hyperparameter Tuning

# %%
print("\n--- Starting Hyperparameter Search with Keras Tuner ---")
# Update directory/project name for h=180
tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=MAX_TRIALS, executions_per_trial=EXECUTIONS_PER_TRIAL,
                        directory='keras_tuner_lstm_wf_h180', project_name=f'{ticker}_lstm_wf_tuning_h180', overwrite=True)
tuner_early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=TUNER_PATIENCE)
tuner.search(x_train_tune, y_train_tune, epochs=TUNER_EPOCHS, validation_data=(x_val_tune, y_val_tune),
             callbacks=[tuner_early_stopping], verbose=1)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
--- Hyperparameter Search Complete ---
Best Hyperparameters Found:
- LSTM Layer 1 Units: {best_hps.get('units_1')} | Dropout 1: {best_hps.get('dropout_1'):.2f}
- LSTM Layer 2 Units: {best_hps.get('units_2')} | Dropout 2: {best_hps.get('dropout_2'):.2f}
- Dense Layer Units: {best_hps.get('dense_units')} | Learning Rate: {best_hps.get('learning_rate')}
""")
tuned_batch_size = best_hps.get('batch_size') if best_hps.get('batch_size') else DEFAULT_BATCH_SIZE

# %% [markdown]
# ## 10. Train Final Initial Model with Best Hyperparameters

# %%
print("\n--- Training Final Initial LSTM Model ---")
start_time_initial_train = time.time()
scaled_train_val_data = scaled_data[:n_train_val]
x_train_val_final, y_train_val_final = create_sequences(scaled_train_val_data, look_back)
if x_train_val_final.size == 0: raise ValueError("Failed to create sequences for final initial training.")
x_train_val_final = np.reshape(x_train_val_final, (x_train_val_final.shape[0], x_train_val_final.shape[1], 1))
final_lstm_model = tuner.hypermodel.build(best_hps)
final_early_stopping = callbacks.EarlyStopping(monitor='loss', patience=FINAL_TRAINING_PATIENCE, restore_best_weights=True)
print(f"Training final initial LSTM on {len(x_train_val_final)} sequences...")
history_final = final_lstm_model.fit(x_train_val_final, y_train_val_final, epochs=FINAL_TRAINING_EPOCHS,
                                     batch_size=tuned_batch_size, callbacks=[final_early_stopping], verbose=1)
end_time_initial_train = time.time()
print(f"Final Initial LSTM Training Complete in {end_time_initial_train - start_time_initial_train:.2f} seconds.")
final_lstm_model.summary()

# %% [markdown]
# ## 11. Walk-Forward Validation (Rolling Forecast) Loop - t+h

# %%
print(f"\n--- Starting LSTM Walk-Forward Validation for {n_test} steps (Predicting {h} steps ahead) ---")
start_time_walk_forward = time.time()

lstm_walk_forward_predictions_h_step = [] # List to store unscaled h-step ahead predictions
history_scaled_buffer = scaled_data[:n_train_val].flatten().tolist() # Initialize with train+val data

for i in range(n_test):
    # 1. Prepare the input sequence from the end of the current history buffer
    if len(history_scaled_buffer) < look_back:
        raise IndexError(f"History buffer length ({len(history_scaled_buffer)}) too short for look_back ({look_back}) at step {i+1}.")
    current_input_sequence = np.array(history_scaled_buffer[-look_back:]).reshape((1, look_back, 1))

    # 2. Iteratively predict 'h' steps ahead
    h_step_predictions_scaled = []
    temp_input_sequence = current_input_sequence.copy() # Use a copy for internal loop

    for step in range(h): # Inner loop for h steps
        next_pred_scaled = final_lstm_model.predict(temp_input_sequence, verbose=0)[0, 0]
        h_step_predictions_scaled.append(next_pred_scaled)
        # Update the temporary sequence for the *next internal prediction*
        next_pred_scaled_reshaped = np.array([[[next_pred_scaled]]]) # Shape (1, 1, 1) for appending
        temp_input_sequence = np.concatenate(
            (temp_input_sequence[:, 1:, :], next_pred_scaled_reshaped), axis=1
        )

    # 3. Store only the prediction for the target step 'h' (index h-1)
    pred_target_h_scaled = h_step_predictions_scaled[h-1]

    # 4. Inverse transform the h-step ahead prediction
    pred_target_h_unscaled = scaler.inverse_transform([[pred_target_h_scaled]])[0, 0]
    lstm_walk_forward_predictions_h_step.append(pred_target_h_unscaled)

    # --- Update MAIN History Buffer with ACTUAL value ---
    # 5. Get the ACTUAL scaled value for the current time step `i` that just occurred
    current_actual_index = n_train_val + i
    actual_scaled_value_i = scaled_data[current_actual_index, 0]

    # 6. Append the *actual* scaled value to the main history buffer
    history_scaled_buffer.append(actual_scaled_value_i)

    # --- Optional: Periodic Retraining ---
    if RETRAIN_FREQUENCY > 0 and (i + 1) % RETRAIN_FREQUENCY == 0 and (i + 1) < n_test:
        print(f"\n--- Retraining LSTM at step {i+1}/{n_test} ---")
        # Retraining logic using history_scaled_buffer... (will be slow)
        # ...
        print("Retraining complete.")
    # --- End Optional Retraining ---

    # Optional: Log progress (adjust frequency for longer runs)
    elif (i + 1) % 50 == 0 or (i+1) == n_test : # Log every 50 steps or on the last step
        print(f"LSTM Walk-Forward (h={h}) Step {i+1}/{n_test} complete.")


end_time_walk_forward = time.time()
total_walk_forward_time = end_time_walk_forward - start_time_walk_forward
print(f"\nLSTM Walk-Forward (h={h}) finished in {total_walk_forward_time:.2f} seconds.")

lstm_walk_forward_predictions_h_step = np.array(lstm_walk_forward_predictions_h_step)

# %% [markdown]
# ## 12. Evaluate Walk-Forward Performance (t+h)

# %%
# Define evaluation metrics function (reusable)
def evaluate_forecast(y_true, y_pred, model_name, horizon):
    """Calculates and prints standard evaluation metrics."""
    y_true_flat = y_true.flatten(); y_pred_flat = y_pred.flatten()
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    try: r2 = r2_score(y_true_flat, y_pred_flat)
    except ValueError: r2 = np.nan
    print(f"\n--- {model_name} Walk-Forward (t+{horizon}) Evaluation Results ---")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4%}, R²: {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

# Evaluate against the actual unscaled test data, shifted by h-1 steps
y_test_actual_h_step = test_data_full['Close'].values[h-1:]

# Check lengths before evaluation
if len(y_test_actual_h_step) != len(lstm_walk_forward_predictions_h_step):
     raise ValueError(f"Length mismatch after loop: Actual evaluation data ({len(y_test_actual_h_step)}) vs Predictions ({len(lstm_walk_forward_predictions_h_step)})")

lstm_wf_h_results = evaluate_forecast(y_test_actual_h_step, lstm_walk_forward_predictions_h_step, f"LSTM ({ticker})", horizon=h)


# %% [markdown]
# ## 13. Visualize Walk-Forward Results (t+h)

# %%
print("\n--- Plotting Walk-Forward Forecasts ---")
prediction_dates = test_data_full.index[h-1:]
# Align plot data if lengths mismatch
if len(prediction_dates) != len(lstm_walk_forward_predictions_h_step):
     min_plot_len = min(len(prediction_dates), len(lstm_walk_forward_predictions_h_step))
     prediction_dates = prediction_dates[:min_plot_len]
     plot_predictions = lstm_walk_forward_predictions_h_step[:min_plot_len]
     plot_actuals = y_test_actual_h_step[:min_plot_len]
else:
     plot_predictions = lstm_walk_forward_predictions_h_step
     plot_actuals = y_test_actual_h_step

results_df_wf = pd.DataFrame({
    'Actual': plot_actuals.flatten(),
    f'LSTM (t+{h})': plot_predictions.flatten()
}, index=prediction_dates)

fig = go.Figure()
fig.add_trace(go.Scatter(x=results_df_wf.index, y=results_df_wf['Actual'], mode='lines', name='Actual Price (Test)', line=dict(color='black')))
fig.add_trace(go.Scatter(x=results_df_wf.index, y=results_df_wf[f'LSTM (t+{h})'], mode='lines', name=f'LSTM Walk-Forward (t+{h})', line=dict(color='green', dash='dash')))
fig.update_layout(
    title=f'LSTM Walk-Forward (t+{h}) Forecast Comparison for {ticker} (Tuned)',
    xaxis_title="Date (Date being forecast)", yaxis_title="Price (USD)", legend_title="Data/Model", template="plotly_white"
)
fig.show()

# %% [markdown]
# ## 14. Walk-Forward Evaluation Period Summary

# %%
print(f"\n--- Walk-Forward Evaluation Summary ---")
print(f"Initial Train+Validation Data End Date: {train_val_data_df.index.max().strftime('%Y-%m-%d')}")
print(f"Walk-Forward Evaluation Period (Test Set Dates): {test_data_full.index.min().strftime('%Y-%m-%d')} to {test_data_full.index.max().strftime('%Y-%m-%d')}")
print(f"Number of Walk-Forward Steps Performed: {n_test}")
print(f"Forecast Horizon Evaluated at each Step: h = {h}")
# Ensure indices exist before formatting dates
if h > 0 and len(test_data_full) >= h:
    print(f"Evaluation Period (Target Dates): {test_data_full.index[h-1].strftime('%Y-%m-%d')} to {test_data_full.index[-1].strftime('%Y-%m-%d')}")
else:
     print("Evaluation Period cannot be determined due to insufficient test data for horizon.")

