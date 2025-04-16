#%%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# --- Configuration ---
TICKERS = ["BTC-USD", "ETH-USD", "LTC-USD", "USDT-USD", "XRP-USD"]
START_DATE = "2017-11-09"
END_DATE = "2025-01-01" # yfinance will fetch up to the last available date if this is in the future

# --- 1. Fetch Data ---
print(f"Fetching daily data for {', '.join(TICKERS)} from {START_DATE} to {END_DATE}...")
try:
    # Download historical data for all tickers at once
    # We only need the 'Close' price, so we can specify that
    # However, downloading all OHLCV first is often more robust with yf
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=True)

    # Check if data was downloaded
    if data.empty:
        print("Error: No data downloaded. Check tickers and date range.")
        exit() # Exit if no data

    # Select only the 'Close' prices
    # The resulting DataFrame might have a multi-level index for columns ('Close', 'BTC-USD'), etc.
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close']
    else:
        # Handle case where only one ticker might have been returned successfully (less likely with multiple tickers)
        if 'Close' in data.columns:
             close_prices = data[['Close']] # Keep it as a DataFrame
             # Rename column to the single ticker if needed - get ticker name
             single_ticker = TICKERS[0] if len(TICKERS) == 1 else 'Unknown' # Basic guess
             if len(data.columns) > 1 and len(TICKERS)==1 : # If other columns like Open, High exist for single ticker
                close_prices = data[['Close']]
                close_prices.columns = [single_ticker]
             elif 'Close' in data.columns: # If only Close column exists
                 close_prices = data[['Close']]
                 close_prices.columns = [single_ticker]
             else: # If Close does not exist
                 print("Error: 'Close' column not found in downloaded data.")
                 exit()

        else:
             print("Error: 'Close' column not found in downloaded data.")
             exit()


    # Remove rows where all tickers have NaN Close prices (e.g., market holidays if applicable, unlikely for crypto)
    close_prices.dropna(how='all', inplace=True)

    if close_prices.empty:
        print("Error: No 'Close' price data available for the selected tickers and dates.")
        exit()

    print("Data fetching complete.")

    # --- 2. Display Descriptive Statistics ---
    print("\n--- Descriptive Statistics for Daily Close Prices ---")
    # Using .round(4) for better readability of prices and stats
    print(close_prices.describe().round(4))

    # --- 3. Generate Separate Plots ---
    print("\n--- Generating Plots ---")

    # Get the list of tickers for which we actually have data (columns in close_prices)
    available_tickers = close_prices.columns

    for ticker in available_tickers:
        print(f"Generating plot for {ticker}...")

        # Create a new figure for each plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the 'Close' price for the current ticker
        ax.plot(close_prices.index, close_prices[ticker], label=f'{ticker} Close Price')

        # Customize the plot
        ax.set_title(f'{ticker} Daily Close Price ({START_DATE} to {END_DATE})')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)

        # Specific y-axis adjustment for USDT for better visibility
        if ticker == "USDT-USD":
            try:
                min_val = close_prices[ticker].min()
                max_val = close_prices[ticker].max()
                # Add a small buffer around the min/max unless they are exactly 1.0
                y_min = min(0.95, min_val * 0.99) if min_val < 1.0 else 0.98
                y_max = max(1.05, max_val * 1.01) if max_val > 1.0 else 1.02
                ax.set_ylim(y_min, y_max)
                print(f"   Adjusted Y-axis for {ticker} to ({y_min:.4f}, {y_max:.4f})")
            except Exception as e:
                print(f"   Could not adjust Y-axis for {ticker}: {e}")


        # Improve layout
        fig.tight_layout()

        # Show the plot for the current ticker
        plt.show()

        # Optional: Close the figure after displaying to free memory
        # plt.close(fig)

    print("\n--- All plots generated ---")

except Exception as e:
    print(f"\nAn error occurred during the process: {e}")
# %%
