import pandas as pd
import numpy as np

input_file = "nse_cleaned_prices.csv"
prices = pd.read_csv(input_file, index_col=0, parse_dates=True)

print(f"Loaded price data for {prices.shape[1]} stocks.")


log_prices = np.log(prices)
log_prices.to_csv("nse_log_prices.csv")


normalized_prices = prices / prices.iloc[0]
normalized_prices.to_csv("nse_normalized_prices.csv")


log_returns = np.log(prices / prices.shift(1)).dropna()
log_returns.to_csv("nse_log_returns.csv")


