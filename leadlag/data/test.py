import pandas as pd
import matplotlib.pyplot as plt

from correlation_method import compute_pearson_correlation, compute_lagged_cross_correlation
from cct_method import compute_local_cct_matrix, compute_cr_distance_matrix


returns_df = pd.read_csv("nse_log_returns.csv", index_col=0, parse_dates=True)

stock_a = "TECHM.NS"
stock_b = "SUNPHARMA.NS"

rx = returns_df[stock_a].to_numpy()
ry = returns_df[stock_b].to_numpy()


p_corr = compute_pearson_correlation(rx, ry)
print(f"Pearson Correlation ({stock_a} vs {stock_b}): {p_corr:.4f}")

lag_results = compute_lagged_cross_correlation(rx, ry, max_lag=15)
print(f"Optimal Fixed Lag: {lag_results['optimal_lag']} days "
      f"(Correlation: {lag_results['max_correlation']:.4f})")

k = 50
print(f"Computing CCT matrix over window k={k}...")
cct_mat = compute_local_cct_matrix(rx, ry, k=k)
cr_mat = compute_cr_distance_matrix(cct_mat)

print(f"CCT Matrix Shape: {cct_mat.shape}")
print(f"CR Distance Range: [{cr_mat.min():.2f}, {cr_mat.max():.2f}]")


plt.figure(figsize=(12, 5))

# Plot Lagged Correlation Curve
plt.subplot(1, 2, 1)
plt.stem(lag_results["lags"], lag_results["correlations"])
plt.title(f"Lagged Cross-Correlation\n({stock_a} vs {stock_b})")
plt.xlabel("Lag (Days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

# Plot CCT Matrix Heatmap
plt.subplot(1, 2, 2)
plt.imshow(cct_mat, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Local CCT")
plt.title(f"Local CCT Matrix (k={k})\n({stock_a} vs {stock_b})")
plt.xlabel(f"{stock_b} Timestamp")
plt.ylabel(f"{stock_a} Timestamp")

plt.tight_layout()
plt.show()