"""performs ols regression and power law analysis on market data"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

base_out_dir = Path("./")
base_out_dir.mkdir(exist_ok=True)

df_all_markets = pd.read_parquet("timebars.parquet").sort_values(["market", "time"])
price_col_name = "mid_price" if "mid_price" in df_all_markets.columns else "price"

df_all_markets["deltaP"] = df_all_markets.groupby("market")[price_col_name].diff()
df_all_markets = df_all_markets.dropna(subset=["deltaP"])
df_all_markets[["Retail", "Institutional"]] = df_all_markets[["Retail", "Institutional"]].fillna(0)

markets = df_all_markets["market"].unique()

for market_name in markets:
    market_out_dir = base_out_dir / market_name
    market_out_dir.mkdir(exist_ok=True)
    
    df = df_all_markets[df_all_markets["market"] == market_name].copy()
    
    if df.empty or len(df) < 2:
        print(f"Skipping market {market_name} due to insufficient data after processing.")
        continue

    if len(df) > 5 and df[["Retail", "Institutional"]].sum().sum() > 0 :
        X_lin = sm.add_constant(df[["Retail", "Institutional"]])
        y_ols = df["deltaP"]
        
        try:
            ols_model = sm.OLS(y_ols, X_lin)
            ols_results = ols_model.fit()
            
            ols_results.params.to_csv(market_out_dir / f"ols_price_impact_coeffs_{market_name}.csv")
            
            with open(market_out_dir / f"ols_summary_{market_name}.txt", "w") as f:
                f.write(ols_results.summary().as_text())
        except Exception as e:
            print(f"OLS failed for market {market_name}: {e}")
    else:
        print(f"Skipping OLS for {market_name} due to insufficient data or zero flow.")

    def plot_power_law_relationship(flow_col, df_market, market_name_str, output_dir, cmap_plot, color_scatter):
        sub_df = df_market[(df_market[flow_col] > 1e-9) & (np.abs(df_market["deltaP"]) > 1e-9)].copy()

        if sub_df.empty or len(sub_df) < 10:
            print(f"Skipping power law plot for {flow_col} in {market_name_str} due to insufficient data after filtering.")
            return

        x_flow = sub_df[flow_col].values
        y_abs_delta_p = np.abs(sub_df["deltaP"].values)

        if x_flow.min() <= 0 or x_flow.max() <= 0:
            print(f"Skipping power law plot for {flow_col} in {market_name_str} due to non-positive flow values for log scale.")
            return
            
        bins_log = np.logspace(np.log10(x_flow.min()), np.log10(x_flow.max()), 60)
        digitized_flow = np.digitize(x_flow, bins_log)
        
        x_binned_mean, y_binned_median = [], []
        for i_bin in range(1, len(bins_log)):
            mask_bin = (digitized_flow == i_bin)
            if mask_bin.any():
                x_binned_mean.append(x_flow[mask_bin].mean())
                y_binned_median.append(np.median(y_abs_delta_p[mask_bin]))
        
        x_binned_mean = np.array(x_binned_mean)
        y_binned_median = np.array(y_binned_median)

        valid_binned_indices = ~np.isnan(x_binned_mean) & ~np.isinf(x_binned_mean) & \
                               ~np.isnan(y_binned_median) & ~np.isinf(y_binned_median) & \
                               (x_binned_mean > 1e-9) & (y_binned_median > 1e-9)
        
        x_binned_clean = x_binned_mean[valid_binned_indices]
        y_binned_clean = y_binned_median[valid_binned_indices]

        if len(x_binned_clean) > 5:
            log_x_binned = np.log10(x_binned_clean)
            log_y_binned = np.log10(y_binned_clean)
            
            try:
                slope, intercept = np.polyfit(log_x_binned, log_y_binned, 1)
                pd.Series([slope, intercept], index=[f"slope_{flow_col}", f"intercept_{flow_col}"])\
                  .to_csv(output_dir / f"loglog_fit_{flow_col}_{market_name_str}.csv")

                xs_fit = np.logspace(np.log10(x_flow.min()), np.log10(x_flow.max()), 200)
                ys_fit = 10**(intercept + slope * np.log10(xs_fit))
            except Exception as e:
                print(f"Polyfit failed for {flow_col} in {market_name_str}: {e}")
                slope, intercept, xs_fit, ys_fit = None, None, None, None
        else:
            print(f"Skipping polyfit for {flow_col} in {market_name_str} due to too few valid binned data points ({len(x_binned_clean)}).")
            slope, intercept, xs_fit, ys_fit = None, None, None, None

        sns.set_style("whitegrid")
        plt.figure(figsize=(7, 5))
        
        valid_plot_indices = (x_flow > 1e-9) & (y_abs_delta_p > 1e-9)
        if not valid_plot_indices.any():
            print(f"No valid data to plot hexbin for {flow_col} in {market_name_str}")
            plt.close()
            return

        plt.hexbin(x_flow[valid_plot_indices], y_abs_delta_p[valid_plot_indices], 
                   gridsize=50, cmap=cmap_plot, xscale="log", yscale="log", mincnt=3, alpha=0.7)
        
        if xs_fit is not None and ys_fit is not None:
            plt.plot(xs_fit, ys_fit, color="black", linestyle="--", linewidth=2, label=f"Log-Log Fit (Slope: {slope:.2f})")
        
        if len(x_binned_clean) > 0:
             plt.scatter(x_binned_clean, y_binned_clean, facecolors='none', edgecolors=color_scatter, s=30, zorder=3, label="Binned Medians")
        
        plt.colorbar(label="Count in Hex Bin")
        plt.xlabel(f"{flow_col} Contracts (Log Scale)")
        plt.ylabel("|ΔP| in Ticks (Log Scale)")
        plt.title(f"{market_name_str}: {flow_col} Volume vs. |ΔP| (Log-Log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{flow_col}_loglog_fit_{market_name_str}.png", dpi=300)
        plt.close()

    plot_power_law_relationship("Retail", df, market_name, market_out_dir, "Oranges", "darkorange")
    plot_power_law_relationship("Institutional", df, market_name, market_out_dir, "Blues", "royalblue")

