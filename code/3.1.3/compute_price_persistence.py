"""
analyzes trade data persistence for retail and institutional labels across markets and horizons, outputting stats and plots
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

rng = np.random.default_rng(42)
out_dir = Path("./")
out_dir.mkdir(exist_ok=True)

df_all_markets = pd.read_parquet("clean_trades.parquet").sort_values(["market", "time"])
df_all_markets = df_all_markets[df_all_markets["label"].isin(["Retail", "Institutional"])]

all_stats_rows, all_diff_rows, all_p_rows = [], [], []

markets = df_all_markets["market"].unique()

for market_name in markets:
    df_market = df_all_markets[df_all_markets["market"] == market_name].copy()

    if df_market.empty or len(df_market) < 101:
        continue

    df_market["deltaP"] = df_market["price"].shift(-1) - df_market["price"]
    for k_horizon in (20, 50, 100):
        df_market[f"cum{k_horizon}"] = df_market["price"].shift(-k_horizon) - df_market["price"]

    df_market = df_market.dropna(subset=["deltaP"] + [f"cum{k_horizon}" for k_horizon in (20,50,100)])

    if df_market.empty:
        continue
        
    df_market = df_market[df_market["deltaP"] != 0].copy()
    if df_market.empty:
        continue

    df_market["abs_delta"] = np.abs(df_market["deltaP"])
    df_market = df_market[df_market["abs_delta"] >= 2].copy()
    
    if df_market.empty:
        continue

    for k_horizon in (20, 50, 100):
        ratio_col_name = f"ratio{k_horizon}"
        df_market[ratio_col_name] = np.where(df_market["abs_delta"] != 0, 
                                             np.abs(df_market[f"cum{k_horizon}"]) / df_market["abs_delta"], 
                                             np.nan)
        
        df_market_filtered_ratio = df_market.dropna(subset=[ratio_col_name]).copy()
        if df_market_filtered_ratio.empty:
            continue

        df_market_filtered_ratio[ratio_col_name] = df_market_filtered_ratio[ratio_col_name].clip(upper=5)
        
        low_quant, hi_quant = df_market_filtered_ratio[ratio_col_name].quantile([.01, .99])
        
        sub_market_df = df_market_filtered_ratio[
            (df_market_filtered_ratio[ratio_col_name] > 1e-9) &
            (np.abs(df_market_filtered_ratio[ratio_col_name] - 1.0) > 1e-9) &
            (df_market_filtered_ratio[ratio_col_name] >= low_quant) &
            (df_market_filtered_ratio[ratio_col_name] <= hi_quant)
        ]

        if sub_market_df.empty:
            continue

        r_market_ratios = sub_market_df[sub_market_df["label"] == "Retail"][ratio_col_name].values
        i_market_ratios = sub_market_df[sub_market_df["label"] == "Institutional"][ratio_col_name].values

        if len(r_market_ratios) < 2 or len(i_market_ratios) < 2:
            if len(r_market_ratios) >=1 :
                 for lab_name, x_vals in [("Retail", r_market_ratios)]:
                    if len(x_vals) > 0:
                        gmean = np.exp(np.mean(np.log(x_vals[x_vals > 0]))) if np.all(x_vals > 0) and len(x_vals[x_vals > 0]) > 0 else np.nan
                        desc_stats = pd.Series(x_vals).describe()
                        all_stats_rows.append([market_name, k_horizon, lab_name, gmean] + desc_stats.tolist())
            if len(i_market_ratios) >=1 :
                 for lab_name, x_vals in [("Institutional", i_market_ratios)]:
                    if len(x_vals) > 0:
                        gmean = np.exp(np.mean(np.log(x_vals[x_vals > 0]))) if np.all(x_vals > 0) and len(x_vals[x_vals > 0]) > 0 else np.nan
                        desc_stats = pd.Series(x_vals).describe()
                        all_stats_rows.append([market_name, k_horizon, lab_name, gmean] + desc_stats.tolist())
            continue


        for lab_name, x_vals in [("Retail", r_market_ratios), ("Institutional", i_market_ratios)]:
            positive_x_vals = x_vals[x_vals > 0]
            gmean = np.exp(np.mean(np.log(positive_x_vals))) if len(positive_x_vals) > 0 else np.nan
            desc_stats = pd.Series(x_vals).describe()
            all_stats_rows.append([market_name, k_horizon, lab_name, gmean] + desc_stats.tolist())

        pick_size = min(10000, len(r_market_ratios), len(i_market_ratios))
        if pick_size > 1:
            diffs_hl = rng.choice(i_market_ratios, pick_size, replace=True) - rng.choice(r_market_ratios, pick_size, replace=True)
            hl_estimator = np.median(diffs_hl)
            
            bootstrap_medians = []
            for _ in range(1000):
                di_b_hl = rng.choice(i_market_ratios, pick_size, replace=True)
                dr_b_hl = rng.choice(r_market_ratios, pick_size, replace=True)
                bootstrap_medians.append(np.median(di_b_hl - dr_b_hl))
            
            ci_lower_hl, ci_upper_hl = np.percentile(bootstrap_medians, [2.5, 97.5])
            all_diff_rows.append([market_name, k_horizon, hl_estimator, ci_lower_hl, ci_upper_hl])
        else:
            all_diff_rows.append([market_name, k_horizon, np.nan, np.nan, np.nan])

        try:
            mwu_p_value = mannwhitneyu(r_market_ratios, i_market_ratios, alternative="two-sided", nan_policy='omit').pvalue
            all_p_rows.append([market_name, k_horizon, mwu_p_value])
        except ValueError as e:
            all_p_rows.append([market_name, k_horizon, np.nan])

        plt.figure(figsize=(7, 5))
        plot_data = sub_market_df[["label", ratio_col_name]].rename(columns={ratio_col_name: "ratio"})
        
        sns.boxplot(x="label", y="ratio", data=plot_data,
                    palette={"Retail": "#E69F00", "Institutional": "#0072B2"},
                    showfliers=False)
        plt.yscale("log")
        plt.ylabel(f"Persistence Ratio (log scale) - Horizon {k_horizon}")
        plt.xlabel("Trader Label")
        plt.title(f"Market: {market_name} - Persistence Ratio |ΔP 0→{k_horizon}| / |ΔP 0→1|\n(Winsorized, 1% Trimmed, Ratio≠1)")
        plt.tight_layout()
        plt.savefig(out_dir / f"persistence_boxplot_{market_name}_h{k_horizon}.png", dpi=300)
        plt.close()

pd.DataFrame(all_stats_rows,
             columns=["market", "horizon", "label", "geom_mean", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
             ).to_csv(out_dir / "ALL_MARKETS_persistence_stats_trimmed.csv", index=False)

pd.DataFrame(all_diff_rows,
             columns=["market", "horizon", "HL_diff_inst_minus_ret", "ci_lower", "ci_upper"]
             ).to_csv(out_dir / "ALL_MARKETS_median_difference_CI.csv", index=False)

pd.DataFrame(all_p_rows, columns=["market", "horizon", "mannwhitney_p"]
             ).to_csv(out_dir / "ALL_MARKETS_mannwhitney_pvalues.csv", index=False)

print(f"Per-market persistence analysis complete. Outputs in {out_dir}")