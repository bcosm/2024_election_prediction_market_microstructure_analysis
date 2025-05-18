"""
analyzes trade data, generating summary statistics and visualizations
for trade size, daily volume share, and hourly trade counts by label
"""
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
sns.set(style="whitegrid")

out_dir = Path("./")
out_dir.mkdir(exist_ok=True)

df = pd.read_parquet("clean_trades.parquet")
df["log_size"] = np.log(df["size"])

(df.groupby("label")["size"].describe()
   .to_csv(out_dir / "size_summary_by_label.csv"))

plt.figure(figsize=(7, 4))
palette = {"Retail": "#E69F00", "Institutional": "#0072B2"}
sns.kdeplot(data=df, x="log_size", hue="label", fill=True,
            common_norm=False, palette=palette, alpha=.4)
plt.title("Density of log(size) by label")
plt.tight_layout()
plt.savefig(out_dir / "log_size_density.png", dpi=300)
plt.close()

daily = (df.set_index("time")
           .groupby("label")["size"]
           .resample("1D").sum()
           .unstack("label").fillna(0))
share = daily.div(daily.sum(axis=1), axis=0)
share.to_csv(out_dir / "daily_volume_share.csv")

plt.figure(figsize=(10, 4))
share.plot.area(color=["#E69F00", "#0072B2"], alpha=.7, ax=plt.gca())
plt.title("Daily volume share")
plt.tight_layout()
plt.savefig(out_dir / "volume_share_over_time.png", dpi=300)
plt.close()

df["hour"] = df["time"].dt.hour
heat = (df.groupby(["hour", "label"]).size()
          .unstack("label").fillna(0))
plt.figure(figsize=(6, 4))
sns.heatmap(heat, cmap="YlGnBu", linewidths=.2)
plt.title("Trade count by hour")
plt.tight_layout()
plt.savefig(out_dir / "hour_of_day_heatmap.png", dpi=300)
plt.close()