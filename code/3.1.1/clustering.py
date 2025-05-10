"""fits a 1-d bayesian gmm on log size to label trades"""

import json, numpy as np, pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture

INPUT_JSON  = "kalshi_trades.json"
OUTPUT_JSON = "kalshi_trades_labeled.json"

with open(INPUT_JSON, "r") as f:
    raw = json.load(f)

rows, trade_refs = [], []
for mkt_idx, (mkt, trades) in enumerate(tqdm(raw.items())):
    for t_idx, t in enumerate(trades):
        rows.append(
            dict(
                size=t["count"],
            )
        )
        trade_refs.append((mkt, t_idx))

df = pd.DataFrame(rows)
df["log_size"] = np.log(df["size"])

X = df["log_size"].values.reshape(-1, 1)
dpgmm = BayesianGaussianMixture(
    n_components=10,
    weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=0.1,
    max_iter=500,
    random_state=42,
)
dpgmm.fit(X)

probs = dpgmm.predict_proba(X)
means = dpgmm.means_.ravel()
sorted_idx = np.argsort(means)

tail_components = sorted_idx[-3:]
inst_prob = probs[:, tail_components].sum(axis=1)

labels = np.full(len(df), "Uncertain", dtype=object)
labels[inst_prob > 0.80] = "Institutional"
labels[inst_prob < 0.20] = "Retail"
df["label"] = labels

print(df["label"].value_counts(normalize=True).round(3))

for (mkt, t_idx), lab in zip(trade_refs, df["label"]):
    raw[mkt][t_idx]["label"] = lab

with open(OUTPUT_JSON, "w") as f:
    json.dump(raw, f, indent=2)

print(f"Labeled JSON written to {OUTPUT_JSON}")