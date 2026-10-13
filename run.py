import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    x, y = data.load_with_noise()
    table = core.benchmark(x, y, k=10)
    methods = [m for m in table.index if m != "none"]
    stab = {m: core.stability(x, y, m, k=10) for m in methods}
    table["stability"] = pd.Series(stab)
    curves = pd.DataFrame({m: core.score_vs_k(x, y, m) for m in ["anova_f", "l1_logistic", "forest_importance"]})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    table["auc_mean"].sort_values().plot.barh(ax=axes[0], color="grey")
    axes[0].set_xlim(table["auc_mean"].min() - 0.01, 1.0)
    axes[0].set_title("ROC AUC, 5-fold CV (selection inside each fold)")
    curves.plot(ax=axes[1], marker="o", color=["black", "dimgrey", "darkgrey"])
    axes[1].set_xlabel("features kept")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Score versus number of kept features")
    fig.tight_layout()
    fig.savefig(OUT / "selection.png", dpi=130)

    table.round(4).to_csv(OUT / "benchmark.csv")
    summary = {
        "features_total": int(x.shape[1]),
        "real_features": int(sum(data.is_real(c) for c in x.columns)),
        "samples": int(len(x)),
        "results": {k: {m: (None if pd.isna(v) else round(float(v), 4)) for m, v in row.items()} for k, row in table.iterrows()},
        "score_vs_k": {m: {int(k): round(float(v), 4) for k, v in curves[m].items()} for m in curves},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
