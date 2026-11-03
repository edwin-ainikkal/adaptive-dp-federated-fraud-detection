"""
evaluation/plot_results.py
----------------------------
Turns the server's metrics.csv files into report-ready figures.

Single run:
    python evaluation/plot_results.py --runs results/metrics.csv --labels "Adaptive DP"

Compare several runs (e.g. adaptive vs fixed baseline vs no-noise), after you
saved each run's metrics.csv under a different name/folder:
    python evaluation/plot_results.py \
        --runs results_adaptive/metrics.csv results_fixed/metrics.csv results_nodp/metrics.csv \
        --labels "Adaptive DP" "Fixed DP" "No DP"

Output PNGs are written to results/plots/ (or --out).
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--labels", nargs="+")
p.add_argument("--out", default="results/plots")
a = p.parse_args()
labels = a.labels or [os.path.basename(os.path.dirname(r)) or r for r in a.runs]
os.makedirs(a.out, exist_ok=True)
dfs = [pd.read_csv(r) for r in a.runs]

def line_plot(metric, title, fname, ylabel=None):
    plt.figure(figsize=(7, 4.2))
    for df, lab in zip(dfs, labels):
        plt.plot(df["round"] + 1, df[metric], marker="o", ms=3, label=lab)
    plt.xlabel("Federated round"); plt.ylabel(ylabel or metric); plt.title(title)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.out, fname), dpi=160); plt.close()

for m, t in [("auc", "Global model ROC-AUC per round"), ("pr_auc", "Precision-Recall AUC per round (better for rare fraud)"),
             ("recall", "Fraud recall per round"),
             ("precision", "Fraud precision per round"), ("f1", "Fraud F1-score per round")]:
    line_plot(m, t, f"{m}_vs_round.png")
line_plot("epsilon", "Cumulative privacy budget spent (epsilon, worst client)", "epsilon_vs_round.png", "epsilon (delta=1e-5)")

# privacy-utility trade-off: final AUC vs final epsilon
plt.figure(figsize=(6, 4.2))
for df, lab in zip(dfs, labels):
    e = df["epsilon"].iloc[-1]
    if e == e:      # runs without DP have no epsilon -> drawn as a horizontal reference line
        plt.scatter(e, df["auc"].iloc[-1], s=80, label=lab)
    else:
        plt.axhline(df["auc"].iloc[-1], ls="--", color="gray", label=f"{lab} (no privacy)")
plt.xlabel("Final epsilon (lower = more private)"); plt.ylabel("Final AUC")
plt.title("Privacy-utility trade-off"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(a.out, "privacy_utility_tradeoff.png"), dpi=160); plt.close()

summary = pd.DataFrame([{"run": lab, **df.iloc[-1][["accuracy", "precision", "recall", "f1", "auc", "pr_auc", "epsilon"]].to_dict()}
                         for df, lab in zip(dfs, labels)])
summary.to_csv(os.path.join(a.out, "summary_table.csv"), index=False)
print(summary.round(4).to_string(index=False))
print(f"\nPlots saved in: {a.out}")
