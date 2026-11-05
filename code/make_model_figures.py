"""Redraw the two model-comparison figures from reports/modeling_v2/cv_results_summary.csv (no retraining).

Usage: python code/make_model_figures.py reports/modeling_v2
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
summ = pd.read_csv(f"{out}/cv_results_summary.csv")
ENS = "Ensemble: Ridge+XGB+CatBoost+SVR"
fam = {"Ridge": "#4C72B0", "Lasso": "#4C72B0", "ElasticNet": "#4C72B0", "Poly2 Ridge": "#4C72B0",
       "Random Forest": "#55A868", "Extra Trees": "#55A868", "SVR (RBF)": "#8172B2", "kNN": "#8172B2",
       "Neural net (MLP)": "#8172B2", ENS: "#333333"}
col = lambda m: fam.get(m, "#DD8452")
models = [m for m in summ.model.unique() if not m.startswith("Baseline")]

full = summ[summ.features == "perf+injury+prior"].set_index("model").loc[models]
order = list(full.sort_values("r2_log_mean", ascending=False).index)      # one row order for every panel
fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
for ax, (metric, title, scale) in zip(axes, [("r2_log", "R² (log salary)", 1), ("mae_usd", "MAE ($M)", 1e6),
                                              ("r2_log_newdeal", "R² (log), new-deal rows", 1)]):
    v, e = full.loc[order, f"{metric}_mean"] / scale, full.loc[order, f"{metric}_std"] / scale
    ax.barh(order, v, xerr=e, color=[col(m) for m in order], capsize=2)
    for y_, val in enumerate(v): ax.text(val + (0.004 if scale == 1 else 0.04), y_, f"{val:.2f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    if metric != "mae_usd": ax.set_xlim(max(0, v.min() - 0.12), v.max() + 0.04)
fig.suptitle("Player-grouped 5-fold CV (3 repeats), features = performance + injury + prior salary; sorted by R² (log salary)\n"
             "blue = linear, green = bagged trees, orange = boosting, purple = kernel/neighbour/neural, black = ensemble", fontsize=11)
fig.tight_layout(); fig.savefig(f"{out}/fig_model_comparison.png", dpi=150); plt.close(fig)

a = summ[summ.features == "perf+injury"].set_index("model").loc[order, "r2_log_mean"]
b = full.loc[order, "r2_log_mean"]
y_ = np.arange(len(order))
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(y_ - 0.2, a, 0.4, label="no prior salary", color="#8C8C8C"); ax.barh(y_ + 0.2, b, 0.4, label="with prior salary", color="#55A868")
ax.set_yticks(y_); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlim(0.4, None)
ax.set_xlabel("R² (log salary), CV mean"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Effect of prior-salary features, by model")
fig.tight_layout(); fig.savefig(f"{out}/fig_prior_salary_effect.png", dpi=150); plt.close(fig)
print("figures written")
