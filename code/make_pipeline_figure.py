"""Draw the evaluation pipeline diagram used in the modeling report.  Usage: python code/make_pipeline_figure.py reports/modeling_v2"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, fc="#EEF2F7", ec="#4C72B0"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, wrap=True)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))
w, h, y = 17, 14, 22
xs = [1, 21, 41, 61, 81]
labels = ["Master table\n5,757 player-seasons\n(2008-2024)\n59 or 62 features",
          "Split players into\n5 folds, no player in\ntrain and test\n(3 repeats)",
          "Inner tuning on the\ntraining part only:\n3 player-grouped folds,\nsmall grid per model",
          "Refit best settings on\nthe training part,\npredict the held-out\nplayers",
          "Scores: R² (log),\nMAE, median % error,\nnew-deal rows"]
for x, t in zip(xs, labels): box(x, y, w, h, t)
for a, b in zip(xs[:-1], xs[1:]): arrow(a + w + 0.6, y + h / 2, b - 0.6, y + h / 2)
box(21, 2, 57, 12, "Time holdout: train on seasons before 2023 (tune inside the training seasons),\ntest on 2023 and later (737 rows)", fc="#F6EFE7", ec="#DD8452")
arrow(9.5, y - 0.6, 21, 9)
arrow(78.5, 9, 89.5, y - 0.6)
fig.tight_layout(); fig.savefig(f"{out}/fig_pipeline.png", dpi=170); print("ok")
