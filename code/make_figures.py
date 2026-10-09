import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
raw, interim, proc, figdir = sys.argv[1:5]
os.makedirs(figdir, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
B1, B2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10})
mr = pd.read_csv(f"{proc}/master_model_ready.csv"); wf = pd.read_csv(f"{proc}/record_waterfall.csv")
sal_raw = pd.read_csv(f"{raw}/public.player_salaries_historical.csv"); sal = pd.read_csv(f"{interim}/salaries_clean.csv")

# 1 salary distribution raw vs log
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].hist(mr.salary / 1e6, bins=40, color=B1, edgecolor=SURF, linewidth=0.8); ax[0].set_xlabel("Salary (millions of dollars)"); ax[0].set_ylabel("Player-seasons")
ax[0].set_title(f"Raw salary: skewness {mr.salary.skew():.2f}", loc="left", fontsize=11, color=INK)
ax[1].hist(mr.log_salary, bins=40, color=B1, edgecolor=SURF, linewidth=0.8); ax[1].set_xlabel("Natural log of salary (dollars)")
ax[1].set_title(f"Log salary: skewness {mr.log_salary.skew():.2f}", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{figdir}/fig1_salary_distribution.png", dpi=200); plt.close(fig)

# 2 waterfall
fig, ax = plt.subplots(figsize=(9, 3.9))
y = np.arange(len(wf))[::-1]; ax.barh(y, wf.rows, color=B1, height=0.62)
for yi, (s, n) in zip(y, zip(wf.step, wf.rows)): ax.text(n + 150, yi, f"{n:,}", va="center", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels(wf.step, fontsize=9); ax.set_xlabel("Salary player-seasons remaining"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, wf.rows.max() * 1.1); ax.set_title("Records remaining after each selection rule", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{figdir}/fig2_record_waterfall.png", dpi=200); plt.close(fig)

# 3 age and position missingness before and after
names = {"br_team_page": "Team payroll pages", "simple_csv": "Simple export", "detailed_csv": "Detailed export"}
order = ["br_team_page", "simple_csv", "detailed_csv"]
b = sal_raw.groupby("source").age.apply(lambda s: s.isna().mean()).reindex(order) * 100
a = sal.groupby("salary_source").age.apply(lambda s: s.isna().mean()).reindex(order) * 100
fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(3); w = 0.36
ax.bar(x - w/2 - 0.01, b, w, color=B1, label="Before cleaning"); ax.bar(x + w/2 + 0.01, a, w, color=B2, label="After cleaning")
for xi, v in zip(x - w/2, b): ax.text(xi - 0.01, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
for xi, v in zip(x + w/2, a): ax.text(xi + 0.01, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([names[o] for o in order]); ax.set_ylabel("Rows with missing age (%)"); ax.set_ylim(0, 115); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9); ax.set_title("Missing age by salary collection source", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{figdir}/fig3_missing_age_before_after.png", dpi=200); plt.close(fig)

# 4 cap normalization
g = mr.groupby("season").agg(med_sal=("salary", "median"), med_pct=("salary_pct_cap", "median"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(g.index, g.med_sal / 1e6, color=B1, lw=2); ax[0].set_ylabel("Median salary (millions of dollars)"); ax[0].set_title("Median salary rises with the cap", loc="left", fontsize=11, color=INK)
ax[1].plot(g.index, g.med_pct * 100, color=B1, lw=2); ax[1].set_ylabel("Median salary as a share of the cap (%)"); ax[1].set_ylim(0, g.med_pct.max() * 100 * 1.3)
ax[1].set_title("Share of the cap is far more stable", loc="left", fontsize=11, color=INK)
for a_ in ax: a_.set_xlabel("Salary season (season ending year)")
fig.tight_layout(); fig.savefig(f"{figdir}/fig4_cap_normalization.png", dpi=200); plt.close(fig)

# 5 top correlations
ID = ["salary_row_id","player_id","player_name","season","team_abbr","salary_source","match_stage","in_injury_window","fold","time_holdout","salary","log_salary","salary_pct_cap"]
F = [c for c in mr.columns if c not in ID]
c = mr[F].corrwith(mr.log_salary).sort_values(key=abs, ascending=False).head(15)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(c)), c.values, color=B1, height=0.62); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=9)
for i, v in enumerate(c.values): ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlabel("Correlation with log salary"); ax.grid(axis="y", visible=False); ax.set_xlim(0, c.max() * 1.12)
ax.set_title("Strongest predictors of log salary (final feature set)", loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f"{figdir}/fig5_top_feature_correlations.png", dpi=200); plt.close(fig)
print(sorted(os.listdir(figdir)))
