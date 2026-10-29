"""Ridge and XGBoost salary models with feature-set ablations.

Usage (from the repo root):
    python code/train_models.py "data/processed/master_integrated (1).csv" reports/modeling

Target: log_salary. Every model is scored with player-grouped cross-validation
(the table's own `fold` column plus 4 reshuffled grouped splits) and on the
time holdout (salary seasons 2023+, trained on earlier seasons only).

Feature sets
    perf               56 prior-season / profile features minus prior salary (53)
    perf+injury        + 6 injury-history features
    perf+prior         + 3 prior-salary features
    perf+injury+prior  all of the above
Injury features only exist for salary seasons 2008-2024, so the four sets are
compared on that common sample. perf and perf+prior are also run on the full
2000-2025 table as a reference.
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
data_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n_repeats, seed0 = 5, 42
log_lines = []
def log(s=""): print(s, flush=True); log_lines.append(s)

df = pd.read_csv(data_path, low_memory=False)

# ------------------------------------------------------------------ features
prior = ["log_prior_salary", "prior_salary_pct_cap", "has_prior_salary"]
injury = ["il_events_l1", "il_days_l1", "il_any_l1", "il_events_2yr", "il_days_2yr", "il_events_cum_since2006"]
# the 56 kept model features are the columns between salary_pct_cap and injury_coverage_l1
cols = list(df.columns)
base56 = cols[cols.index("salary_pct_cap") + 1: cols.index("injury_coverage_l1")]
perf = [c for c in base56 if c not in prior]
assert len(base56) == 56 and len(perf) == 53, (len(base56), len(perf))
never = {"salary", "log_salary", "salary_pct_cap"} | {c for c in cols if c.startswith("contract_")}
FEATURE_SETS = {
    "perf": perf,
    "perf+injury": perf + injury,
    "perf+prior": perf + prior,
    "perf+injury+prior": perf + injury + prior,
}
for name, fs in FEATURE_SETS.items():
    assert not (set(fs) & never), f"target-side column in {name}"

common = df[injury].notna().all(axis=1)
SAMPLES = {"common_2008_2024": df[common].reset_index(drop=True), "full_2000_2025": df.reset_index(drop=True)}
RUNS = [(s, f) for s in ["common_2008_2024"] for f in FEATURE_SETS] + \
       [("full_2000_2025", "perf"), ("full_2000_2025", "perf+prior")]
for s, d in SAMPLES.items():
    log(f"- Sample {s}: {len(d):,} rows, {d.player_id.nunique():,} players, seasons {d.season.min()}-{d.season.max()}, "
        f"time-holdout rows {int(d.time_holdout.sum()):,}")
for s, d in SAMPLES.items():
    assert not d[perf + prior].isna().any().any(), f"missing values in {s}"


# "new-deal" rows: prior salary is known and the salary moved by more than 10%
def new_deal_mask(d):
    ps = np.exp(d.log_prior_salary)
    return (d.has_prior_salary == 1) & ((d.salary / ps - 1).abs() > 0.10)


# ------------------------------------------------------------------ models
def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40)))

XGB_GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=w)
            for d in (3, 5) for lr in (0.03, 0.08) for n in (300, 700) for w in (1, 8)]
def xgb(params):
    return XGBRegressor(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                        n_jobs=2, random_state=seed0, verbosity=0, **params)


def grouped_folds(d, k, seed):
    """Assign whole players to k folds at random (fold column = repeat 0)."""
    if seed is None:
        return d.fold.to_numpy()
    players = d.player_id.unique()
    rng = np.random.default_rng(seed)
    f = dict(zip(rng.permutation(players), np.arange(len(players)) % k))
    return d.player_id.map(f).to_numpy()


# tune XGBoost once, on pre-holdout seasons of the common sample, with the richest feature set
t0 = time.time()
tune = SAMPLES["common_2008_2024"]
tune = tune[tune.time_holdout == 0].reset_index(drop=True)
tf = grouped_folds(tune, 3, seed=7)
Xt, yt = tune[FEATURE_SETS["perf+injury+prior"]], tune.log_salary
scores = []
for p in XGB_GRID:
    err = []
    for k in range(3):
        m = xgb(p).fit(Xt[tf != k], yt[tf != k])
        err.append(mean_squared_error(yt[tf == k], m.predict(Xt[tf == k])) ** 0.5)
    scores.append((np.mean(err), p))
scores.sort(key=lambda t: t[0])
best_xgb = scores[0][1]
log(f"- XGBoost tuning ({len(XGB_GRID)} settings, 3 player-grouped folds, seasons before the holdout): "
    f"best {best_xgb} with RMSE(log) {scores[0][0]:.4f}; worst {scores[-1][0]:.4f} ({time.time() - t0:.0f}s)")
MODELS = {"Ridge": ridge, "XGBoost": lambda: xgb(best_xgb)}


def metrics(y, p, sal, nd):
    pd_ = np.exp(p)
    out = dict(r2_log=r2_score(y, p), rmse_log=mean_squared_error(y, p) ** 0.5,
               mae_usd=mean_absolute_error(sal, pd_), r2_usd=r2_score(sal, pd_),
               medape=float(np.median(np.abs(pd_ / sal - 1))))
    if nd.sum() > 30:
        out.update(r2_log_newdeal=r2_score(y[nd], p[nd]), mae_usd_newdeal=mean_absolute_error(sal[nd], pd_[nd]))
    return out


# ------------------------------------------------------------------ cross-validation
cv_rows, oof_store = [], {}
for sname, fset in RUNS:
    d = SAMPLES[sname]
    X, y, sal, nd = d[FEATURE_SETS[fset]], d.log_salary.to_numpy(), d.salary.to_numpy(), new_deal_mask(d).to_numpy()
    for mname, make in MODELS.items():
        t0 = time.time()
        for r in range(n_repeats):
            folds = grouped_folds(d, 5, None if r == 0 else seed0 + r)
            assert all(len(set(d.player_id[folds == k]) & set(d.player_id[folds != k])) == 0 for k in range(5))
            oof = np.zeros(len(d))
            for k in range(5):
                tr, te = folds != k, folds == k
                oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
            cv_rows.append(dict(sample=sname, features=fset, model=mname, repeat=r, **metrics(y, oof, sal, nd)))
            if r == 0:
                oof_store[(sname, fset, mname)] = oof
        log(f"- CV {mname:7s} {sname} {fset:18s} done ({time.time() - t0:.0f}s)")

# naive baselines on the same folds (repeat 0): training mean, and "same as last year"
for sname in SAMPLES:
    d = SAMPLES[sname]; y, sal, nd = d.log_salary.to_numpy(), d.salary.to_numpy(), new_deal_mask(d).to_numpy()
    folds = d.fold.to_numpy(); mean_pred = np.zeros(len(d))
    for k in range(5):
        mean_pred[folds == k] = y[folds != k].mean()
    cv_rows.append(dict(sample=sname, features="-", model="Baseline: mean", repeat=0, **metrics(y, mean_pred, sal, nd)))
    has = (d.has_prior_salary == 1).to_numpy()
    cv_rows.append(dict(sample=sname, features="prior salary only", model="Baseline: last year's salary (rows with prior salary)", repeat=0,
                        **metrics(y[has], d.log_prior_salary.to_numpy()[has], sal[has], nd[has])))

cv = pd.DataFrame(cv_rows)
cv.to_csv(f"{out}/cv_results_by_repeat.csv", index=False)
mcols = ["r2_log", "rmse_log", "mae_usd", "r2_usd", "medape", "r2_log_newdeal", "mae_usd_newdeal"]
summ = cv.groupby(["sample", "model", "features"], sort=False)[mcols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
summ.to_csv(f"{out}/cv_results_summary.csv", index=False)

# ------------------------------------------------------------------ time holdout
ho_rows = []
for sname, fset in RUNS:
    d = SAMPLES[sname]
    tr, te = (d.time_holdout == 0).to_numpy(), (d.time_holdout == 1).to_numpy()
    X, y, sal, nd = d[FEATURE_SETS[fset]], d.log_salary.to_numpy(), d.salary.to_numpy(), new_deal_mask(d).to_numpy()
    for mname, make in MODELS.items():
        p = make().fit(X[tr], y[tr]).predict(X[te])
        ho_rows.append(dict(sample=sname, features=fset, model=mname, train_rows=int(tr.sum()), test_rows=int(te.sum()),
                            **metrics(y[te], p, sal[te], nd[te])))
ho = pd.DataFrame(ho_rows)
ho.to_csv(f"{out}/holdout_results.csv", index=False)

# ------------------------------------------------------------------ out-of-fold predictions
d = SAMPLES["common_2008_2024"]
oof = d[["player_id", "player_name", "season", "team_abbr", "salary", "log_salary", "has_prior_salary"]].copy()
oof["new_deal_row"] = new_deal_mask(d).astype(int).to_numpy()
for (s, f, mname), p in oof_store.items():
    if s == "common_2008_2024":
        oof[f"pred_salary_{mname}_{f}"] = np.round(np.exp(p))
oof.to_csv(f"{out}/oof_predictions_common_sample.csv", index=False)

# ------------------------------------------------------------------ figures
C = {"Ridge": "#4C72B0", "XGBoost": "#DD8452"}
common_s = summ[(summ["sample"] == "common_2008_2024") & summ.model.isin(MODELS)]
order = list(FEATURE_SETS)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (metric, label) in zip(axes, [("r2_log", "R² (log salary)"), ("mae_usd", "MAE ($M)"), ("r2_log_newdeal", "R² (log), new-deal rows only")]):
    x = np.arange(len(order)); w = 0.38
    for i, mname in enumerate(MODELS):
        s = common_s[common_s.model == mname].set_index("features").loc[order]
        scale = 1e6 if metric == "mae_usd" else 1
        ax.bar(x + (i - 0.5) * w, s[f"{metric}_mean"] / scale, w, yerr=s[f"{metric}_std"] / scale, color=C[mname], label=mname, capsize=3)
        for xi, v in zip(x + (i - 0.5) * w, s[f"{metric}_mean"] / scale):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15, fontsize=9); ax.set_title(label)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
fig.suptitle("Player-grouped 5-fold CV, 5 repeats (mean ± sd), salary seasons 2008-2024", fontsize=11)
fig.tight_layout(); fig.savefig(f"{out}/fig1_model_comparison.png", dpi=150); plt.close(fig)

best_key = max(((m, f) for m in MODELS for f in order),
               key=lambda t: common_s[(common_s.model == t[0]) & (common_s.features == t[1])].r2_log_mean.iloc[0])
bm, bf = best_key
p = oof_store[("common_2008_2024", bf, bm)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
nd = new_deal_mask(d).to_numpy()
ax.scatter(np.exp(d.log_salary[~nd]) / 1e6, np.exp(p[~nd]) / 1e6, s=5, alpha=0.3, color="#999999", label="salary within 10% of prior / no prior")
ax.scatter(np.exp(d.log_salary[nd]) / 1e6, np.exp(p[nd]) / 1e6, s=5, alpha=0.4, color=C[bm], label="new-deal rows (salary moved >10%)")
lim = [0, max(d.salary.max(), np.exp(p).max()) / 1e6]
ax.plot(lim, lim, color="black", lw=0.8); ax.set_xlabel("Actual salary ($M)"); ax.set_ylabel("Predicted salary ($M)")
ax.set_title(f"Out-of-fold predictions: {bm}, {bf}"); ax.legend(frameon=False, fontsize=8, markerscale=3)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
res = d.log_salary.to_numpy() - p
g = pd.DataFrame({"season": d.season, "res": res}).groupby("season").res
ax.axhline(0, color="black", lw=0.8)
ax.plot(g.mean().index, g.mean().values, marker="o", color=C[bm], label="mean residual")
ax.fill_between(g.mean().index, g.quantile(0.25).values, g.quantile(0.75).values, color=C[bm], alpha=0.2, label="25th-75th percentile")
ax.set_xlabel("Salary season"); ax.set_ylabel("log(actual) - log(predicted)"); ax.set_title("Residuals by season")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{out}/fig2_predicted_vs_actual_and_residuals.png", dpi=150); plt.close(fig)

# importance: XGBoost gain and Ridge standardized coefficients, full feature set, fit on the whole common sample
fs = FEATURE_SETS["perf+injury+prior"]
xm = xgb(best_xgb).fit(d[fs], d.log_salary)
gain = pd.Series(xm.get_booster().get_score(importance_type="gain")).reindex(fs).fillna(0)
gain = gain / gain.sum()
rm = ridge().fit(d[fs], d.log_salary)
coef = pd.Series(rm[-1].coef_, index=fs)
imp = pd.DataFrame({"xgb_gain_share": gain, "ridge_std_coef": coef})
imp["is_injury"] = imp.index.isin(injury); imp["is_prior_salary"] = imp.index.isin(prior)
imp.sort_values("xgb_gain_share", ascending=False).to_csv(f"{out}/feature_importance.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
def colr(i): return "#C44E52" if i in injury else ("#55A868" if i in prior else "#8C8C8C")
top = gain.sort_values().tail(20)
axes[0].barh(top.index, top.values, color=[colr(i) for i in top.index]); axes[0].set_title("XGBoost: share of total gain (top 20)")
topc = coef.reindex(coef.abs().sort_values().tail(20).index)
axes[1].barh(topc.index, topc.values, color=[colr(i) for i in topc.index]); axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_title("Ridge: standardized coefficients (top 20 by size)")
for ax in axes: ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
fig.text(0.5, 0.005, "green = prior salary, red = injury, grey = performance/profile", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{out}/fig3_feature_importance.png", dpi=150); plt.close(fig)
inj_rank = gain.rank(ascending=False)[injury].sort_values()
log(f"- Injury features' XGBoost gain share combined: {gain[injury].sum():.1%}; best-ranked injury feature: {inj_rank.index[0]} (#{int(inj_rank.iloc[0])} of {len(fs)})")
log(f"- Prior-salary features' XGBoost gain share combined: {gain[prior].sum():.1%}")

json.dump(dict(xgb_params=best_xgb, ridge_alpha_full_fit=float(rm[-1].alpha_), best_cv=dict(model=bm, features=bf)),
          open(f"{out}/run_config.json", "w"), indent=1)
open(f"{out}/run_log.md", "w").write("# Modeling run log\n\n" + "\n".join(log_lines) + "\n")
print("done")
