"""Wider model comparison: linear, tree, boosting, kernel, neighbour and neural models.

Usage (from the repo root):
    python code/train_more_models.py "data/processed/master_integrated (1).csv" reports/modeling_v2
    SMOKE=1 python code/train_more_models.py ...        # time one outer fold per model, write nothing

Same data, same player-grouped folds and same feature sets as train_models.py, with two changes:
  * hyperparameters are tuned INSIDE every training fold (inner 3-fold, grouped by player), so no
    test row ever influences tuning (this fixes the tuning/test overlap in the first run);
  * 3 repeats instead of 5 to keep the run time reasonable with many more models.

Feature sets (common sample = salary seasons 2008-2024, 5,757 rows)
    perf+injury        53 prior-season/profile features + 6 injury-history features, NO prior salary
    perf+injury+prior  the above + 3 prior-salary features
XGBoost is run with five "regression types": squared error on log salary, absolute error,
pseudo-Huber, gamma regression on dollars (log link), and squared error on salary as % of cap.
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
data_path, out = sys.argv[1], sys.argv[2]
SMOKE = os.environ.get("SMOKE") == "1"
if not SMOKE:
    os.makedirs(out, exist_ok=True)
n_repeats, seed0 = (1 if SMOKE else 3), 42
log_lines = []
def log(s=""): print(s, flush=True); log_lines.append(s)

df = pd.read_csv(data_path, low_memory=False)
prior = ["log_prior_salary", "prior_salary_pct_cap", "has_prior_salary"]
injury = ["il_events_l1", "il_days_l1", "il_any_l1", "il_events_2yr", "il_days_2yr", "il_events_cum_since2006"]
cols = list(df.columns)
base56 = cols[cols.index("salary_pct_cap") + 1: cols.index("injury_coverage_l1")]
perf = [c for c in base56 if c not in prior]
assert len(base56) == 56 and len(perf) == 53
never = {"salary", "log_salary", "salary_pct_cap"} | {c for c in cols if c.startswith("contract_")}
FEATURE_SETS = {"perf+injury": perf + injury, "perf+injury+prior": perf + injury + prior}
for n, fs in FEATURE_SETS.items():
    assert not (set(fs) & never), n

d = df[df[injury].notna().all(axis=1)].reset_index(drop=True)
assert not d[perf + prior].isna().any().any()
log(f"- Sample: {len(d):,} rows, {d.player_id.nunique():,} players, seasons {d.season.min()}-{d.season.max()}, "
    f"time-holdout rows {int(d.time_holdout.sum()):,}")
cap = (d.salary / d.salary_pct_cap).to_numpy()          # salary cap of that season (public, constant within a season)
sal = d.salary.to_numpy(); ylog = d.log_salary.to_numpy()
ps = np.exp(d.log_prior_salary)
nd_mask = ((d.has_prior_salary == 1) & ((d.salary / ps - 1).abs() > 0.10)).to_numpy()

# ---------------------------------------------------------------- targets
# each target: (training y, function turning model output back into log-dollars)
TARGETS = {
    "log":    (ylog,                     lambda p, rows: p),
    "pctcap": (d.salary_pct_cap.to_numpy(), lambda p, rows: np.log(np.clip(p, 1e-4, None) * cap[rows])),
    "gammaM": (sal / 1e6,                lambda p, rows: np.log(np.clip(p, 1e-2, None) * 1e6)),
}

# ---------------------------------------------------------------- models: (pipeline factory, grid, target)
def P(*steps): return Pipeline(list(steps))
sc = lambda: ("sc", StandardScaler())
XG = dict(subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=seed0, verbosity=0)
xgb_grid = {"m__max_depth": [3, 5], "m__min_child_weight": [1, 8]}
def xgb(**kw): return lambda: P(("m", XGBRegressor(learning_rate=0.03, n_estimators=400, **XG, **kw)))

MODELS = {
    # linear family
    "Ridge":            (lambda: P(sc(), ("m", Ridge())), {"m__alpha": np.logspace(-1, 3.5, 10)}, "log"),
    "Lasso":            (lambda: P(sc(), ("m", Lasso(max_iter=20000))), {"m__alpha": np.logspace(-4, -1, 7)}, "log"),
    "ElasticNet":       (lambda: P(sc(), ("m", ElasticNet(max_iter=20000))),
                         {"m__alpha": np.logspace(-4, -1, 5), "m__l1_ratio": [0.2, 0.6]}, "log"),
    "Poly2 Ridge":      (lambda: P(sc(), ("pf", PolynomialFeatures(2, include_bias=False)), ("sc2", StandardScaler()), ("m", Ridge())),
                         {"m__alpha": [300, 3000, 30000]}, "log"),
    # bagged trees
    "Random Forest":    (lambda: P(("m", RandomForestRegressor(n_estimators=200, max_features=0.4, n_jobs=2, random_state=seed0))),
                         {"m__min_samples_leaf": [3, 10]}, "log"),
    "Extra Trees":      (lambda: P(("m", ExtraTreesRegressor(n_estimators=200, max_features=0.6, n_jobs=2, random_state=seed0))),
                         {"m__min_samples_leaf": [3, 10]}, "log"),
    # boosting
    "XGBoost":          (xgb(), xgb_grid, "log"),
    "LightGBM":         (lambda: P(("m", LGBMRegressor(learning_rate=0.03, n_estimators=400, subsample=0.8, subsample_freq=1,
                                                      colsample_bytree=0.8, n_jobs=2, random_state=seed0, verbose=-1))),
                         {"m__num_leaves": [8, 24], "m__min_child_samples": [20, 50]}, "log"),
    "CatBoost":         (lambda: P(("m", CatBoostRegressor(iterations=600, learning_rate=0.05, thread_count=2, random_seed=seed0, verbose=0))),
                         {"m__depth": [4, 6]}, "log"),
    # XGBoost with different regression types
    "XGB abs-error":    (xgb(objective="reg:absoluteerror"), xgb_grid, "log"),
    "XGB pseudo-Huber": (xgb(objective="reg:pseudohubererror", huber_slope=0.5, base_score=15.0), xgb_grid, "log"),
    "XGB gamma ($)":    (xgb(objective="reg:gamma"), xgb_grid, "gammaM"),
    "XGB % of cap":     (xgb(), xgb_grid, "pctcap"),
    # kernel / neighbours / neural
    "SVR (RBF)":        (lambda: P(sc(), ("m", SVR(kernel="rbf", gamma="scale", epsilon=0.1))), {"m__C": [1, 3, 10]}, "log"),
    "kNN":              (lambda: P(sc(), ("m", KNeighborsRegressor(weights="distance"))), {"m__n_neighbors": [10, 25]}, "log"),
    "Neural net (MLP)": (lambda: P(sc(), ("m", TransformedTargetRegressor(
                             MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, n_iter_no_change=20,
                                          max_iter=600, random_state=seed0), transformer=StandardScaler()))),
                         {"m__regressor__alpha": [1e-1, 1.0]}, "log"),
}
ENSEMBLE = ("Ensemble: Ridge+XGB+CatBoost+SVR", ["Ridge", "XGBoost", "CatBoost", "SVR (RBF)"])   # chosen before seeing results


def fit_predict(name, X, tr, te, d_groups, rows_te):
    make, grid, tname = MODELS[name]
    y = TARGETS[tname][0]
    gs = GridSearchCV(make(), grid, cv=GroupKFold(3), scoring="neg_mean_squared_error", n_jobs=1, refit=True)
    gs.fit(X[tr], y[tr], groups=d_groups[tr])
    return TARGETS[tname][1](gs.predict(X[te]), rows_te), gs.best_params_


def metrics(y, p, s, nd):
    q = np.exp(p)
    o = dict(r2_log=r2_score(y, p), rmse_log=mean_squared_error(y, p) ** 0.5, mae_usd=mean_absolute_error(s, q),
             r2_usd=r2_score(s, q), medape=float(np.median(np.abs(q / s - 1))))
    if nd.sum() > 30:
        o.update(r2_log_newdeal=r2_score(y[nd], p[nd]), mae_usd_newdeal=mean_absolute_error(s[nd], q[nd]))
    return o


def grouped_folds(k, seed):
    if seed is None: return d.fold.to_numpy()
    pl = d.player_id.unique(); rng = np.random.default_rng(seed)
    f = dict(zip(rng.permutation(pl), np.arange(len(pl)) % k))
    return d.player_id.map(f).to_numpy()


groups = d.player_id.to_numpy()
if SMOKE:
    folds = grouped_folds(5, None)
    X = d[FEATURE_SETS["perf+injury+prior"]].to_numpy(); tr, te = folds != 0, folds == 0
    for name in MODELS:
        t = time.time(); p, bp = fit_predict(name, X, tr, te, groups, np.where(te)[0])
        print(f"{name:20s} {time.time() - t:6.1f}s  r2_log={r2_score(ylog[te], p):.3f}  {bp}", flush=True)
    sys.exit()

# ---------------------------------------------------------------- cross-validation
cv_rows, oof_all, chosen = [], {}, {}
for fset, feats in FEATURE_SETS.items():
    X = d[feats].to_numpy()
    for name in MODELS:
        t0 = time.time()
        for r in range(n_repeats):
            folds = grouped_folds(5, None if r == 0 else seed0 + r)
            assert all(len(set(groups[folds == k]) & set(groups[folds != k])) == 0 for k in range(5))
            oof = np.zeros(len(d))
            for k in range(5):
                tr, te = folds != k, folds == k
                oof[te], bp = fit_predict(name, X, tr, te, groups, np.where(te)[0])
                chosen.setdefault((fset, name), []).append(str(bp))
            oof_all[(fset, name, r)] = oof
            cv_rows.append(dict(features=fset, model=name, repeat=r, **metrics(ylog, oof, sal, nd_mask)))
        log(f"- CV {name:18s} {fset:18s} R2(log) {np.mean([x['r2_log'] for x in cv_rows[-n_repeats:]]):.3f}  ({time.time() - t0:.0f}s)")
    ename, members = ENSEMBLE
    for r in range(n_repeats):
        p = np.mean([oof_all[(fset, m, r)] for m in members], axis=0)
        oof_all[(fset, ename, r)] = p
        cv_rows.append(dict(features=fset, model=ename, repeat=r, **metrics(ylog, p, sal, nd_mask)))

# naive baselines (repeat 0 folds)
folds = d.fold.to_numpy(); mp = np.zeros(len(d))
for k in range(5): mp[folds == k] = ylog[folds != k].mean()
cv_rows.append(dict(features="-", model="Baseline: mean", repeat=0, **metrics(ylog, mp, sal, nd_mask)))
has = (d.has_prior_salary == 1).to_numpy()
cv_rows.append(dict(features="prior salary only", model="Baseline: last year's salary (rows with prior salary)", repeat=0,
                    **metrics(ylog[has], d.log_prior_salary.to_numpy()[has], sal[has], nd_mask[has])))

cv = pd.DataFrame(cv_rows); cv.to_csv(f"{out}/cv_results_by_repeat.csv", index=False)
mcols = ["r2_log", "rmse_log", "mae_usd", "r2_usd", "medape", "r2_log_newdeal", "mae_usd_newdeal"]
summ = cv.groupby(["features", "model"], sort=False)[mcols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.reset_index()
summ.to_csv(f"{out}/cv_results_summary.csv", index=False)

# ---------------------------------------------------------------- time holdout (train < 2023 seasons, tune inside)
tr, te = (d.time_holdout == 0).to_numpy(), (d.time_holdout == 1).to_numpy()
ho = []
ho_pred = {}
for fset, feats in FEATURE_SETS.items():
    X = d[feats].to_numpy()
    for name in MODELS:
        p, bp = fit_predict(name, X, tr, te, groups, np.where(te)[0]); ho_pred[(fset, name)] = p
        ho.append(dict(features=fset, model=name, best_params=str(bp), train_rows=int(tr.sum()), test_rows=int(te.sum()),
                       **metrics(ylog[te], p, sal[te], nd_mask[te])))
    p = np.mean([ho_pred[(fset, m)] for m in ENSEMBLE[1]], axis=0)
    ho.append(dict(features=fset, model=ENSEMBLE[0], best_params="-", train_rows=int(tr.sum()), test_rows=int(te.sum()),
                   **metrics(ylog[te], p, sal[te], nd_mask[te])))
pd.DataFrame(ho).to_csv(f"{out}/holdout_results.csv", index=False)

# ---------------------------------------------------------------- out-of-fold predictions (repeat 0)
o = d[["player_id", "player_name", "season", "team_abbr", "salary", "has_prior_salary"]].copy()
o["new_deal_row"] = nd_mask.astype(int)
for (f, m, r), p in oof_all.items():
    if r == 0: o[f"pred_{m}__{f}"] = np.round(np.exp(p))
o.to_csv(f"{out}/oof_predictions.csv", index=False)
json.dump({f"{k[0]} | {k[1]}": v for k, v in chosen.items()}, open(f"{out}/chosen_hyperparameters.json", "w"), indent=1)

# ---------------------------------------------------------------- figure
order_models = list(MODELS) + [ENSEMBLE[0]]
fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True)
fam = {"Ridge": "#4C72B0", "Lasso": "#4C72B0", "ElasticNet": "#4C72B0", "Poly2 Ridge": "#4C72B0",
       "Random Forest": "#55A868", "Extra Trees": "#55A868", "SVR (RBF)": "#8172B2", "kNN": "#8172B2",
       "Neural net (MLP)": "#8172B2", ENSEMBLE[0]: "#333333"}
col = lambda m: fam.get(m, "#DD8452")
for ax, (metric, title, scale) in zip(axes, [("r2_log", "R² (log salary)", 1), ("mae_usd", "MAE ($M)", 1e6),
                                              ("r2_log_newdeal", "R² (log), new-deal rows", 1)]):
    s = summ[summ.features == "perf+injury+prior"].set_index("model").loc[order_models]
    s = s.loc[s.sort_values("r2_log_mean", ascending=False).index]      # same row order in every panel (panels share the y labels)
    ax.barh(s.index, s[f"{metric}_mean"] / scale, xerr=s[f"{metric}_std"] / scale, color=[col(m) for m in s.index], capsize=2)
    for y_, v in enumerate(s[f"{metric}_mean"] / scale): ax.text(v, y_, f" {v:.2f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    if metric != "mae_usd": ax.set_xlim(max(0, s[f"{metric}_mean"].min() - 0.15), None)
fig.suptitle("Player-grouped 5-fold CV (3 repeats), features = performance + injury + prior salary\n"
             "blue = linear, green = bagged trees, orange = boosting, purple = kernel/neighbour/neural", fontsize=11)
fig.tight_layout(); fig.savefig(f"{out}/fig_model_comparison.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
a = summ[summ.features == "perf+injury"].set_index("model").loc[order_models, "r2_log_mean"]
b = summ[summ.features == "perf+injury+prior"].set_index("model").loc[order_models, "r2_log_mean"]
y_ = np.arange(len(order_models))
ax.barh(y_ - 0.2, a, 0.4, label="no prior salary", color="#8C8C8C"); ax.barh(y_ + 0.2, b, 0.4, label="with prior salary", color="#55A868")
ax.set_yticks(y_); ax.set_yticklabels(order_models); ax.invert_yaxis(); ax.set_xlim(0.4, None)
ax.set_xlabel("R² (log salary), CV mean"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Effect of prior-salary features, by model")
fig.tight_layout(); fig.savefig(f"{out}/fig_prior_salary_effect.png", dpi=150); plt.close(fig)

open(f"{out}/run_log.md", "w").write("# Wider model comparison run log\n\n" + "\n".join(log_lines) + "\n")
print("done")
