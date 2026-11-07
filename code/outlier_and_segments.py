"""Outlier detection and segmented-model experiments (answers the TA's suggestion).

Usage: python code/outlier_and_segments.py "data/processed/master_integrated (1).csv" reports/modeling_v3

Same sample, folds, and feature sets as train_more_models.py. XGBoost uses fixed settings
(depth 5, min child weight 8, 400 trees, lr 0.03; the most common inner-CV pick) so every
variant differs only in what is being tested. Segments are defined from information known
BEFORE the season (prior-season minutes per game, prior-season salary), never from the target.
Outlier removal is decided inside each training fold only; test rows are never filtered.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
data_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.read_csv(data_path, low_memory=False)
prior = ["log_prior_salary", "prior_salary_pct_cap", "has_prior_salary"]
injury = ["il_events_l1", "il_days_l1", "il_any_l1", "il_events_2yr", "il_days_2yr", "il_events_cum_since2006"]
cols = list(df.columns)
base56 = cols[cols.index("salary_pct_cap") + 1: cols.index("injury_coverage_l1")]
perf = [c for c in base56 if c not in prior]
FS = {"no prior": perf + injury, "with prior": perf + injury + prior}
d = df[df[injury].notna().all(axis=1)].reset_index(drop=True)
y = d.log_salary.to_numpy(); sal = d.salary.to_numpy(); n = len(d)
mpg = d.mpg_l1.to_numpy(); ps = np.exp(d.log_prior_salary.to_numpy()); hasp = d.has_prior_salary.to_numpy() == 1
SEG = {  # name -> integer labels per row
    "mpg": np.digitize(mpg, [15, 25]),                                  # <15, 15-25, >=25 minutes per game last season
    "prior salary": np.where(~hasp, 0, np.digitize(ps / (sal / d.salary_pct_cap.to_numpy()), [0.05, 0.20]) + 1),  # none / <5% / 5-20% / >=20% of cap
}
SEGNAME = {"mpg": ["<15 mpg", "15-25 mpg", ">=25 mpg"], "prior salary": ["no prior", "<5% cap", "5-20% cap", ">=20% cap"]}

def mk(): return XGBRegressor(learning_rate=0.03, n_estimators=400, max_depth=5, min_child_weight=8, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=42, verbosity=0)
def fold_ids(rep):
    if rep == 0: return d.fold.to_numpy()
    rng = np.random.RandomState(42 + rep); pl = d.player_id.unique(); rng.shuffle(pl)
    m = {p: i % 5 for i, p in enumerate(pl)}; return d.player_id.map(m).to_numpy()
def inner_resid(X, yy, g):
    r = np.zeros(len(yy))
    for a, b in GroupKFold(3).split(X, yy, g):
        r[b] = yy[b] - mk().fit(X[a], yy[a]).predict(X[b])
    return r

rows = []; oof_keep = {}
for fsname, feats in FS.items():
    X = d[feats].to_numpy(float); g = d.player_id.to_numpy()
    for rep in range(3):
        f = fold_ids(rep)
        P = {k: np.zeros(n) for k in ["global", "drop5", "drop10", "seg_mpg", "seg_prior salary", "iso_drop"]}
        for k in range(5):
            tr, te = np.where(f != k)[0], np.where(f == k)[0]
            m = mk().fit(X[tr], y[tr]); P["global"][te] = m.predict(X[te])
            # (1) residual-based outlier removal, decided from inner out-of-fold residuals on the training part
            r = inner_resid(X[tr], y[tr], g[tr])
            for pct, key in [(95, "drop5"), (90, "drop10")]:
                keep = np.abs(r) <= np.percentile(np.abs(r), pct)
                P[key][te] = mk().fit(X[tr][keep], y[tr][keep]).predict(X[te])
            # (2) unsupervised outliers in feature space (Isolation Forest, top 5% most isolated)
            iso = IsolationForest(n_estimators=200, random_state=0).fit(X[tr]); s = iso.score_samples(X[tr])
            keep = s >= np.percentile(s, 5)
            P["iso_drop"][te] = mk().fit(X[tr][keep], y[tr][keep]).predict(X[te])
            # (3) separate model per segment (segment from pre-season info), fallback to global if too small
            for sname, lab in SEG.items():
                for s_ in np.unique(lab):
                    tr_s, te_s = tr[lab[tr] == s_], te[lab[te] == s_]
                    if len(te_s) == 0: continue
                    mm = mk().fit(X[tr_s], y[tr_s]) if len(tr_s) >= 100 else m
                    P[f"seg_{sname}"][te_s] = mm.predict(X[te_s])
        for v, p in P.items():
            ok = np.ones(n, bool)
            rows.append(dict(feats=fsname, rep=rep, variant=v, segment="ALL", n=n, r2=r2_score(y, p),
                             mae=np.mean(np.abs(np.exp(p) - sal)) / 1e6, medpe=np.median(np.abs(np.exp(p) - sal) / sal)))
            for sname, lab in SEG.items():
                for s_, nm in enumerate(SEGNAME[sname]):
                    mk_ = lab == s_
                    if mk_.sum() < 30: continue
                    rows.append(dict(feats=fsname, rep=rep, variant=v, segment=f"{sname}: {nm}", n=int(mk_.sum()),
                                     r2=r2_score(y[mk_], p[mk_]), mae=np.mean(np.abs(np.exp(p[mk_]) - sal[mk_])) / 1e6,
                                     medpe=np.median(np.abs(np.exp(p[mk_]) - sal[mk_]) / sal[mk_])))
        if rep == 0: oof_keep[fsname] = P["global"].copy()
        print(fsname, "rep", rep, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{out}/segment_outlier_by_repeat.csv", index=False)
S = R.groupby(["feats", "variant", "segment"]).agg(n=("n", "first"), r2=("r2", "mean"), mae=("mae", "mean"), medpe=("medpe", "mean")).reset_index()
S.to_csv(f"{out}/segment_outlier_summary.csv", index=False)
# segment context: spread of the target inside each segment (restricted range)
ctx = []
for sname, lab in SEG.items():
    for s_, nm in enumerate(SEGNAME[sname]):
        mk_ = lab == s_; ctx.append(dict(segment=f"{sname}: {nm}", n=int(mk_.sum()), share=mk_.mean(), median_salary_M=np.median(sal[mk_]) / 1e6,
                                         sd_log_salary=y[mk_].std(), var_ratio_vs_all=y[mk_].var() / y.var()))
pd.DataFrame(ctx).to_csv(f"{out}/segment_context.csv", index=False)
# outlier lists from with-prior OOF residuals
res = y - oof_keep["with prior"]; thr = 2 * 1.4826 * np.median(np.abs(res - np.median(res)))
o = d[["player_name", "season", "mpg_l1", "salary"]].copy() if "player_name" in d else d[["player_id", "season", "mpg_l1", "salary"]].copy()
o["pred"] = np.exp(oof_keep["with prior"]); o["log_resid"] = res; o["is_outlier"] = np.abs(res) > thr
o.to_csv(f"{out}/oof_residuals_with_prior.csv", index=False)
json.dump(dict(mad_threshold_log=float(thr), n_outliers=int(o.is_outlier.sum()), share=float(o.is_outlier.mean()),
               share_over=float((res < -thr).mean()), share_under=float((res > thr).mean())), open(f"{out}/outlier_summary.json", "w"), indent=1)
print(pd.read_json(f"{out}/outlier_summary.json", typ="series"))
