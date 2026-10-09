"""Build the master player-season dataset from the cleaned NBA Data tables.
One row per salary player-season (season = season-ending year). All predictors come from seasons
before the salary season (prior-season rule), except age, position, and static player attributes."""
import sys, hashlib, numpy as np, pandas as pd
interim, raw, ext, out = sys.argv[1:5]
WINDOW_START, MIN_MP, MIN_G_SAME, MIN_SALARY, PRUNE_R = 2000, 200, 10, 100000, 0.90
LOG = []
def log(s=""): print(s); LOG.append(s)
rd = lambda n: pd.read_csv(f"{raw}/public.{n}.csv", low_memory=False)

sal = pd.read_csv(f"{interim}/salaries_clean.csv")
st = pd.read_csv(f"{interim}/season_stats_clean.csv")
ep = pd.read_csv(f"{interim}/epm_clean.csv")
pl = pd.read_csv(f"{interim}/players_clean.csv")
cap = pd.read_csv(f"{ext}/nba_salary_cap_history_realgm.csv").set_index("season_end_year").salary_cap_usd

# ---------------- player-season features (season t)
f = st.copy()
f["mpg"] = f.mp / f.g
f["gs_share"] = f.gs / f.g
f["g_share"] = f.g / f.groupby("season").g.transform("max")
for c in ["pts","trb","ast","stl","blk","tov","x3pa"]:
    f[f"{c}_36"] = np.where(f.mp > 0, f[c] / f.mp * 36, np.nan)
for c in ["pts_36","trb_36","ast_36","x3pa_36","ts_percent"]:
    q = f[f.mp >= MIN_MP].groupby("season")[c].mean()
    f[f"{c}_adj"] = f[c] - f.season.map(q)
BASE = ["g","mp","mpg","gs_share","g_share","pts","pts_36","trb_36","ast_36","stl_36","blk_36","tov_36","x3pa_36",
        "ts_percent","usg_percent","per","ws","ws_48","obpm","dbpm","bpm","vorp","o_rtg","d_rtg",
        "ast_percent","trb_percent","tov_percent","stl_percent","blk_percent","x3p_ar","f_tr",
        "pts_36_adj","trb_36_adj","ast_36_adj","x3pa_36_adj","ts_percent_adj"]
LAG2 = ["g_share","mp","pts_36","ts_percent","usg_percent","per","ws","ws_48","bpm","vorp"]
TREND = ["bpm","ws","vorp","per"]

def lag(df, cols, k, suffix):
    t = df[["player_id","season"]+cols].copy(); t["season"] = t.season + k
    return t.rename(columns={c: f"{c}_{suffix}" for c in cols})

base = sal[sal.player_id.notna()].copy()
n_all = len(sal); n_linked = len(base)
m = base.merge(lag(f, BASE, 1, "l1"), on=["player_id","season"], how="left") \
        .merge(lag(f, LAG2, 2, "l2"), on=["player_id","season"], how="left")
same = f[["player_id","season","g","mp"]].rename(columns={"g":"g_same","mp":"mp_same"})
m = m.merge(same, on=["player_id","season"], how="left")
for c in LAG2: m[f"{c}_avg2"] = m[[f"{c}_l1", f"{c}_l2"]].mean(axis=1)
m["has_l2"] = m.mp_l2.notna().astype(int)
for c in TREND: m[f"{c}_trend"] = (m[f"{c}_l1"] - m[f"{c}_l2"]).where(m.has_l2 == 1, 0.0)

# career to date (through the last observed season before the salary season)
f = f.sort_values(["player_id","season"])
cum = f[["player_id","season"]].copy()
cum["career_g"] = f.groupby("player_id").g.cumsum()
cum["career_mp"] = f.groupby("player_id").mp.cumsum()
cum["career_ws"] = f.groupby("player_id").ws.cumsum()
cum["career_vorp"] = f.groupby("player_id").vorp.cumsum()
bpm_q = f.bpm.where(f.mp >= MIN_MP)
cum["peak_bpm"] = bpm_q.groupby(f.player_id).cummax()
cum["peak_bpm"] = cum.groupby("player_id").peak_bpm.ffill()

# awards (counts through prior seasons, flag for the immediately prior season)
asg = rd("all_star_selections"); asg = asg[(asg.lg=="NBA") & asg.player_id.notna()]
eot = rd("end_of_season_teams"); eot = eot[(eot.lg=="NBA") & eot.player_id.notna()]
log(f"- End-of-season team types present: {sorted(eot.type.dropna().unique())}")
aw = pd.concat([
    asg[["player_id","season"]].drop_duplicates().assign(all_star=1),
    eot[eot.type=="All-NBA"][["player_id","season"]].drop_duplicates().assign(all_nba=1),
    eot[eot.type=="All-Defense"][["player_id","season"]].drop_duplicates().assign(all_def=1)]).fillna(0)
aw = aw.groupby(["player_id","season"], as_index=False)[["all_star","all_nba","all_def"]].max().sort_values(["player_id","season"])
for c in ["all_star","all_nba","all_def"]: aw[f"{c}_count_prior"] = aw.groupby("player_id")[c].cumsum()
cum = cum.merge(aw[["player_id","season","all_star_count_prior","all_nba_count_prior","all_def_count_prior"]], on=["player_id","season"], how="outer").sort_values(["player_id","season"])
for c in ["career_g","career_mp","career_ws","career_vorp","peak_bpm","all_star_count_prior","all_nba_count_prior","all_def_count_prior"]:
    cum[c] = cum.groupby("player_id")[c].ffill()
cum["season"] = cum.season.astype(int)
m["season_prev"] = m.season - 1
m = m.sort_values("season_prev")
m = pd.merge_asof(m, cum.sort_values("season"), left_on="season_prev", right_on="season", by="player_id", direction="backward", suffixes=("","_cum")).drop(columns=["season_cum"])
for c in ["all_star_count_prior","all_nba_count_prior","all_def_count_prior","career_g","career_mp","career_ws","career_vorp"]: m[c] = m[c].fillna(0)
m = m.merge(lag(aw.rename(columns={"all_star":"all_star_l1","all_nba":"all_nba_l1","all_def":"all_def_l1"}), ["all_star_l1","all_nba_l1","all_def_l1"], 1, "tmp").rename(columns=lambda c: c.replace("_l1_tmp","_l1")), on=["player_id","season"], how="left")
for c in ["all_star_l1","all_nba_l1","all_def_l1"]: m[c] = m[c].fillna(0)

# EPM
e1 = ep.dropna(subset=["br_player_id"]).rename(columns={"br_player_id":"player_id"})[["player_id","season","epm_off","epm_def","epm_tot"]]
m = m.merge(lag(e1, ["epm_off","epm_def","epm_tot"], 1, "l1"), on=["player_id","season"], how="left")

# players: experience, draft, physical
pm = pl.drop_duplicates("player_id").set_index("player_id")
m["exp_prior"] = (m.season - m.player_id.map(pm.year_from)).clip(lower=0)
m["height_cm"] = m.player_id.map(pm.height_cm); m["weight_kg"] = m.player_id.map(pm.weight_kg)
dd = rd("dim_draft_history"); dd = dd[dd.league=="NBA"].copy()
dd["has_pick"] = dd.pick_overall.notna(); dd = dd.sort_values(["has_pick","season"], ascending=[False, True]).drop_duplicates("player_id")
dd = dd.set_index("player_id")
in_draft = m.player_id.isin(dd.index)
m["draft_pick"] = m.player_id.map(dd.pick_overall)
m["draft_season"] = m.player_id.map(dd.season)
m["undrafted_or_unlisted"] = (~in_draft).astype(int)
m.loc[~in_draft, "draft_pick"] = 61
m["years_since_draft"] = (m.season - m.draft_season - 1).clip(lower=0).fillna(m.exp_prior)
m["age_sq"] = m.age ** 2

# targets and context
m["cap_usd"] = m.season.map(cap)
m["log_salary"] = np.log(m.salary)
m["salary_pct_cap"] = m.salary / m.cap_usd
m["in_injury_window"] = (m.season >= 2005).astype(int)
m["fold"] = m.player_id.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 5)
m["time_holdout"] = (m.season >= 2023).astype(int)
m["pos_primary"] = m.pos_primary.fillna("unknown")
m = m.sort_values("salary_row_id").reset_index(drop=True)
allcols = pd.concat([m, sal[sal.player_id.isna()]], ignore_index=True)
allcols.to_csv(f"{out}/master_all.csv", index=False)
log(f"- Master table before filtering: {len(allcols):,} salary rows ({n_linked:,} linked to a player identifier, {n_all-n_linked:,} not linked), {m.shape[1]} columns.")

# ---------------- evidence for filter choices
ev = m[(m.season >= WINDOW_START) & (m.mp_l1 >= MIN_MP)].copy()
ev["played_same"] = (ev.g_same.fillna(0) >= MIN_G_SAME)
log("\n### Evidence for the record filters (window start, prior-season minutes at least %d)" % MIN_MP)
log("Median salary as a share of the cap, by whether the player logged at least %d games in the salary season:" % MIN_G_SAME)
log(ev.groupby(["played_same","salary_source"]).salary_pct_cap.agg(["size","median"]).round(4).to_string())
mp_b = pd.cut(m.mp_l1, [0,50,100,200,400,800,5000]); 
log("Share of prior-season rows with true shooting percentage outside [0.30, 0.75] or box plus-minus beyond +/-10, by prior-season minutes bucket:")
ex = ((m.ts_percent_l1 < 0.30) | (m.ts_percent_l1 > 0.75) | (m.bpm_l1.abs() > 10))
log(ex.groupby(mp_b, observed=True).agg(["size","mean"]).round(3).to_string() if False else pd.DataFrame({"rows": m.groupby(mp_b, observed=True).size(), "extreme_share": ex.groupby(mp_b, observed=True).mean().round(3)}).to_string())

# ---------------- record filters (waterfall)
w = [("Salary rows after cleaning", n_all)]
d = m.copy(); w.append(("Linked to a player identifier", len(d)))
d = d[d.season >= WINDOW_START]; w.append((f"Salary season {WINDOW_START} or later", len(d)))
d = d[d.mp_l1.notna()]; w.append(("Has a prior-season statistics row", len(d)))
d = d[d.mp_l1 >= MIN_MP]; w.append((f"Prior-season minutes at least {MIN_MP}", len(d)))
d = d[d.g_same.fillna(0) >= MIN_G_SAME]; w.append((f"Played at least {MIN_G_SAME} games in the salary season", len(d)))
d = d[d.salary >= MIN_SALARY]; w.append((f"Salary at least {MIN_SALARY:,} dollars", len(d)))
d = d[d.cap_usd.notna() & d.age.notna()]; w.append(("Has cap and age", len(d)))
log("\n### Record waterfall")
for lbl, n in w: log(f"- {lbl}: {n:,}")
pd.DataFrame(w, columns=["step","rows"]).to_csv(f"{out}/record_waterfall.csv", index=False)

# ---------------- feature set
POS = pd.get_dummies(d.pos_primary, prefix="pos").astype(int)
d = pd.concat([d, POS], axis=1)
feat = {}
def add(names, group): 
    for n in names: feat[n] = group
add(["age","age_sq","exp_prior","years_since_draft","draft_pick","undrafted_or_unlisted","height_cm","weight_kg"], "player profile")
add(list(POS.columns), "position")
add([f"{c}_l1" for c in BASE if not c.endswith("_adj")], "prior-season performance")
add([f"{c}_l1" for c in BASE if c.endswith("_adj")], "era-adjusted prior-season rates")
add([f"{c}_avg2" for c in LAG2] + [f"{c}_trend" for c in TREND] + ["has_l2"], "two-season history and trend")
add(["career_g","career_mp","career_ws","career_vorp","peak_bpm"], "career to date")
add(["all_star_l1","all_nba_l1","all_def_l1","all_star_count_prior","all_nba_count_prior","all_def_count_prior"], "honors")
add(["cap_usd"], "context")
EPM = ["epm_off_l1","epm_def_l1","epm_tot_l1"]
X = d[list(feat)].copy()
miss = X.isna().mean(); miss = miss[miss > 0].sort_values(ascending=False)
log("\n### Missing values in candidate features after record filters")
log(miss.round(4).to_string() if len(miss) else "none")
med = X.median(numeric_only=True)
n_imp = X.isna().sum(); n_imp = n_imp[n_imp > 0]
X = X.fillna(med)
log(f"- Imputed with the median of the filtered table: {int(n_imp.sum())} values across {len(n_imp)} features (largest: {n_imp.sort_values(ascending=False).head(4).to_dict()}).")
for c in ["ts_percent_l1"]:
    X[c] = X[c].clip(0, 1)

# ---------------- correlation pruning
tgt = d[["log_salary","salary_pct_cap"]]
cor_t = pd.DataFrame({t: X.corrwith(tgt[t]) for t in tgt}).abs().mean(axis=1).sort_values(ascending=False)
protected = set(POS.columns) | {"has_l2","undrafted_or_unlisted","cap_usd"}
num = [c for c in X.columns if c not in protected]
C = X[num].corr().abs()
kept = []; dropped = []
for c in cor_t[num].sort_values(ascending=False).index:
    hit = [k for k in kept if C.loc[c, k] > PRUNE_R]
    if hit: dropped.append((c, hit[0], round(float(C.loc[c, hit[0]]),3)))
    else: kept.append(c)
final = kept + [c for c in X.columns if c in protected]
log(f"\n### Correlation pruning (drop a feature if |r| > {PRUNE_R} with a kept feature that is more correlated with the targets)")
log(f"- Candidate features: {X.shape[1]}; kept: {len(final)}; dropped: {len(dropped)}")
for a, b, r in dropped: log(f"  - dropped {a} (r = {r} with {b})")
sd = X[final].std(); log(f"- Near-constant features among kept (std < 1e-6): {list(sd[sd < 1e-6].index)}")

ID = ["salary_row_id","player_id","player_name","season","team_abbr","salary_source","match_stage","in_injury_window","fold","time_holdout"]
TGT = ["salary","log_salary","salary_pct_cap"]
mr = pd.concat([d[ID + TGT].reset_index(drop=True), X[final].reset_index(drop=True)], axis=1)
mr.to_csv(f"{out}/master_model_ready.csv", index=False)
ex_cols = ["salary_row_id"] + EPM
d[ex_cols].to_csv(f"{out}/master_epm_extension.csv", index=False)

# optional extension: the player's salary in the previous season (kept out of the main table on purpose)
ps = sal[["player_id","season","salary"]].dropna(subset=["player_id"]).drop_duplicates(["player_id","season"])
ps = ps.assign(season=ps.season + 1).rename(columns={"salary":"prior_salary"})
px = d[["salary_row_id","player_id","season"]].merge(ps, on=["player_id","season"], how="left")
px["log_prior_salary"] = np.log(px.prior_salary)
px["prior_salary_pct_cap"] = px.prior_salary / (px.season - 1).map(cap)
px[["salary_row_id","prior_salary","log_prior_salary","prior_salary_pct_cap"]].to_csv(f"{out}/master_prior_salary_extension.csv", index=False)
log(f"\n### Final model-ready table: {mr.shape[0]:,} rows, {len(final)} features, {len(TGT)} target columns, {len(ID)} identifier and split columns")
log(f"- Seasons {mr.season.min()} to {mr.season.max()}; players {mr.player_id.nunique():,}; rows with the Estimated Plus-Minus extension available: {int(d.epm_tot_l1.notna().sum()):,}")
log(f"- Targets: salary median {mr.salary.median():,.0f} dollars, mean {mr.salary.mean():,.0f}; share of cap median {mr.salary_pct_cap.median():.4f}; skewness salary {mr.salary.skew():.2f}, log salary {mr.log_salary.skew():.2f}")
log("- Correlation of final features with log salary (top 12): " + ", ".join(f"{k} {v:.2f}" for k, v in X[final].corrwith(d.log_salary).abs().sort_values(ascending=False).head(12).items()))
log(f"- Prior-salary extension (not a feature in the main table): available for {int(px.prior_salary.notna().sum()):,} of {len(px):,} rows")
log(f"- Duplicate player-seasons: {int(mr.duplicated(['player_id','season']).sum())}; fold sizes: {mr.fold.value_counts().sort_index().to_dict()}; time holdout rows: {int(mr.time_holdout.sum())}")
fd = pd.DataFrame({"feature": list(feat), "group": list(feat.values())})
fd["kept_after_pruning"] = fd.feature.isin(final)
fd = fd.merge(pd.DataFrame(dropped, columns=["feature","dropped_for","r_with"]), on="feature", how="left")
fd["abs_corr_with_targets"] = fd.feature.map(cor_t).round(3)
fd.to_csv(f"{out}/feature_dictionary_draft.csv", index=False)
open(f"{out}/build_log.md","w").write("# Master dataset build log\n\n" + "\n".join(LOG) + "\n")
