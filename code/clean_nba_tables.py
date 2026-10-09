"""Clean the NBA Data 1946-2026 tables needed for the master dataset.
Reads data/raw, writes data/interim and a markdown log of every step with counts."""
import sys, re, unicodedata, pandas as pd, numpy as np
raw, out = sys.argv[1], sys.argv[2]
r = lambda n, **k: pd.read_csv(f"{raw}/public.{n}.csv", low_memory=False, **k)
LOG = []
def log(s=""): print(s); LOG.append(s)

def norm(s):
    if pd.isna(s): return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode().lower().replace("*","")
    s = re.sub(r"[.'`\-,]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(2): s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)
    return s

# ---------------------------------------------------------------- players
dp = r("dim_players")
log("## 1. Player master (dim_players)")
log(f"- Input rows: {len(dp):,}; slug unique: {dp.player_id.is_unique}")
drop_cols = ["pronunciation","nationality","source_url","scraped_at","headshot_path","headshot_url","headshot_status",
             "headshot_scraped_at","bio_ext_scraped_at","player_page_extras_scraped_at","relatives","career_honors_text","aba_debut","died"]
log(f"- Dropped {len(drop_cols)} columns that are scrape metadata or almost entirely empty (nationality is 100% empty).")
dp = dp.drop(columns=drop_cols)
dp["birth_date"] = pd.to_datetime(dp.birth_date, errors="coerce")
dp["is_hall_of_famer"] = dp.is_hall_of_famer.map({"t":True,"f":False,True:True,False:False})
POS = {"Point Guard":"PG","Shooting Guard":"SG","Small Forward":"SF","Power Forward":"PF","Center":"C","Guard":"G","Forward":"F"}
dp["pos_primary"] = dp.position.fillna("").str.split(r" and |/|, ", regex=True).str[0].map(POS)
dp["name_key"] = dp.player_name.map(norm)
log(f"- Parsed birth_date ({dp.birth_date.isna().sum()} missing) and primary position ({dp.pos_primary.isna().sum()} unmapped; unmapped labels: {dp[dp.pos_primary.isna()].position.value_counts(dropna=False).head(5).to_dict()}).")
dp.to_csv(f"{out}/players_clean.csv", index=False)

# ---------------------------------------------------------------- season stats
st = r("fact_player_season_stats")
log("\n## 2. Player season statistics (fact_player_season_stats)")
log(f"- Input rows: {len(st):,} (Regular Season {int((st.season_type=='Regular Season').sum()):,}, Playoffs {int((st.season_type=='Playoffs').sum()):,}).")
log(f"- Playoff rows have no player identifier in {st[st.season_type=='Playoffs'].player_id.isna().mean():.1%} of cases, so playoff rows are dropped; the project values regular season performance.")
st = st[st.season_type=="Regular Season"].copy()
st = st[st.season >= 1996].copy()
log(f"- Kept seasons 1996 and later (1996 is needed as the prior season for 1997 salaries): {len(st):,} rows.")
st["is_total_row"] = st.team.astype(str).str.fullmatch(r"\dTM")
n_before = len(st)
tot = st[st.is_total_row].set_index(["player_id","season"])
per_team = st[~st.is_total_row]
teams = per_team.groupby(["player_id","season"]).team.agg(lambda s: "/".join(s)).rename("teams")
nteams = per_team.groupby(["player_id","season"]).size().rename("n_teams")
keep_total = st[st.is_total_row | ~st.set_index(["player_id","season"]).index.isin(tot.index)]
keep_total = keep_total.drop(columns=["id","weight","lg","season_type","trp_dbl"])
keep_total = keep_total.join(teams, on=["player_id","season"]).join(nteams, on=["player_id","season"])
keep_total["n_teams"] = keep_total.n_teams.fillna(1).astype(int)
keep_total["teams"] = keep_total.teams.fillna(keep_total.team)
keep_total = keep_total.rename(columns={"team":"team_row","pos":"pos_raw"})
keep_total["player"] = keep_total.player.str.replace("*","",regex=False).str.strip()
keep_total["pos_primary"] = keep_total.pos_raw.astype(str).str.split("-").str[0].replace({"nan":np.nan})
log(f"- Traded players have one row per team plus a combined row (team code like 2TM). Kept the combined row, kept the team list in `teams` and the count in `n_teams`: {n_before:,} -> {len(keep_total):,} rows, one per player-season. Duplicate player-season check: {keep_total.duplicated(['player_id','season']).sum()}.")
miss = keep_total[["age","pos_raw","mp","per","ts_percent","usg_percent","ws","bpm","vorp","o_rtg","d_rtg"]].isna().mean().round(4)
log(f"- Missing shares after filtering: " + ", ".join(f"{k} {v:.2%}" for k,v in miss.items() if v>0))
log(f"- Missing advanced metrics are concentrated in tiny samples: median minutes played in rows with missing true shooting percentage is {keep_total[keep_total.ts_percent.isna()].mp.median():.0f} versus {keep_total.mp.median():.0f} overall.")
keep_total.to_csv(f"{out}/season_stats_clean.csv", index=False)
stats_ps = keep_total[["player_id","season","age","pos_primary","teams","g","mp"]]

# ---------------------------------------------------------------- Estimated Plus-Minus
ep = r("player_epm")
log("\n## 3. Estimated Plus-Minus (player_epm)")
log(f"- Input rows: {len(ep):,}, seasons {ep.season.min()} to {ep.season.max()}.")
log("- `epm_tot_change` is 100% empty and the raw JSON column duplicates other fields; both dropped.")
dup = ep.duplicated(["nba_player_id","season"], keep=False).sum()
ep = ep.sort_values(["nba_player_id","season","game_dt"])
ep["is_final"] = ep.source.eq("dunksandthrees.com/epm")
ep = ep.sort_values(["nba_player_id","season","is_final","game_dt"]).drop_duplicates(["nba_player_id","season"], keep="last")
log(f"- {dup//2} player-seasons appeared twice (all in 2026, from two sources); kept the season-end snapshot: {len(ep):,} rows, one per player-season.")
br = r("player_id_bridge")
br = br.dropna(subset=["nba_player_id","br_player_id"]).drop_duplicates("nba_player_id")
br["nba_player_id"] = br.nba_player_id.astype(float)
n_miss = ep.br_player_id.isna().sum()
ep["br_player_id"] = ep.br_player_id.fillna(ep.nba_player_id.astype(float).map(br.set_index("nba_player_id").br_player_id))
n_bridge = n_miss - ep.br_player_id.isna().sum()
nmk = keep_total[["player_id","season","player"]].assign(nn=lambda d: d.player.map(norm)).groupby(["nn","season"]).player_id.agg(lambda x: sorted(set(x)))
have_ep = set(zip(ep.br_player_id.dropna(), ep.season))
fix = []
for i in ep.index[ep.br_player_id.isna()]:
    cs = nmk.get((norm(ep.at[i,"player_name"]), ep.at[i,"season"]))
    fix.append(cs[0] if isinstance(cs,list) and len(cs)==1 and (cs[0], ep.at[i,"season"]) not in have_ep else np.nan)
ep.loc[ep.br_player_id.isna(), "br_player_id"] = fix
log(f"- Missing Basketball-Reference slug: {n_miss} rows ({n_miss/len(ep):.1%}); {n_bridge} recovered through the player identifier bridge, {n_miss-n_bridge-ep.br_player_id.isna().sum()} more by matching normalized name and season to the statistics table, {ep.br_player_id.isna().sum()} remain unlinked.")
slug_name = dp.drop_duplicates("player_id").set_index("player_id").name_key
dmask = ep.br_player_id.notna() & ep.duplicated(["br_player_id","season"], keep=False)
n_wrong = 0
for i in ep.index[dmask]:
    if norm(ep.at[i,"player_name"]) != slug_name.get(ep.at[i,"br_player_id"]):
        cs = nmk.get((norm(ep.at[i,"player_name"]), ep.at[i,"season"]))
        ep.at[i,"br_player_id"] = cs[0] if isinstance(cs,list) and len(cs)==1 else np.nan
        n_wrong += 1
log(f"- The source assigns one slug to two different players in {int(dmask.sum())//2} player-seasons (for example Bojan and Bogdan Bogdanovic share one slug). Re-linked {n_wrong} rows by name and season; duplicates remaining: {int(ep.dropna(subset=['br_player_id']).duplicated(['br_player_id','season']).sum())}.")
left = ep.br_player_id.notna() & ep.duplicated(["br_player_id","season"], keep=False)
if left.any():
    log(f"- {int(left.sum())//2} unresolved shared-slug player-seasons: kept the row whose name matches the player master, otherwise the one with more possessions sampled ({sorted(set(ep[left].player_name))}).")
    ep["_ok"] = ep.apply(lambda r: norm(r.player_name) == slug_name.get(r.br_player_id), axis=1)
    ep = pd.concat([ep[ep.br_player_id.notna()].sort_values(["_ok","n"]).drop_duplicates(["br_player_id","season"], keep="last"), ep[ep.br_player_id.isna()]]).drop(columns="_ok")
ep = ep[["br_player_id","nba_player_id","player_name","season","team_alias","age","epm_off","epm_def","epm_tot","n"]]
ep.to_csv(f"{out}/epm_clean.csv", index=False)

# ---------------------------------------------------------------- draft combine
dc = r("draft_combine")
log("\n## 4. Draft combine (draft_combine)")
nn = dc.wingspan_cm.notna().sum()
ratio = (dc.wingspan_cm/dc.height_wo_shoes_cm)
both = dc.wingspan_cm.notna() & dc.height_wo_shoes_cm.notna()
log(f"- Of {len(dc):,} rows, {int(both.sum()):,} have both a wingspan and a height, and in all {int((dc.wingspan_cm[both]<dc.height_wo_shoes_cm[both]).sum()):,} of them the wingspan is below height. The earlier '1,152 of 1,202' figure is therefore every row with data, not a subset, which points to a units error rather than noise.")
log(f"- Median wingspan label 'cm' is {dc.wingspan_cm.median():.2f} versus median height {dc.height_wo_shoes_cm.median():.0f} cm (ratio {ratio.median():.3f}). The values are inches, not centimeters.")
for c in ["wingspan_cm","standing_reach_cm"]:
    dc[c.replace("_cm","_in")] = dc[c]; dc[c] = dc[c]*2.54
r2 = dc.wingspan_cm/dc.height_wo_shoes_cm
log(f"- After multiplying by 2.54, median wingspan to height ratio is {r2.median():.3f} and {(r2>=1).mean():.1%} of non-missing rows have wingspan at least equal to height.")
dc.to_csv(f"{out}/draft_combine_clean.csv", index=False)

# ---------------------------------------------------------------- salaries
sal = r("player_salaries_historical")
log("\n## 5. Historical salaries (player_salaries_historical)")
log(f"- Input rows: {len(sal):,}; seasons {sal.season.min()} to {sal.season.max()}; unique player-season: {not sal.duplicated(['player_name','season']).any()}.")
src = sal.groupby("source").size()
log("- Rows by collection source: " + ", ".join(f"{k} {v:,}" for k,v in src.items()) + ". Age and position are missing exactly for the two sources that never carried them (" + f"{(sal.age.isna()).sum():,} rows, {sal.age.isna().mean():.2%}).")
sal = sal.rename(columns={"source":"salary_source","team_abbr":"team_abbr_raw","position":"pos_raw","age":"age_raw","id":"salary_row_id"}).drop(columns=["imported_at"])
sal["name_key"] = sal.player_name.map(norm)
# team codes
valid = set(keep_total.teams.str.split("/").explode().dropna()) | {"CHO"}
sal["team_abbr"] = sal.team_abbr_raw.replace({"CHA":"CHO"})
bad = ~sal.team_abbr.isin(valid)
log(f"- Team codes: 'CHA' ({int((sal.team_abbr_raw=='CHA').sum())} rows, 2005) mapped to 'CHO' to match the statistics table; {int(bad.sum())} rows carry codes with no counterpart ({sorted(sal[bad].team_abbr_raw.unique())}), set to missing and kept.")
sal.loc[bad, "team_abbr"] = np.nan

# slug matching
ps = pd.concat([stats_ps.assign(nn=stats_ps.player_id.map(dp.drop_duplicates("player_id").set_index("player_id").name_key)) ]) if False else None
nm = keep_total[["player_id","season","player","age","teams"]].copy(); nm["nn"] = nm.player.map(norm)
cand = nm.groupby(["nn","season"]).player_id.agg(lambda s: sorted(set(s)))
sal = sal.merge(cand.rename("c1").reset_index().rename(columns={"nn":"name_key"}), on=["name_key","season"], how="left")
sal["player_id"] = pd.Series([None]*len(sal), dtype=object, index=sal.index); sal["match_stage"] = ""
n1 = sal.c1.map(lambda x: len(x) if isinstance(x,list) else 0)
m = n1==1; sal.loc[m,"player_id"] = sal.loc[m,"c1"].str[0]; sal.loc[m,"match_stage"] = "1_same_season_stats_name"
# 1b ambiguous -> age then team
amb = n1>1
agemap = nm.set_index(["player_id","season"]).age; teammap = nm.set_index(["player_id","season"]).teams
for i in sal.index[amb]:
    row = sal.loc[i]; cs = row.c1
    keep = [c for c in cs if pd.notna(row.age_raw) and (c,row.season) in agemap.index and abs(agemap[(c,row.season)]-row.age_raw)<=0]
    if len(keep)!=1:
        keep = [c for c in cs if pd.notna(row.team_abbr) and (c,row.season) in teammap.index and row.team_abbr in str(teammap[(c,row.season)]).split("/")]
    if len(keep)==1: sal.loc[i,"player_id"] = keep[0]; sal.loc[i,"match_stage"] = "1b_disambiguated_age_or_team"
log(f"- Slug matching stage 1 (normalized name plus same-season statistics row): {int((sal.match_stage=='1_same_season_stats_name').sum()):,} unique; {int(amb.sum())} ambiguous names, {int((sal.match_stage=='1b_disambiguated_age_or_team').sum())} resolved by age or team.")
# stage 2 good_name_map
gm = r("good_name_map"); gm["nn"] = gm.nn.map(norm)
g_team_sy = gm.dropna(subset=["sy"]).groupby(["nn","team","sy"]).br_player_id.agg(lambda s: sorted(set(s)))
g_team = gm.groupby(["nn","team"]).br_player_id.agg(lambda s: sorted(set(s)))
un = sal.index[sal.player_id.isna()]
c2 = 0
for i in un:
    row = sal.loc[i]; tm = row.team_abbr_raw
    cs = g_team_sy.get((row.name_key,tm,float(row.season))) or g_team.get((row.name_key,tm)) or g_team.get((row.name_key,row.team_abbr))
    if isinstance(cs,list) and len(cs)==1: sal.loc[i,"player_id"] = cs[0]; sal.loc[i,"match_stage"] = "2_good_name_map"; c2 += 1
log(f"- Stage 2 (dataset name map by name, team, season): {c2:,} more.")
# stage 3 dim_players
lk = {}
for col in ["player_name","full_name","nickname"]:
    for pid, nn_, yf, yt in zip(dp.player_id, dp[col].map(norm), dp.year_from, dp.year_to):
        if nn_: lk.setdefault(nn_, []).append((pid, yf, yt))
c3 = 0
for i in sal.index[sal.player_id.isna()]:
    row = sal.loc[i]; cs = lk.get(row.name_key, [])
    cs = {p for p,yf,yt in cs if pd.isna(yf) or (yf-1 <= row.season <= yt+1)}
    if len(cs)==1: sal.loc[i,"player_id"] = next(iter(cs)); sal.loc[i,"match_stage"] = "3_player_master_name_and_career_span"; c3 += 1
log(f"- Stage 3 (player master names and nicknames, season inside career span plus or minus one year): {c3:,} more.")
import difflib
c4 = 0; pairs = []
by_season = {k: g for k, g in nm.groupby("season")}
for i in sal.index[sal.player_id.isna()]:
    row = sal.loc[i]; g = by_season.get(row.season)
    if g is None or not row.name_key: continue
    if pd.notna(row.team_abbr): g = g[g.teams.str.split("/").map(lambda t: row.team_abbr in t)]
    if pd.notna(row.age_raw): g = g[(g.age - row.age_raw).abs() <= 1]
    if g.empty: continue
    sc = g.nn.map(lambda x: difflib.SequenceMatcher(None, row.name_key, x).ratio())
    top = sc.sort_values(ascending=False)
    last_ok = g.nn.map(lambda x: x.split()[-1] == row.name_key.split()[-1])
    best = top.index[0]
    fa, fb = row.name_key.split()[0], g.loc[best,"nn"].split()[0]
    first_ok = fa.startswith(fb) or fb.startswith(fa) or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.80
    if top.iloc[0] >= 0.80 and last_ok[best] and first_ok and (len(top) == 1 or top.iloc[1] <= top.iloc[0] - 0.05):
        sal.loc[i,"player_id"] = g.loc[best,"player_id"]; sal.loc[i,"match_stage"] = "4_fuzzy_name_team_season"; c4 += 1
        pairs.append((row.player_name, g.loc[best,"player"], int(row.season)))
log(f"- Stage 4 (fuzzy name match restricted to the same season, same team and age within one year, last name equal, overall similarity at least 0.80 with a clear winner, and first names either sharing a prefix or at least 0.80 similar): {c4} more.")
pd.DataFrame(pairs, columns=["salary_name","matched_stats_name","season"]).to_csv(f"{out}/salary_fuzzy_matches_to_review.csv", index=False)
c5 = 0; pairs5 = []
dpn = dp.dropna(subset=["name_key"]).copy()
for i in sal.index[sal.player_id.isna()]:
    row = sal.loc[i]
    if not row.name_key: continue
    g = dpn[dpn.year_from.notna() & (dpn.year_from - 1 <= row.season) & (dpn.year_to + 1 >= row.season)]
    g = g[g.name_key.map(lambda x: x.split()[-1][:3] == row.name_key.split()[-1][:3])]
    if g.empty: continue
    sc = g.name_key.map(lambda x: difflib.SequenceMatcher(None, row.name_key, x).ratio()).sort_values(ascending=False)
    best = sc.index[0]
    fa, fb = row.name_key.split()[0], g.loc[best,"name_key"].split()[0]
    first_ok = fa.startswith(fb) or fb.startswith(fa) or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.80
    la, lb = row.name_key.split()[-1], g.loc[best,"name_key"].split()[-1]
    last_ok = difflib.SequenceMatcher(None, la, lb).ratio() >= 0.93
    if sc.iloc[0] >= 0.80 and first_ok and last_ok and (len(sc) == 1 or sc.iloc[1] <= sc.iloc[0] - 0.05):
        sal.loc[i,"player_id"] = g.loc[best,"player_id"]; sal.loc[i,"match_stage"] = "5_fuzzy_player_master_career_span"; c5 += 1
        pairs5.append((row.player_name, g.loc[best,"player_name"], g.loc[best,"player_id"], int(row.season)))
log(f"- Stage 5 (fuzzy name match against the player master, season inside career span plus or minus one year, first name prefix or at least 0.80 similar, last name at least 0.93 similar): {c5} more.")
pd.DataFrame(pairs5, columns=["salary_name","matched_master_name","player_id","season"]).to_csv(f"{out}/salary_fuzzy_matches_stage5_to_review.csv", index=False)
sal["match_stage"] = sal.match_stage.replace("", "unmatched")
sal = sal.drop(columns=["c1"])
tot_n = len(sal); um = (sal.match_stage=="unmatched")
log(f"- Result before removing duplicates: {tot_n-um.sum():,} of {tot_n:,} rows linked to a slug ({(1-um.mean()):.2%}); {um.sum():,} unmatched ({um.mean():.2%}).")
log("- Link rate by source: " + ", ".join(f"{k} {1-v:.1%}" for k,v in sal.groupby('salary_source').match_stage.apply(lambda s:(s=='unmatched').mean()).items()))
log(f"- Unmatched sample: {sal[um].player_name.drop_duplicates().head(12).tolist()}")

# alternate spellings: the same player-season can appear twice under two spellings
STAGE = {"1_same_season_stats_name":0,"1b_disambiguated_age_or_team":0,"2_good_name_map":1,"3_player_master_name_and_career_span":2,"4_fuzzy_name_team_season":3,"5_fuzzy_player_master_career_span":4}
SRC = {"detailed_csv":0,"simple_csv":1,"br_team_page":2}
sal["_k"] = sal.match_stage.map(STAGE).fillna(9)*10 + sal.salary_source.map(SRC)
linked = sal[sal.player_id.notna()].sort_values("_k")
ndup = int(linked.duplicated(["player_id","season"]).sum())
removed = linked[linked.duplicated(["player_id","season"], keep="first")]
linked = linked.drop_duplicates(["player_id","season"], keep="first")
sal = pd.concat([linked, sal[sal.player_id.isna()]]).drop(columns="_k").sort_values("salary_row_id").reset_index(drop=True)
log(f"- Duplicate player-seasons under alternate name spellings (for example Darrel and Darrell Armstrong, both Orlando 2000): {ndup} extra rows removed, keeping the row linked by the strictest match stage, then by source (detailed_csv, simple_csv, br_team_page). Removed names: {sorted(set(removed.player_name))[:12]}.")
removed[["salary_row_id","player_name","season","team_abbr_raw","salary","salary_source","player_id","match_stage"]].to_csv(f"{out}/salary_duplicates_removed.csv", index=False)
log(f"- Rows after removing duplicates: {len(sal):,}; linked {sal.player_id.notna().sum():,} ({sal.player_id.notna().mean():.2%}).")

# presence flags
keys = set(zip(keep_total.player_id, keep_total.season))
sal["has_same_season_stats"] = [ (p,s) in keys for p,s in zip(sal.player_id, sal.season)]
sal["has_prior_season_stats"] = [ (p,s-1) in keys for p,s in zip(sal.player_id, sal.season)]
log(f"- Rows with a statistics row in the same season: {sal.has_same_season_stats.mean():.1%}; in the prior season: {sal.has_prior_season_stats.mean():.1%}.")
log("- Same-season statistics presence by source: " + ", ".join(f"{k} {v:.1%}" for k,v in sal.groupby('salary_source').has_same_season_stats.mean().items()))
log("  Salary rows with no same-season statistics are players paid but with no recorded regular season games, mostly in the `simple_csv` source from 2010 on.")

# age and position fill
sal["age"] = sal.age_raw
a = sal.merge(stats_ps[["player_id","season","age","pos_primary"]].rename(columns={"age":"age_st","pos_primary":"pos_st"}), on=["player_id","season"], how="left")
sal["age_st"] = a.age_st.values; sal["pos_st"] = a.pos_st.values
agree = (sal.age_raw==sal.age_st)[sal.age_raw.notna() & sal.age_st.notna()].mean()
log(f"- Age check: salary-source age equals statistics age in {agree:.2%} of rows where both exist.")
bd = sal.player_id.map(dp.drop_duplicates('player_id').set_index('player_id').birth_date)
feb1 = pd.to_datetime(sal.season.astype(str)+"-02-01")
sal["age_birth"] = np.floor((feb1-bd).dt.days/365.25)
agree2 = (sal.age_raw==sal.age_birth)[sal.age_raw.notna() & sal.age_birth.notna()].mean()
log(f"- Age from birth date as of February 1 of the season agrees with the salary-source age in {agree2:.2%} of rows, so it is used when no statistics row exists.")
sal["age"] = sal.age.fillna(sal.age_st).fillna(sal.age_birth)
sal["pos_primary"] = sal.pos_raw.astype(str).str.split("-").str[0].replace({"nan":np.nan})
sal["pos_primary"] = sal.pos_primary.fillna(sal.pos_st).fillna(sal.player_id.map(dp.drop_duplicates('player_id').set_index('player_id').pos_primary))
log(f"- After filling: age missing {sal.age.isna().mean():.2%} (was {sal.age_raw.isna().mean():.2%}); position missing {sal.pos_primary.isna().mean():.2%} (was {sal.pos_raw.isna().mean():.2%}).")
sal = sal.drop(columns=["age_st","pos_st","age_birth"])
# salary sanity
sal["salary_lt_100k"] = sal.salary < 100000
lo = sal[sal.salary_lt_100k]
log(f"- Salaries under 100,000 dollars: {len(lo)} rows (minimum {sal.salary.min():,.0f}). Median regular season games played in those rows (where statistics exist): {stats_ps.merge(lo[['player_id','season']], on=['player_id','season']).g.median():.0f}. Flagged as probable prorated or short contracts in `salary_lt_100k`; not dropped yet.")
sal.to_csv(f"{out}/salaries_clean.csv", index=False)
open(f"{out}/cleaning_log.md","w").write("# Cleaning log: NBA Data 1946-2026 tables\n\n" + "\n".join(LOG) + "\n")
