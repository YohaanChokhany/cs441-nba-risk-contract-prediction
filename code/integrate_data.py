"""Merge injuries and contracts onto the master player-season table.
"""
import sys, re, unicodedata, numpy as np, pandas as pd
from collections import defaultdict

interim, proc = sys.argv[1:3]
cover_start, cover_end = 2006, 2023      # 2005 has about half the events; log ends 2023-04-16
max_episode_days = 365                   # max days for one episode
log_lines = []
def log(s=""): print(s); log_lines.append(s)


fold = str.maketrans({"ı": "i", "İ": "I", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L"})


# turn a name into a match key
def norm(s):
    if pd.isna(s): return None
    s = str(s).translate(fold)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("*", "")
    s = re.sub(r"[.'`\-,]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(2): s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)
    return s


# inputs
m = pd.read_csv(f"{proc}/master_model_ready.csv")
pl = pd.read_csv(f"{interim}/players_clean.csv", low_memory=False)
sal = pd.read_csv(f"{interim}/salaries_clean.csv", usecols=["player_id", "player_name", "season"]).dropna(subset=["player_id", "season"])
inj = pd.read_csv(f"{proc}/injuries_clean.csv", low_memory=False,
                  usecols=["date", "player_name", "event_direction", "conflicting_direction", "notes"])
con = pd.read_csv(f"{proc}/contracts_clean.csv")
has_sal = set(zip(sal.player_id, sal.season.astype(int)))
log(f"- Base table: {len(m):,} player-season rows, {m.shape[1]} columns ({m.player_id.nunique():,} players, seasons {m.season.min()}-{m.season.max()}).")

# link names to player ids
pl["birth_date"] = pd.to_datetime(pl.birth_date, errors="coerce")
n_no_years = int(pl.year_from.isna().sum())
pl = pl.dropna(subset=["year_from", "year_to"])
lk = defaultdict(list)
for r in pl.itertuples():
    for col in {norm(r.player_name), norm(r.full_name)} - {None, ""}:
        lk[col].append(r)
# salary-source names count as aliases
byid = pl.set_index("player_id")
for r in sal.drop_duplicates(["player_id", "player_name"]).itertuples():
    k = norm(r.player_name)
    if k and r.player_id in byid.index and not any(x.player_id == r.player_id for x in lk[k]):
        q = byid.loc[r.player_id]
        lk[k].append(type("P", (), dict(player_id=r.player_id, year_from=q.year_from, year_to=q.year_to, birth_date=q.birth_date)))
log(f"- Player universe: {len(pl):,} players with a known career span ({n_no_years:,} without one were left out); {len(lk):,} distinct name keys.")


# name keys for an injury name
def variants(name):
    born = None
    mt = re.search(r"\(b\.\s*(\d{4}-\d{2}-\d{2})\)", name)
    if mt:
        born = pd.Timestamp(mt.group(1)); name = name.replace(mt.group(0), " ")
    out = []
    for part in name.split("/"):
        if not part.strip(): continue
        out.append(re.sub(r"\([^)]*\)", " ", part))
        out.append(re.sub(r"[()]", " ", part))
    keys = {norm(v) for v in out} - {None, ""}
    return keys, born


# match a name to one player id
def resolve(keys, born, season):
    cand = {}
    for k in keys:
        for r in lk.get(k, []):
            if r.year_from - 1 <= season <= r.year_to + 1: cand[r.player_id] = r
    if born is not None:
        by_birth = {p: r for p, r in cand.items() if r.birth_date == born}
        if by_birth: cand = by_birth
    if len(cand) == 1: return next(iter(cand)), "unique"
    if not cand: return None, "no_match"
    paid = [p for p in cand if (p, season) in has_sal]
    if len(paid) == 1: return paid[0], "tiebreak_salary"
    return None, "ambiguous"


# injuries data
log("\n## Injuries")
inj = inj.dropna(subset=["player_name"]).copy()
inj["date"] = pd.to_datetime(inj.date)
inj["season"] = inj.date.dt.year + (inj.date.dt.month >= 9).astype(int)   # sep-dec count as next season
n_raw = len(inj)
conf = inj.conflicting_direction.astype(str).str.lower().eq("true")
inj = inj[~conf]
log(f"- Cleaned injury transactions: {n_raw:,}; {int(conf.sum())} with conflicting direction left out (kept in injuries_clean.csv for review).")
inj = inj[inj.season >= cover_start].copy()
log(f"- Events in seasons {cover_start} or later (earlier years are sparse): {len(inj):,}.")
inj["kind"] = np.where(inj.event_direction == "relinquished", "placed", "activated")

uniq = inj[["player_name", "season"]].drop_duplicates()
res = [resolve(*variants(n), s) for n, s in zip(uniq.player_name, uniq.season)]
uniq["player_id"] = [r[0] for r in res]; uniq["link"] = [r[1] for r in res]
inj = inj.merge(uniq, on=["player_name", "season"], how="left")
log("- Event linking to player_id: " + ", ".join(f"{k} {v:,}" for k, v in inj.link.value_counts().items()) +
    f" -> {inj.player_id.notna().mean():.1%} of events linked.")
bad = inj[inj.player_id.isna()].groupby(["player_name", "link"]).agg(events=("date", "size"), first=("season", "min"), last=("season", "max")).reset_index().sort_values("events", ascending=False)
bad.to_csv(f"{proc}/injury_unlinked_names.csv", index=False)
log(f"- Unlinked names saved to injury_unlinked_names.csv ({len(bad):,} names). Many are players who never reached a salary row in the base table, which is harmless.")

ev = inj.dropna(subset=["player_id"]).copy()
ev["order"] = (ev.kind == "placed").astype(int)          # same-day activation goes first
ev = ev.sort_values(["player_id", "date", "order"])
data_end = ev.date.max()
eps = []
for pid, g in ev.groupby("player_id", sort=False):
    start = None
    for d, k in zip(g.date, g.kind):
        if k == "placed":
            if start is None: start = d              # repeat placement keeps the first start
        elif start is not None:
            eps.append((pid, start, d, True)); start = None
    if start is not None: eps.append((pid, start, pd.NaT, False))
eps = pd.DataFrame(eps, columns=["player_id", "start", "end", "paired"])
cap = eps.start + pd.Timedelta(days=max_episode_days)
eps["end"] = np.minimum(eps.end.fillna(data_end), cap)
n_unpaired = int((~eps.paired).sum())
log(f"- Episodes (placement to activation): {len(eps):,}; {n_unpaired:,} placements never activated in the log (end set to the log end or the {max_episode_days}-day cap); "
    f"median length of paired episodes: {(eps.end - eps.start)[eps.paired].dt.days.median():.0f} days.")

# days out per season window (sep 1 to aug 31), clipped to the window
days = defaultdict(float)
for r in eps.itertuples():
    for s in range(cover_start, cover_end + 1):
        w0, w1 = pd.Timestamp(s - 1, 9, 1), pd.Timestamp(s, 9, 1)
        ov = (min(r.end, w1) - max(r.start, w0)).days
        if ov > 0: days[(r.player_id, s)] += ov
cnt = ev[(ev.kind == "placed") & ev.season.between(cover_start, cover_end)].groupby(["player_id", "season"]).size()
dser = pd.Series(days, name="il_days", dtype=float)
dser.index = pd.MultiIndex.from_tuples(dser.index, names=["player_id", "season"])
ips = pd.concat([cnt.rename("il_events"), dser], axis=1).fillna(0).reset_index()
ips.to_csv(f"{proc}/injury_player_season.csv", index=False)
log("- IL placements per season window (season-ending year): " + ", ".join(f"{int(a)}: {int(b)}" for a, b in ips.groupby("season").il_events.sum().items()) + ". The 2023 window stops at 2023-04-16.")
log(f"- Player-season injury table: {len(ips):,} rows, {ips.player_id.nunique():,} players (saved to injury_player_season.csv, window season = season-ending year).")

# cumulative events by season
seasons = list(range(cover_start, cover_end + 1))
ew = ips.pivot_table(index="player_id", columns="season", values="il_events", aggfunc="sum").reindex(columns=seasons).fillna(0)
cum = ew.cumsum(axis=1).stack().rename("il_events_cum").reset_index()


# shift seasons forward k years
def prior(tbl, cols, k):
    t = tbl[["player_id", "season"] + cols].copy(); t["season"] = t.season + k
    return t.rename(columns={c: f"{c}_L{k}" for c in cols})


m = m.copy()
m["injury_coverage_l1"] = (m.season - 1).between(cover_start, cover_end).astype(int)
m["injury_coverage_l2"] = (m.season - 2).between(cover_start, cover_end).astype(int)
m = (m.merge(prior(ips, ["il_events", "il_days"], 1), on=["player_id", "season"], how="left")
       .merge(prior(ips, ["il_events", "il_days"], 2), on=["player_id", "season"], how="left")
       .merge(prior(cum.rename(columns={"il_events_cum": "il_events_cum"}), ["il_events_cum"], 1), on=["player_id", "season"], how="left"))
c1, c2 = m.injury_coverage_l1 == 1, (m.injury_coverage_l1 == 1) & (m.injury_coverage_l2 == 1)
# no events = 0 in covered years, nan outside
m["il_events_l1"] = m.il_events_L1.fillna(0).where(c1)
m["il_days_l1"] = m.il_days_L1.fillna(0).where(c1)
m["il_any_l1"] = (m.il_events_l1 > 0).astype(float).where(c1)
m["il_events_2yr"] = (m.il_events_L1.fillna(0) + m.il_events_L2.fillna(0)).where(c2)
m["il_days_2yr"] = (m.il_days_L1.fillna(0) + m.il_days_L2.fillna(0)).where(c2)
m["il_events_cum_since2006"] = m.il_events_cum_L1.fillna(0).where(c1)
m = m.drop(columns=[c for c in m.columns if re.search(r"_L[12]$", c)])
inj_cols = ["injury_coverage_l1", "injury_coverage_l2", "il_events_l1", "il_days_l1", "il_any_l1",
            "il_events_2yr", "il_days_2yr", "il_events_cum_since2006"]
log(f"- Injury features present (non-missing) for {m.il_events_l1.notna().mean():.1%} of rows (seasons whose prior season is {cover_start}-{cover_end}); "
    f"{(m.il_any_l1 == 1).sum():,} rows ({(m.il_any_l1 == 1).sum() / m.il_any_l1.notna().sum():.1%} of covered rows) had at least one IL placement in the prior season.")

# contracts
log("\n## Contracts")
# contract_start is the start year, season is the end year, so +1
res = [resolve({norm(n)}, None, s + 1) for n, s in zip(con.name, con.contract_start)]
con["player_id"] = [r[0] for r in res]
log(f"- Contract rows: {len(con)}; linked to a player_id: {con.player_id.notna().sum()} ({'; '.join(f'{k} {v}' for k, v in pd.Series([r[1] for r in res]).value_counts().items())}).")
con = con.dropna(subset=["player_id"]).copy()
con["contract_id"] = con.player_id + "_" + con.contract_start.astype(str)
stat_cols = ["gp", "w", "l", "min", "pts", "fgm", "fga", "fg_pct", "3pm", "3pa", "3p_pct", "ftm", "fta", "ft_pct",
             "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "pf", "plus_minus"]
rows = []
for r in con.itertuples():
    for s in range(r.contract_start + 1, r.contract_end + 2):
        rows.append((r.player_id, s, r.contract_id, r.contract_start + 1, r.contract_end + 1, r.contract_years, r.avg_salary, r.age, r.low_sample))
cs = pd.DataFrame(rows, columns=["player_id", "season", "contract_id", "contract_first_season", "contract_last_season",
                                 "contract_years", "contract_avg_salary", "contract_age_at_start", "contract_low_sample"])
cs["contract_year_num"] = cs.season - cs.contract_first_season + 1
cs["contract_years_left"] = cs.contract_last_season - cs.season
cs["contract_is_final_year"] = (cs.contract_years_left == 0).astype(int)
cs = cs.sort_values(["player_id", "season", "contract_first_season"]).drop_duplicates(["player_id", "season"], keep="last")  # keep the latest overlapping deal
stats = con[["contract_id"] + stat_cols].rename(columns={c: f"contract_period_{c}" for c in stat_cols})
cs = cs.merge(stats, on="contract_id", how="left")
m = m.merge(cs.drop(columns=["contract_first_season", "contract_last_season"]), on=["player_id", "season"], how="left")
m["has_contract_record"] = m.contract_id.notna().astype(int)
log(f"- Base rows covered by a contract record: {int(m.has_contract_record.sum()):,} of {len(m):,} ({m.has_contract_record.mean():.1%}). "
    f"The contracts file has only {len(con)} contracts starting {int(con.contract_start.min())}-{int(con.contract_start.max())}, so it can only describe a small slice.")
hit = m[m.has_contract_record == 1]
if len(hit):
    ratio = (hit.salary / hit.contract_avg_salary)
    log(f"- Check of the season alignment: season salary / contract average salary has median {ratio.median():.2f} and "
        f"{((ratio > 0.5) & (ratio < 1.5)).mean():.0%} of rows fall between 0.5 and 1.5; correlation of log values {np.corrcoef(np.log(hit.salary), np.log(hit.contract_avg_salary))[0, 1]:.3f}.")
con_cols = [c for c in m.columns if c.startswith("contract_") or c == "has_contract_record"]

# checks and outputs
assert len(m) == len(pd.read_csv(f"{proc}/master_model_ready.csv")), "row count changed during integration"
assert not m.duplicated(["player_id", "season"]).any(), "duplicate player-seasons after integration"
assert m.loc[m.injury_coverage_l1 == 0, ["il_events_l1", "il_days_l1"]].isna().all().all(), "injury values leaked outside coverage"
log(f"\n- Integrated table: {len(m):,} rows x {m.shape[1]} columns; no duplicate player-seasons; row count equals the base table.")

m.to_csv(f"{proc}/master_integrated.csv", index=False)
groups = ([(c, "injury history (prior seasons)", "predictor; NaN where the injury log does not cover the prior season, see injury_coverage_l1") for c in inj_cols]
          + [(c, "contract terms", "describes the contract covering this season; not known before signing, check before using for a new-contract prediction")
             for c in con_cols if not c.startswith("contract_period_") and c not in ("contract_avg_salary", "contract_id", "contract_low_sample")]
          + [("contract_avg_salary", "contract target-side", "equals the salary target in effect; never use as a predictor of salary")]
          + [(c, "contract-period outcome stats", "stats earned during the contract itself; leakage if used to predict its salary") for c in con_cols if c.startswith("contract_period_")]
          + [("contract_id", "identifier", ""), ("contract_low_sample", "contract quality flag", "from contracts_clean.csv")])
pd.DataFrame(groups, columns=["column", "group", "note"]).to_csv(f"{proc}/integrated_feature_groups.csv", index=False)
open(f"{proc}/integration_log.md", "w").write("# Integration log\n\n" + "\n".join(log_lines) + "\n")
