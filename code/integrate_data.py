"""Integrate the three project datasets (NBA tables, injuries, contracts) into one player-season table.

Base table: data/processed/master_model_ready.csv (one row per salary player-season, season = season-ending year).
Adds, keyed on (player_id, season):
  * injury history from data/processed/injuries_clean.csv (a transaction log with no player id, linked by name)
  * contract terms from data/processed/contracts_clean.csv (no player id, linked by name)

Prior-season rule (same as build_master.py): injury features for salary season t use only events up to the end
of season t-1 (Sep 1 of t-2 to Aug 31 of t-1). Contract columns describe the contract covering season t and are
kept separate and labelled in integrated_feature_groups.csv, because they are not all known before signing.

Usage (repository root): python code/integrate_data.py data/interim data/processed
"""
import sys, re, unicodedata, numpy as np, pandas as pd
from collections import defaultdict

interim, proc = sys.argv[1:3]
COVER_START, COVER_END = 2006, 2023      # the 2005 window has about half the events of 2006; the log ends 2023-04-16
MAX_EPISODE_DAYS = 365                   # cap for a placement with no (or a very late) activation
LOG = []
def log(s=""): print(s); LOG.append(s)


FOLD = str.maketrans({"ı": "i", "İ": "I", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L"})


def norm(s):
    """Same name normaliser as clean_nba_tables.py so keys line up across the pipeline."""
    if pd.isna(s): return None
    s = str(s).translate(FOLD)    # letters that NFKD does not decompose and ascii-encoding would drop
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("*", "")
    s = re.sub(r"[.'`\-,]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(2): s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)
    return s


# ---------------------------------------------------------------- inputs
m = pd.read_csv(f"{proc}/master_model_ready.csv")
pl = pd.read_csv(f"{interim}/players_clean.csv", low_memory=False)
sal = pd.read_csv(f"{interim}/salaries_clean.csv", usecols=["player_id", "player_name", "season"]).dropna(subset=["player_id", "season"])
inj = pd.read_csv(f"{proc}/injuries_clean.csv", low_memory=False,
                  usecols=["date", "player_name", "event_direction", "conflicting_direction", "notes"])
con = pd.read_csv(f"{proc}/contracts_clean.csv")
has_sal = set(zip(sal.player_id, sal.season.astype(int)))
log(f"- Base table: {len(m):,} player-season rows, {m.shape[1]} columns ({m.player_id.nunique():,} players, seasons {m.season.min()}-{m.season.max()}).")

# ---------------------------------------------------------------- name -> player_id linking
pl["birth_date"] = pd.to_datetime(pl.birth_date, errors="coerce")
n_no_years = int(pl.year_from.isna().sum())
pl = pl.dropna(subset=["year_from", "year_to"])
lk = defaultdict(list)
for r in pl.itertuples():
    for col in {norm(r.player_name), norm(r.full_name)} - {None, ""}:
        lk[col].append(r)
# names a player appears under in the salary source also count as aliases (e.g. Enes Kanter / Enes Freedom)
byid = pl.set_index("player_id")
for r in sal.drop_duplicates(["player_id", "player_name"]).itertuples():
    k = norm(r.player_name)
    if k and r.player_id in byid.index and not any(x.player_id == r.player_id for x in lk[k]):
        q = byid.loc[r.player_id]
        lk[k].append(type("P", (), dict(player_id=r.player_id, year_from=q.year_from, year_to=q.year_to, birth_date=q.birth_date)))
log(f"- Player universe: {len(pl):,} players with a known career span ({n_no_years:,} without one were left out); {len(lk):,} distinct name keys.")


def variants(name):
    """Candidate name keys for one injury-log name: slash-separated aliases, parenthetical text kept or dropped,
    and an optional '(b. YYYY-MM-DD)' birth date used to disambiguate."""
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


def resolve(keys, born, season):
    """Return (player_id or None, reason). Candidates must share a name key and be active within a year of
    `season`; a birth date, then a salary row in that season, break ties. Ambiguity is left unlinked, not guessed."""
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


# ---------------------------------------------------------------- injuries: link, pair episodes, aggregate
log("\n## Injuries")
inj = inj.dropna(subset=["player_name"]).copy()
inj["date"] = pd.to_datetime(inj.date)
inj["season"] = inj.date.dt.year + (inj.date.dt.month >= 9).astype(int)   # season-ending year; Sep-Dec belong to next season
n_raw = len(inj)
conf = inj.conflicting_direction.astype(str).str.lower().eq("true")
inj = inj[~conf]
log(f"- Cleaned injury transactions: {n_raw:,}; {int(conf.sum())} with conflicting direction left out (kept in injuries_clean.csv for review).")
inj = inj[inj.season >= COVER_START].copy()
log(f"- Events in seasons {COVER_START} or later (earlier years are sparse): {len(inj):,}.")
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
ev["order"] = (ev.kind == "placed").astype(int)          # same-day activation closes the previous episode first
ev = ev.sort_values(["player_id", "date", "order"])
data_end = ev.date.max()
eps = []
for pid, g in ev.groupby("player_id", sort=False):
    start = None
    for d, k in zip(g.date, g.kind):
        if k == "placed":
            if start is None: start = d              # a repeat placement while still out is counted but does not restart the clock
        elif start is not None:
            eps.append((pid, start, d, True)); start = None
    if start is not None: eps.append((pid, start, pd.NaT, False))
eps = pd.DataFrame(eps, columns=["player_id", "start", "end", "paired"])
cap = eps.start + pd.Timedelta(days=MAX_EPISODE_DAYS)
eps["end"] = np.minimum(eps.end.fillna(data_end), cap)
n_unpaired = int((~eps.paired).sum())
log(f"- Episodes (placement to activation): {len(eps):,}; {n_unpaired:,} placements never activated in the log (end set to the log end or the {MAX_EPISODE_DAYS}-day cap); "
    f"median length of paired episodes: {(eps.end - eps.start)[eps.paired].dt.days.median():.0f} days.")

# days out per season window (Sep 1 of s-1 to Aug 31 of s); an episode is clipped to the window, so no information
# from after the window enters its season.
days = defaultdict(float)
for r in eps.itertuples():
    for s in range(COVER_START, COVER_END + 1):
        w0, w1 = pd.Timestamp(s - 1, 9, 1), pd.Timestamp(s, 9, 1)
        ov = (min(r.end, w1) - max(r.start, w0)).days
        if ov > 0: days[(r.player_id, s)] += ov
cnt = ev[(ev.kind == "placed") & ev.season.between(COVER_START, COVER_END)].groupby(["player_id", "season"]).size()
dser = pd.Series(days, name="il_days", dtype=float)
dser.index = pd.MultiIndex.from_tuples(dser.index, names=["player_id", "season"])
ips = pd.concat([cnt.rename("il_events"), dser], axis=1).fillna(0).reset_index()
ips.to_csv(f"{proc}/injury_player_season.csv", index=False)
log("- IL placements per season window (season-ending year): " + ", ".join(f"{int(a)}: {int(b)}" for a, b in ips.groupby("season").il_events.sum().items()) + ". The 2023 window stops at 2023-04-16.")
log(f"- Player-season injury table: {len(ips):,} rows, {ips.player_id.nunique():,} players (saved to injury_player_season.csv, window season = season-ending year).")

# wide -> cumulative events through each season
seasons = list(range(COVER_START, COVER_END + 1))
ew = ips.pivot_table(index="player_id", columns="season", values="il_events", aggfunc="sum").reindex(columns=seasons).fillna(0)
cum = ew.cumsum(axis=1).stack().rename("il_events_cum").reset_index()


def prior(tbl, cols, k):
    t = tbl[["player_id", "season"] + cols].copy(); t["season"] = t.season + k
    return t.rename(columns={c: f"{c}_L{k}" for c in cols})


m = m.copy()
m["injury_coverage_l1"] = (m.season - 1).between(COVER_START, COVER_END).astype(int)
m["injury_coverage_l2"] = (m.season - 2).between(COVER_START, COVER_END).astype(int)
m = (m.merge(prior(ips, ["il_events", "il_days"], 1), on=["player_id", "season"], how="left")
       .merge(prior(ips, ["il_events", "il_days"], 2), on=["player_id", "season"], how="left")
       .merge(prior(cum.rename(columns={"il_events_cum": "il_events_cum"}), ["il_events_cum"], 1), on=["player_id", "season"], how="left"))
c1, c2 = m.injury_coverage_l1 == 1, (m.injury_coverage_l1 == 1) & (m.injury_coverage_l2 == 1)
# no record inside the covered years means zero events; outside the covered years the value is unknown (NaN)
m["il_events_l1"] = m.il_events_L1.fillna(0).where(c1)
m["il_days_l1"] = m.il_days_L1.fillna(0).where(c1)
m["il_any_l1"] = (m.il_events_l1 > 0).astype(float).where(c1)
m["il_events_2yr"] = (m.il_events_L1.fillna(0) + m.il_events_L2.fillna(0)).where(c2)
m["il_days_2yr"] = (m.il_days_L1.fillna(0) + m.il_days_L2.fillna(0)).where(c2)
m["il_events_cum_since2005"] = m.il_events_cum_L1.fillna(0).where(c1)
m = m.drop(columns=[c for c in m.columns if re.search(r"_L[12]$", c)])
INJ_COLS = ["injury_coverage_l1", "injury_coverage_l2", "il_events_l1", "il_days_l1", "il_any_l1",
            "il_events_2yr", "il_days_2yr", "il_events_cum_since2005"]
log(f"- Injury features present (non-missing) for {m.il_events_l1.notna().mean():.1%} of rows (seasons whose prior season is {COVER_START}-{COVER_END}); "
    f"{(m.il_any_l1 == 1).sum():,} rows ({(m.il_any_l1 == 1).sum() / m.il_any_l1.notna().sum():.1%} of covered rows) had at least one IL placement in the prior season.")

# ---------------------------------------------------------------- contracts: link, expand to seasons, join
log("\n## Contracts")
# contract_start is the first season's starting year, master `season` is the season-ending year, hence the +1
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
cs = cs.sort_values(["player_id", "season", "contract_first_season"]).drop_duplicates(["player_id", "season"], keep="last")  # overlapping deals: keep the latest
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
CON_COLS = [c for c in m.columns if c.startswith("contract_") or c == "has_contract_record"]

# ---------------------------------------------------------------- checks and outputs
assert len(m) == len(pd.read_csv(f"{proc}/master_model_ready.csv")), "row count changed during integration"
assert not m.duplicated(["player_id", "season"]).any(), "duplicate player-seasons after integration"
assert m.loc[m.injury_coverage_l1 == 0, ["il_events_l1", "il_days_l1"]].isna().all().all(), "injury values leaked outside coverage"
log(f"\n- Integrated table: {len(m):,} rows x {m.shape[1]} columns; no duplicate player-seasons; row count equals the base table.")

m.to_csv(f"{proc}/master_integrated.csv", index=False)
groups = ([(c, "injury history (prior seasons)", "predictor; NaN where the injury log does not cover the prior season, see injury_coverage_l1") for c in INJ_COLS]
          + [(c, "contract terms", "describes the contract covering this season; not known before signing, check before using for a new-contract prediction")
             for c in CON_COLS if not c.startswith("contract_period_") and c not in ("contract_avg_salary", "contract_id", "contract_low_sample")]
          + [("contract_avg_salary", "contract target-side", "equals the salary target in effect; never use as a predictor of salary")]
          + [(c, "contract-period outcome stats", "stats earned during the contract itself; leakage if used to predict its salary") for c in CON_COLS if c.startswith("contract_period_")]
          + [("contract_id", "identifier", ""), ("contract_low_sample", "contract quality flag", "from contracts_clean.csv")])
pd.DataFrame(groups, columns=["column", "group", "note"]).to_csv(f"{proc}/integrated_feature_groups.csv", index=False)
open(f"{proc}/integration_log.md", "w").write("# Integration log\n\n" + "\n".join(LOG) + "\n")
