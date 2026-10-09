import sys, pandas as pd
proc = sys.argv[1]
fd = pd.read_csv(f"{proc}/feature_dictionary_draft.csv")
STAT = {"g":"games played","mp":"minutes played","mpg":"minutes per game","gs_share":"share of games started","g_share":"games played as a share of the most games any player logged that season (availability proxy)",
 "pts":"total points","pts_36":"points per 36 minutes","trb_36":"rebounds per 36 minutes","ast_36":"assists per 36 minutes","stl_36":"steals per 36 minutes","blk_36":"blocks per 36 minutes",
 "tov_36":"turnovers per 36 minutes","x3pa_36":"three point attempts per 36 minutes","ts_percent":"true shooting percentage","usg_percent":"usage percentage","per":"player efficiency rating",
 "ws":"win shares","ws_48":"win shares per 48 minutes","obpm":"offensive box plus-minus","dbpm":"defensive box plus-minus","bpm":"box plus-minus","vorp":"value over replacement player",
 "o_rtg":"offensive rating","d_rtg":"defensive rating","ast_percent":"assist percentage","trb_percent":"rebound percentage","tov_percent":"turnover percentage","stl_percent":"steal percentage",
 "blk_percent":"block percentage","x3p_ar":"three point attempt rate","f_tr":"free throw attempt rate",
 "pts_36_adj":"points per 36 minutes minus the season average of qualified players","trb_36_adj":"rebounds per 36 minutes minus the season average","ast_36_adj":"assists per 36 minutes minus the season average",
 "x3pa_36_adj":"three point attempts per 36 minutes minus the season average","ts_percent_adj":"true shooting percentage minus the season average"}
OTHER = {"age":"age on February 1 of the salary season","age_sq":"age squared (for linear models)","exp_prior":"seasons since the player's first season, before the salary season","years_since_draft":"full seasons since the draft (seasons since debut if undrafted)",
 "draft_pick":"overall draft pick (61 if undrafted or not listed)","undrafted_or_unlisted":"1 if the player has no draft record","height_cm":"height in centimeters","weight_kg":"weight in kilograms",
 "career_g":"career games through the prior season (1996 onward)","career_mp":"career minutes through the prior season (1996 onward)","career_ws":"career win shares through the prior season (1996 onward)",
 "career_vorp":"career value over replacement player through the prior season (1996 onward)","peak_bpm":"best single-season box plus-minus so far, seasons with at least 200 minutes",
 "all_star_l1":"1 if an All-Star in the prior season","all_nba_l1":"1 if All-NBA in the prior season","all_def_l1":"1 if All-Defense in the prior season","all_star_count_prior":"All-Star selections through the prior season",
 "all_nba_count_prior":"All-NBA selections through the prior season","all_def_count_prior":"All-Defense selections through the prior season","cap_usd":"league salary cap for the salary season (context for eras)","has_l2":"1 if the player also has a statistics row two seasons back"}
def desc(f):
    if f in OTHER: return OTHER[f]
    if f.startswith("pos_"): return f"1 if primary position is {f[4:]}"
    for suf, lab in [("_l1","prior season"),("_avg2","average of the two prior seasons"),("_trend","prior season minus two seasons back (0 if no earlier row)")]:
        if f.endswith(suf) and f[:-len(suf)] in STAT: return f"{STAT[f[:-len(suf)]]}, {lab}"
    return ""
fd["description"] = fd.feature.map(desc)
fd["source_tables"] = fd.group.map({"player profile":"dim_players, dim_draft_history, salaries","position":"salaries, season statistics, dim_players","prior-season performance":"fact_player_season_stats",
  "era-adjusted prior-season rates":"fact_player_season_stats","two-season history and trend":"fact_player_season_stats","career to date":"fact_player_season_stats",
  "honors":"all_star_selections, end_of_season_teams","context":"external cap history (RealGM)"})
fd = fd.rename(columns={"dropped_for":"dropped_because_correlated_with","r_with":"absolute_correlation"})
fd.to_csv(f"{proc}/feature_dictionary.csv", index=False)
print(fd.groupby("group").agg(candidates=("feature","size"), kept=("kept_after_pruning","sum")).to_string()); print("missing descriptions:", fd[fd.description==""].feature.tolist())
print("kept:", fd[fd.kept_after_pruning].feature.tolist())
