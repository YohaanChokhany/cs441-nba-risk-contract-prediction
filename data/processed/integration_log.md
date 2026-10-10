# Integration log

- Base table: 8,620 player-season rows, 69 columns (1,582 players, seasons 2000-2025).
- Player universe: 5,416 players with a known career span (61 without one were left out); 5,386 distinct name keys.

## Injuries
- Cleaned injury transactions: 37,661; 3 with conflicting direction left out (kept in injuries_clean.csv for review).
- Events in seasons 2006 or later (earlier years are sparse): 30,308.
- Event linking to player_id: unique 30,135, no_match 138, tiebreak_salary 35 -> 99.5% of events linked.
- Unlinked names saved to injury_unlinked_names.csv (57 names). Many are players who never reached a salary row in the base table, which is harmless.
- Episodes (placement to activation): 14,245; 478 placements never activated in the log (end set to the log end or the 365-day cap); median length of paired episodes: 6 days.
- IL placements per season window (season-ending year): 2006: 908, 2007: 1021, 2008: 835, 2009: 816, 2010: 786, 2011: 943, 2012: 579, 2013: 634, 2014: 600, 2015: 765, 2016: 800, 2017: 971, 2018: 1228, 2019: 875, 2020: 846, 2021: 944, 2022: 1498, 2023: 903. The 2023 window stops at 2023-04-16.
- Player-season injury table: 6,659 rows, 1,668 players (saved to injury_player_season.csv, window season = season-ending year).
- Injury features present (non-missing) for 70.6% of rows (seasons whose prior season is 2006-2023); 4,132 rows (67.9% of covered rows) had at least one IL placement in the prior season.

## Contracts
- Contract rows: 197; linked to a player_id: 197 (unique 197).
- Base rows covered by a contract record: 585 of 8,620 (6.8%). The contracts file has only 197 contracts starting 2011-2019, so it can only describe a small slice.
- Check of the season alignment: season salary / contract average salary has median 1.00 and 92% of rows fall between 0.5 and 1.5; correlation of log values 0.835.

- Integrated table: 8,620 rows x 109 columns; no duplicate player-seasons; row count equals the base table.
