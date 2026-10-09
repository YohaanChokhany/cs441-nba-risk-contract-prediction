# Master dataset build log

- End-of-season team types present: ['All-Defense', 'All-NBA', 'All-Rookie']
- Master table before filtering: 14,945 salary rows (14,549 linked to a player identifier, 396 not linked), 109 columns.

### Evidence for the record filters (window start, prior-season minutes at least 200)
Median salary as a share of the cap, by whether the player logged at least 10 games in the salary season:
                           size  median
played_same salary_source              
False       br_team_page    157  0.0459
            detailed_csv    177  0.0156
            simple_csv      420  0.0199
True        br_team_page   1851  0.0745
            detailed_csv   5234  0.0558
            simple_csv     1564  0.0639
Share of prior-season rows with true shooting percentage outside [0.30, 0.75] or box plus-minus beyond +/-10, by prior-season minutes bucket:
             rows  extreme_share
mp_l1                           
(0, 50]       406          0.468
(50, 100]     318          0.195
(100, 200]    526          0.053
(200, 400]    862          0.010
(400, 800]   1589          0.001
(800, 5000]  7891          0.003

### Record waterfall
- Salary rows after cleaning: 14,945
- Linked to a player identifier: 14,549
- Salary season 2000 or later: 13,303
- Has a prior-season statistics row: 10,578
- Prior-season minutes at least 200: 9,403
- Played at least 10 games in the salary season: 8,649
- Salary at least 100,000 dollars: 8,620
- Has cap and age: 8,620

### Missing values in candidate features after record filters
none
- Imputed with the median of the filtered table: 0 values across 0 features (largest: {}).

### Correlation pruning (drop a feature if |r| > 0.9 with a kept feature that is more correlated with the targets)
- Candidate features: 76; kept: 53; dropped: 23
  - dropped vorp_avg2 (r = 0.92 with ws_avg2)
  - dropped ws_l1 (r = 0.928 with ws_avg2)
  - dropped per_l1 (r = 0.936 with per_avg2)
  - dropped bpm_l1 (r = 0.917 with bpm_avg2)
  - dropped pts_36_l1 (r = 0.959 with pts_36_avg2)
  - dropped mp_l1 (r = 0.905 with mp_avg2)
  - dropped career_vorp (r = 0.945 with career_ws)
  - dropped pts_36_adj_l1 (r = 0.934 with pts_36_avg2)
  - dropped career_mp (r = 0.914 with career_ws)
  - dropped usg_percent_avg2 (r = 0.916 with pts_36_avg2)
  - dropped all_nba_count_prior (r = 0.919 with all_star_count_prior)
  - dropped exp_prior (r = 0.909 with career_g)
  - dropped ts_percent_adj_l1 (r = 0.921 with ts_percent_l1)
  - dropped ast_36_l1 (r = 0.984 with ast_percent_l1)
  - dropped ast_36_adj_l1 (r = 0.987 with ast_percent_l1)
  - dropped g_l1 (r = 0.937 with g_share_l1)
  - dropped age_sq (r = 0.996 with age)
  - dropped trb_percent_l1 (r = 0.995 with trb_36_l1)
  - dropped trb_36_adj_l1 (r = 0.998 with trb_36_l1)
  - dropped x3p_ar_l1 (r = 0.923 with x3pa_36_l1)
  - dropped blk_36_l1 (r = 0.991 with blk_percent_l1)
  - dropped x3pa_36_adj_l1 (r = 0.904 with x3pa_36_l1)
  - dropped stl_36_l1 (r = 0.992 with stl_percent_l1)
- Near-constant features among kept (std < 1e-6): []

### Final model-ready table: 8,620 rows, 53 features, 3 target columns, 10 identifier and split columns
- Seasons 2000 to 2025; players 1,582; rows with the Estimated Plus-Minus extension available: 7,689
- Targets: salary median 4,000,000 dollars, mean 6,925,372; share of cap median 0.0618; skewness salary 2.40, log salary -0.12
- Correlation of final features with log salary (top 12): mpg_l1 0.61, pts_l1 0.58, ws_avg2 0.57, mp_avg2 0.56, gs_share_l1 0.54, peak_bpm 0.54, bpm_avg2 0.53, per_avg2 0.53, pts_36_avg2 0.53, obpm_l1 0.52, vorp_l1 0.50, career_ws 0.47
- Duplicate player-seasons: 0; fold sizes: {0: 1808, 1: 1845, 2: 1650, 3: 1625, 4: 1692}; time holdout rows: 1101
