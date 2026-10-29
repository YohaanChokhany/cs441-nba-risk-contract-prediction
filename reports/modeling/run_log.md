# Modeling run log

- Sample common_2008_2024: 5,757 rows, 1,182 players, seasons 2008-2024, time-holdout rows 737
- Sample full_2000_2025: 8,620 rows, 1,582 players, seasons 2000-2025, time-holdout rows 1,101
- XGBoost tuning (16 settings, 3 player-grouped folds, seasons before the holdout): best {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 300, 'min_child_weight': 8} with RMSE(log) 0.6203; worst 0.6420 (37s)
- CV Ridge   common_2008_2024 perf               done (2s)
- CV XGBoost common_2008_2024 perf               done (15s)
- CV Ridge   common_2008_2024 perf+injury        done (2s)
- CV XGBoost common_2008_2024 perf+injury        done (17s)
- CV Ridge   common_2008_2024 perf+prior         done (2s)
- CV XGBoost common_2008_2024 perf+prior         done (14s)
- CV Ridge   common_2008_2024 perf+injury+prior  done (2s)
- CV XGBoost common_2008_2024 perf+injury+prior  done (16s)
- CV Ridge   full_2000_2025 perf               done (2s)
- CV XGBoost full_2000_2025 perf               done (15s)
- CV Ridge   full_2000_2025 perf+prior         done (3s)
- CV XGBoost full_2000_2025 perf+prior         done (15s)
- Injury features' XGBoost gain share combined: 4.0%; best-ranked injury feature: il_days_2yr (#24 of 62)
- Prior-salary features' XGBoost gain share combined: 36.9%
