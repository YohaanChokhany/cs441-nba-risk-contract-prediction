# Injury Dataset Cleaning Log

Input: data/raw/injuries/NBA Player Injury Stats(1951 - 2023).csv
Input SHA256: 967ebaf701738b33f1a99629dcb8aacc413ef1967d7ee6bbe9c6b85a7b023fe7

Rows: 37,667 raw → 37,661 cleaned; 2 duplicate records removed.
Excluded four user-approved unusable-player records; corrected one user-approved activation. Original values and reasons are in injury_exceptions.csv.
Coverage retained: 1951-12-25 through 2023-04-16; all years retained.
Corrected Cavailers to Cavaliers in 1 row(s).
Whitespace/Unicode changes by field: {'date': 0, 'team': 0, 'acquired': 17623, 'relinquished': 20044, 'notes': 0}.

The export index is preserved as source_row_id; source_row_ids records all raw rows represented by each cleaned record.
Duplicates are compared on the five data fields after whitespace and team corrections; removed records are saved separately.
Structural empty Acquired/Relinquished fields are preserved. No missing values are imputed.
Player name keys retain slash aliases, parenthetical disambiguators and suffixes. They are comparison aids, not unique identifiers.
Event direction describes the populated field, not a verified clinical injury category. Any new both/neither-field records remain unknown and flagged.
Explicit placed-on-IL/activated-from-IL note wording is checked against field direction. Conflicts remain unchanged and are flagged for review; this check does not interpret all possible note wording.
No year selection, injury categories, episode pairing, durations, season aggregation or cross-dataset integration was performed.

## Review Flags

- invalid_date: 0
- both_player_fields: 0
- missing_player: 0
- suspect_player_name: 0
- missing_team: 0
- missing_notes: 0
- conflicting_direction: 3

Flagged rows remain in injuries_clean.csv and are also exported to injury_rows_to_review.csv. No ambiguous identity or transaction is silently corrected.
