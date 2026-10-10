# Running and Reviewing Injury Dataset Cleaning

Run from the repository root with Python and Pandas installed:

```bash
python code/clean_injuries.py
```

The existing requirements.txt includes Pandas. No additional packages are required. The injury script runs independently; run_pipeline.sh is unchanged.

To select another input or output location:

```bash
python code/clean_injuries.py --input 'data/raw/injuries/NBA Player Injury Stats(1951 - 2023).csv' --output-dir data/processed
```

The approved exception corrections are checked against the exact original record contents. A revised source that changes those records causes the script to stop for review rather than apply a correction to an unrelated row.

## Files

| File | Purpose |
| --- | --- |
| data/raw/injuries/NBA Player Injury Stats(1951 - 2023).csv | Unmodified source CSV |
| data/raw/injuries/source_manifest.json | Source URL, retrieval date, checksum and initial profile |
| code/clean_injuries.py | Reproducible standalone cleaning script |
| data/processed/injuries_clean.csv | Cleaned transaction table |
| data/processed/injury_duplicates_removed.csv | Two removed duplicate copies |
| data/processed/injury_exceptions.csv | Five original exception records, actions and reasons |
| data/processed/injury_rows_to_review.csv | Three retained records with conflicting field and note direction |
| data/processed/injury_cleaning_summary.json | Machine-readable counts and schema |
| data/processed/injury_cleaning_log.md | Human-readable decisions and results |
| reports/injury_cleaning_contribution.md | Report-ready injury contribution under the six report headings |

## Cleaned Schema

| Columns | Meaning |
| --- | --- |
| date, team, acquired, relinquished, notes | Five cleaned source data fields |
| player_name | Name from the populated player field |
| event_direction | acquired, relinquished or unknown; transaction direction, not clinical category |
| player_name_key | Comparison aid; preserves aliases, suffixes and disambiguators; not a unique player ID |
| source_row_id | Retained record's original export index |
| source_row_ids | All original export IDs represented by the record, separated by semicolons |
| invalid_date, both_player_fields, missing_player, suspect_player_name, missing_team, missing_notes, conflicting_direction | Specific quality checks; direction conflicts compare explicit placed-on-IL/activated-from-IL wording with the populated field |
| needs_review | Whether any of the specific checks is triggered |

Read source_row_ids as text. Boolean audit flags are exported as True/False. Blank player fields represent the unpopulated side of a transaction. Retain the raw file and audit outputs when sharing this contribution.

## Scope and Status

All years are retained. Four unusable identities are excluded, two duplicate copies are removed, and one activation field is corrected. Three conflicting-direction records (source IDs 12214, 12699 and 12790) are preserved unchanged with conflicting_direction and needs_review set to True. The cleaned table has 37,661 rows and 18 columns. Raw files are unchanged. No categorization, episode pairing, temporal aggregation, integration or modeling is performed. The cleaned table is not proof of comprehensive historical injury coverage.

The report contribution is a local draft for incorporation into the team's shared report. The team must reconcile the full report's dataset description and supply project title, project number and actual effort percentages. No report submission is performed by this script.
