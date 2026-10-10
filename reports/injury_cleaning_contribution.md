# Injury Dataset Data Preparation Contribution

## Select Data

We used the NBA Player Injury Stats dataset from Kaggle, retaining its full December 25, 1951 through April 16, 2023 coverage. The raw file contains 37,667 transaction records, five substantive fields—Date, Team, Acquired, Relinquished and Notes—and an exported row index. These fields provide the date, team, player and description needed to study recorded roster availability transactions. We retained the full historical range so later analysis can choose an appropriate observation window without losing source information during cleaning.

The injury log is a transaction dataset rather than a complete clinical injury registry. It includes illness, rest and other reasons for inactivity, as well as activations. Earlier exploratory analysis identified sparse coverage before 2005; retaining those years does not imply that reporting is equally complete across time. Any later modeling subset should account for this limitation and the source's April 2023 endpoint.

## Clean Data

We cleaned the file using a reproducible Pandas script while preserving the original raw CSV. Column names were standardized to lowercase, whitespace was trimmed and collapsed, Unicode text was normalized, and dates were parsed and written in YYYY-MM-DD format. The Acquired and Relinquished fields contained surrounding whitespace in 17,623 and 20,044 populated values, respectively. Other substantive fields required no whitespace or Unicode changes. All dates parsed successfully. One occurrence of the team label “Cavailers” was corrected to “Cavaliers.”

The exported row index was retained as source_row_id for traceability. Duplicate detection compared the five substantive fields after cleaning, excluding that index. This identified two duplicate records: Aaron Gordon's October 24, 2017 activation and Jacob Evans's February 25, 2019 placement. We retained one copy of each transaction, saved the removed copies separately, and recorded both original row IDs in source_row_ids. The earlier Data Understanding report's statement that there were no exact duplicates therefore needs qualification: the index made each raw row unique, but two transaction records were otherwise identical.

Empty values in Acquired and Relinquished generally represent the transaction's direction rather than missing player information. We preserved these structural blanks instead of imputing names. Four unusable-player records were excluded, and one activation was corrected as follows.

| Source row ID | Date | Issue | Action |
| --- | --- | --- | --- |
| 525 | 1987-02-01 | Player field contains Kings | Excluded; player identity cannot be determined from the row |
| 28651 | 2018-12-27 | Player field contains 76ers | Excluded; player identity cannot be determined from the row |
| 29351 | 2019-03-16 | Both player fields are empty | Excluded; no player is identified |
| 30322 | 2019-12-31 | Player field contains 11/25/2019 | Excluded; a date appears in place of a name |
| 14793 | 2010-01-16 | Kyrylo Fesenko appears in both fields, with Notes stating activated from IL | Retained Acquired and cleared Relinquished |

Original exception values and the reasons for each action are saved separately. We made no attempt to guess the four missing identities. These exclusions remove fewer than 0.02% of raw records, but may slightly undercount activity for the affected teams and dates. The activation correction follows the explicit transaction description rather than an inferred injury history.

Player names retain suffixes, slash-separated aliases, parenthetical disambiguators and birth-date annotations. These details can distinguish different players and should not be discarded solely to simplify names. Notes remain free text; an unspecified injury cause is not treated as evidence that a player was healthy. No missing values were imputed.

We also checked explicit “placed on IL” and “activated from IL” wording against the populated player field. Three retained records populate Acquired while describing a placement: Kenny Thomas on March 9, 2008 (source row 12214), Shaun Livingston on November 5, 2008 (12699), and Larry Hughes on November 13, 2008 (12790). We preserved their original fields and Notes and set conflicting_direction and needs_review to True. Without independent corroboration, we cannot establish whether the field or description is incorrect. These records are also saved in the review file, and subsequent analysis should resolve or account for their direction ambiguity.

Validation confirmed that every raw row is accounted for in the cleaned records, duplicate provenance or exclusions; every original year remains represented; no duplicate substantive records remain; and exactly one player field is populated in every cleaned record. Apart from the three direction conflicts, no records trigger the implemented checks for invalid dates, missing players, obvious non-name player values, missing teams or missing notes. These checks do not establish that all names, notes or reported events are factually correct, and the direction check covers explicit wording rather than every possible expression.

## Construct Data

Three convenience fields were derived: player_name selects the populated player field; event_direction records acquired or relinquished; and player_name_key provides a lowercase, accent-normalized comparison string. The key retains alias and disambiguation information and is not a unique player identifier. Event direction describes the roster transaction and does not by itself establish a clinical injury type.

Source provenance and validation flags were also added. No synthetic transaction records, injury categories, episode durations, games-missed estimates or player-season features were generated during this cleaning contribution.

## Integrate Data

This contribution produces a standalone cleaned injury transaction table. It performs no cross-dataset joins or player identity matching. The original name information, dates and team labels remain available for the team's later integration work. Injury records have not been aggregated to the salary or contract dataset's grain.

## Format Data

The cleaned table is saved as a comma-delimited CSV with lowercase snake_case column names, ISO dates and records ordered by date and source row ID. Unpopulated player fields remain empty. Audit columns use Boolean flags. The source_row_ids field uses semicolons to list raw IDs represented by a retained transaction and should be read as text.

Names, notes, identifiers and quality flags are retained for interpretation and auditing. They should not automatically be treated as model predictors. No categorical encoding or numeric scaling is applied at this stage.

## Dataset Description

The cleaned injury dataset contains **37,661 records and 18 columns**, covering December 25, 1951 through April 16, 2023. The row count reconciles as 37,667 raw records minus two duplicate copies and four unusable-player records. One retained activation was corrected without changing the row count, and three direction conflicts were retained with review flags.

The 18 columns comprise five source data fields, three convenience fields, two provenance fields and eight quality flags. This is the size of the standalone cleaned injury table, not the final integrated modeling dataset or its predictor count. Historical reporting gaps, unspecified causes, unresolved direction conflicts, possible unrecognized name errors and the source cutoff remain limitations for subsequent analysis.

Source: [NBA Injury Stats 1951–2023](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023), retrieved October 9, 2026. Exact input checksum and schema are recorded in data/raw/injuries/source_manifest.json. Cleaning evidence is recorded in data/processed/injury_cleaning_summary.json and injury_cleaning_log.md.
