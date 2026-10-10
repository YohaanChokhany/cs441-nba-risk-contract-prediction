"""Clean the injury transaction log without filtering years or integrating datasets.

Run from the repository root: python code/clean_injuries.py
Raw data stays unchanged. Approved exceptions are archived; other anomalies are flagged.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def tidy(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()
    return text if text else pd.NA


def name_key(value):
    """A comparison aid, not a unique player ID; keep aliases and suffixes."""
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[.'`\-,]", " ", text)).strip()


def clean(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path, dtype="string")
    required = {"Date", "Team", "Acquired", "Relinquished", "Notes"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing required fields: {sorted(required - set(raw.columns))}")
    extras = set(raw.columns) - required - {"Unnamed: 0"}
    if extras:
        raise ValueError(f"Unexpected columns require review: {sorted(extras)}")

    df = raw[["Date", "Team", "Acquired", "Relinquished", "Notes"]].copy()
    df.columns = [c.lower() for c in df.columns]
    # Retain the original export index as provenance, never as a feature.
    df.insert(0, "source_row_id", raw["Unnamed: 0"] if "Unnamed: 0" in raw else pd.Series(range(len(raw))).astype("string"))
    if df.source_row_id.isna().any() or not df.source_row_id.is_unique:
        raise ValueError("Source row IDs must be populated and unique")
    changed = {}
    for col in ["date", "team", "acquired", "relinquished", "notes"]:
        before = df[col].fillna("")
        df[col] = df[col].map(tidy).astype("string")
        changed[col] = int((before != df[col].fillna("")).sum())
    team_fixes = int(df.team.eq("Cavailers").sum())
    df["team"] = df.team.replace({"Cavailers": "Cavaliers"})
    parsed = pd.to_datetime(df.date, format="%Y-%m-%d", errors="coerce")
    df["invalid_date"] = parsed.isna()
    df["date"] = parsed.dt.strftime("%Y-%m-%d").astype("string").fillna(df.date)

    fields = ["date", "team", "acquired", "relinquished", "notes"]
    duplicate_mask = df.duplicated(fields, keep="first")
    groups = df.groupby(fields, dropna=False, sort=False).source_row_id.transform(lambda s: ";".join(s))
    df["source_row_ids"] = groups
    duplicates = df.loc[duplicate_mask].copy()
    df = df.loc[~duplicate_mask].copy()
    # User-approved, source-specific decisions. Check content rather than relying
    # solely on an export row number, so a changed input cannot be miscorrected.
    exclusions = {
        "525": ("1987-02-01", "Kings", None, "Kings", "placed on IL with right foot injury"),
        "28651": ("2018-12-27", "76ers", None, "76ers", "placed on IL"),
        "29351": ("2019-03-16", "Nets", None, None, "activated from IL"),
        "30322": ("2019-12-31", "Thunder", "11/25/2019", None, "activated from IL"),
    }
    decisions = {**exclusions, "14793": ("2010-01-16", "Jazz", "Kyrylo Fesenko", "Kyrylo Fesenko", "activated from IL")}
    for source_id, expected in decisions.items():
        row = df.loc[df.source_row_id.eq(source_id), fields]
        if len(row) != 1:
            raise ValueError(f"Approved exception {source_id} missing or nonunique; review input")
        actual = tuple(None if pd.isna(v) else v for v in row.iloc[0])
        if actual != expected:
            raise ValueError(f"Approved exception {source_id} changed; review input")
    exceptions = df.loc[df.source_row_id.isin(decisions)].copy()
    exceptions["action"] = exceptions.source_row_id.map(lambda s: "correct_activation" if s == "14793" else "exclude_unusable_player")
    exceptions["reason"] = exceptions.source_row_id.map({
        "525": "Team name in place of player", "28651": "Team name in place of player",
        "29351": "Both player fields empty", "30322": "Date in place of player",
        "14793": "Same player in both fields; notes explicitly say activated from IL",
    })
    exceptions["approved_change"] = exceptions.source_row_id.map(lambda s: "Set relinquished to empty; retain acquired" if s == "14793" else "Remove from cleaned output; preserve original here")
    df = df.loc[~df.source_row_id.isin(exclusions)].copy()
    df.loc[df.source_row_id.eq("14793"), "relinquished"] = pd.NA
    acquired, relinquished = df.acquired.notna(), df.relinquished.notna()
    df["both_player_fields"] = acquired & relinquished
    df["missing_player"] = ~acquired & ~relinquished
    df["player_name"] = df.acquired.combine_first(df.relinquished)
    conflicting = df.both_player_fields & df.acquired.ne(df.relinquished).fillna(False)
    df.loc[conflicting, "player_name"] = pd.NA
    df["event_direction"] = "unknown"
    df.loc[acquired & ~relinquished, "event_direction"] = "acquired"
    df.loc[relinquished & ~acquired, "event_direction"] = "relinquished"
    # Preserve the source fields when they conflict with explicit note wording.
    # A conflict does not establish which source field is factually correct.
    note_activation = df.notes.str.contains(r"\bactivated from IL\b", case=False, na=False)
    note_placement = df.notes.str.contains(r"\bplaced on IL\b", case=False, na=False)
    df["conflicting_direction"] = (
        (note_activation & df.event_direction.ne("acquired"))
        | (note_placement & df.event_direction.ne("relinquished"))
    )
    df["player_name_key"] = df.player_name.map(name_key)
    # Flag obvious non-name values; do not guess replacements or discard them.
    team_labels = set(df.team.dropna())
    def suspect_name(value):
        if pd.isna(value):
            return False
        # Birth-date annotations disambiguate real players and are retained.
        without_birth = re.sub(r"\(b\. \d{4}-\d{2}-\d{2}\)", "", str(value))
        return bool(re.search(r"\d", without_birth) or value in team_labels)

    df["suspect_player_name"] = df.player_name.map(suspect_name)
    df["missing_team"] = df.team.isna()
    df["missing_notes"] = df.notes.isna()
    flags = ["invalid_date", "both_player_fields", "missing_player", "suspect_player_name", "missing_team", "missing_notes", "conflicting_direction"]
    df["needs_review"] = df[flags].any(axis=1)
    df = df.sort_values(["date", "source_row_id"], kind="stable").reset_index(drop=True)
    df.to_csv(output_dir / "injuries_clean.csv", index=False, date_format="%Y-%m-%d")
    duplicates.to_csv(output_dir / "injury_duplicates_removed.csv", index=False)
    exceptions.to_csv(output_dir / "injury_exceptions.csv", index=False)
    df.loc[df.needs_review].to_csv(output_dir / "injury_rows_to_review.csv", index=False)
    dates = pd.to_datetime(df.date, errors="coerce")
    summary = {
        "input_filename": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_rows": len(raw), "output_rows": len(df),
        "duplicates_removed": len(duplicates), "whitespace_or_unicode_changes": changed,
        "team_typo_corrections": team_fixes,
        "unusable_player_rows_excluded": len(exclusions), "activation_field_corrections": 1,
        "date_min": str(dates.min().date()), "date_max": str(dates.max().date()),
        "years_filtered": 0, "flags": {c: int(df[c].sum()) for c in flags},
        "rows_needing_review": int(df.needs_review.sum()),
        "output_columns": list(df.columns),
    }
    assert len(raw) == len(df) + len(duplicates) + len(exclusions)
    assert not df.duplicated(fields).any()
    (output_dir / "injury_cleaning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Injury Dataset Cleaning Log", "", f"Input: {input_path.as_posix()}",
             f"Input SHA256: {summary['input_sha256']}", "",
             f"Rows: {len(raw):,} raw → {len(df):,} cleaned; {len(duplicates)} duplicate records removed.",
             "Excluded four user-approved unusable-player records; corrected one user-approved activation. Original values and reasons are in injury_exceptions.csv.",
             f"Coverage retained: {summary['date_min']} through {summary['date_max']}; all years retained.",
             f"Corrected Cavailers to Cavaliers in {team_fixes} row(s).",
             f"Whitespace/Unicode changes by field: {changed}.", "",
             "The export index is preserved as source_row_id; source_row_ids records all raw rows represented by each cleaned record.",
             "Duplicates are compared on the five data fields after whitespace and team corrections; removed records are saved separately.",
             "Structural empty Acquired/Relinquished fields are preserved. No missing values are imputed.",
             "Player name keys retain slash aliases, parenthetical disambiguators and suffixes. They are comparison aids, not unique identifiers.",
             "Event direction describes the populated field, not a verified clinical injury category. Any new both/neither-field records remain unknown and flagged.",
             "Explicit placed-on-IL/activated-from-IL note wording is checked against field direction. Conflicts remain unchanged and are flagged for review; this check does not interpret all possible note wording.",
             "No year selection, injury categories, episode pairing, durations, season aggregation or cross-dataset integration was performed.", "",
             "## Review Flags", ""]
    lines += [f"- {c}: {summary['flags'][c]}" for c in flags]
    lines += ["", "Flagged rows remain in injuries_clean.csv and are also exported to injury_rows_to_review.csv. No ambiguous identity or transaction is silently corrected.", ""]
    (output_dir / "injury_cleaning_log.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/injuries/NBA Player Injury Stats(1951 - 2023).csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    clean(args.input, args.output_dir)
