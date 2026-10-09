# ------------------------------------------------------------
# Clean the Current NBA Players Contracts History dataset
# Base R only, no packages needed.
# ------------------------------------------------------------

in_file  <- "data/raw/contracts/nba_contracts_history.csv"   # run from the repo root
out_file <- "data/processed/contracts_clean.csv"
options(scipen = 999)                        # no 1.2e+07 in the saved file

# NOTE: the Kaggle download is named .xls but is really a CSV (saved here as .csv),
# so read.csv() is the right reader (not readxl).
raw <- read.csv(in_file, stringsAsFactors = FALSE, check.names = FALSE)
cat("Raw rows:", nrow(raw), " columns:", ncol(raw), "\n\n")
df <- raw

# 1. Tidy column names -----------------------------------------
names(df) <- tolower(names(df))
names(df) <- gsub("%", "_pct", names(df), fixed = TRUE)
names(df) <- gsub("+/-", "plus_minus", names(df), fixed = TRUE)
cat("Columns:", paste(names(df), collapse = ", "), "\n\n")

# 2. Trim stray whitespace in player names ---------------------
before <- df$name
df$name <- trimws(gsub("\\s+", " ", df$name))
cat("Names changed by trimming:", sum(before != df$name), "\n")

# 3. Missing values --------------------------------------------
cat("Missing values in whole table:", sum(is.na(df)), "\n")
cat("Empty-string names:", sum(df$name == ""), "\n\n")

# 4. Remove exact duplicate rows -------------------------------
dup_rows <- duplicated(df)
cat("Exact duplicate rows:", sum(dup_rows), "\n")
print(df[dup_rows, c("name", "contract_start", "contract_end", "avg_salary")])
df <- df[!dup_rows, ]
cat("Rows after dedupe:", nrow(df), "\n\n")

# 5. Same player + same start year: keep one row --------------
# Spencer Dinwiddie has two 2016 rows (a Bulls deal that was waived in
# preseason and the Nets deal he played under). Rule: keep the row with
# the latest contract_end, i.e. the longer deal (here the Nets one).
df <- df[order(df$name, df$contract_start, -df$contract_end), ]
same_key <- duplicated(df[, c("name", "contract_start")])
cat("Rows sharing player + start year (dropped):", sum(same_key), "\n")
print(df[same_key, c("name", "contract_start", "contract_end", "avg_salary")])
df <- df[!same_key, ]
cat("Rows after dropping:", nrow(df), "\n\n")

# 6. Consistency checks (counts of rows that FAIL) -------------
checks <- c(
  end_before_start   = sum(df$contract_end < df$contract_start),
  gp_over_82         = sum(df$gp > 82),
  gp_not_w_plus_l    = sum(df$gp != df$w + df$l),
  reb_not_oreb_dreb  = sum(df$reb != df$oreb + df$dreb),
  fgm_over_fga       = sum(df$fgm > df$fga),
  tpm_over_tpa       = sum(df$`3pm` > df$`3pa`),
  ftm_over_fta       = sum(df$ftm > df$fta),
  pts_not_2fg_3p_ft  = sum(df$pts != 2 * df$fgm + df$`3pm` + df$ftm),
  fg_pct_mismatch    = sum(abs(round(df$fgm / df$fga * 100, 1) - df$fg_pct) > 0.15),
  nonpositive_salary = sum(df$avg_salary <= 0),
  age_out_of_range   = sum(df$age < 18 | df$age > 45)
)
print(checks)
cat("\n")

# 7. Flag small samples (keep them, but mark them) -------------
df$low_sample <- as.integer(df$gp < 20 | df$min < 500)
cat("Low-sample rows (GP < 20 or MIN < 500):", sum(df$low_sample), "\n")
cat("Rows with 0 three-point attempts:", sum(df$`3pa` == 0), "\n")
cat("Rows with < 20 three-point attempts:", sum(df$`3pa` < 20), "\n")
cat("Rows with < 20 free-throw attempts:", sum(df$fta < 20), "\n\n")

# 8. Add contract length and a clean join key ------------------
# contract_end is the START year of the final season (Al Horford: 2011-2015
# = seasons 2011-12 through 2015-16 = 5 years), so add 1.
df$contract_years <- df$contract_end - df$contract_start + 1

strip_accents <- function(x) iconv(x, to = "ASCII//TRANSLIT")
df$name_key <- tolower(strip_accents(df$name))
df$name_key <- gsub("[^a-z ]", "", df$name_key)   # drop . ' - digits etc.
df$name_key <- trimws(gsub("\\s+", " ", df$name_key))

# 9. Save -------------------------------------------------------
df <- df[order(df$contract_start, df$name), ]
write.csv(df, out_file, row.names = FALSE)

cat("Clean rows:", nrow(df), " unique players:", length(unique(df$name_key)),
    " columns:", ncol(df), "\nSaved to", out_file, "\n")
