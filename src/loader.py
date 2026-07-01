"""
src/loader.py
-------------
Responsible for loading, merging, and cleaning all WFE raw Excel files
into a single analysis-ready DataFrame.

Key decisions made here:
- Only the 'Data' sheet is read from each file 
- Blank/separator rows (where Year is null) are dropped
- All columns are cast to their correct types
- A proper business_date column (YYYY-MM-01) is constructed from Year + Month
- Column names are normalised to snake_case for consistent downstream use
- No business logic here — this is pure ingestion and typing
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Resolve paths relative to the project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "wfe_combined.parquet"

# Month abbreviation → zero-padded month number
MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# Numeric columns that need float casting
NUMERIC_COLS = [
    "value", "nominal", "ytd", "pct_change_ytd",
    "pct_change_mtm", "pct_change_yty"
    ]

# Rename map: original Excel column → snake_case
COLUMN_RENAME = {
    "Year":             "year",
    "Month":            "month",
    "Region":           "region",
    "Indicator Name":   "indicator_name",
    "ExchangeName":     "exchange_name",
    "CurrencyName":     "currency_name",
    "Value":            "value",
    "Nominal":          "nominal",
    "DataType":         "data_type",
    "YTD":              "ytd",
    "% Change (YTD)":   "pct_change_ytd",
    "% Change (MTM)":   "pct_change_mtm",
    "% Change (YTY)":   "pct_change_yty",
    "AggregationType":  "aggregation_type",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_business_date(df: pd.DataFrame) -> pd.Series:
    """
    Construct a proper datetime from the 'year' (int) and 'month' (str abbrev)
    columns. Returns a Series of dtype datetime64[ns], day always set to 1.
    Rows where year or month cannot be parsed get NaT.
    """
    month_num = df["month"].str.strip().str.lower().map(MONTH_MAP)
    year_str = df["year"].astype(str)
    date_str = year_str + "-" + month_num + "-01"
    return pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")


def _load_single_file(filepath: Path) -> pd.DataFrame:
    """
    Load the 'Data' sheet from a single WFE Excel file.
    - Drops blank separator rows (where 'Year' is NaN after stripping)
    - Renames columns to snake_case
    - Returns a raw but row-filtered DataFrame
    """
    logger.info(f"Loading: {filepath.name}")

    df = pd.read_excel(filepath, sheet_name="Data", dtype=str)

    # Drop rows where 'Year' is blank — these are Excel separator rows
    df = df[df["Year"].notna() & (df["Year"].str.strip() != "")]

    # Rename to snake_case
    df = df.rename(columns=COLUMN_RENAME)

    # Tag source file for traceability
    df["source_file"] = filepath.name

    logger.info(f"  → {len(df):,} valid rows loaded")
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all columns to their correct types:
    - year → int
    - numeric columns → float (coerce errors → NaN)
    - string columns → stripped strings (NaN stays NaN)
    - business_date → datetime
    """
    # Year as integer
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"].str.strip(), errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype("int64")

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # String columns: strip whitespace
    str_cols = ["region", "indicator_name", "exchange_name", "currency_name",
                "data_type", "aggregation_type", "month"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Build proper date column
    df["business_date"] = _parse_business_date(df)

    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns that are useful for analysis but not in the raw data:
    - half_year: 'H1' or 'H2'
    - value_real: alias for value (placeholder for future FX adjustment)
    - indicator_category: top-level grouping extracted from indicator_name
    """
    # Half-year label
    month_num = df["business_date"].dt.month
    df["half_year"] = month_num.apply(lambda m: "H1" if m <= 6 else "H2")

    # value_real: same as value for now 
    # (no FX Foreign Exchange normalisation in scope)
    df["value_real"] = df["value"]

    # Top-level indicator category (text before the first ' - ')
    df["indicator_category"] = (
        df["indicator_name"]
        .str.split(" - ")
        .str[0]
        .str.strip()
    )

    return df


# ── Public API ───────────────────────────────────────────────────────────────

def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load and concatenate all WFE Excel files from raw_dir.
    Returns a cleaned, typed, analysis-ready DataFrame.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw WFE .xlsx files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with ~700k rows and all types correctly cast.
    """
    xlsx_files = sorted([
        f for f in raw_dir.iterdir()
        if f.suffix.lower() in (".xlsx", ".xls")
        and not f.name.startswith("~")  # skip Excel temp files
    ])

    if not xlsx_files:
        raise FileNotFoundError(f"No Excel files found in {raw_dir}")

    logger.info(f"Found {len(xlsx_files)} files to load")

    frames = [_load_single_file(f) for f in xlsx_files]
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined shape before cleaning: {df.shape}")

    df = _cast_types(df)
    df = _add_derived_columns(df)

    # Drop rows where we couldn't parse a valid date 
    # (data quality issue — logged)
    bad_dates = df["business_date"].isna().sum()
    if bad_dates > 0:
        logger.warning(f"Dropping {bad_dates:,} rows with unparseable business_date")
        df = df[df["business_date"].notna()]

    # Sort chronologically
    df = df.sort_values("business_date").reset_index(drop=True)

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Date range: {df['business_date'].min()} → {df['business_date'].max()}")
    logger.info(f"Regions: {sorted(df['region'].dropna().unique().tolist())}")

    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_FILE) -> None:
    """
    Persist the cleaned DataFrame as Parquet for fast downstream loading.
    Parquet preserves dtypes perfectly, including datetime and Int64.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data to {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_processed(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    """
    Load the pre-processed Parquet file. Much faster than re-reading Excel.
    Call this in notebooks after the first run.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found at {path}. "
            "Run load_raw() and save_processed() first."
        )
    df = pd.read_parquet(path)
    logger.info(f"Loaded processed data: {df.shape} from {path}")
    return df
