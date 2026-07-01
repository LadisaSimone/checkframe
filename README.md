# checkframe

A pluggable data quality framework combining statistical, ML, and validation checks with an LLM layer that explains findings in plain language.
---

## Overview

This project implements an **Intelligent Data Quality Control Framework** for regulatory reporting, built on World Federation of Exchanges (WFE) market data covering January 2021 to December 2023.

The framework automatically validates, checks, and flags data quality issues across 626,630 rows of global exchange market data — ensuring that only clean, compliant data reaches regulatory submissions.

### What it does
- **Loads and cleans** 6 semi-annual WFE Excel exports into a single analysis-ready dataset
- **Validates** exchange codes and currency compliance against ISO 4217
- **Runs basic checks** for nulls, negative values, zero values, and string lengths
- **Runs advanced checks** for statistical outliers (Z-score) and month-on-month spikes
- **Runs ML checks** using Isolation Forest anomaly detection
- **Generates plain-language reports** for regulatory stakeholders using the Claude LLM API

---

## Project Structure

```
checkframe/
│
├── data/
│   ├── raw/                    # WFE Excel files (not tracked in Git)
│   └── processed/              # Parquet cache (auto-generated, not tracked)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis + market growth + VIX
│   ├── 02_checks.ipynb         # All data quality checks + results
│   └── 03_ml.ipynb             # Isolation Forest anomaly detection
│
├── src/
│   ├── __init__.py
│   ├── loader.py               # Data ingestion, cleaning, type casting
│   ├── checks.py               # CheckRegistry + all check functions
│   └── reporter.py             # LLM explainability via Claude API
│
├── reports/
│   └── dq_report.md            # Auto-generated regulatory report
│
├── environment.yml             # Conda environment specification
├── .env.example                # API key template
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/LadisaSimone/checkframe.git
cd checkframe
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate checkframe
python -m ipykernel install --user --name checkframe --display-name "checkframe"
```

### 3. Add the data files
Download the WFE Excel files from [statistics.world-exchanges.org](https://statistics.world-exchanges.org) and place them in `data/raw/`. The expected files are:
```
data/raw/
├── all_markets_202101_202106_*.xlsx
├── all_markets_202107_202112_*.xlsx
├── all_markets_202201_202206_*.xlsx
├── all_markets_202207_202212_*.xlsx
├── all_markets_202301_202306_*.xlsx
└── all_markets_202307_202312_*.xlsx
```

Also download the VIX historical data from [CBOE](https://www.cboe.com/tradable_products/vix/vix_historical_data/) and place it at:
```
data/raw/vix_historical.csv
```

### 4. Configure the API key
Copy `.env.example` to `.env` and add your Anthropic API key:
```bash
cp .env.example .env
```
Edit `.env`:
```
ANTHROPIC_API_KEY=your-key-here
```

---

## Running the Notebooks

Run the notebooks **in order** — each one builds on the previous:

| Notebook | Description | Est. runtime |
|----------|-------------|--------------|
| `01_eda.ipynb` | Data loading, cleaning, EDA, market growth analysis, VIX overlay | ~2 min |
| `02_checks.ipynb` | All data quality checks, results summary, deep dives | ~1 min |
| `03_ml.ipynb` | Feature engineering, Isolation Forest, overlap analysis, LLM report | ~3 min |

> **First run:** `01_eda.ipynb` loads all Excel files and saves a Parquet cache to `data/processed/`. Subsequent runs use `load_processed()` for instant loading.

---

## Framework Architecture

### CheckRegistry
The core of the framework. Every check is a pure function with the signature:
```python
def my_check(df: pd.DataFrame) -> CheckResult:
    ...
```

Checks are registered by category and run uniformly:
```python
registry = CheckRegistry()
registry.register('basic', check_null_value)
registry.register('advanced', check_outliers_zscore)
registry.register('ml', check_isolation_forest)

results = registry.run_all(df)
summary = registry.summary(results)
```

The architecture supports **50 basic, 20 advanced, and 10 ML checks** — adding a new check requires only implementing the function and registering it with one line.

### Check inventory

| ID | Category | Description | Result |
|----|----------|-------------|--------|
| VAL_001 | Validation | Exchange code format (MIC proxy) | ✅ PASS |
| VAL_002 | Validation | Currency ISO 4217 compliance | ❌ 0.61% |
| BAS_001 | Basic | Null value check (Stock aggregation) | ❌ 65.5% |
| BAS_002 | Basic | Negative monetary value check | ✅ PASS |
| BAS_003 | Basic | Zero market cap check | ❌ 3.6% |
| BAS_004 | Basic | Exchange name string length | ✅ PASS |
| ADV_001 | Advanced | Z-score outlier detection (threshold=3.5) | ❌ 0.22% |
| ADV_002 | Advanced | Month-on-month spike detection (threshold=5x) | ❌ 4.54% |
| ML_001 | ML | Isolation Forest anomaly detection | ❌ 2.0% |

---

## Key Findings

- **626,630 rows** loaded from 6 WFE Excel files covering Jan 2021 – Dec 2023
- **65.5% null rate** in Stock-type values — structural reporting gaps, not data corruption
- **Croatian Kuna (HRK)** flagged as the sole unmapped currency — retired when Croatia adopted the Euro in January 2023
- **1,741 ML-only anomalies** not caught by Z-score — genuine multivariate patterns invisible to univariate checks
- **Bolsa Latinoamericana de Valores (Latinex)** — highest anomaly rate (13.6%) due to simultaneous erratic behaviour across REITs, Investment Funds, and Bonds

---

## Data Sources

| Source | URL | Usage |
|--------|-----|-------|
| WFE Market Data | [statistics.world-exchanges.org](https://statistics.world-exchanges.org) | Primary dataset |
| ISO 20022 MIC codes | [iso20022.org/market-identifier-codes](https://www.iso20022.org/market-identifier-codes) | Exchange code validation |
| CBOE VIX | [cboe.com/tradable_products/vix](https://www.cboe.com/tradable_products/vix/vix_historical_data/) | Volatility context |
| ESMA Report | [esma.europa.eu](https://www.esma.europa.eu/sites/default/files/2024-04/ESMA12-1209242288-852_2023_Report_on_Quality_and_Use_of_Data.pdf) | Regulatory reference |

---

## Dependencies

Key libraries used:

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scipy` | Z-score computation |
| `scikit-learn` | Isolation Forest |
| `plotly` | Interactive visualisations |
| `anthropic` | LLM report generation |
| `pyarrow` | Parquet serialisation |

Full dependency list in `environment.yml`.

---

## Limitations

- Data covers 2021–2023 only — 2024 data was not available in the provided WFE export
- All values are in local currencies — cross-region absolute comparisons require FX normalisation
- `pct_change_ytd` and `pct_change_yty` are 100% null — structural export limitation
- Isolation Forest contamination parameter (2%) is a business assumption, not derived from labelled data
- Full MIC validation requires joining with the ISO 20022 reference dataset

---