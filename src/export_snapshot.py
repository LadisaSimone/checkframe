"""
src/export_snapshot.py
-----------------------
Exports a static snapshot of CheckRegistry results to dashboard/sample_results.json
so the Streamlit dashboard can display results without needing the full WFE
dataset or an API key.

Usage:
    python -m src.export_snapshot
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.loader import load_processed
from src.checks import (
    CheckRegistry,
    check_exchange_code_format,
    check_currency_iso4217,
    check_null_value,
    check_negative_value,
    check_zero_value,
    check_string_length,
    check_outliers_zscore,
    check_month_on_month_spike,
    check_isolation_forest,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = _PROJECT_ROOT / "dashboard" / "sample_results.json"


def build_registry() -> CheckRegistry:
    registry = CheckRegistry()
    registry.register('validation', check_exchange_code_format)
    registry.register('validation', check_currency_iso4217)
    registry.register('basic', check_null_value)
    registry.register('basic', check_negative_value)
    registry.register('basic', check_zero_value)
    registry.register('basic', check_string_length)
    registry.register('advanced', check_outliers_zscore)
    registry.register('advanced', check_month_on_month_spike)
    registry.register('ml', check_isolation_forest)
    return registry


def export_snapshot(output_path: Path = OUTPUT_PATH) -> None:
    """Run the full check registry and save a JSON snapshot for the dashboard."""
    df = load_processed()
    registry = build_registry()
    all_results = registry.run_all(df)

    checks = [
        {
            "check_id": r.check_id,
            "category": r.category,
            "check_name": r.check_name,
            "passed": r.passed,
            "total_rows": r.total_rows,
            "failed_rows": r.failed_rows,
            "failure_rate": round(r.failure_rate, 6),
            "message": r.message,
        }
        for results in all_results.values()
        for r in results
    ]

    snapshot = {
        "metadata": {
            "dataset_name": "WFE Market Data",
            "total_rows": int(len(df)),
            "date_range": {
                "start": df["business_date"].min().strftime("%Y-%m-%d"),
                "end": df["business_date"].max().strftime("%Y-%m-%d"),
            },
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info(f"Snapshot saved to {output_path}")


if __name__ == "__main__":
    export_snapshot()
