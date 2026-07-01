"""
dashboard/app.py
-----------------
Streamlit dashboard for the checkframe data quality framework.

Loads a static sample snapshot (sample_results.json, sample_report.md) so it
runs standalone with no dataset, API key, or setup beyond `pip install streamlit`.

Usage:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = APP_DIR / "sample_results.json"
REPORT_PATH = APP_DIR / "sample_report.md"

st.set_page_config(page_title="checkframe", page_icon="✅", layout="wide")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 18px 36px;
        margin: 4px 0;
        border-radius: 8px;
        border: 1.5px solid #9a9a9a;
        background-color: #5c5c5c;
        color: #e8e8e8;
        transition: box-shadow 0.15s ease, background-color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_report(path: Path) -> str:
    return path.read_text()


snapshot = load_results(RESULTS_PATH)
metadata = snapshot["metadata"]
checks_df = pd.DataFrame(snapshot["checks"])
checks_df["failure_rate_pct"] = (checks_df["failure_rate"] * 100).round(2)

report_markdown = load_report(REPORT_PATH)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.info(
    "This dashboard shows results from a sample run on WFE market data "
    "(2021-2023, 626,630 rows). To run on your own data, see README setup "
    "instructions."
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("checkframe")
st.caption(
    "A pluggable data quality framework combining statistical, ML, and "
    "validation checks with an LLM layer that explains findings in plain language."
)

# ── Summary metrics ────────────────────────────────────────────────────────
total_checks = len(checks_df)
n_passed = int(checks_df["passed"].sum())
n_failed = total_checks - n_passed
pass_rate = (n_passed / total_checks * 100) if total_checks else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows Processed", f"{metadata['total_rows']:,}")
col2.metric("Checks Run", total_checks)
col3.metric("Passed / Failed", f"{n_passed} / {n_failed}")
col4.metric("Overall Pass Rate", f"{pass_rate:.0f}%")

st.caption(
    f"Dataset: {metadata['dataset_name']} · "
    f"{metadata['date_range']['start']} to {metadata['date_range']['end']}"
)

st.divider()

overview_tab, report_tab = st.tabs(["📊 Overview", "📄 Executive Report"])

with overview_tab:
    # ── Results table ────────────────────────────────────────────────────
    st.subheader("Check Results")

    categories = ["All"] + sorted(checks_df["category"].unique().tolist())
    selected_category = st.selectbox("Filter by category", categories)

    filtered_df = (
        checks_df if selected_category == "All"
        else checks_df[checks_df["category"] == selected_category]
    )

    display_cols = [
        "check_id", "category", "check_name", "passed",
        "total_rows", "failed_rows", "failure_rate_pct", "message",
    ]

    def _highlight_passed(passed: bool) -> str:
        color = "#d4edda" if passed else "#f8d7da"
        return f"background-color: {color}"

    styled_table = (
        filtered_df[display_cols]
        .style.map(_highlight_passed, subset=["passed"])
    )
    st.dataframe(styled_table, width="stretch", hide_index=True)

    st.divider()

    # ── Failure rate by category ─────────────────────────────────────────
    st.subheader("Failure Rate by Category")

    category_summary = (
        checks_df.groupby("category")[["failed_rows", "total_rows"]]
        .sum()
        .reset_index()
    )
    category_summary["failure_rate_pct"] = (
        category_summary["failed_rows"] / category_summary["total_rows"] * 100
    ).round(2)

    fig = px.bar(
        category_summary.sort_values("failure_rate_pct"),
        x="failure_rate_pct",
        y="category",
        orientation="h",
        labels={"failure_rate_pct": "Failure Rate (%)", "category": "Category"},
        title="Failure Rate by Check Category",
        color="category",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ── Failed check details ─────────────────────────────────────────────
    st.subheader("Failed Check Details")

    failed_checks = checks_df[~checks_df["passed"]]
    if failed_checks.empty:
        st.success("No failed checks in this snapshot.")
    else:
        for _, row in failed_checks.iterrows():
            with st.expander(
                f"❌ {row['check_id']} — {row['check_name']} "
                f"({row['failed_rows']:,} rows, {row['failure_rate_pct']}%)"
            ):
                st.write(row["message"])

with report_tab:
    st.markdown(report_markdown)
