"""
src/reporter.py
---------------
LLM-powered explainability layer for the Data Quality Control Framework.

Takes structured CheckResult objects and generates plain-language summaries
suitable for Regulatory Reporting stakeholders who are not data scientists.

I used the Anthropic Claude API to translate technical findings into:
- Executive summaries per check
- Actionable recommendations
- Regulatory risk assessments
- Overall framework narrative

Usage:
    from src.reporter import Reporter
    reporter = Reporter()  # reads ANTHROPIC_API_KEY from environment
    narrative = reporter.explain_check(result)
    full_report = reporter.generate_full_report(all_results)
"""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import anthropic
import pandas as pd

from src.checks import CheckResult

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior data quality analyst,
writing reports for the Regulatory Reporting team. Your audience understands financial 
markets but is not technical — they do not know what Z-scores or Isolation Forests are.

Your job is to translate data quality check results into clear, concise, actionable 
language. Always:
- Lead with the business impact, not the technical method
- Quantify findings with the actual numbers provided
- Give a clear recommendation (fix it, monitor it, or accept it)
- Keep responses under 150 words unless asked for more
- Never use jargon like "outlier detection", "contamination parameter", or "Z-score"
- Instead say things like "unusual values", "values that deviate significantly from 
  historical patterns", "values requiring manual review"
"""

CHECK_EXPLANATION_TEMPLATE = """
A data quality check has been run on WFE exchange market data (2021-2023).

Check name: {check_name}
Check ID: {check_id}
Category: {category}
Result: {result}
Total rows checked: {total_rows:,}
Rows with issues: {failed_rows:,} ({failure_rate:.1%} of total)
Technical finding: {message}

Please write a 3-sentence explanation for a regulatory reporting manager:
1. What was checked and what was found
2. What this means for regulatory reporting
3. What action should be taken
"""

FULL_REPORT_TEMPLATE = """
Below are the results of a complete data quality framework run on WFE market data 
covering 626,630 rows from January 2021 to December 2023.

VALIDATION CHECKS:
{validation_summary}

BASIC CHECKS:
{basic_summary}

ADVANCED CHECKS:
{advanced_summary}

ML CHECKS:
{ml_summary}

Please write an executive summary (200-250 words) for the Head of Regulatory Reporting 
that covers:
1. Overall data quality status (one sentence verdict)
2. The 2-3 most important findings that need immediate attention
3. Findings that are expected and require no action
4. Recommended next steps before the next regulatory submission
"""

CURRENCY_FINDINGS_TEMPLATE = """
A currency validation check found that {failed_count:,} rows ({failure_rate:.1%}) 
contain currency names that cannot be automatically mapped to official ISO 4217 codes.

The unmapped currencies are: {unmapped_list}

One specific finding: Croatian Kuna (HRK) cannot be mapped because Croatia adopted 
the Euro in January 2023, making HRK a retired currency.

Please write a 3-sentence explanation for a compliance officer covering:
1. What the problem is in plain terms
2. The regulatory risk if left unresolved
3. The recommended fix
"""

ANOMALY_FINDINGS_TEMPLATE = """
An automated pattern detection system analysed {total_rows:,} rows of market data 
and found {anomaly_count:,} rows ({anomaly_rate:.1%}) with unusual patterns.

Key finding: {ml_only_count:,} of these were only detected by the advanced system 
(not by simpler statistical checks), meaning they represent subtle multi-dimensional 
patterns that standard analysis would miss entirely.

The most affected exchanges are: {top_exchanges}

Please write a 3-sentence explanation for a regulatory reporting manager covering:
1. What was found in plain language
2. Why the advanced detection matters compared to simpler checks
3. What the reporting team should do with these flagged rows
"""


# ── Reporter class ────────────────────────────────────────────────────────────

@dataclass
class ReportSection:
    """A single section of the generated report."""
    title: str
    check_id: str
    narrative: str
    passed: bool
    failed_rows: int
    failure_rate: float


class Reporter:
    """
    Generates plain-language regulatory reports from CheckResult objects
    using the Claude API.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY
        environment variable.
    model : str
        Claude model to use. Defaults to claude-sonnet-4-20250514.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514"):

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        logger.info(f"Reporter initialised with model: {model}")

    def _call_claude(self, prompt: str, max_tokens: int = 300) -> str:
        """Make a single Claude API call and return the text response."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"[Report generation failed: {e}]"

    def explain_check(self, result: CheckResult) -> ReportSection:
        """
        Generate a plain-language explanation for a single check result.

        Parameters
        ----------
        result : CheckResult
            The check result to explain.

        Returns
        -------
        ReportSection
            Structured report section with narrative.
        """
        prompt = CHECK_EXPLANATION_TEMPLATE.format(
            check_name=result.check_name,
            check_id=result.check_id,
            category=result.category,
            result="PASSED" if result.passed else "FAILED",
            total_rows=result.total_rows,
            failed_rows=result.failed_rows,
            failure_rate=result.failure_rate,
            message=result.message,
        )

        narrative = self._call_claude(prompt)
        logger.info(f"Generated explanation for {result.check_id}")

        return ReportSection(
            title=result.check_name,
            check_id=result.check_id,
            narrative=narrative,
            passed=result.passed,
            failed_rows=result.failed_rows,
            failure_rate=result.failure_rate,
        )

    def explain_currency_findings(self,
                                   val_002_result: CheckResult) -> str:
        """Generate a plain-language explanation of VAL_002 currency findings."""
        unmapped = val_002_result.details.get('unmapped_currencies', [])
        unmapped_str = ', '.join(unmapped[:10])
        if len(unmapped) > 10:
            unmapped_str += f' and {len(unmapped) - 10} more'

        prompt = CURRENCY_FINDINGS_TEMPLATE.format(
            failed_count=val_002_result.failed_rows,
            failure_rate=val_002_result.failure_rate,
            unmapped_list=unmapped_str,
        )

        return self._call_claude(prompt, max_tokens=200)

    def explain_ml_findings(self,
                             ml_result: CheckResult,
                             ml_only_count: int,
                             top_exchanges: List[str]) -> str:
        """Generate a plain-language explanation of ML anomaly findings."""
        prompt = ANOMALY_FINDINGS_TEMPLATE.format(
            total_rows=ml_result.total_rows,
            anomaly_count=ml_result.failed_rows,
            anomaly_rate=ml_result.failure_rate,
            ml_only_count=ml_only_count,
            top_exchanges=', '.join(top_exchanges[:5]),
        )

        return self._call_claude(prompt, max_tokens=200)

    def generate_full_report(self,
                              all_results: Dict[str, List[CheckResult]]) -> str:
        """
        Generate a complete executive summary report from all check results.

        Parameters
        ----------
        all_results : dict
            Output of CheckRegistry.run_all(df).

        Returns
        -------
        str
            Full executive summary narrative.
        """
        def summarise_category(results: List[CheckResult]) -> str:
            lines = []
            for r in results:
                status = "PASSED" if r.passed else "FAILED"
                lines.append(
                    f"- {r.check_id} ({status}): {r.failed_rows:,} issues "
                    f"({r.failure_rate:.1%}) — {r.message[:100]}"
                )
            return '\n'.join(lines) if lines else "No checks run."

        prompt = FULL_REPORT_TEMPLATE.format(
            validation_summary=summarise_category(
                all_results.get('validation', [])),
            basic_summary=summarise_category(
                all_results.get('basic', [])),
            advanced_summary=summarise_category(
                all_results.get('advanced', [])),
            ml_summary=summarise_category(
                all_results.get('ml', [])),
        )

        return self._call_claude(prompt, max_tokens=400)

    def generate_report_markdown(self,
                                  all_results: Dict[str, List[CheckResult]],
                                  ml_only_count: int = 0,
                                  top_anomalous_exchanges: List[str] = None
                                  ) -> str:
        """
        Generate a full markdown report with per-check narratives and
        an executive summary.

        Parameters
        ----------
        all_results : dict
            Output of CheckRegistry.run_all(df).
        ml_only_count : int
            Number of ML-only anomalies (not caught by Z-score).
        top_anomalous_exchanges : list
            Top exchanges by ML anomaly count.

        Returns
        -------
        str
            Full markdown report string.
        """
        sections = []

        sections.append("# Data Quality Framework — Regulatory Report")
        sections.append(
            f"**Generated by:** checkframe  \n"
            f"**Data:** WFE Market Data 2021–2023  \n"
            f"**Model:** {self.model}\n"
        )

        # Executive summary
        sections.append("## Executive Summary\n")
        exec_summary = self.generate_full_report(all_results)
        sections.append(exec_summary)

        # Per-check sections
        for category, results in all_results.items():
            if not results:
                continue
            sections.append(f"\n## {category.title()} Checks\n")
            for result in results:
                section = self.explain_check(result)
                status_emoji = "✅" if section.passed else "❌"
                sections.append(
                    f"### {status_emoji} {section.check_id} — {section.title}\n"
                )
                sections.append(
                    f"**Failures:** {section.failed_rows:,} "
                    f"({section.failure_rate:.1%})\n"
                )
                sections.append(section.narrative)
                sections.append("")

        # Special sections
        if 'validation' in all_results and len(all_results['validation']) > 1:
            val_002 = all_results['validation'][1]
            sections.append("\n## Currency Compliance Deep Dive\n")
            sections.append(self.explain_currency_findings(val_002))

        if 'ml' in all_results and all_results['ml'] and ml_only_count > 0:
            ml_result = all_results['ml'][0]
            sections.append("\n## ML Anomaly Detection Deep Dive\n")
            sections.append(self.explain_ml_findings(
                ml_result,
                ml_only_count,
                top_anomalous_exchanges or []
            ))

        return '\n'.join(sections)

    def save_report(self,
                    report_markdown: str,
                    output_path: str = "reports/dq_report.md") -> None:
        """Save the generated report to a markdown file."""
        import pathlib
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_markdown)
        logger.info(f"Report saved to {output_path}")
        print(f"Report saved to {output_path}")