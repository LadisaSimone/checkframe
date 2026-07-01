"""
src/checks.py
-------------
Intelligent Data Quality Control Framework for WFE regulatory reporting.

Architecture:
- CheckResult      : dataclass holding the output of any single check
- CheckRegistry    : registry that stores and runs all checks by category
- Validation checks: structural/regulatory rules (exchange code, ISO 4217)
- Basic checks     : simple statistical rules (nulls, negatives, string lengths)
- Advanced checks  : statistical methods (outlier detection, correlation)

Design principles:
- Every check is a pure function: (df) -> CheckResult
- Checks are registered by category and can be run selectively
- Results are always structured the same way for easy reporting
- The registry is designed to scale to 50 basic, 20 advanced, 10 ML checks
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── ISO 4217 official currency codes ─────────────────────────────────────────
# Full official list — used for currency validation check
ISO_4217_CODES = {
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BOV",
    "BRL", "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHE", "CHF",
    "CHW", "CLF", "CLP", "CNY", "COP", "COU", "CRC", "CUC", "CUP", "CVE",
    "CZK", "DJF", "DKK", "DOP", "DZD", "EGP", "ERN", "ETB", "EUR", "FJD",
    "FKP", "GBP", "GEL", "GHS", "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD",
    "HNL", "HTG", "HUF", "IDR", "ILS", "INR", "IQD", "IRR", "ISK", "JMD",
    "JOD", "JPY", "KES", "KGS", "KHR", "KMF", "KPW", "KRW", "KWD", "KYD",
    "KZT", "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA",
    "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MXV",
    "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB",
    "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB",
    "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD", "SHP", "SLE", "SLL",
    "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB", "TJS", "TMT",
    "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "USN",
    "UYI", "UYU", "UYW", "UZS", "VED", "VES", "VND", "VUV", "WST", "XAF",
    "XAG", "XAU", "XBA", "XBB", "XBC", "XBD", "XCD", "XDR", "XOF", "XPD",
    "XPF", "XPT", "XSU", "XTS", "XUA", "XXX", "YER", "ZAR", "ZMW", "ZWL",
}

# Currency name → ISO code mapping (WFE uses full names, not codes)
CURRENCY_NAME_TO_ISO = {
    "azerbaijani new manat": "AZN",
    "bermudian dollar": "BMD",
    "cfa franc": "XAF",  # flagged as ambiguous — needs manual review
    "chinese yuan renminbi": "CNY",
    "ghana cedi": "GHS",
    "israeli shekel": "ILS",
    "jordanian dollar": "JOD",  # likely misprint of jordanian dinar
    "kwanza": "AOA",
    "mauritius rupee": "MUR",
    "morocco dirham": "MAD",
    "new turkish lira": "TRY",
    "peruvian nuevo sol": "PEN",
    "qatari rial": "QAR",
    "rwanda franc": "RWF",
    "saudi arabian riyal": "SAR",
    "sri lanka rupee": "LKR",
    "taiwan dollar": "TWD",
    "afghan afghani": "AFN", "albanian lek": "ALL", "algerian dinar": "DZD",
    "angolan kwanza": "AOA", "argentine peso": "ARS", "armenian dram": "AMD",
    "aruban florin": "AWG", "australian dollar": "AUD", "azerbaijani manat": "AZN",
    "bahamian dollar": "BSD", "bahraini dinar": "BHD", "bangladeshi taka": "BDT",
    "barbadian dollar": "BBD", "belarusian ruble": "BYN", "belize dollar": "BZD",
    "botswana pula": "BWP", "brazilian real": "BRL", "british pound": "GBP",
    "brunei dollar": "BND", "bulgarian lev": "BGN", "burundian franc": "BIF",
    "cambodian riel": "KHR", "canadian dollar": "CAD", "cape verdean escudo": "CVE",
    "cayman islands dollar": "KYD", "central african cfa franc": "XAF",
    "chilean peso": "CLP", "chinese renminbi": "CNY", "chinese yuan": "CNY",
    "colombian peso": "COP", "comorian franc": "KMF", "congolese franc": "CDF",
    "costa rican colon": "CRC", "croatian kuna": "HRK", "czech koruna": "CZK",
    "danish krone": "DKK", "djiboutian franc": "DJF", "dominican peso": "DOP",
    "east caribbean dollar": "XCD", "egyptian pound": "EGP", "eritrean nakfa": "ERN",
    "ethiopian birr": "ETB", "euro": "EUR", "fijian dollar": "FJD",
    "gambian dalasi": "GMD", "georgian lari": "GEL", "ghanaian cedi": "GHS",
    "gibraltar pound": "GIP", "guatemalan quetzal": "GTQ", "guinean franc": "GNF",
    "guyanese dollar": "GYD", "haitian gourde": "HTG", "honduran lempira": "HNL",
    "hong kong dollar": "HKD", "hungarian forint": "HUF", "icelandic krona": "ISK",
    "icelandic króna": "ISK", "indian rupee": "INR", "indonesian rupiah": "IDR",
    "iranian rial": "IRR", "iraqi dinar": "IQD", "israeli new shekel": "ILS",
    "jamaican dollar": "JMD", "japanese yen": "JPY", "jordanian dinar": "JOD",
    "kazakhstani tenge": "KZT", "kenyan shilling": "KES", "kuwaiti dinar": "KWD",
    "kyrgyzstani som": "KGS", "lao kip": "LAK", "lebanese pound": "LBP",
    "lesotho loti": "LSL", "liberian dollar": "LRD", "libyan dinar": "LYD",
    "macanese pataca": "MOP", "macedonian denar": "MKD", "malagasy ariary": "MGA",
    "malawian kwacha": "MWK", "malaysian ringgit": "MYR", "maldivian rufiyaa": "MVR",
    "mauritanian ouguiya": "MRU", "mauritian rupee": "MUR", "mexican peso": "MXN",
    "moldovan leu": "MDL", "mongolian togrog": "MNT", "moroccan dirham": "MAD",
    "mozambican metical": "MZN", "myanmar kyat": "MMK", "namibian dollar": "NAD",
    "nepalese rupee": "NPR", "new taiwan dollar": "TWD", "new zealand dollar": "NZD",
    "nicaraguan cordoba": "NIO", "nigerian naira": "NGN", "norwegian krone": "NOK",
    "omani rial": "OMR", "pakistani rupee": "PKR", "panamanian balboa": "PAB",
    "papua new guinean kina": "PGK", "paraguayan guarani": "PYG",
    "peruvian sol": "PEN", "philippine peso": "PHP", "philippine piso": "PHP",
    "polish zloty": "PLN", "qatari riyal": "QAR", "romanian leu": "RON",
    "russian ruble": "RUB", "rwandan franc": "RWF", "saudi riyal": "SAR",
    "seychellois rupee": "SCR", "sierra leonean leone": "SLL",
    "singapore dollar": "SGD", "solomon islands dollar": "SBD",
    "somali shilling": "SOS", "south african rand": "ZAR",
    "south korean won": "KRW", "south sudanese pound": "SSP",
    "sri lankan rupee": "LKR", "sudanese pound": "SDG",
    "surinamese dollar": "SRD", "swazi lilangeni": "SZL", "swedish krona": "SEK",
    "swiss franc": "CHF", "syrian pound": "SYP", "são tomé and príncipe dobra": "STN",
    "taiwanese dollar": "TWD", "tajikistani somoni": "TJS",
    "tanzanian shilling": "TZS", "thai baht": "THB", "tongan paanga": "TOP",
    "trinidad and tobago dollar": "TTD", "tunisian dinar": "TND",
    "turkish lira": "TRY", "turkmenistan manat": "TMT", "ugandan shilling": "UGX",
    "ukrainian hryvnia": "UAH", "united arab emirates dirham": "AED",
    "uruguayan peso": "UYU", "us dollar": "USD", "uzbekistani sum": "UZS",
    "vanuatu vatu": "VUV", "venezuelan bolivar": "VES", "vietnamese dong": "VND",
    "west african cfa franc": "XOF", "yemeni rial": "YER",
    "zambian kwacha": "ZMW", "zimbabwean dollar": "ZWL",
    # Common aliases
    "pound sterling": "GBP", "renminbi": "CNY", "won": "KRW",
    "baht": "THB", "ringgit": "MYR", "rupiah": "IDR", "ruble": "RUB",
    "hryvnia": "UAH", "zloty": "PLN", "forint": "HUF", "koruna": "CZK",
    "krona": "SEK", "krone": "NOK", "lira": "TRY", "shekel": "ILS",
    "dirham": "AED", "riyal": "SAR", "dinar": "BHD", "franc": "CHF",
    "peso": "MXN", "sol": "PEN", "real": "BRL", "rupee": "INR",
    "tenge": "KZT", "som": "KGS", "manat": "AZN", "lari": "GEL",
    "dram": "AMD", "leu": "RON", "lev": "BGN", "kuna": "HRK",
    "namibian dollar": "NAD", "moroccan dirham": "MAD",
    "bahraini dinar": "BHD", "kuwaiti dinar": "KWD",
    "omani rial": "OMR", "qatari riyal": "QAR",
    "seychelles rupee": "SCR", "kenyan shilling": "KES",
    "egyptian pound": "EGP", "nigerian naira": "NGN",
    "ghanaian cedi": "GHS", "tanzanian shilling": "TZS",
    "ugandan shilling": "UGX", "zambian kwacha": "ZMW",
    "botswana pula": "BWP", "mauritian rupee": "MUR",
    "rwandan franc": "RWF", "ethiopian birr": "ETB",
    "mozambican metical": "MZN", "angolan kwanza": "AOA",
    "congolese franc": "CDF", "burundian franc": "BIF",
    "comorian franc": "KMF", "guinea franc": "GNF",
    "sierra leone leone": "SLL", "liberian dollar": "LRD",
    "cape verde escudo": "CVE", "sao tome and principe dobra": "STN",
    "djiboutian franc": "DJF", "eritrean nakfa": "ERN",
    "somali shilling": "SOS", "south sudanese pound": "SSP",
    "sudanese pound": "SDG", "malagasy ariary": "MGA",
    "malawian kwacha": "MWK", "lesotho loti": "LSL",
    "swazi lilangeni": "SZL", "zimbabwean dollar": "ZWL",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """
    Standardised output for every data quality check.
    All checks return exactly this structure — enables uniform reporting.
    """
    check_id: str                        # Unique identifier e.g. 'VAL_001'
    check_name: str                      # Human-readable name
    category: str                        # 'validation' | 'basic' | 'advanced' | 'ml'
    passed: bool                         # True = no issues found
    total_rows: int                      # Rows checked
    failed_rows: int                     # Rows that failed
    failure_rate: float                  # failed_rows / total_rows
    failed_sample: pd.DataFrame          # Sample of failing rows (up to 20)
    message: str                         # Plain-language summary
    details: Dict = field(default_factory=dict)  # Extra metadata


class CheckRegistry:
    """
    Central registry for all data quality checks.

    Designed to scale to 50 basic + 20 advanced + 10 ML checks.
    Checks are registered by category and can be run selectively.

    Usage:
        registry = CheckRegistry()
        registry.register('basic', my_check_fn)
        results = registry.run_all(df)
        results = registry.run_category(df, 'validation')
    """

    def __init__(self):
        self._checks: Dict[str, List[Callable]] = {
            'validation': [],
            'basic': [],
            'advanced': [],
            'ml': [],
        }

    def register(self, category: str, fn: Callable) -> None:
        """Register a check function under a category."""
        if category not in self._checks:
            raise ValueError(f"Unknown category '{category}'. "
                             f"Choose from: {list(self._checks.keys())}")
        self._checks[category].append(fn)
        logger.debug(f"Registered {category} check: {fn.__name__}")

    def run_category(self, df: pd.DataFrame, category: str) -> List[CheckResult]:
        """Run all checks in a single category and return results."""
        results = []
        for fn in self._checks[category]:
            try:
                result = fn(df)
                results.append(result)
                status = "✓ PASS" if result.passed else "✗ FAIL"
                logger.info(f"[{category.upper()}] {status} | {result.check_name} "
                            f"| {result.failed_rows:,} failures "
                            f"({result.failure_rate:.1%})")
            except Exception as e:
                logger.error(f"[{category.upper()}] ERROR in {fn.__name__}: {e}")
        return results

    def run_all(self, df: pd.DataFrame) -> Dict[str, List[CheckResult]]:
        """Run all registered checks across all categories."""
        return {cat: self.run_category(df, cat) for cat in self._checks}

    def summary(self, results: Dict[str, List[CheckResult]]) -> pd.DataFrame:
        """Convert results dict to a clean summary DataFrame."""
        rows = []
        for category, res_list in results.items():
            for r in res_list:
                rows.append({
                    'check_id': r.check_id,
                    'category': r.category,
                    'check_name': r.check_name,
                    'passed': r.passed,
                    'total_rows': r.total_rows,
                    'failed_rows': r.failed_rows,
                    'failure_rate_pct': round(r.failure_rate * 100, 2),
                    'message': r.message,
                })
        return pd.DataFrame(rows)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_result(check_id, check_name, category, df_all, df_failed, message,
                 details=None) -> CheckResult:
    """Convenience constructor for CheckResult."""
    n_total = len(df_all)
    n_failed = len(df_failed)
    return CheckResult(
        check_id=check_id,
        check_name=check_name,
        category=category,
        passed=n_failed == 0, #if n_failed == 0 then passed = True
        total_rows=n_total,
        failed_rows=n_failed,
        failure_rate=n_failed / n_total if n_total > 0 else 0.0,
        failed_sample=df_failed.head(20),
        message=message,
        details=details or {},
    )


# ── Validation checks ─────────────────────────────────────────────────────────

def check_exchange_code_format(df: pd.DataFrame) -> CheckResult:
    """
    VAL_001 — Exchange code must be a 4-character alphanumeric code (MIC format).

    The ISO 20022 Market Identifier Code (MIC) standard requires exactly
    4 alphanumeric characters. Since WFE data contains exchange names rather
    than MIC codes, we validate the exchange_name is non-null and non-empty
    as a proxy, and flag any that match a known malformed pattern.

    Note: A full MIC validation requires joining with the ISO 20022 MIC
    reference dataset (www.iso20022.org/market-identifier-codes).
    """
    # In WFE data, exchange codes are not present directly.
    # We validate that exchange_name is non-null and non-empty (prerequisite
    # for any downstream MIC lookup), and flag suspiciously short names.
    df_check = df[df['exchange_name'].notna()].copy()

    # Flag exchange names that are suspiciously short (< 3 chars) —
    # likely malformed or placeholder values
    mask_failed = df_check['exchange_name'].str.strip().str.len() < 3

    failed = df_check[mask_failed][
        ['business_date', 'region', 'exchange_name', 'indicator_name']
    ]

    n_null = df['exchange_name'].isna().sum()
    message = (
        f"Found {len(failed):,} rows with exchange_name shorter than 3 characters. "
        f"Additionally {n_null:,} rows have null exchange_name. "
        f"Full MIC validation requires ISO 20022 reference join."
    )

    return _make_result(
        check_id='VAL_001',
        check_name='Exchange code format (MIC proxy)',
        category='validation',
        df_all=df_check,
        df_failed=failed,
        message=message,
        details={'null_exchange_name': int(n_null)},
    )


def check_currency_iso4217(df: pd.DataFrame) -> CheckResult:
    """
    VAL_002 — Currency must be a valid ISO 4217 currency code.

    WFE data stores currency as full name (e.g. 'Egyptian Pound') rather
    than ISO code (e.g. 'EGP'). We map known names to ISO codes and flag
    any currency names that cannot be resolved to a valid ISO 4217 code.
    """
    df_check = df[df['currency_name'].notna()].copy()

    # Map currency name to ISO code (case-insensitive)
    df_check['currency_iso'] = (
        df_check['currency_name']
        .str.strip()
        .str.lower()
        .map(CURRENCY_NAME_TO_ISO)
    )

    # Flag rows where mapping failed or ISO code not in official list
    mask_no_mapping = df_check['currency_iso'].isna()
    mask_invalid = (~mask_no_mapping) & (~df_check['currency_iso'].isin(ISO_4217_CODES))
    mask_failed = mask_no_mapping | mask_invalid

    failed = df_check[mask_failed][
        ['business_date', 'region', 'exchange_name', 'currency_name', 'currency_iso']
    ].drop_duplicates('currency_name')

    # List unique unmapped currencies for reporting
    unmapped = sorted(df_check[mask_no_mapping]['currency_name'].unique().tolist())

    message = (
        f"Found {df_check[mask_failed]['currency_name'].nunique()} unique currency names "
        f"that could not be mapped to a valid ISO 4217 code. "
        f"Unmapped: {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}"
    )

    return _make_result(
        check_id='VAL_002',
        check_name='Currency ISO 4217 compliance',
        category='validation',
        df_all=df_check,
        df_failed=failed,
        message=message,
        details={
            'unmapped_currencies': unmapped,
            'total_unique_currencies': int(df_check['currency_name'].nunique()),
        },
    )


# ── Basic checks ──────────────────────────────────────────────────────────────

def check_null_value(df: pd.DataFrame) -> CheckResult:
    """
    BAS_001 — Value column should not be null for Stock aggregation type.

    For rows where aggregation_type == 'Stock' (point-in-time snapshots),
    a null value means the exchange failed to report. This is the most
    critical basic check for regulatory completeness.
    """
    df_check = df[df['aggregation_type'] == 'Stock'].copy()
    failed = df_check[df_check['value'].isna()][
        ['business_date', 'region', 'exchange_name', 'indicator_name', 'value']
    ]

    message = (
        f"{len(failed):,} Stock-type rows have null value "
        f"({len(failed)/len(df_check):.1%} of Stock rows). "
        f"These represent exchanges that did not submit data for the period."
    )

    return _make_result(
        check_id='BAS_001',
        check_name='Null value check (Stock aggregation)',
        category='basic',
        df_all=df_check,
        df_failed=failed,
        message=message,
    )


def check_negative_value(df: pd.DataFrame) -> CheckResult:
    """
    BAS_002 — Monetary values should not be negative.

    Market capitalisation, traded value, and similar monetary indicators
    cannot be negative by definition. Negative values indicate data entry
    errors or sign convention inconsistencies.
    """
    df_check = df[df['value'].notna() & (df['data_type'] == 'Monetary')].copy()
    failed = df_check[df_check['value'] < 0][
        ['business_date', 'region', 'exchange_name', 'indicator_name', 'value']
    ]

    message = (
        f"{len(failed):,} Monetary rows have negative value. "
        f"{'No issues found.' if len(failed) == 0 else 'These require immediate investigation.'}"
    )

    return _make_result(
        check_id='BAS_002',
        check_name='Negative monetary value check',
        category='basic',
        df_all=df_check,
        df_failed=failed,
        message=message,
    )


def check_zero_value(df: pd.DataFrame) -> CheckResult:
    """
    BAS_003 — Zero values in market cap indicators may indicate missing data
    reported as zero rather than null.
    """
    df_check = df[
        df['value'].notna() &
        df['indicator_name'].str.contains('Market Capitalisation', na=False)
    ].copy()

    failed = df_check[df_check['value'] == 0][
        ['business_date', 'region', 'exchange_name', 'indicator_name', 'value']
    ]

    message = (
        f"{len(failed):,} Market Capitalisation rows have value == 0. "
        f"These may represent genuine zero-cap exchanges or misreported nulls."
    )

    return _make_result(
        check_id='BAS_003',
        check_name='Zero market cap check',
        category='basic',
        df_all=df_check,
        df_failed=failed,
        message=message,
    )


def check_string_length(df: pd.DataFrame) -> CheckResult:
    """
    BAS_004 — Exchange names should have reasonable string length.
    Names shorter than 3 or longer than 100 characters are suspicious.
    """
    df_check = df[df['exchange_name'].notna()].copy()
    name_len = df_check['exchange_name'].str.strip().str.len()
    mask_failed = (name_len < 3) | (name_len > 100)

    failed = df_check[mask_failed][
        ['business_date', 'exchange_name', 'region']
    ].drop_duplicates('exchange_name')

    message = (
        f"{len(failed):,} unique exchange names have suspicious length "
        f"(< 3 or > 100 characters)."
    )

    return _make_result(
        check_id='BAS_004',
        check_name='Exchange name string length check',
        category='basic',
        df_all=df_check,
        df_failed=df_check[mask_failed],
        message=message,
    )


# ── Advanced checks ───────────────────────────────────────────────────────────

def check_outliers_zscore(df: pd.DataFrame,
                          z_threshold: float = 3.5) -> CheckResult:
    """
    ADV_001 — Outlier detection using Z-score per exchange per indicator.

    For each exchange-indicator combination, we compute the Z-score of each
    monthly value. Values beyond z_threshold standard deviations are flagged.
    Z-score is computed on log-transformed values to handle skewed distributions
    typical in financial data (e.g. Iranian Rial nominal values).

    Threshold of 3.5 is chosen over the standard 3.0 to reduce false positives
    in volatile market periods (e.g. H1 2022).
    """
    df_check = df[df['value'].notna() & (df['value'] > 0)].copy()

    # Log-transform to handle extreme scale differences across currencies
    df_check['log_value'] = np.log1p(df_check['value'])

    # Compute Z-score within each exchange × indicator group
    def zscore_group(group):
        if len(group) < 4:  # Not enough data points for meaningful Z-score
            group['zscore'] = np.nan
            return group
        group['zscore'] = np.abs(stats.zscore(group['log_value'], nan_policy='omit'))
        return group

    df_check = df_check.groupby(
        ['exchange_name', 'indicator_name'], group_keys=False
    ).apply(zscore_group)

    failed = df_check[df_check['zscore'] > z_threshold][
        ['business_date', 'region', 'exchange_name', 'indicator_name',
         'value', 'zscore']
    ].sort_values('zscore', ascending=False)

    message = (
        f"{len(failed):,} rows flagged as outliers (|Z-score| > {z_threshold}) "
        f"using log-transformed values within exchange-indicator groups. "
        f"Top offender: {failed.iloc[0]['exchange_name'] if len(failed) > 0 else 'N/A'}"
    )

    return _make_result(
        check_id='ADV_001',
        check_name=f'Z-score outlier detection (threshold={z_threshold})',
        category='advanced',
        df_all=df_check,
        df_failed=failed,
        message=message,
        details={'z_threshold': z_threshold, 'log_transformed': True},
    )


def check_month_on_month_spike(df: pd.DataFrame,
                                spike_threshold: float = 5.0) -> CheckResult:
    """
    ADV_002 — Month-on-month value change exceeding spike_threshold × (5x)
    is flagged as a potential data quality issue.

    Computes MoM ratio per exchange-indicator pair. A ratio > 5.0 means
    the value more than quintupled in one month — highly unusual outside
    of extreme market events.
    """
    df_check = df[df['value'].notna() & (df['value'] > 0)].copy()
    df_check = df_check.sort_values('business_date')

    # Compute MoM ratio within each exchange × indicator group
    df_check['prev_value'] = df_check.groupby(
        ['exchange_name', 'indicator_name']
    )['value'].shift(1)

    df_check['mom_ratio'] = df_check['value'] / df_check['prev_value']

    # Flag extreme spikes (either direction)
    mask_spike = (
        (df_check['mom_ratio'] > spike_threshold) |  # value grew more than 5x in one month → suspicious spike up
        (df_check['mom_ratio'] < 1 / spike_threshold)  # which is < 0.2 → value dropped to less than 20% of previous month → suspicious spike down
    ) & df_check['prev_value'].notna()

    failed = df_check[mask_spike][
        ['business_date', 'region', 'exchange_name', 'indicator_name',
         'value', 'prev_value', 'mom_ratio']
    ].sort_values('mom_ratio', ascending=False)

    message = (
        f"{len(failed):,} rows show month-on-month ratio > {spike_threshold}x "
        f"or < 1/{spike_threshold}x. "
        f"May indicate data errors or extreme market events requiring contextual review."
    )

    return _make_result(
        check_id='ADV_002',
        check_name=f'Month-on-month spike detection (threshold={spike_threshold}x)',
        category='advanced',
        df_all=df_check,
        df_failed=failed,
        message=message,
        details={'spike_threshold': spike_threshold},
    )

# ── ML checks ─────────────────────────────────────────────────────────────────


def check_isolation_forest(df: pd.DataFrame,
                            contamination: float = 0.02,
                            n_estimators: int = 200) -> CheckResult:
    """
    ML_001 — Isolation Forest multivariate anomaly detection.

    Detects anomalies that univariate checks (Z-score, MoM) cannot catch
    by modelling the joint distribution of multiple features simultaneously.

    Features used:
    - log_value          : log-transformed value (scale normalisation)
    - log_mom_ratio      : log month-on-month change ratio
    - rolling_zscore     : 6-month rolling Z-score per exchange-indicator
    - region_enc         : label-encoded region
    - month_num          : month number for seasonality

    Rows with incomplete feature vectors are excluded from scoring.
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import LabelEncoder
        from scipy import stats as scipy_stats
    except ImportError:
        raise ImportError("scikit-learn and scipy required for ML checks. "
                          "Run: pip install scikit-learn scipy")

    # Prepare feature subset
    df_check = df[
        df['value'].notna() &
        (df['value'] > 0) &
        df['indicator_name'].str.contains('Market Capitalisation', na=False)
    ].copy()

    df_check = df_check.sort_values(['exchange_name', 'indicator_name', 'business_date'])

    # Feature engineering
    df_check['log_value'] = np.log1p(df_check['value'])

    df_check['prev_value'] = df_check.groupby(
        ['exchange_name', 'indicator_name']
    )['value'].shift(1)

    df_check['mom_ratio'] = np.where(
        df_check['prev_value'].notna() & (df_check['prev_value'] > 0),
        df_check['value'] / df_check['prev_value'],
        np.nan
    )
    df_check['log_mom_ratio'] = np.log1p(df_check['mom_ratio'].clip(0))

    def rolling_zscore(group, window=6):
        roll_mean = group['log_value'].rolling(window, min_periods=3).mean()
        roll_std  = group['log_value'].rolling(window, min_periods=3).std()
        group['rolling_zscore'] = (
            (group['log_value'] - roll_mean) / roll_std.replace(0, np.nan)
        )
        return group

    df_check = df_check.groupby(
        ['exchange_name', 'indicator_name'], group_keys=False
    ).apply(rolling_zscore)

    le = LabelEncoder()
    df_check['region_enc'] = le.fit_transform(df_check['region'].fillna('Unknown'))
    df_check['month_num']  = df_check['business_date'].dt.month

    FEATURES = ['log_value', 'log_mom_ratio', 'rolling_zscore',
                'region_enc', 'month_num']

    df_model = df_check[FEATURES + ['business_date', 'exchange_name',
                                     'indicator_name', 'region', 'value']].dropna()

    if len(df_model) < 10:
        return _make_result(
            check_id='ML_001',
            check_name='Isolation Forest anomaly detection',
            category='ml',
            df_all=df_check,
            df_failed=pd.DataFrame(),
            message='Insufficient data for ML check (< 10 complete rows)',
        )

    X = df_model[FEATURES].values

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)

    df_model['anomaly_label'] = model.predict(X)
    df_model['anomaly_score'] = model.score_samples(X)

    failed = df_model[df_model['anomaly_label'] == -1][
        ['business_date', 'region', 'exchange_name',
         'indicator_name', 'value', 'anomaly_score']
    ].sort_values('anomaly_score')

    n_anomalies = len(failed)
    message = (
        f"Isolation Forest detected {n_anomalies:,} anomalies "
        f"({n_anomalies/len(df_model):.2%} of scored rows) "
        f"using contamination={contamination}. "
        f"Most anomalous: {failed.iloc[0]['exchange_name'] if n_anomalies > 0 else 'N/A'}"
    )

    return _make_result(
        check_id='ML_001',
        check_name='Isolation Forest anomaly detection',
        category='ml',
        df_all=df_model,
        df_failed=failed,
        message=message,
        details={
            'contamination': contamination,
            'n_estimators': n_estimators,
            'features': FEATURES,
            'n_scored_rows': len(df_model),
        },
    )