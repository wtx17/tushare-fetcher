"""Core constants, dataclasses, exceptions and date helpers for sync_tushare.

This module has no internal dependencies and holds only pure value types so it
can be imported freely by every other layer.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("sync_tushare")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = SCRIPT_DIR / "使用说明"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"
DEFAULT_API_URL = "https://tx.xiaodefa.top/"

DATE_FORMAT = "%Y%m%d"
MANIFEST_VERSION = 1
CATALOG_VERSION = 1

KNOWN_INCOME_MISSING_FIELDS = frozenset(
    {
        "net_after_nr_lp_correct",
        "credit_impa_loss",
        "net_expo_hedging_benefits",
        "oth_impair_loss_assets",
        "total_opcost",
        "amodcost_fin_assets",
        "oth_income",
        "asset_disp_income",
        "continued_net_profit",
        "end_net_profit",
    }
)

KNOWN_EXPRESS_MISSING_FIELDS = frozenset(
    {
        "eps_last_year",
        "growth_assets",
        "growth_bps",
        "is_audit",
        "np_last_year",
        "op_last_year",
        "or_last_year",
        "remark",
        "tp_last_year",
        "yoy_dedu_np",
        "yoy_eps",
        "yoy_equity",
        "yoy_op",
        "yoy_roe",
        "yoy_sales",
        "yoy_tp",
    }
)

KNOWN_DAILY_BASIC_MISSING_FIELDS = frozenset({"limit_status"})


class SyncError(RuntimeError):
    """Base exception for controlled synchronization failures."""


class ConfigurationError(SyncError):
    """Raised when local configuration is incomplete or invalid."""


class SourceSchemaError(SyncError):
    """Raised when the upstream response is incompatible with the docs."""


class SchemaDriftError(SyncError):
    """Raised when local documentation changed after data was written."""


class DatasetRunError(SyncError):
    """Raised after one or more independent partitions failed."""


class OverwriteGuardError(SyncError):
    """Raised when a refresh would shrink an existing partition's data."""


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    name: str
    source_type: str

    @property
    def pandas_dtype(self) -> str:
        lowered = self.source_type.lower()
        if "float" in lowered or "double" in lowered or "decimal" in lowered:
            return "Float64"
        if "int" in lowered:
            return "Int64"
        return "string"


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    api_name: str
    doc_name: str
    mode: str
    page_size: int
    allowed_missing_fields: frozenset[str] = frozenset()


@dataclasses.dataclass
class FetchResult:
    frame: Any
    pages: int
    raw_rows: int
    last_page_rows: int
    cross_page_duplicates: int
    missing_fields: set[str]
    warnings: list[str]


@dataclasses.dataclass
class PartitionResult:
    key: str
    relative_path: str
    query: dict[str, Any]
    rows: int
    raw_rows: int
    pages: int
    last_page_rows: int
    cross_page_duplicates: int
    missing_fields: list[str]
    warnings: list[str]
    sha256: str
    bytes: int
    fetched_at: str
    subqueries: dict[str, Any] | None = None

    def to_manifest_entry(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result.pop("key")
        if self.subqueries is None:
            result.pop("subqueries")
        return result


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    # Put the smaller datasets first so a backfill establishes broad coverage
    # before entering the slower 12-report-type statement loops.
    DatasetSpec("fina_indicator", "fina_indicator_vip", "财务指标.md", "period", 5_000),
    DatasetSpec("forecast", "forecast_vip", "业绩预告.md", "period", 5_000),
    DatasetSpec(
        "express",
        "express_vip",
        "业绩快报.md",
        "period",
        5_000,
        KNOWN_EXPRESS_MISSING_FIELDS,
    ),
    DatasetSpec("stk_holdernumber", "stk_holdernumber", "股东人数.md", "event", 3_000),
    DatasetSpec("stk_holdertrade", "stk_holdertrade", "股东增减持.md", "event", 3_000),
    DatasetSpec("index_member_all", "index_member_all", "申万行业成分.md", "industry", 2_000),
    DatasetSpec("ci_index_member", "ci_index_member", "中信行业成分.md", "industry", 5_000),
    DatasetSpec(
        "income",
        "income_vip",
        "income.md",
        "statement",
        5_000,
        KNOWN_INCOME_MISSING_FIELDS,
    ),
    DatasetSpec("balancesheet", "balancesheet_vip", "balancesheet.md", "statement", 5_000),
    DatasetSpec("cashflow", "cashflow_vip", "cashflow.md", "statement", 5_000),
    # daily_basic only returns one trade date per market-wide query, so it is
    # deliberately partitioned and requested one calendar date at a time.
    DatasetSpec(
        "daily_basic",
        "daily_basic",
        "daily_basic.md",
        "daily",
        6_000,
        KNOWN_DAILY_BASIC_MISSING_FIELDS,
    ),
)

DATASET_BY_NAME = {spec.name: spec for spec in DATASET_SPECS}
API_ALIAS_TO_DATASET = {spec.api_name: spec.name for spec in DATASET_SPECS}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_yyyymmdd(value: str, label: str = "date") -> dt.date:
    try:
        parsed = dt.datetime.strptime(value, DATE_FORMAT).date()
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must use YYYYMMDD, got {value!r}") from exc
    if parsed.strftime(DATE_FORMAT) != value:
        raise ConfigurationError(f"{label} must use zero-padded YYYYMMDD, got {value!r}")
    return parsed


def format_date(value: dt.date) -> str:
    return value.strftime(DATE_FORMAT)


def quarter_ends(start_date: dt.date, end_date: dt.date) -> list[str]:
    """Return quarter-end strings within the closed date interval."""
    if start_date > end_date:
        raise ConfigurationError("start_date must not be after end_date")
    result: list[str] = []
    for year in range(start_date.year, end_date.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            value = dt.date(year, month, day)
            if start_date <= value <= end_date:
                result.append(format_date(value))
    return result


def event_year_ranges(start_date: dt.date, end_date: dt.date) -> list[tuple[str, str, str]]:
    """Return (year, query_start, query_end) partitions for an interval."""
    if start_date > end_date:
        raise ConfigurationError("start_date must not be after end_date")
    result: list[tuple[str, str, str]] = []
    for year in range(start_date.year, end_date.year + 1):
        lower = max(start_date, dt.date(year, 1, 1))
        upper = min(end_date, dt.date(year, 12, 31))
        result.append((str(year), format_date(lower), format_date(upper)))
    return result


def calendar_dates(start_date: dt.date, end_date: dt.date) -> list[str]:
    """Return every calendar date in the closed interval as YYYYMMDD."""
    if start_date > end_date:
        raise ConfigurationError("start_date must not be after end_date")
    day_count = (end_date - start_date).days + 1
    return [
        format_date(start_date + dt.timedelta(days=offset))
        for offset in range(day_count)
    ]


__all__ = [
    "LOGGER",
    "SCRIPT_DIR",
    "DEFAULT_DOCS_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_API_URL",
    "DATE_FORMAT",
    "MANIFEST_VERSION",
    "CATALOG_VERSION",
    "KNOWN_INCOME_MISSING_FIELDS",
    "KNOWN_EXPRESS_MISSING_FIELDS",
    "KNOWN_DAILY_BASIC_MISSING_FIELDS",
    "SyncError",
    "ConfigurationError",
    "SourceSchemaError",
    "SchemaDriftError",
    "DatasetRunError",
    "OverwriteGuardError",
    "FieldSpec",
    "DatasetSpec",
    "FetchResult",
    "PartitionResult",
    "DATASET_SPECS",
    "DATASET_BY_NAME",
    "API_ALIAS_TO_DATASET",
    "now_iso",
    "parse_yyyymmdd",
    "format_date",
    "quarter_ends",
    "event_year_ranges",
    "calendar_dates",
]
