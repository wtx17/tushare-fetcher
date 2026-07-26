"""Documentation parsing, frame normalization and content validation.

Pure data logic: no network access and no filesystem writes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from _models import (
    ConfigurationError,
    DATASET_SPECS,
    FieldSpec,
    LOGGER,
    SourceSchemaError,
    parse_yyyymmdd,
)


def parse_output_fields(doc_path: Path) -> list[FieldSpec]:
    """Parse the first Markdown output-parameter table in a local API doc."""
    if not doc_path.is_file():
        raise ConfigurationError(f"documentation file not found: {doc_path}")

    fields: list[FieldSpec] = []
    active = False
    heading_seen = False
    for raw_line in doc_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if re.match(r"^#{2,6}\s*输出参数\s*$", line):
            active = True
            heading_seen = True
            continue
        if active and re.match(r"^#{1,6}\s+", line):
            break
        if not active or not line.startswith("|"):
            continue

        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        name, source_type = cells[0], cells[1]
        if name == "名称" or re.fullmatch(r":?-+:?", name or ""):
            continue
        if not name:
            continue
        fields.append(FieldSpec(name=name, source_type=source_type or "str"))

    if not heading_seen:
        raise ConfigurationError(f"no 输出参数 section found in {doc_path}")
    if not fields:
        raise ConfigurationError(f"no output fields parsed from {doc_path}")

    names = [field.name for field in fields]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigurationError(f"duplicate output fields in {doc_path}: {duplicates}")
    return fields


def schema_payload(fields: Sequence[FieldSpec]) -> list[dict[str, str]]:
    return [
        {
            "name": field.name,
            "source_type": field.source_type,
            "parquet_type": field.pandas_dtype,
        }
        for field in fields
    ]


def schema_hash(fields: Sequence[FieldSpec]) -> str:
    payload = json.dumps(schema_payload(fields), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def empty_frame(fields: Sequence[FieldSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        {field.name: pd.Series([], dtype=field.pandas_dtype) for field in fields}
    )


def _coerce_numeric(series: pd.Series, field: FieldSpec) -> pd.Series:
    blank = series.astype("string").str.strip().eq("").fillna(False)
    converted = pd.to_numeric(series.mask(blank, pd.NA), errors="coerce")
    invalid = series.notna() & ~blank & converted.isna()
    if invalid.any():
        examples = series.loc[invalid].astype("string").drop_duplicates().head(5).tolist()
        raise SourceSchemaError(
            f"field {field.name!r} contains non-numeric values, examples={examples}"
        )
    if field.pandas_dtype == "Int64":
        non_null = converted.dropna()
        fractional = (non_null % 1).abs() > 1e-12
        if fractional.any():
            examples = non_null.loc[fractional].head(5).tolist()
            raise SourceSchemaError(
                f"integer field {field.name!r} contains fractional values, examples={examples}"
            )
    return converted.astype(field.pandas_dtype)


def normalize_frame(
    frame: pd.DataFrame,
    fields: Sequence[FieldSpec],
    allowed_missing_fields: Iterable[str] = (),
    dataset_name: str = "",
) -> tuple[pd.DataFrame, set[str]]:
    """Align a response to the documented order and stable nullable dtypes."""
    expected_names = [field.name for field in fields]
    allowed = set(allowed_missing_fields)
    missing = {name for name in expected_names if name not in frame.columns}

    # An empty Tushare response can legitimately have no columns at all.  It
    # does not prove that the source lacks the documented schema.
    reported_missing = missing if len(frame) else set()
    unexpected_missing = reported_missing - allowed
    if unexpected_missing:
        raise SourceSchemaError(
            f"{dataset_name or 'dataset'} response is missing documented fields: "
            f"{sorted(unexpected_missing)}"
        )

    normalized = frame.copy()
    field_by_name = {field.name: field for field in fields}
    for name in missing:
        normalized[name] = pd.Series(pd.NA, index=normalized.index, dtype=field_by_name[name].pandas_dtype)

    normalized = normalized.loc[:, expected_names]
    for field in fields:
        series = normalized[field.name]
        if field.pandas_dtype in {"Float64", "Int64"}:
            normalized[field.name] = _coerce_numeric(series, field)
        else:
            normalized[field.name] = series.astype("string")
    return normalized, reported_missing


def row_hashes(frame: pd.DataFrame) -> list[int]:
    if frame.empty:
        return []
    values = pd.util.hash_pandas_object(frame, index=False).astype("uint64")
    return [int(value) for value in values.tolist()]


def validate_period_frame(frame: pd.DataFrame, period: str, dataset_name: str) -> None:
    if frame.empty:
        return
    values = frame["end_date"].dropna().astype("string")
    invalid = values.ne(period)
    if invalid.any():
        examples = values.loc[invalid].drop_duplicates().head(5).tolist()
        raise SourceSchemaError(
            f"{dataset_name} period={period} returned other end_date values: {examples}"
        )


def period_is_historical(period: str, grace_days: int = 120) -> bool:
    """Whether a report period is old enough that market-wide data must exist."""
    period_date = parse_yyyymmdd(period, "period")
    return period_date <= dt.date.today() - dt.timedelta(days=grace_days)


def validate_event_frame(
    frame: pd.DataFrame,
    query_start: str,
    query_end: str,
    dataset_name: str,
) -> None:
    if frame.empty:
        return
    values = frame["ann_date"].astype("string")
    missing = values.isna()
    invalid = missing | values.lt(query_start) | values.gt(query_end)
    if invalid.any():
        examples = values.loc[invalid].drop_duplicates().head(5).tolist()
        raise SourceSchemaError(
            f"{dataset_name} returned ann_date values outside {query_start}..{query_end}: {examples}"
        )


def industry_overlap_mask(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.Series:
    if frame.empty:
        return pd.Series([], index=frame.index, dtype="bool")
    in_date = frame["in_date"].astype("string")
    out_date = frame["out_date"].astype("string")
    starts_before_end = in_date.isna() | in_date.le(end_date)
    ends_after_start = out_date.isna() | out_date.eq("") | out_date.ge(start_date)
    return (starts_before_end & ends_after_start).fillna(False)


def load_all_fields(docs_dir: Path) -> dict[str, list[FieldSpec]]:
    result = {
        spec.name: parse_output_fields(docs_dir / spec.doc_name)
        for spec in DATASET_SPECS
    }
    total = sum(len(fields) for fields in result.values())
    LOGGER.info("parsed %s documented output fields across %s datasets", total, len(result))
    return result


__all__ = [
    "parse_output_fields",
    "schema_payload",
    "schema_hash",
    "empty_frame",
    "normalize_frame",
    "row_hashes",
    "validate_period_frame",
    "period_is_historical",
    "validate_event_frame",
    "industry_overlap_mask",
    "load_all_fields",
]
