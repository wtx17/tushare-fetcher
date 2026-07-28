"""Atomic storage, manifests, partition orchestration and offline verification."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.dataset as pyarrow_dataset
import pyarrow.parquet as pyarrow_parquet

from _fetch import (
    ApiFetcher,
    fetch_daily_partition,
    fetch_event_partition,
    fetch_industry_partition,
    fetch_period_partition,
    fetch_statement_partition,
)
from _models import (
    CATALOG_VERSION,
    ConfigurationError,
    DatasetRunError,
    DatasetSpec,
    FieldSpec,
    LOGGER,
    MANIFEST_VERSION,
    OverwriteGuardError,
    PartitionResult,
    SchemaDriftError,
    SourceSchemaError,
    SyncError,
    calendar_dates,
    event_year_ranges,
    format_date,
    now_iso,
    parse_yyyymmdd,
    quarter_ends,
)
from _schema import (
    industry_overlap_mask,
    schema_hash,
    schema_payload,
    validate_daily_frame,
    validate_event_frame,
    validate_period_frame,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(
            temporary,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        metadata = pyarrow_parquet.read_metadata(temporary)
        if metadata.num_rows != len(frame):
            raise SyncError(
                f"Parquet row-count mismatch for {path}: "
                f"expected {len(frame)}, wrote {metadata.num_rows}"
            )
        checksum = sha256_file(temporary)
        size = temporary.stat().st_size
        os.replace(temporary, path)
        return checksum, size
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON root must be an object: {path}")
    return value


def dataset_root(output_dir: Path, spec: DatasetSpec) -> Path:
    return output_dir / spec.name


def manifest_path(output_dir: Path, spec: DatasetSpec) -> Path:
    return dataset_root(output_dir, spec) / "_manifest.json"


def catalog_path(output_dir: Path) -> Path:
    return output_dir / "_catalog.json"


def partition_file(output_dir: Path, spec: DatasetSpec, key: str) -> Path:
    root = dataset_root(output_dir, spec)
    if spec.mode in {"statement", "period"}:
        return root / "period" / key / "data.parquet"
    if spec.mode == "event":
        return root / "ann_year" / key / "data.parquet"
    if spec.mode == "daily":
        return root / "trade_date" / key / "data.parquet"
    if spec.mode == "industry":
        return root / "data.parquet"
    raise AssertionError(f"unknown dataset mode: {spec.mode}")


def new_manifest(
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    docs_dir: Path,
    api_url: str,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": spec.name,
        "api_name": spec.api_name,
        "doc_path": str((docs_dir / spec.doc_name).resolve()),
        "source_url": api_url.rstrip("/"),
        "schema_hash": schema_hash(fields),
        "fields": schema_payload(fields),
        "allowed_missing_fields": sorted(spec.allowed_missing_fields),
        "range_start": None,
        "range_end": None,
        "partitions": {},
        "updated_at": None,
    }


def prepare_manifest(
    output_dir: Path,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    docs_dir: Path,
    api_url: str,
    force: bool,
    requested_start: str,
    requested_end: str,
) -> dict[str, Any]:
    existing = load_json(manifest_path(output_dir, spec))
    if existing is None:
        return new_manifest(spec, fields, docs_dir, api_url)

    expected_hash = schema_hash(fields)
    if existing.get("schema_hash") != expected_hash:
        if not force:
            raise SchemaDriftError(
                f"{spec.name} documentation schema changed; rerun backfill with --force "
                "over the entire stored range"
            )
        old_start = existing.get("range_start")
        old_end = existing.get("range_end")
        if old_start and requested_start > old_start:
            raise SchemaDriftError(
                f"{spec.name} --force range starts after existing data ({old_start})"
            )
        if old_end and requested_end < old_end:
            raise SchemaDriftError(
                f"{spec.name} --force range ends before existing data ({old_end})"
            )
        return new_manifest(spec, fields, docs_dir, api_url)

    expected_names = [field.name for field in fields]
    manifest_names = [field.get("name") for field in existing.get("fields", [])]
    if expected_names != manifest_names:
        raise SchemaDriftError(f"{spec.name} manifest field order differs from documentation")
    prepared = copy.deepcopy(existing)
    prepared["allowed_missing_fields"] = sorted(spec.allowed_missing_fields)
    return prepared


def partition_is_valid(
    output_dir: Path,
    manifest_entry: Mapping[str, Any] | None,
    query: Mapping[str, Any],
    expected_schema_hash: str,
) -> bool:
    if not manifest_entry or manifest_entry.get("query") != dict(query):
        return False
    relative_path = manifest_entry.get("relative_path")
    if not isinstance(relative_path, str):
        return False
    path = output_dir / relative_path
    if not path.is_file():
        return False
    if manifest_entry.get("schema_hash", expected_schema_hash) != expected_schema_hash:
        return False
    if path.stat().st_size != manifest_entry.get("bytes"):
        return False
    return sha256_file(path) == manifest_entry.get("sha256")


def build_partition_result(
    output_dir: Path,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    key: str,
    query: dict[str, Any],
    frame: pd.DataFrame,
    stats: Mapping[str, Any],
) -> PartitionResult:
    target = partition_file(output_dir, spec, key)
    checksum, size = atomic_write_parquet(frame, target)
    relative = str(target.relative_to(output_dir))
    return PartitionResult(
        key=key,
        relative_path=relative,
        query=query,
        rows=len(frame),
        raw_rows=int(stats.get("raw_rows", len(frame))),
        pages=int(stats.get("pages", 0)),
        last_page_rows=int(stats.get("last_page_rows", 0)),
        cross_page_duplicates=int(stats.get("cross_page_duplicates", 0)),
        missing_fields=list(stats.get("missing_fields", [])),
        warnings=list(stats.get("warnings", [])),
        sha256=checksum,
        bytes=size,
        fetched_at=now_iso(),
        subqueries=stats.get("subqueries"),
    )


def check_overwrite_safety(
    spec: DatasetSpec,
    key: str,
    new_frame: pd.DataFrame,
    old_entry: Mapping[str, Any] | None,
    allow_shrink: bool,
) -> None:
    """Refuse to overwrite an existing partition with fewer rows.

    For a fixed query the upstream row count only grows over time (late
    filings, restatements add rows), so any shrink signals a likely throttle
    or upstream regression rather than a legitimate change.  An empty fetch
    over a previously non-empty partition is always refused, even with
    ``allow_shrink``.
    """
    if old_entry is None:
        return
    old_rows = int(old_entry.get("rows", 0))
    new_rows = len(new_frame)
    if new_rows == 0 and old_rows > 0:
        raise OverwriteGuardError(
            f"{spec.name} partition={key} new fetch returned 0 rows but existing "
            f"partition has {old_rows}; refusing overwrite (use --force to refresh)"
        )
    if not allow_shrink and new_rows < old_rows:
        raise OverwriteGuardError(
            f"{spec.name} partition={key} row count shrank {old_rows} -> {new_rows}; "
            "refusing overwrite (use --allow-shrink or --force to override)"
        )


def _run_partition_task(
    fetcher: ApiFetcher,
    output_dir: Path,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    key: str,
    query: dict[str, Any],
    old_entry: Mapping[str, Any] | None,
    force: bool,
    allow_shrink: bool,
) -> PartitionResult:
    if spec.mode == "statement":
        frame, stats = fetch_statement_partition(fetcher, spec, fields, query["period"])
    elif spec.mode == "period":
        frame, stats = fetch_period_partition(fetcher, spec, fields, query["period"])
    elif spec.mode == "event":
        frame, stats = fetch_event_partition(
            fetcher,
            spec,
            fields,
            query["start_date"],
            query["end_date"],
        )
    elif spec.mode == "daily":
        frame, stats = fetch_daily_partition(
            fetcher,
            spec,
            fields,
            query["trade_date"],
        )
    elif spec.mode == "industry":
        frame, stats = fetch_industry_partition(
            fetcher,
            spec,
            fields,
            query["start_date"],
            query["end_date"],
        )
    else:
        raise AssertionError(f"unknown mode {spec.mode}")
    # When refreshing an existing partition without --force, refuse to write a
    # frame that is smaller than what is already stored.  The old file and
    # manifest entry are left untouched because this function raises before
    # build_partition_result performs any write.
    if not force:
        check_overwrite_safety(spec, key, frame, old_entry, allow_shrink)
    return build_partition_result(output_dir, spec, fields, key, query, frame, stats)


def targets_for_range(
    spec: DatasetSpec,
    start_date: dt.date,
    end_date: dt.date,
) -> list[tuple[str, dict[str, Any]]]:
    if spec.mode in {"statement", "period"}:
        return [(period, {"period": period}) for period in quarter_ends(start_date, end_date)]
    if spec.mode == "event":
        return [
            (year, {"start_date": lower, "end_date": upper})
            for year, lower, upper in event_year_ranges(start_date, end_date)
        ]
    if spec.mode == "daily":
        return [
            (trade_date, {"trade_date": trade_date})
            for trade_date in calendar_dates(start_date, end_date)
        ]
    if spec.mode == "industry":
        return [
            (
                "all",
                {"start_date": format_date(start_date), "end_date": format_date(end_date)},
            )
        ]
    raise AssertionError(f"unknown mode {spec.mode}")


def update_targets(
    spec: DatasetSpec,
    manifest: Mapping[str, Any],
    archive_start: dt.date,
    as_of: dt.date,
    financial_lookback_quarters: int,
    event_lookback_days: int,
    daily_lookback_days: int = 7,
) -> list[tuple[str, dict[str, Any]]]:
    if spec.mode in {"statement", "period"}:
        all_periods = quarter_ends(archive_start, as_of)
        existing = set(manifest.get("partitions", {}))
        missing = [period for period in all_periods if period not in existing]
        recent = all_periods[-financial_lookback_quarters:]
        selected = sorted(set(missing) | set(recent))
        return [(period, {"period": period}) for period in selected]

    if spec.mode == "event":
        overlap_start = max(archive_start, as_of - dt.timedelta(days=event_lookback_days))
        ranges = event_year_ranges(overlap_start, as_of)
        targets: list[tuple[str, dict[str, Any]]] = []
        for year, _lower, upper in ranges:
            year_start = max(archive_start, dt.date(int(year), 1, 1))
            targets.append(
                (
                    year,
                    {"start_date": format_date(year_start), "end_date": upper},
                )
            )
        return targets

    if spec.mode == "daily":
        all_dates = calendar_dates(archive_start, as_of)
        existing = set(manifest.get("partitions", {}))
        missing = [trade_date for trade_date in all_dates if trade_date not in existing]
        recent = all_dates[-daily_lookback_days:]
        selected = sorted(set(missing) | set(recent))
        return [
            (trade_date, {"trade_date": trade_date})
            for trade_date in selected
        ]

    if spec.mode == "industry":
        return [
            (
                "all",
                {"start_date": format_date(archive_start), "end_date": format_date(as_of)},
            )
        ]
    raise AssertionError(f"unknown mode {spec.mode}")


def process_dataset(
    fetcher: ApiFetcher,
    output_dir: Path,
    docs_dir: Path,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    targets: Sequence[tuple[str, dict[str, Any]]],
    range_start: str,
    range_end: str,
    workers: int,
    force: bool,
    resume: bool,
    allow_shrink: bool = False,
) -> dict[str, Any]:
    manifest = prepare_manifest(
        output_dir,
        spec,
        fields,
        docs_dir,
        fetcher.api_url,
        force=force,
        requested_start=range_start,
        requested_end=range_end,
    )
    manifest.setdefault("partitions", {})
    # Persist policy metadata even when every data partition is reused.  This
    # keeps an older manifest auditable after an explicit proxy whitelist
    # change without forcing an otherwise unnecessary historical redownload.
    atomic_write_json(manifest_path(output_dir, spec), manifest)
    expected_hash = schema_hash(fields)

    pending: list[tuple[str, dict[str, Any], Mapping[str, Any] | None]] = []
    for key, query in targets:
        existing_entry = manifest["partitions"].get(key)
        if resume and not force and partition_is_valid(
            output_dir,
            existing_entry,
            query,
            expected_hash,
        ):
            LOGGER.info("%s partition=%s already verified; skipping", spec.name, key)
            continue
        pending.append((key, query, existing_entry))

    if not pending:
        LOGGER.info("%s has no pending partitions", spec.name)
    failures: list[tuple[str, BaseException]] = []
    completed = 0

    # A statement partition already performs 12 sequential calls and retains
    # wider frames in memory, so cap it slightly below the general worker count.
    effective_workers = max(1, min(workers, len(pending) or 1))
    if spec.mode == "statement":
        effective_workers = min(effective_workers, 4)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_map: dict[Future[PartitionResult], tuple[str, dict[str, Any]]] = {
            executor.submit(
                _run_partition_task,
                fetcher,
                output_dir,
                spec,
                fields,
                key,
                query,
                old_entry,
                force,
                allow_shrink,
            ): (key, query)
            for key, query, old_entry in pending
        }
        for future in as_completed(future_map):
            key, _query = future_map[future]
            try:
                result = future.result()
            except BaseException as exc:
                failures.append((key, exc))
                LOGGER.error("%s partition=%s failed: %s", spec.name, key, exc)
                continue

            entry = result.to_manifest_entry()
            entry["schema_hash"] = expected_hash
            manifest["partitions"][result.key] = entry
            old_start = manifest.get("range_start")
            old_end = manifest.get("range_end")
            manifest["range_start"] = min(filter(None, [old_start, range_start]))
            manifest["range_end"] = max(filter(None, [old_end, range_end]))
            manifest["updated_at"] = now_iso()
            atomic_write_json(manifest_path(output_dir, spec), manifest)
            completed += 1
            LOGGER.info(
                "%s partition=%s complete (%s/%s pending) rows=%s pages=%s bytes=%s",
                spec.name,
                result.key,
                completed,
                len(pending),
                result.rows,
                result.pages,
                result.bytes,
            )

    if failures:
        detail = "; ".join(f"{key}: {exc}" for key, exc in failures[:10])
        raise DatasetRunError(
            f"{spec.name} failed {len(failures)} partition(s); successful partitions were kept. {detail}"
        )
    return manifest


def _verify_partition_dates(
    path: Path,
    spec: DatasetSpec,
    entry: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    query = entry.get("query", {})
    if spec.mode in {"statement", "period"}:
        frame = pd.read_parquet(path, columns=["end_date"])
        try:
            validate_period_frame(frame, query["period"], spec.name)
        except BaseException as exc:
            errors.append(str(exc))
    elif spec.mode == "event":
        frame = pd.read_parquet(path, columns=["ann_date"])
        try:
            validate_event_frame(
                frame,
                query["start_date"],
                query["end_date"],
                spec.name,
            )
        except BaseException as exc:
            errors.append(str(exc))
    elif spec.mode == "daily":
        frame = pd.read_parquet(path, columns=["trade_date"])
        try:
            validate_daily_frame(
                frame,
                query["trade_date"],
                spec.name,
            )
        except BaseException as exc:
            errors.append(str(exc))
    elif spec.mode == "industry":
        frame = pd.read_parquet(path, columns=["in_date", "out_date"])
        if not industry_overlap_mask(frame, query["start_date"], query["end_date"]).all():
            errors.append(f"{spec.name} contains industry rows outside the requested overlap window")
    return errors


def verify_dataset(
    output_dir: Path,
    docs_dir: Path,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    catalog: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    path = manifest_path(output_dir, spec)
    manifest = load_json(path)
    if manifest is None:
        return {}, [f"missing manifest: {path}"]

    expected_names = [field.name for field in fields]
    manifest_names = [field.get("name") for field in manifest.get("fields", [])]
    if manifest.get("schema_hash") != schema_hash(fields):
        errors.append(f"{spec.name} manifest schema hash differs from documentation")
    if manifest_names != expected_names:
        errors.append(f"{spec.name} manifest field order differs from documentation")
    if manifest.get("allowed_missing_fields", []) != sorted(spec.allowed_missing_fields):
        errors.append(f"{spec.name} manifest missing-field policy differs from the sync policy")

    partitions = manifest.get("partitions", {})
    if not isinstance(partitions, dict) or not partitions:
        errors.append(f"{spec.name} has no manifest partitions")
        partitions = {}

    total_rows = 0
    total_bytes = 0
    empty_partitions = 0
    warning_count = 0
    for key, entry in sorted(partitions.items()):
        relative = entry.get("relative_path")
        if not isinstance(relative, str):
            errors.append(f"{spec.name} partition={key} has no relative_path")
            continue
        parquet_path = output_dir / relative
        if not parquet_path.is_file():
            errors.append(f"{spec.name} partition={key} file is missing: {parquet_path}")
            continue
        if parquet_path.stat().st_size != entry.get("bytes"):
            errors.append(f"{spec.name} partition={key} byte size differs from manifest")
        if sha256_file(parquet_path) != entry.get("sha256"):
            errors.append(f"{spec.name} partition={key} checksum differs from manifest")

        parquet = pyarrow_parquet.ParquetFile(parquet_path)
        parquet_names = parquet.schema_arrow.names
        if parquet_names != expected_names:
            errors.append(f"{spec.name} partition={key} Parquet schema/order differs")
        rows = parquet.metadata.num_rows
        if rows != entry.get("rows"):
            errors.append(f"{spec.name} partition={key} row count differs from manifest")
        total_rows += rows
        total_bytes += parquet_path.stat().st_size
        empty_partitions += int(rows == 0)
        warning_count += len(entry.get("warnings", []))
        if entry.get("last_page_rows", 0) >= spec.page_size and spec.mode != "statement":
            errors.append(f"{spec.name} partition={key} pagination did not record a terminal short page")
        if spec.mode == "statement":
            subqueries = entry.get("subqueries", {})
            if set(subqueries) != {str(value) for value in range(1, 13)}:
                errors.append(f"{spec.name} partition={key} lacks all 12 report-type results")
            for report_type, stats in subqueries.items():
                if stats.get("last_page_rows", 0) >= spec.page_size:
                    errors.append(
                        f"{spec.name} partition={key} report_type={report_type} lacks a terminal page"
                    )
        errors.extend(_verify_partition_dates(parquet_path, spec, entry))

        missing = set(entry.get("missing_fields", []))
        unapproved = missing - spec.allowed_missing_fields
        if unapproved:
            errors.append(
                f"{spec.name} partition={key} has unapproved missing fields: "
                f"{sorted(unapproved)}"
            )
        if missing:
            expected_warning = (
                "source omitted documented fields: " + ", ".join(sorted(missing))
            )
            if expected_warning not in entry.get("warnings", []):
                errors.append(
                    f"{spec.name} partition={key} missing-field warning is absent from manifest"
                )
        if missing and rows:
            check = pd.read_parquet(parquet_path, columns=sorted(missing))
            non_null = [name for name in check if check[name].notna().any()]
            if non_null:
                errors.append(
                    f"{spec.name} partition={key} manifest-declared missing columns contain "
                    f"values: {non_null}"
                )

    dataset_dir = dataset_root(output_dir, spec)
    temporary_files = [
        str(path)
        for path in dataset_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".tmp") or ".tmp." in path.name)
    ]
    if temporary_files:
        errors.append(f"{spec.name} has temporary files: {temporary_files[:5]}")

    if partitions:
        try:
            arrow_data = pyarrow_dataset.dataset(dataset_dir, format="parquet")
            if arrow_data.schema.names != expected_names:
                errors.append(f"{spec.name} directory schema is not the documented schema")
            if arrow_data.count_rows() != total_rows:
                errors.append(f"{spec.name} directory row count differs from manifest sum")
            # Loading one column through pandas proves the endpoint directory is
            # directly consumable without materializing every wide column.
            sample = pd.read_parquet(dataset_dir, columns=[expected_names[0]])
            if len(sample) != total_rows:
                errors.append(f"{spec.name} pandas directory read returned the wrong row count")
        except BaseException as exc:
            errors.append(f"{spec.name} directory-level Parquet read failed: {exc}")

    if catalog and manifest.get("range_start") and manifest.get("range_end"):
        archive_start = parse_yyyymmdd(catalog["archive_start"], "catalog archive_start")
        archive_end = parse_yyyymmdd(catalog["archive_end"], "catalog archive_end")
        if spec.mode in {"statement", "period"}:
            expected_keys = set(quarter_ends(archive_start, archive_end))
        elif spec.mode == "event":
            expected_keys = {year for year, _start, _end in event_year_ranges(archive_start, archive_end)}
        elif spec.mode == "daily":
            expected_keys = set(calendar_dates(archive_start, archive_end))
        else:
            expected_keys = {"all"}
        missing_keys = expected_keys - set(partitions)
        if missing_keys:
            errors.append(
                f"{spec.name} is missing expected partitions: {sorted(missing_keys)[:10]}"
            )

    summary = {
        "dataset": spec.name,
        "rows": total_rows,
        "partitions": len(partitions),
        "empty_partitions": empty_partitions,
        "bytes": total_bytes,
        "warnings": warning_count,
    }
    return summary, errors


def write_catalog(
    output_dir: Path,
    start_date: str,
    end_date: str,
    selected: Sequence[DatasetSpec],
    command: str,
) -> None:
    existing = load_json(catalog_path(output_dir)) or {}
    old_start = existing.get("archive_start")
    old_end = existing.get("archive_end")
    catalog = {
        "catalog_version": CATALOG_VERSION,
        "archive_start": min(filter(None, [old_start, start_date])),
        "archive_end": max(filter(None, [old_end, end_date])),
        "datasets": sorted(
            set(existing.get("datasets", [])) | {spec.name for spec in selected}
        ),
        "last_command": command,
        "updated_at": now_iso(),
    }
    atomic_write_json(catalog_path(output_dir), catalog)


__all__ = [
    "ApiFetcher",
    "sha256_file",
    "atomic_write_json",
    "atomic_write_parquet",
    "load_json",
    "dataset_root",
    "manifest_path",
    "catalog_path",
    "partition_file",
    "new_manifest",
    "prepare_manifest",
    "partition_is_valid",
    "build_partition_result",
    "check_overwrite_safety",
    "targets_for_range",
    "update_targets",
    "process_dataset",
    "verify_dataset",
    "write_catalog",
]
