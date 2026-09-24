"""Manifests, partition orchestration and offline verification."""

from __future__ import annotations

import copy
import datetime as dt
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
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
    DatasetRunError,
    DatasetSpec,
    FieldSpec,
    LOGGER,
    MANIFEST_VERSION,
    OverwriteGuardError,
    SchemaDriftError,
    SourceSchemaError,
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
from _storage import (
    archive_lock, atomic_write_json, atomic_write_parquet, commit_partition,
    load_json, recover_transactions, sha256_file,
)


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


def check_overwrite_safety(
    spec: DatasetSpec,
    key: str,
    new_frame: pd.DataFrame,
    old_entry: Mapping[str, Any] | None,
    allow_shrink: bool,
) -> None:
    """Permit corrections/de-duplication; reject empty or >50% unique-row loss.

    This is a throttle sanity check, not a proof of source completeness.
    Callers pass the unique count of the SAME query window, not raw file size.
    """
    if old_entry is None or allow_shrink:
        return
    old_rows = int(old_entry.get("rows", 0))
    new_rows = len(new_frame.drop_duplicates())
    if new_rows == 0 and old_rows > 0:
        raise OverwriteGuardError(
            f"{spec.name} partition={key} new fetch returned 0 rows but existing "
            f"query has {old_rows}; use --allow-shrink after reviewing the source"
        )
    if new_rows < old_rows * 0.5:
        raise OverwriteGuardError(
            f"{spec.name} partition={key} unique row count shrank {old_rows} -> {new_rows}; "
            "refusing >50% loss (use --allow-shrink or --force to override)"
        )


def _run_partition_task(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    query: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    return frame.drop_duplicates(ignore_index=True), stats


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
    discovered_periods: Sequence[str] = (),
    historical_periods: Sequence[str] = (),
    refresh_recent: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    if spec.mode in {"statement", "period"}:
        all_periods = quarter_ends(archive_start, as_of)
        existing = set(manifest.get("partitions", {}))
        missing = [period for period in all_periods if period not in existing]
        recent = all_periods[-financial_lookback_quarters:] if refresh_recent else []
        selected = sorted(set(missing) | set(recent) | set(discovered_periods) | set(historical_periods))
        return [(period, {"period": period}) for period in selected]

    if spec.mode == "event":
        overlap_start = update_window_start(manifest, archive_start, as_of, event_lookback_days)
        targets: list[tuple[str, dict[str, Any]]] = []
        for year, lower, upper in event_year_ranges(archive_start, as_of):
            old = manifest.get("partitions", {}).get(year)
            if not old:
                targets.append((year, {"start_date": lower, "end_date": upper}))
            elif upper >= format_date(overlap_start):
                targets.append((year, {"start_date": max(lower, format_date(overlap_start)), "end_date": upper}))
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


def update_window_start(
    manifest: Mapping[str, Any], archive_start: dt.date, as_of: dt.date, lookback_days: int,
) -> dt.date:
    state = manifest.get("update_state", {})
    # Keep the original window on a failed first migration, even if some
    # successful partition commits have already extended range_end.
    pending_start = state.get("pending_window_start")
    through = state.get("discovery_through") or manifest.get("range_end")
    anchor = min(as_of, parse_yyyymmdd(through)) if through else as_of
    start = max(archive_start, anchor - dt.timedelta(days=lookback_days - 1))
    if pending_start:
        start = min(start, parse_yyyymmdd(pending_start))
    return start


def historical_rotation(
    manifest: Mapping[str, Any], archive_start: dt.date, as_of: dt.date,
    recent_quarters: int, count: int,
) -> list[str]:
    all_periods = quarter_ends(archive_start, as_of)
    eligible = all_periods[:-recent_quarters] if recent_quarters else all_periods
    candidates = [p for p in eligible if p in manifest.get("partitions", {})]
    cursor = manifest.get("update_state", {}).get("history_cursor", "")
    ordered = [p for p in candidates if p > cursor] + [p for p in candidates if p <= cursor]
    return ordered[:count]


def _prepare_candidate(
    output_dir: Path, spec: DatasetSpec, key: str, query: dict,
    frame: pd.DataFrame, stats: dict, force: bool, allow_shrink: bool, run_id: str,
) -> tuple[pd.DataFrame, dict, dict]:
    target = partition_file(output_dir, spec, key)
    old = pd.read_parquet(target) if target.exists() else None
    stored_query = dict(query)
    if spec.mode == "industry":
        from patch_industry_data import PATCH_ID, audit_membership, patch_table
        table = pa.Table.from_pandas(frame, preserve_index=False)
        scope = {"range_start": query["start_date"], "range_end": query["end_date"]}
        if spec.name == "index_member_all":
            cleaned, report = patch_table(table, scope)
            stats["industry_patch_id"] = PATCH_ID
        else:
            cleaned = table
            report = {"action": "review_only", "audit": audit_membership(table, scope), "blockers": []}
        report_path = output_dir / "_runs" / run_id / f"{spec.name}_audit.json"
        atomic_write_json(report_path, report)
        stats["industry_audit"] = str(report_path.relative_to(output_dir))
        if report["blockers"]:
            raise SourceSchemaError("industry patch requires review: " + "; ".join(report["blockers"]))
        frame = cleaned.to_pandas().drop_duplicates(ignore_index=True)
        audit = report.get("audit_after", report.get("audit", {}))
        if audit.get("conflicts"):
            stats.setdefault("warnings", []).append(f"industry audit: {len(audit['conflicts'])} unresolved conflicts; see {report_path}")
        LOGGER.info("%s industry audit: %s", spec.name, report_path)
    comparison = old
    if old is not None and spec.mode == "event":
        # Replace the complete fetched date window, including removals, while
        # retaining every row outside it in the existing yearly partition.
        mask = old["ann_date"].between(query["start_date"], query["end_date"]).fillna(False)
        comparison = old.loc[mask]
        manifest = load_json(manifest_path(output_dir, spec)) or {}
        old_query = manifest.get("partitions", {}).get(key, {}).get("query", {})
        stored_query = {
            "start_date": min(query["start_date"], old_query.get("start_date", query["start_date"])),
            "end_date": max(query["end_date"], old_query.get("end_date", query["end_date"])),
        }
        stats["refresh_window"] = dict(query)
    if not force and comparison is not None:
        check_overwrite_safety(spec, key, frame, {"rows": len(comparison.drop_duplicates())}, allow_shrink)
    if old is not None and spec.mode == "event":
        outside = old.loc[~mask]
        if not outside.empty:
            frame = pd.concat([outside, frame], ignore_index=True).drop_duplicates(ignore_index=True)
    return frame, stored_query, stats


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
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or uuid.uuid4().hex
    manifest = prepare_manifest(
        output_dir, spec, fields, docs_dir, fetcher.api_url, force,
        range_start, range_end,
    )
    expected_hash = schema_hash(fields)
    pending = []
    for key, query in targets:
        entry = manifest["partitions"].get(key, {})
        current_policy = entry.get("pagination_overlap_rows") == fetcher.page_overlap and entry.get("deduplicated")
        if spec.mode == "industry":
            current_policy = current_policy and bool(entry.get("industry_audit"))
            if spec.name == "index_member_all":
                from patch_industry_data import PATCH_ID
                current_policy = current_policy and entry.get("industry_patch_id") == PATCH_ID
        if resume and not force and current_policy and partition_is_valid(output_dir, entry, query, expected_hash):
            LOGGER.info("%s partition=%s already verified; skipping", spec.name, key)
            continue
        pending.append((key, query))

    failures: list[tuple[str, Exception]] = []
    effective_workers = max(1, min(workers, len(pending) or 1))
    if spec.mode == "statement":
        effective_workers = min(effective_workers, 4)
    # Workers only fetch. One writer commits backups, data and manifests in order.
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_run_partition_task, fetcher, spec, fields, query): (key, query)
            for key, query in pending
        }
        for future in as_completed(futures):
            key, query = futures[future]
            try:
                frame, stats = future.result()
                frame, stored_query, stats = _prepare_candidate(
                    output_dir, spec, key, query, frame, stats, force, allow_shrink, run_id,
                )
            except Exception as exc:
                failures.append((key, exc))
                LOGGER.error("%s partition=%s failed: %s", spec.name, key, exc)
                continue
            entry = {
                **stats, "relative_path": str(partition_file(output_dir, spec, key).relative_to(output_dir)),
                "query": stored_query, "schema_hash": expected_hash, "fetched_at": now_iso(),
                "pagination_overlap_rows": fetcher.page_overlap,
            }
            missing = entry.get("missing_fields", [])
            if missing:
                warning = "source omitted documented fields: " + ", ".join(sorted(missing))
                entry["warnings"] = sorted(set(entry.get("warnings", [])) | {warning})
            updated = copy.deepcopy(manifest)
            updated["range_start"] = min(filter(None, [manifest.get("range_start"), range_start]))
            updated["range_end"] = max(filter(None, [manifest.get("range_end"), range_end]))
            updated["updated_at"] = now_iso()
            # Commit errors must abort this dataset: a pending journal may need
            # recovery before any later manifest write is safe.
            manifest = commit_partition(
                output_dir, partition_file(output_dir, spec, key), manifest_path(output_dir, spec),
                updated, key, frame, entry, run_id,
            )
    if failures:
        detail = "; ".join(f"{key}: {exc}" for key, exc in failures[:10])
        raise DatasetRunError(
            f"{spec.name} failed {len(failures)} partition(s); successful partitions were kept. {detail}"
        )
    if not pending:
        atomic_write_json(manifest_path(output_dir, spec), manifest)
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
        except Exception as exc:
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
        except Exception as exc:
            errors.append(str(exc))
    elif spec.mode == "daily":
        frame = pd.read_parquet(path, columns=["trade_date"])
        try:
            validate_daily_frame(
                frame,
                query["trade_date"],
                spec.name,
            )
        except Exception as exc:
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
        except Exception as exc:
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
    "check_overwrite_safety",
    "targets_for_range",
    "update_targets",
    "process_dataset",
    "update_window_start",
    "historical_rotation",
    "archive_lock",
    "recover_transactions",
    "verify_dataset",
    "write_catalog",
]
