#!/usr/bin/env python3
"""Download the locally documented Tushare datasets to partitioned Parquet.

The script deliberately keeps credentials outside the source tree.  Run it from
the ``quant_data`` conda environment, where ``TUSHARE_TOKEN`` is configured.

This file is the command-line entry point and a re-export facade so that the
historical ``import sync_tushare as sync`` surface keeps working; the real
logic lives in :mod:`_models`, :mod:`_schema`, :mod:`_fetch` and
:mod:`_pipeline`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

# Facade re-export (keeps `import sync_tushare as sync; sync.X` working).
from _models import *  # noqa: F401,F403
from _schema import *  # noqa: F401,F403
from _fetch import *  # noqa: F401,F403
from _pipeline import *  # noqa: F401,F403

from _fetch import ApiFetcher
from _models import API_ALIAS_TO_DATASET, ConfigurationError, DATASET_BY_NAME, DatasetSpec
from _pipeline import (
    process_dataset,
    targets_for_range,
    update_targets,
    verify_dataset,
    write_catalog,
)
from _schema import load_all_fields


LOGGER = logging.getLogger("sync_tushare")


def parse_api_selection(value: str) -> list[DatasetSpec]:
    if value.strip().lower() == "all":
        return list(DATASET_SPECS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ConfigurationError("--apis must be 'all' or a comma-separated dataset list")
    normalized: list[str] = []
    for name in requested:
        canonical = API_ALIAS_TO_DATASET.get(name, name)
        if canonical not in DATASET_BY_NAME:
            raise ConfigurationError(
                f"unknown dataset {name!r}; choices={sorted(DATASET_BY_NAME)}"
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return [DATASET_BY_NAME[name] for name in normalized]


def build_fetcher(args: argparse.Namespace) -> ApiFetcher:
    token = os.environ.get("TUSHARE_TOKEN", "")
    return ApiFetcher(
        token=token,
        api_url=args.api_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        request_interval=args.request_interval,
    )


def run_backfill(args: argparse.Namespace) -> int:
    start = parse_yyyymmdd(args.start_date, "--start-date")
    end = parse_yyyymmdd(args.end_date, "--end-date")
    if start > end:
        raise ConfigurationError("--start-date must not be after --end-date")
    selected = parse_api_selection(args.apis)
    fields_by_dataset = load_all_fields(args.docs_dir)
    fetcher = build_fetcher(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, BaseException]] = []
    for spec in selected:
        targets = targets_for_range(spec, start, end)
        LOGGER.info("starting %s with %s target partition(s)", spec.name, len(targets))
        try:
            process_dataset(
                fetcher=fetcher,
                output_dir=args.output_dir,
                docs_dir=args.docs_dir,
                spec=spec,
                fields=fields_by_dataset[spec.name],
                targets=targets,
                range_start=args.start_date,
                range_end=args.end_date,
                workers=args.workers,
                force=args.force,
                resume=True,
                allow_shrink=args.allow_shrink,
            )
        except BaseException as exc:
            failures.append((spec.name, exc))
            LOGGER.error("dataset %s did not complete: %s", spec.name, exc)

    if failures:
        detail = "; ".join(f"{name}: {exc}" for name, exc in failures)
        raise DatasetRunError(
            f"backfill incomplete for {len(failures)} dataset(s). Rerun the same command to resume. {detail}"
        )
    write_catalog(args.output_dir, args.start_date, args.end_date, selected, "backfill")
    return 0


def run_update(args: argparse.Namespace) -> int:
    catalog = load_json(catalog_path(args.output_dir))
    if not catalog:
        raise ConfigurationError(
            f"no catalog found at {catalog_path(args.output_dir)}; run backfill first"
        )
    archive_start = parse_yyyymmdd(catalog["archive_start"], "catalog archive_start")
    as_of = parse_yyyymmdd(args.as_of, "--as-of")
    if as_of < archive_start:
        raise ConfigurationError("--as-of precedes the stored archive start")

    selected = parse_api_selection(args.apis)
    fields_by_dataset = load_all_fields(args.docs_dir)
    fetcher = build_fetcher(args)
    failures: list[tuple[str, BaseException]] = []

    for spec in selected:
        existing_manifest = load_json(manifest_path(args.output_dir, spec))
        if existing_manifest is None:
            existing_manifest = new_manifest(
                spec,
                fields_by_dataset[spec.name],
                args.docs_dir,
                fetcher.api_url,
            )
        targets = update_targets(
            spec,
            existing_manifest,
            archive_start,
            as_of,
            args.financial_lookback_quarters,
            args.event_lookback_days,
        )
        LOGGER.info("updating %s with %s target partition(s)", spec.name, len(targets))
        try:
            process_dataset(
                fetcher=fetcher,
                output_dir=args.output_dir,
                docs_dir=args.docs_dir,
                spec=spec,
                fields=fields_by_dataset[spec.name],
                targets=targets,
                range_start=catalog["archive_start"],
                range_end=args.as_of,
                workers=args.workers,
                force=False,
                resume=False,
                allow_shrink=args.allow_shrink,
            )
        except BaseException as exc:
            failures.append((spec.name, exc))
            LOGGER.error("dataset %s update did not complete: %s", spec.name, exc)

    if failures:
        detail = "; ".join(f"{name}: {exc}" for name, exc in failures)
        raise DatasetRunError(f"update incomplete for {len(failures)} dataset(s). {detail}")
    write_catalog(
        args.output_dir,
        catalog["archive_start"],
        args.as_of,
        selected,
        "update",
    )
    return 0


def _smoke_params(spec: DatasetSpec, args: argparse.Namespace) -> dict[str, Any]:
    if spec.mode in {"statement", "period"}:
        params: dict[str, Any] = {"period": args.period}
        if spec.mode == "statement":
            params["report_type"] = "1"
        return params
    if spec.mode == "event":
        return {"start_date": args.event_start, "end_date": args.event_end}
    return {"is_new": "Y"}


def run_smoke(args: argparse.Namespace) -> int:
    selected = parse_api_selection(args.apis)
    fields_by_dataset = load_all_fields(args.docs_dir)
    fetcher = build_fetcher(args)
    failures: list[tuple[str, BaseException]] = []

    for spec in selected:
        fields = fields_by_dataset[spec.name]
        fields_csv = ",".join(field.name for field in fields)
        base = _smoke_params(spec, args)
        try:
            page0 = fetcher.query_page(
                spec.api_name,
                fields_csv,
                {**base, "limit": 10, "offset": 0},
            )
            page1 = fetcher.query_page(
                spec.api_name,
                fields_csv,
                {**base, "limit": 10, "offset": 10},
            )
            normalized0, missing0 = normalize_frame(
                page0,
                fields,
                spec.allowed_missing_fields,
                spec.name,
            )
            normalized1, missing1 = normalize_frame(
                page1,
                fields,
                spec.allowed_missing_fields,
                spec.name,
            )
            overlap = len(set(row_hashes(normalized0)) & set(row_hashes(normalized1)))
            LOGGER.info(
                "SMOKE %s page0=%s page1=%s overlap=%s missing=%s",
                spec.name,
                len(page0),
                len(page1),
                overlap,
                sorted(missing0 | missing1),
            )
        except BaseException as exc:
            failures.append((spec.name, exc))
            LOGGER.error("SMOKE %s failed: %s", spec.name, exc)
    if failures:
        raise DatasetRunError(
            "smoke test failures: " + "; ".join(f"{name}: {exc}" for name, exc in failures)
        )
    return 0


def run_verify(args: argparse.Namespace) -> int:
    selected = parse_api_selection(args.apis)
    fields_by_dataset = load_all_fields(args.docs_dir)
    catalog = load_json(catalog_path(args.output_dir))
    all_errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    for spec in selected:
        summary, errors = verify_dataset(
            args.output_dir,
            args.docs_dir,
            spec,
            fields_by_dataset[spec.name],
            catalog,
        )
        if summary:
            summaries.append(summary)
            LOGGER.info(
                "VERIFY %s rows=%s partitions=%s empty=%s bytes=%s warnings=%s",
                summary["dataset"],
                summary["rows"],
                summary["partitions"],
                summary["empty_partitions"],
                summary["bytes"],
                summary["warnings"],
            )
        for error in errors:
            LOGGER.error("VERIFY %s", error)
        all_errors.extend(errors)

    if all_errors:
        raise DatasetRunError(f"verification failed with {len(all_errors)} error(s)")
    LOGGER.info(
        "verification complete: datasets=%s rows=%s bytes=%s",
        len(summaries),
        sum(item["rows"] for item in summaries),
        sum(item["bytes"] for item in summaries),
    )
    return 0


def add_network_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-url",
        default=os.environ.get("TUSHARE_API_URL", DEFAULT_API_URL),
        help="Tushare-compatible API base URL (token always comes from TUSHARE_TOKEN)",
    )
    parser.add_argument("--timeout", type=int, default=120, help="request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.1,
        help="minimum global interval between API requests in seconds",
    )


def add_common_options(parser: argparse.ArgumentParser, include_network: bool = True) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--apis",
        default="all",
        help="all or comma-separated dataset names",
    )
    if include_network:
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="partition workers; the configured third-party proxy requires 1 for correctness",
        )
        add_network_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser("backfill", help="backfill a closed date range")
    add_common_options(backfill)
    backfill.add_argument("--start-date", required=True)
    backfill.add_argument("--end-date", required=True)
    backfill.add_argument("--force", action="store_true", help="refresh completed partitions")
    backfill.add_argument(
        "--allow-shrink",
        action="store_true",
        help="permit overwriting an existing partition with fewer rows than stored",
    )
    backfill.set_defaults(handler=run_backfill)

    update = subparsers.add_parser("update", help="incrementally update an existing archive")
    add_common_options(update)
    update.add_argument("--as-of", default=format_date(dt.date.today()))
    update.add_argument("--financial-lookback-quarters", type=int, default=8)
    update.add_argument("--event-lookback-days", type=int, default=365)
    update.add_argument(
        "--allow-shrink",
        action="store_true",
        help="permit overwriting an existing partition with fewer rows than stored",
    )
    update.set_defaults(handler=run_update)

    verify = subparsers.add_parser("verify", help="verify local Parquet and manifests offline")
    add_common_options(verify, include_network=False)
    verify.set_defaults(handler=run_verify)

    smoke = subparsers.add_parser("smoke", help="run two small read-only pages per API")
    add_common_options(smoke)
    smoke.add_argument("--period", default="20241231")
    smoke.add_argument("--event-start", default="20240101")
    smoke.add_argument("--event-end", default="20240131")
    smoke.set_defaults(handler=run_smoke)
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if hasattr(args, "workers") and args.workers < 1:
        raise ConfigurationError("--workers must be at least 1")
    if (
        hasattr(args, "workers")
        and args.workers > 1
        and "xiaodefa.top" in (urlparse(args.api_url).hostname or "")
    ):
        raise ConfigurationError(
            "the configured third-party proxy silently returns empty data under concurrent load; "
            "use --workers 1"
        )
    if hasattr(args, "timeout") and args.timeout < 1:
        raise ConfigurationError("--timeout must be at least 1")
    if hasattr(args, "max_retries") and args.max_retries < 1:
        raise ConfigurationError("--max-retries must be at least 1")
    if hasattr(args, "request_interval") and args.request_interval < 0:
        raise ConfigurationError("--request-interval must not be negative")
    if hasattr(args, "financial_lookback_quarters") and args.financial_lookback_quarters < 1:
        raise ConfigurationError("--financial-lookback-quarters must be at least 1")
    if hasattr(args, "event_lookback_days") and args.event_lookback_days < 1:
        raise ConfigurationError("--event-lookback-days must be at least 1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        validate_cli_args(args)
        args.output_dir = args.output_dir.resolve()
        args.docs_dir = args.docs_dir.resolve()
        return int(args.handler(args))
    except KeyboardInterrupt:
        LOGGER.error("interrupted; completed partitions remain resumable")
        return 130
    except SyncError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
