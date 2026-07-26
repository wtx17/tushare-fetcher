"""Rate limiting, the Tushare API facade and per-partition fetch strategies."""

from __future__ import annotations

import hashlib
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import tushare as ts

from _models import (
    ConfigurationError,
    DatasetSpec,
    FetchResult,
    FieldSpec,
    LOGGER,
    SourceSchemaError,
    SyncError,
)
from _schema import (
    empty_frame,
    industry_overlap_mask,
    normalize_frame,
    period_is_historical,
    row_hashes,
    validate_event_frame,
    validate_period_frame,
)


NON_RETRYABLE_MESSAGE_PARTS = (
    "token不对",
    "token 不对",
    "权限",
    "积分",
    "参数错误",
    "字段错误",
    "接口不存在",
    "api_name不存在",
)


class RateLimiter:
    def __init__(self, minimum_interval: float) -> None:
        self.minimum_interval = max(0.0, minimum_interval)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.minimum_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.minimum_interval
        if delay:
            time.sleep(delay)


class ApiFetcher:
    """Thread-safe facade around one Tushare client per worker thread."""

    def __init__(
        self,
        token: str,
        api_url: str,
        timeout: int = 120,
        max_retries: int = 5,
        request_interval: float = 0.1,
        query_override: Callable[[str, str, Mapping[str, Any]], pd.DataFrame] | None = None,
    ) -> None:
        if not token and query_override is None:
            raise ConfigurationError("TUSHARE_TOKEN is not set in the active environment")
        self._token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.rate_limiter = RateLimiter(request_interval)
        self._local = threading.local()
        self._query_override = query_override

    def _client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = ts.pro_api(self._token, timeout=self.timeout)
            client._DataApi__http_url = self.api_url
            self._local.client = client
        return client

    @staticmethod
    def _is_non_retryable(exc: BaseException) -> bool:
        message = str(exc).lower().replace(" ", "")
        return any(part.lower().replace(" ", "") in message for part in NON_RETRYABLE_MESSAGE_PARTS)

    def query_page(
        self,
        api_name: str,
        fields_csv: str,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                if self._query_override is not None:
                    result = self._query_override(api_name, fields_csv, params)
                else:
                    result = self._client().query(api_name, fields=fields_csv, **dict(params))
                if not isinstance(result, pd.DataFrame):
                    raise SourceSchemaError(
                        f"{api_name} returned {type(result).__name__}, expected pandas.DataFrame"
                    )
                return result
            except SourceSchemaError:
                raise
            except BaseException as exc:
                if self._is_non_retryable(exc) or attempt >= self.max_retries:
                    raise SyncError(f"{api_name} request failed: {exc}") from exc
                delay = min(30.0, (2 ** (attempt - 1)) + random.random())
                LOGGER.warning(
                    "%s transient request failure (attempt %s/%s); retrying in %.1fs: %s",
                    api_name,
                    attempt,
                    self.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def fetch_paginated(
        self,
        spec: DatasetSpec,
        fields: Sequence[FieldSpec],
        base_params: Mapping[str, Any],
    ) -> FetchResult:
        fields_csv = ",".join(field.name for field in fields)
        page_frames: list[pd.DataFrame] = []
        seen_hashes: set[int] = set()
        seen_page_fingerprints: set[str] = set()
        missing_fields: set[str] = set()
        warnings: list[str] = []
        cross_page_duplicates = 0
        raw_rows = 0
        pages = 0
        last_page_rows = 0
        offset = 0

        while True:
            params = dict(base_params)
            params.update(limit=spec.page_size, offset=offset)
            raw_page = self.query_page(spec.api_name, fields_csv, params)
            pages += 1
            page_rows = len(raw_page)
            last_page_rows = page_rows
            raw_rows += page_rows
            if page_rows > spec.page_size:
                raise SourceSchemaError(
                    f"{spec.api_name} returned {page_rows} rows for limit={spec.page_size}"
                )

            normalized, page_missing = normalize_frame(
                raw_page,
                fields,
                allowed_missing_fields=spec.allowed_missing_fields,
                dataset_name=spec.name,
            )
            missing_fields.update(page_missing)

            hashes = row_hashes(normalized)
            fingerprint = hashlib.sha256(
                b"".join(value.to_bytes(8, "little", signed=False) for value in hashes)
            ).hexdigest()
            if hashes and fingerprint in seen_page_fingerprints:
                raise SyncError(
                    f"{spec.api_name} repeated an earlier page at offset={offset}; "
                    "the source may be ignoring offset"
                )
            if hashes:
                seen_page_fingerprints.add(fingerprint)

            # Compare against rows from prior pages only.  Identical rows that
            # coexist in one upstream page remain intact as raw source data.
            keep = [value not in seen_hashes for value in hashes]
            overlap = len(keep) - sum(keep)
            if overlap:
                cross_page_duplicates += overlap
                normalized = normalized.loc[keep].reset_index(drop=True)
            seen_hashes.update(hashes)
            page_frames.append(normalized)

            if page_rows < spec.page_size:
                break
            offset += page_rows

        if cross_page_duplicates:
            warnings.append(
                f"removed {cross_page_duplicates} exact row overlaps across pagination pages"
            )
        if missing_fields:
            warnings.append(
                "source omitted documented fields: " + ", ".join(sorted(missing_fields))
            )

        frame = (
            pd.concat(page_frames, ignore_index=True)
            if page_frames
            else empty_frame(fields)
        )
        if frame.empty:
            frame = empty_frame(fields)
        return FetchResult(
            frame=frame,
            pages=pages,
            raw_rows=raw_rows,
            last_page_rows=last_page_rows,
            cross_page_duplicates=cross_page_duplicates,
            missing_fields=missing_fields,
            warnings=warnings,
        )


def fetch_period_partition(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    period: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = fetcher.fetch_paginated(spec, fields, {"period": period})
    validate_period_frame(result.frame, period, spec.name)
    if spec.name == "fina_indicator" and result.frame.empty and period_is_historical(period):
        raise SourceSchemaError(
            f"{spec.name} historical period={period} unexpectedly returned zero rows; "
            "refusing to persist a likely throttled response"
        )
    stats = {
        "raw_rows": result.raw_rows,
        "pages": result.pages,
        "last_page_rows": result.last_page_rows,
        "cross_page_duplicates": result.cross_page_duplicates,
        "missing_fields": sorted(result.missing_fields),
        "warnings": result.warnings,
    }
    return result.frame, stats


def fetch_statement_partition(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    period: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    subqueries: dict[str, Any] = {}
    all_missing: set[str] = set()
    warnings: list[str] = []
    total_pages = 0
    total_raw_rows = 0
    total_cross_page_duplicates = 0
    last_page_rows = 0

    for report_type in range(1, 13):
        key = str(report_type)
        result = fetcher.fetch_paginated(
            spec,
            fields,
            {"period": period, "report_type": key},
        )
        validate_period_frame(result.frame, period, spec.name)
        if report_type == 1 and result.frame.empty and period_is_historical(period):
            raise SourceSchemaError(
                f"{spec.name} historical period={period} report_type=1 unexpectedly returned "
                "zero rows; refusing to persist a likely throttled response"
            )
        if not result.frame.empty:
            returned_types = set(result.frame["report_type"].dropna().astype("string").tolist())
            if returned_types - {key}:
                raise SourceSchemaError(
                    f"{spec.name} period={period} report_type={key} returned types "
                    f"{sorted(returned_types)}"
                )
        frames.append(result.frame)
        all_missing.update(result.missing_fields)
        warnings.extend(result.warnings)
        total_pages += result.pages
        total_raw_rows += result.raw_rows
        total_cross_page_duplicates += result.cross_page_duplicates
        last_page_rows = result.last_page_rows
        subqueries[key] = {
            "rows": len(result.frame),
            "raw_rows": result.raw_rows,
            "pages": result.pages,
            "last_page_rows": result.last_page_rows,
            "cross_page_duplicates": result.cross_page_duplicates,
            "missing_fields": sorted(result.missing_fields),
            "warnings": result.warnings,
        }
        LOGGER.info(
            "%s period=%s report_type=%s rows=%s pages=%s",
            spec.name,
            period,
            key,
            len(result.frame),
            result.pages,
        )

    combined = pd.concat(frames, ignore_index=True) if frames else empty_frame(fields)
    if combined.empty:
        combined = empty_frame(fields)
    stats = {
        "raw_rows": total_raw_rows,
        "pages": total_pages,
        "last_page_rows": last_page_rows,
        "cross_page_duplicates": total_cross_page_duplicates,
        "missing_fields": sorted(all_missing),
        "warnings": sorted(set(warnings)),
        "subqueries": subqueries,
    }
    return combined, stats


def fetch_event_partition(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    query_start: str,
    query_end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = fetcher.fetch_paginated(
        spec,
        fields,
        {"start_date": query_start, "end_date": query_end},
    )
    validate_event_frame(result.frame, query_start, query_end, spec.name)
    stats = {
        "raw_rows": result.raw_rows,
        "pages": result.pages,
        "last_page_rows": result.last_page_rows,
        "cross_page_duplicates": result.cross_page_duplicates,
        "missing_fields": sorted(result.missing_fields),
        "warnings": result.warnings,
    }
    return result.frame, stats


def fetch_industry_partition(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    subqueries: dict[str, Any] = {}
    total_pages = 0
    total_raw_rows = 0
    total_cross_page_duplicates = 0
    all_missing: set[str] = set()
    warnings: list[str] = []
    last_page_rows = 0

    for is_new in ("Y", "N"):
        result = fetcher.fetch_paginated(spec, fields, {"is_new": is_new})
        if not result.frame.empty:
            returned = set(result.frame["is_new"].dropna().astype("string").tolist())
            if returned - {is_new}:
                raise SourceSchemaError(
                    f"{spec.name} is_new={is_new} returned status values {sorted(returned)}"
                )
        frames.append(result.frame)
        total_pages += result.pages
        total_raw_rows += result.raw_rows
        total_cross_page_duplicates += result.cross_page_duplicates
        all_missing.update(result.missing_fields)
        warnings.extend(result.warnings)
        last_page_rows = result.last_page_rows
        subqueries[is_new] = {
            "rows_before_date_filter": len(result.frame),
            "raw_rows": result.raw_rows,
            "pages": result.pages,
            "last_page_rows": result.last_page_rows,
            "cross_page_duplicates": result.cross_page_duplicates,
            "missing_fields": sorted(result.missing_fields),
            "warnings": result.warnings,
        }

    combined = pd.concat(frames, ignore_index=True) if frames else empty_frame(fields)
    if combined.empty:
        filtered = empty_frame(fields)
    else:
        mask = industry_overlap_mask(combined, start_date, end_date)
        filtered = combined.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise SourceSchemaError(
            f"{spec.name} returned no industry relationships overlapping {start_date}..{end_date}"
        )
    stats = {
        "raw_rows": total_raw_rows,
        "pages": total_pages,
        "last_page_rows": last_page_rows,
        "cross_page_duplicates": total_cross_page_duplicates,
        "missing_fields": sorted(all_missing),
        "warnings": sorted(set(warnings)),
        "subqueries": subqueries,
        "rows_before_date_filter": len(combined),
        "rows_after_date_filter": len(filtered),
    }
    return filtered, stats


__all__ = [
    "NON_RETRYABLE_MESSAGE_PARTS",
    "RateLimiter",
    "ApiFetcher",
    "fetch_period_partition",
    "fetch_statement_partition",
    "fetch_event_partition",
    "fetch_industry_partition",
]
