"""Rate limiting, the Tushare API facade and per-partition fetch strategies."""

from __future__ import annotations

import hashlib
import datetime as dt
from email.utils import parsedate_to_datetime
import random
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import requests

from _models import (
    ConfigurationError,
    DatasetSpec,
    FetchResult,
    FieldSpec,
    LOGGER,
    PAGE_OVERLAP,
    FINANCIAL_UPDATE_DATES,
    SourceSchemaError,
    SyncError,
    calendar_dates,
    format_date,
    parse_yyyymmdd,
)
from _schema import (
    empty_frame,
    industry_overlap_mask,
    normalize_frame,
    period_is_historical,
    row_hashes,
    validate_daily_frame,
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
    """Strict API transport with one reusable HTTP session per worker thread."""

    def __init__(
        self,
        token: str,
        api_url: str,
        timeout: int = 120,
        max_retries: int = 5,
        request_interval: float = 0.1,
        query_override: Callable[[str, str, Mapping[str, Any]], pd.DataFrame] | None = None,
        page_overlap: int = PAGE_OVERLAP,
    ) -> None:
        if not token and query_override is None:
            raise ConfigurationError("TUSHARE_TOKEN is not set in the active environment")
        self._token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.rate_limiter = RateLimiter(request_interval)
        self._local = threading.local()
        self._sessions: list[requests.Session] = []
        self._sessions_lock = threading.Lock()
        self._query_override = query_override
        self.page_overlap = page_overlap

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    def close(self) -> None:
        """Release connections after all workers have finished."""
        with self._sessions_lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()
        self._local = threading.local()

    def _request(self, api_name: str, fields_csv: str, params: Mapping[str, Any]) -> pd.DataFrame:
        query = dict(params)
        query.setdefault("ts_type_name", self.api_url)
        # Preserve the installed proxy SDK's wire protocol.
        payload = {"api_name": api_name, "token": self._token,
                   "params": query, "fields": fields_csv}
        with self._session().post(f"{self.api_url}/{api_name}", json=payload,
                                  timeout=self.timeout, allow_redirects=False) as response:
            if not 200 <= response.status_code < 300:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            try:
                result = response.json()
            except ValueError as exc:
                raise SourceSchemaError(f"{api_name}: HTTP {response.status_code}, invalid JSON") from exc
            if not isinstance(result, dict) or type(result.get("code")) is not int:
                raise SourceSchemaError(f"{api_name}: missing or invalid business code")
            if result["code"] != 0:
                message = str(result.get("msg", "unspecified business error"))
                if self._token:
                    message = message.replace(self._token, "[REDACTED]")
                raise SyncError(f"business code={result['code']}: {message}")
            data = result.get("data")
            if not isinstance(data, dict):
                raise SourceSchemaError(f"{api_name}: missing data object")
            columns, items = data.get("fields"), data.get("items")
            if (not isinstance(columns, list) or not columns
                    or not all(isinstance(column, str) and column for column in columns)
                    or len(set(columns)) != len(columns)
                    or not isinstance(items, list)
                    or any(not isinstance(row, list) or len(row) != len(columns) for row in items)):
                raise SourceSchemaError(f"{api_name}: invalid fields/items structure")
            return pd.DataFrame(items, columns=columns)

    @staticmethod
    def _is_non_retryable(exc: BaseException) -> bool:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            return status not in {408, 429} and not 500 <= status < 600
        message = str(exc).lower().replace(" ", "")
        return any(part.lower().replace(" ", "") in message for part in NON_RETRYABLE_MESSAGE_PARTS)

    def query_page(
        self,
        api_name: str,
        fields_csv: str,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        context = {key: params[key] for key in (
            "ann_date", "f_ann_date", "start_date", "end_date", "trade_date",
            "period", "report_type", "is_new", "offset", "limit"
        ) if key in params}
        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()
            started = time.monotonic()
            LOGGER.debug("%s request start params=%s attempt=%s/%s", api_name, context,
                         attempt, self.max_retries)
            try:
                if self._query_override is not None:
                    result = self._query_override(api_name, fields_csv, params)
                else:
                    result = self._request(api_name, fields_csv, params)
                if not isinstance(result, pd.DataFrame):
                    raise SourceSchemaError(
                        f"{api_name} returned {type(result).__name__}, expected pandas.DataFrame"
                    )
                LOGGER.debug("%s request complete params=%s rows=%s elapsed=%.2fs",
                             api_name, context, len(result), time.monotonic() - started)
                return result
            except SourceSchemaError:
                raise
            except Exception as exc:
                message = str(exc).replace(self._token, "[REDACTED]") if self._token else str(exc)
                if self._is_non_retryable(exc) or attempt >= self.max_retries:
                    raise SyncError(f"{api_name} request failed params={context}: {message}") from exc
                delay = min(30.0, (2 ** (attempt - 1)) + random.random())
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            seconds = float(retry_after)
                        except ValueError:
                            try:
                                seconds = (parsedate_to_datetime(retry_after)
                                           - dt.datetime.now(dt.timezone.utc)).total_seconds()
                            except (ValueError, TypeError, OverflowError):
                                seconds = 0.0
                        if seconds > 60:
                            raise SyncError(f"{api_name}: {message}; Retry-After exceeds 60s; rerun later") from exc
                        delay = max(delay, seconds)
                LOGGER.warning(
                    "%s transient request failure params=%s elapsed=%.2fs (attempt %s/%s); retrying in %.1fs: %s",
                    api_name,
                    context,
                    time.monotonic() - started,
                    attempt,
                    self.max_retries,
                    delay,
                    message,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def fetch_paginated(
        self,
        spec: DatasetSpec,
        fields: Sequence[FieldSpec],
        base_params: Mapping[str, Any],
    ) -> FetchResult:
        if not 0 <= self.page_overlap < spec.page_size:
            raise ConfigurationError("pagination overlap must be >= 0 and less than page size")
        fields_csv = ",".join(field.name for field in fields)
        page_frames: list[pd.DataFrame] = []
        seen_page_fingerprints: set[str] = set()
        missing_fields: set[str] = set()
        warnings: list[str] = []
        raw_rows = 0
        pages = 0
        last_page_rows = 0
        offset = 0
        started = last_progress = time.monotonic()

        while True:
            params = dict(base_params)
            params.update(limit=spec.page_size, offset=offset)
            raw_page = self.query_page(spec.api_name, fields_csv, params)
            pages += 1
            page_rows = len(raw_page)
            last_page_rows = page_rows
            raw_rows += page_rows
            now = time.monotonic()
            if now - last_progress >= 10:
                LOGGER.info("%s pagination params=%s offset=%s pages=%s raw_rows=%s elapsed=%.1fs",
                            spec.name, dict(base_params), offset, pages, raw_rows, now - started)
                last_progress = now
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

            page_frames.append(normalized.drop_duplicates(ignore_index=True))

            if page_rows < spec.page_size:
                break
            # Offset belongs to the raw server result, never to deduplicated rows.
            offset += spec.page_size - self.page_overlap

        frame = pd.concat(page_frames, ignore_index=True) if page_frames else empty_frame(fields)
        unique_page_rows = len(frame)
        frame = frame.drop_duplicates(ignore_index=True)
        cross_page_duplicates = unique_page_rows - len(frame)
        duplicates_removed = raw_rows - len(frame)
        if duplicates_removed:
            warnings.append(
                f"removed {duplicates_removed} exact duplicate rows "
                f"({cross_page_duplicates} across pagination pages)"
            )
        if missing_fields:
            warnings.append(
                "source omitted documented fields: " + ", ".join(sorted(missing_fields))
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
            duplicates_removed=duplicates_removed,
        )


def discover_financial_periods(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    start: dt.date,
    end: dt.date,
    archive_start: dt.date,
) -> tuple[set[str], dict[str, Any]]:
    """Discover report periods by a validated exact publication-date predicate.

    Do not combine this with period or start_date/end_date: the proxy's period
    predicate overrides ranges, and its ranges use ann_date even for income.
    """
    date_field = FINANCIAL_UPDATE_DATES[spec.name]
    names = {"ts_code", "end_date", date_field}
    if spec.mode == "statement":
        names.add("report_type")
    projection = [field for field in fields if field.name in names]
    if {field.name for field in projection} != names:
        raise ConfigurationError(f"{spec.name} lacks discovery fields {names}")
    periods: set[str] = set()
    pages = raw_rows = 0
    started = time.monotonic()
    total_days = (end - start).days + 1
    report_types = [str(value) for value in range(1, 13)] if spec.mode == "statement" else [None]
    LOGGER.info("%s discovery start %s=%s..%s days=%s report_types=%s minimum_requests=%s",
                spec.name, date_field, format_date(start), format_date(end), total_days,
                len(report_types), total_days * len(report_types))
    for day_number, day in enumerate(calendar_dates(start, end), 1):
        for report_type in report_types:
            params = {date_field: day}
            if report_type is not None:
                params["report_type"] = report_type
            result = fetcher.fetch_paginated(spec, projection, params)
            frame = result.frame
            pages += result.pages
            raw_rows += result.raw_rows
            if frame.empty:
                continue
            if not frame[date_field].eq(day).fillna(False).all():
                raise SourceSchemaError(f"{spec.name} ignored {date_field}={day}; discovery aborted")
            if report_type is not None and not frame["report_type"].eq(report_type).fillna(False).all():
                raise SourceSchemaError(f"{spec.name} ignored report_type={report_type}")
            for period in frame["end_date"].unique():
                try:
                    parsed = parse_yyyymmdd(period, "discovered end_date")
                except ConfigurationError as exc:
                    raise SourceSchemaError(f"{spec.name}: {exc}") from exc
                if parsed >= archive_start:
                    periods.add(period)
        LOGGER.info("%s discovery progress %s=%s days=%s/%s pages=%s raw_rows=%s periods=%s elapsed=%.1fs",
                    spec.name, date_field, day, day_number, total_days, pages, raw_rows,
                    len(periods), time.monotonic() - started)
    audit = {
        "date_field": date_field, "start_date": format_date(start),
        "end_date": format_date(end), "pages": pages, "raw_rows": raw_rows,
        "periods": sorted(periods),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    LOGGER.info("%s discovery complete %s=%s..%s pages=%s raw_rows=%s elapsed=%.1fs periods=%s",
                spec.name, date_field, audit["start_date"], audit["end_date"], pages,
                raw_rows, audit["elapsed_seconds"], audit["periods"])
    return periods, audit


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
        "duplicates_removed": result.duplicates_removed,
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
            if result.frame["report_type"].isna().any() or returned_types - {key}:
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
            "duplicates_removed": result.duplicates_removed,
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
        "duplicates_removed": sum(item["duplicates_removed"] for item in subqueries.values()),
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
        "duplicates_removed": result.duplicates_removed,
        "missing_fields": sorted(result.missing_fields),
        "warnings": result.warnings,
    }
    return result.frame, stats


def fetch_daily_partition(
    fetcher: ApiFetcher,
    spec: DatasetSpec,
    fields: Sequence[FieldSpec],
    trade_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # daily_basic's date-range parameters do not provide a market-wide range
    # download.  Each partition therefore uses exactly one trade_date query.
    result = fetcher.fetch_paginated(
        spec,
        fields,
        {"trade_date": trade_date},
    )
    validate_daily_frame(result.frame, trade_date, spec.name)
    stats = {
        "raw_rows": result.raw_rows,
        "pages": result.pages,
        "last_page_rows": result.last_page_rows,
        "cross_page_duplicates": result.cross_page_duplicates,
        "duplicates_removed": result.duplicates_removed,
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
            if result.frame["is_new"].isna().any() or returned - {is_new}:
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
            "duplicates_removed": result.duplicates_removed,
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
        "duplicates_removed": sum(item["duplicates_removed"] for item in subqueries.values()),
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
    "discover_financial_periods",
    "fetch_period_partition",
    "fetch_statement_partition",
    "fetch_event_partition",
    "fetch_daily_partition",
    "fetch_industry_partition",
]
