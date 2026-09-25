"""Transport failures must never masquerade as successful empty pages."""
import datetime as dt
import json
import logging

import pytest
import requests

import _fetch
import sync_tushare as sync


def response(status=200, payload=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload if payload is not None else {
        "code": 0, "data": {"fields": ["ts_code"], "items": []}
    }).encode()
    result._content_consumed = True
    result.headers.update(headers or {})
    return result


@pytest.fixture
def transport(monkeypatch):
    class Session:
        def __init__(self):
            self.calls = []
            self.responses = []
            self.closed = False

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        def close(self):
            self.closed = True

    session = Session()
    created = []

    def factory():
        created.append(session)
        return session

    sleeps = []
    monkeypatch.setattr(_fetch.requests, "Session", factory)
    monkeypatch.setattr(_fetch.time, "sleep", sleeps.append)
    fetcher = _fetch.ApiFetcher("secret-token", "https://example.invalid/", request_interval=0,
                                max_retries=3, page_overlap=1)
    yield fetcher, session, created, sleeps
    fetcher.close()


def test_empty_success_protocol_connection_reuse_and_close(transport):
    fetcher, session, created, _ = transport
    session.responses = [response(), response()]
    params = {"ann_date": "20260925", "offset": 0, "limit": 100}
    for _ in range(2):
        frame = fetcher.query_page("forecast_vip", "ts_code", params)
        assert frame.empty and list(frame.columns) == ["ts_code"]
    assert len(created) == 1
    assert "ts_type_name" not in params
    url, kwargs = session.calls[0]
    assert url == "https://example.invalid/forecast_vip"
    assert kwargs == {"json": {"api_name": "forecast_vip", "token": "secret-token",
                             "params": {**params, "ts_type_name": "https://example.invalid"},
                             "fields": "ts_code"}, "timeout": 120, "allow_redirects": False}
    fetcher.close()
    assert session.closed


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404])
def test_permanent_http_failure_never_empty_or_retried(transport, status):
    fetcher, session, _, sleeps = transport
    session.responses = [response(status)]
    with pytest.raises(sync.SyncError, match=f"HTTP {status}"):
        fetcher.query_page("income_vip", "ts_code", {"offset": 0})
    assert len(session.calls) == 1 and not sleeps


@pytest.mark.parametrize("failure", [response(408), response(429, headers={"Retry-After": "5"}),
                                     response(500), requests.Timeout("timed out")])
def test_transient_failure_retries_identical_page(transport, failure, caplog):
    fetcher, session, _, sleeps = transport
    session.responses = [failure, response()]
    with caplog.at_level(logging.DEBUG, logger="sync_tushare"):
        assert fetcher.query_page("income_vip", "ts_code", {"offset": 4800}).empty
    assert session.calls[0] == session.calls[1]
    assert len(sleeps) == 1
    if isinstance(failure, requests.Response) and failure.status_code == 429:
        assert sleeps == [5]
    assert "secret-token" not in caplog.text
    assert "4800" in caplog.text and "elapsed=" in caplog.text


def test_retry_exhaustion_raises(transport):
    fetcher, session, _, sleeps = transport
    session.responses = [response(503) for _ in range(3)]
    with pytest.raises(sync.SyncError, match="HTTP 503"):
        fetcher.query_page("income_vip", "ts_code", {})
    assert len(session.calls) == 3 and len(sleeps) == 2


def test_long_retry_after_fails_without_early_retry(transport):
    fetcher, session, _, sleeps = transport
    session.responses = [response(429, headers={"Retry-After": "120"})]
    with pytest.raises(sync.SyncError, match="rerun later"):
        fetcher.query_page("income_vip", "ts_code", {})
    assert not sleeps


@pytest.mark.parametrize("payload", [[], {}, {"code": False}, {"code": 0},
    {"code": 0, "data": {"fields": ["ts_code"], "items": None}},
    {"code": 0, "data": {"fields": ["ts_code"], "items": [[1, 2]]}},
    {"code": 0, "data": {"fields": ["ts_code", "ts_code"], "items": []}}])
def test_malformed_success_is_error(transport, payload):
    fetcher, session, _, sleeps = transport
    session.responses = [response(payload=payload)]
    with pytest.raises(sync.SourceSchemaError):
        fetcher.query_page("income_vip", "ts_code", {})
    assert not sleeps


def test_invalid_json_is_error(transport):
    fetcher, session, _, _ = transport
    invalid = response()
    invalid._content = b'<html>upstream failed</html>'
    session.responses = [invalid]
    with pytest.raises(sync.SourceSchemaError, match="invalid JSON"):
        fetcher.query_page("income_vip", "ts_code", {})


def test_business_permission_error_redacts_token(transport):
    fetcher, session, _, sleeps = transport
    session.responses = [response(payload={"code": -1, "msg": "权限不足 secret-token"})]
    with pytest.raises(sync.SyncError) as error:
        fetcher.query_page("income_vip", "ts_code", {})
    assert "secret-token" not in str(error.value) and not sleeps


def test_pagination_does_not_advance_after_http_failure(transport):
    import dataclasses
    fetcher, session, _, _ = transport
    spec = dataclasses.replace(sync.DATASET_BY_NAME["forecast"], page_size=3)
    fields = [sync.FieldSpec("ts_code", "str")]
    def page(values):
        return response(payload={"code": 0, "data": {"fields": ["ts_code"],
                                                     "items": [[v] for v in values]}})
    session.responses = [page(["A", "B", "C"]), response(500), page(["C", "D"])]
    result = fetcher.fetch_paginated(spec, fields, {})
    assert [call[1]["json"]["params"]["offset"] for call in session.calls] == [0, 2, 2]
    assert list(result.frame.ts_code) == ["A", "B", "C", "D"]
    assert result.pages == 2


def test_empty_discovery_logs_daily_progress(caplog):
    import pandas as pd
    fetcher = _fetch.ApiFetcher("", "unused", request_interval=0,
                                query_override=lambda *args: pd.DataFrame())
    fields = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)["income"]
    with caplog.at_level(logging.INFO, logger="sync_tushare"):
        periods, audit = _fetch.discover_financial_periods(fetcher, sync.DATASET_BY_NAME["income"],
            fields, dt.date(2026, 9, 24), dt.date(2026, 9, 25), dt.date(2016, 1, 1))
    assert not periods and audit["pages"] == 24
    assert audit["elapsed_seconds"] >= 0
    assert "minimum_requests=24" in caplog.text
    assert "days=1/2" in caplog.text and "days=2/2" in caplog.text
    assert "discovery complete" in caplog.text


def test_http_failure_preserves_partition_and_update_cursor(transport, monkeypatch, tmp_path):
    fetcher, session, _, _ = transport
    fields = [sync.FieldSpec(name, "str") for name in ("ts_code", "ann_date", "end_date")]
    monkeypatch.setattr(sync, "load_all_fields", lambda _: {"forecast": fields})
    monkeypatch.setattr(sync, "build_fetcher", lambda _: fetcher)
    session.responses = [response(payload={"code": 0, "data": {
        "fields": [field.name for field in fields],
        "items": [["600000.SH", "20240320", "20240331"]]}})]
    common = ["--apis", "forecast", "--output-dir", str(tmp_path)]
    assert sync.main(["backfill", *common, "--start-date", "20240101", "--end-date", "20240401"]) == 0
    spec = sync.DATASET_BY_NAME["forecast"]
    partition = sync.partition_file(tmp_path, spec, "20240331")
    before = partition.read_bytes()
    manifest_file = sync.manifest_path(tmp_path, spec)
    manifest = sync.load_json(manifest_file)
    manifest["update_state"] = {"discovery_through": "20240401"}
    sync.atomic_write_json(manifest_file, manifest)
    session.responses = [response(503) for _ in range(3)]
    assert sync.main(["update", *common, "--as-of", "20240402"]) == 1
    assert sync.load_json(manifest_file)["update_state"]["discovery_through"] == "20240401"
    assert partition.read_bytes() == before
    assert session.closed
