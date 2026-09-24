from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

import sync_tushare as sync


EXPECTED_FIELD_COUNTS = {
    "income": 94,
    "balancesheet": 158,
    "cashflow": 97,
    "fina_indicator": 167,
    "forecast": 12,
    "express": 32,
    "stk_holdernumber": 4,
    "stk_holdertrade": 13,
    "index_member_all": 11,
    "ci_index_member": 11,
    "daily_basic": 19,
}


def test_parse_all_documented_fields() -> None:
    parsed = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)
    assert {name: len(fields) for name, fields in parsed.items()} == EXPECTED_FIELD_COUNTS
    assert sum(EXPECTED_FIELD_COUNTS.values()) == 618
    for fields in parsed.values():
        assert len({field.name for field in fields}) == len(fields)


def test_rolling_ten_year_quarters_and_event_years() -> None:
    start = dt.date(2016, 7, 20)
    end = dt.date(2026, 7, 20)
    periods = sync.quarter_ends(start, end)
    assert len(periods) == 40
    assert periods[0] == "20160930"
    assert periods[-1] == "20260630"

    years = sync.event_year_ranges(start, end)
    assert years[0] == ("2016", "20160720", "20161231")
    assert years[-1] == ("2026", "20260101", "20260720")
    assert len(years) == 11

    dates = sync.calendar_dates(dt.date(2024, 1, 30), dt.date(2024, 2, 2))
    assert dates == ["20240130", "20240131", "20240201", "20240202"]


def test_normalize_known_missing_fields_and_reject_other_missing() -> None:
    fields = [
        sync.FieldSpec("ts_code", "str"),
        sync.FieldSpec("value", "float"),
        sync.FieldSpec("new_value", "float"),
    ]
    raw = pd.DataFrame({"ts_code": ["000001.SZ"], "value": ["1.25"]})
    normalized, missing = sync.normalize_frame(
        raw,
        fields,
        allowed_missing_fields={"new_value"},
        dataset_name="demo",
    )
    assert list(normalized.columns) == ["ts_code", "value", "new_value"]
    assert normalized["value"].dtype.name == "Float64"
    assert normalized["new_value"].isna().all()
    assert missing == {"new_value"}

    with pytest.raises(sync.SourceSchemaError):
        sync.normalize_frame(raw, fields, dataset_name="demo")


def test_express_proxy_missing_fields_are_filled_and_warned() -> None:
    spec = sync.DATASET_BY_NAME["express"]
    fields = sync.parse_output_fields(sync.DEFAULT_DOCS_DIR / spec.doc_name)
    values: dict[str, object] = {}
    for field in fields:
        if field.name in sync.KNOWN_EXPRESS_MISSING_FIELDS:
            continue
        if field.name == "ts_code":
            values[field.name] = "000001.SZ"
        elif field.name == "ann_date":
            values[field.name] = "20260101"
        elif field.name == "end_date":
            values[field.name] = "20251231"
        elif field.pandas_dtype == "Float64":
            values[field.name] = "1.0"
        elif field.pandas_dtype == "Int64":
            values[field.name] = "1"
        else:
            values[field.name] = "sample"

    def fake_query(_api: str, _fields: str, _params: dict[str, object]) -> pd.DataFrame:
        return pd.DataFrame([values])

    fetcher = sync.ApiFetcher(
        token="test-token",
        api_url="https://example.invalid",
        query_override=fake_query,
        request_interval=0,
    )
    frame, stats = sync.fetch_period_partition(fetcher, spec, fields, "20251231")

    assert set(stats["missing_fields"]) == sync.KNOWN_EXPRESS_MISSING_FIELDS
    assert stats["warnings"] == [
        "source omitted documented fields: "
        + ", ".join(sorted(sync.KNOWN_EXPRESS_MISSING_FIELDS))
    ]
    assert frame[list(sync.KNOWN_EXPRESS_MISSING_FIELDS)].isna().all().all()

    unexpected = pd.DataFrame({"ts_code": ["000001.SZ"]})
    with pytest.raises(sync.SourceSchemaError):
        sync.normalize_frame(
            unexpected,
            fields,
            allowed_missing_fields=spec.allowed_missing_fields,
            dataset_name=spec.name,
        )


def test_overlapping_pagination_deduplicates_all_rows_and_uses_raw_stride() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "unused.md", "period", 3)
    fields = [sync.FieldSpec("code", "str")]
    offsets = []
    source = ["A", "A", "B", "C", "D"]

    def query(_api, _fields, params):
        offsets.append(params["offset"])
        return pd.DataFrame({"code": source[params["offset"]:params["offset"] + params["limit"]]})

    fetcher = sync.ApiFetcher("t", "https://example.invalid", query_override=query,
                              request_interval=0, page_overlap=1)
    result = fetcher.fetch_paginated(spec, fields, {})
    assert offsets == [0, 2, 4]
    assert result.pages == 3
    assert result.raw_rows == 7
    assert result.last_page_rows == 1
    assert result.cross_page_duplicates == 2
    assert result.duplicates_removed == 3
    assert result.frame["code"].tolist() == ["A", "B", "C", "D"]


def test_industry_overlap_uses_closed_intervals() -> None:
    frame = pd.DataFrame(
        {
            "in_date": pd.Series(
                ["20100101", "20160720", "20260720", "20260721", pd.NA],
                dtype="string",
            ),
            "out_date": pd.Series(
                ["20160719", "20160720", pd.NA, pd.NA, "20170101"],
                dtype="string",
            ),
        }
    )
    mask = sync.industry_overlap_mask(frame, "20160720", "20260720")
    assert mask.tolist() == [False, True, True, False, True]


def test_atomic_partition_files_are_readable_as_one_directory(tmp_path: Path) -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "unused.md", "period", 100)
    frame1 = pd.DataFrame(
        {
            "ts_code": pd.Series(["000001.SZ"], dtype="string"),
            "end_date": pd.Series(["20240331"], dtype="string"),
        }
    )
    frame2 = pd.DataFrame(
        {
            "ts_code": pd.Series(["000002.SZ"], dtype="string"),
            "end_date": pd.Series(["20240630"], dtype="string"),
        }
    )
    sync.atomic_write_parquet(frame1, sync.partition_file(tmp_path, spec, "20240331"))
    sync.atomic_write_parquet(frame2, sync.partition_file(tmp_path, spec, "20240630"))

    combined = pd.read_parquet(tmp_path / "demo")
    assert len(combined) == 2
    assert set(combined["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert not list((tmp_path / "demo").rglob("*.tmp"))


def test_update_targets_add_missing_and_refresh_recent() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "unused.md", "period", 100)
    manifest = {"partitions": {"20240331": {}}}
    targets = sync.update_targets(
        spec,
        manifest,
        archive_start=dt.date(2024, 1, 1),
        as_of=dt.date(2024, 7, 20),
        financial_lookback_quarters=1,
        event_lookback_days=365,
    )
    assert targets == [("20240630", {"period": "20240630"})]


def test_daily_targets_use_one_trade_date_per_partition() -> None:
    spec = sync.DATASET_BY_NAME["daily_basic"]
    targets = sync.targets_for_range(
        spec,
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 3),
    )
    assert targets == [
        ("20240101", {"trade_date": "20240101"}),
        ("20240102", {"trade_date": "20240102"}),
        ("20240103", {"trade_date": "20240103"}),
    ]


def test_daily_fetch_queries_and_validates_one_trade_date() -> None:
    spec = sync.DATASET_BY_NAME["daily_basic"]
    fields = sync.parse_output_fields(sync.DEFAULT_DOCS_DIR / spec.doc_name)
    calls: list[dict[str, object]] = []

    def fake_query(_api: str, _fields: str, params: dict[str, object]) -> pd.DataFrame:
        calls.append(dict(params))
        values: dict[str, object] = {}
        for field in fields:
            if field.name in sync.KNOWN_DAILY_BASIC_MISSING_FIELDS:
                continue
            if field.name == "ts_code":
                values[field.name] = "000001.SZ"
            elif field.name == "trade_date":
                values[field.name] = "20240102"
            elif field.pandas_dtype == "Float64":
                values[field.name] = "1.0"
            elif field.pandas_dtype == "Int64":
                values[field.name] = "1"
            else:
                values[field.name] = "sample"
        return pd.DataFrame([values])

    fetcher = sync.ApiFetcher(
        token="test-token",
        api_url="https://example.invalid",
        query_override=fake_query,
        request_interval=0,
    )
    frame, stats = sync.fetch_daily_partition(fetcher, spec, fields, "20240102")

    assert calls == [{"trade_date": "20240102", "limit": 6000, "offset": 0}]
    assert frame["trade_date"].tolist() == ["20240102"]
    assert frame["limit_status"].dtype.name == "Int64"
    assert frame["limit_status"].isna().all()
    assert stats["pages"] == 1
    assert stats["missing_fields"] == ["limit_status"]
    assert stats["warnings"] == ["source omitted documented fields: limit_status"]

    invalid = frame.copy()
    invalid["trade_date"] = "20240103"
    with pytest.raises(sync.SourceSchemaError, match="other trade_date"):
        sync.validate_daily_frame(invalid, "20240102", spec.name)

    with pytest.raises(sync.SourceSchemaError, match="close"):
        sync.normalize_frame(
            frame.drop(columns=["close"]),
            fields,
            allowed_missing_fields=spec.allowed_missing_fields,
            dataset_name=spec.name,
        )


def test_daily_update_refreshes_recent_dates_and_fills_gaps() -> None:
    spec = sync.DATASET_BY_NAME["daily_basic"]
    manifest = {
        "partitions": {
            "20240101": {},
            "20240103": {},
            "20240104": {},
            "20240105": {},
        }
    }
    targets = sync.update_targets(
        spec,
        manifest,
        archive_start=dt.date(2024, 1, 1),
        as_of=dt.date(2024, 1, 5),
        financial_lookback_quarters=1,
        event_lookback_days=1,
        daily_lookback_days=2,
    )
    assert targets == [
        ("20240102", {"trade_date": "20240102"}),
        ("20240104", {"trade_date": "20240104"}),
        ("20240105", {"trade_date": "20240105"}),
    ]


def test_proxy_concurrency_is_rejected_before_download() -> None:
    parser = sync.build_parser()
    args = parser.parse_args(
        [
            "backfill",
            "--start-date",
            "20240101",
            "--end-date",
            "20241231",
            "--workers",
            "2",
        ]
    )
    with pytest.raises(sync.ConfigurationError, match="silently returns empty data"):
        sync.validate_cli_args(args)


def test_old_market_wide_period_is_treated_as_nonempty_sentinel() -> None:
    assert sync.period_is_historical("20201231")
    assert not sync.period_is_historical("20991231")


def test_partition_validation_checks_query_and_checksum(tmp_path: Path) -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "unused.md", "period", 100)
    target = sync.partition_file(tmp_path, spec, "20240331")
    frame = pd.DataFrame({"ts_code": pd.Series(["000001.SZ"], dtype="string")})
    checksum, size = sync.atomic_write_parquet(frame, target)
    fields = [sync.FieldSpec("ts_code", "str")]
    entry = {
        "relative_path": str(target.relative_to(tmp_path)),
        "query": {"period": "20240331"},
        "bytes": size,
        "sha256": checksum,
        "schema_hash": sync.schema_hash(fields),
    }
    assert sync.partition_is_valid(
        tmp_path,
        entry,
        {"period": "20240331"},
        sync.schema_hash(fields),
    )
    assert not sync.partition_is_valid(
        tmp_path,
        entry,
        {"period": "20240630"},
        sync.schema_hash(fields),
    )


def test_verify_checks_only_manifest_declared_missing_fields(tmp_path: Path) -> None:
    spec = sync.DatasetSpec(
        "demo",
        "demo_api",
        "demo.md",
        "period",
        100,
        frozenset({"sometimes_missing"}),
    )
    fields = [
        sync.FieldSpec("ts_code", "str"),
        sync.FieldSpec("end_date", "str"),
        sync.FieldSpec("sometimes_missing", "float"),
    ]
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "demo.md").write_text(
        "## 输出参数\n\n| 名称 | 类型 | 描述 |\n| --- | --- | --- |\n"
        "| ts_code | str | code |\n"
        "| end_date | str | date |\n"
        "| sometimes_missing | float | value |\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "data"
    target = sync.partition_file(output_dir, spec, "20240331")
    frame = pd.DataFrame(
        {
            "ts_code": pd.Series(["000001.SZ"], dtype="string"),
            "end_date": pd.Series(["20240331"], dtype="string"),
            "sometimes_missing": pd.Series([1.0], dtype="Float64"),
        }
    )
    checksum, size = sync.atomic_write_parquet(frame, target)
    manifest = sync.new_manifest(spec, fields, docs_dir, "https://example.invalid")
    manifest["range_start"] = "20240101"
    manifest["range_end"] = "20241231"
    manifest["partitions"]["20240331"] = {
        "relative_path": str(target.relative_to(output_dir)),
        "query": {"period": "20240331"},
        "rows": 1,
        "raw_rows": 1,
        "pages": 1,
        "last_page_rows": 1,
        "cross_page_duplicates": 0,
        "missing_fields": [],
        "warnings": [],
        "sha256": checksum,
        "bytes": size,
        "fetched_at": sync.now_iso(),
        "schema_hash": sync.schema_hash(fields),
    }
    sync.atomic_write_json(sync.manifest_path(output_dir, spec), manifest)

    summary, errors = sync.verify_dataset(output_dir, docs_dir, spec, fields, None)
    assert summary["rows"] == 1
    assert errors == []

    manifest["allowed_missing_fields"] = []
    sync.atomic_write_json(sync.manifest_path(output_dir, spec), manifest)
    _summary, errors = sync.verify_dataset(output_dir, docs_dir, spec, fields, None)
    assert errors == ["demo manifest missing-field policy differs from the sync policy"]


def test_prepare_manifest_refreshes_missing_field_policy(tmp_path: Path) -> None:
    old_spec = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
    new_spec = sync.DatasetSpec(
        "demo",
        "demo_api",
        "demo.md",
        "period",
        100,
        frozenset({"proxy_field"}),
    )
    fields = [sync.FieldSpec("ts_code", "str")]
    manifest = sync.new_manifest(old_spec, fields, tmp_path, "https://example.invalid")
    sync.atomic_write_json(sync.manifest_path(tmp_path, old_spec), manifest)

    prepared = sync.prepare_manifest(
        tmp_path,
        new_spec,
        fields,
        tmp_path,
        "https://example.invalid",
        force=False,
        requested_start="20240101",
        requested_end="20241231",
    )
    assert prepared["allowed_missing_fields"] == ["proxy_field"]


def test_overwrite_guard_refuses_empty_over_nonempty() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
    new_empty = pd.DataFrame({"ts_code": pd.Series([], dtype="string")})
    # Empty responses are rejected by default; an explicit override is available.
    with pytest.raises(sync.OverwriteGuardError, match="0 rows"):
        sync.check_overwrite_safety(spec, "k", new_empty, {"rows": 5}, allow_shrink=False)
    sync.check_overwrite_safety(spec, "k", new_empty, {"rows": 5}, allow_shrink=True)


def test_overwrite_guard_refuses_row_shrink() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
    smaller = pd.DataFrame({"ts_code": pd.Series(["A", "B"], dtype="string")})
    with pytest.raises(sync.OverwriteGuardError, match="shrank 10 -> 2"):
        sync.check_overwrite_safety(spec, "k", smaller, {"rows": 10}, allow_shrink=False)


def test_allow_shrink_permits_large_shrink() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
    smaller = pd.DataFrame({"ts_code": pd.Series(["A", "B"], dtype="string")})
    # a non-empty partial shrink is allowed with the flag...
    sync.check_overwrite_safety(spec, "k", smaller, {"rows": 10}, allow_shrink=True)
    # ...but an equal-size refresh is always fine.
    equal = pd.DataFrame({"ts_code": pd.Series(["A", "B"], dtype="string")})
    sync.check_overwrite_safety(spec, "k", equal, {"rows": 2}, allow_shrink=False)


def test_overwrite_guard_skipped_for_new_partition() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
    new_empty = pd.DataFrame({"ts_code": pd.Series([], dtype="string")})
    # No existing entry means there is nothing to protect; any frame is accepted.
    sync.check_overwrite_safety(spec, "k", new_empty, None, allow_shrink=False)


def _demo_period_fetcher(row_counts: list[int]) -> sync.ApiFetcher:
    """Return a fetcher whose successive paginated calls return row_counts[0], etc."""
    state = {"i": 0, "n": row_counts[0]}

    def make_frame(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": pd.Series([f"{i:06d}.SZ" for i in range(n)], dtype="string"),
                "end_date": pd.Series(["20240331"] * n, dtype="string"),
                "value": pd.Series(list(range(n)), dtype="Int64"),
            }
        )

    def fake_query(_api: str, _fields: str, _params: dict[str, object]) -> pd.DataFrame:
        frame = make_frame(state["n"])
        state["i"] += 1
        if state["i"] < len(row_counts):
            state["n"] = row_counts[state["i"]]
        return frame

    return sync.ApiFetcher(
        token="t",
        api_url="https://example.invalid",
        query_override=fake_query,
        request_interval=0,
        page_overlap=0,
    )


_DEMO_SPEC = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
_DEMO_FIELDS = [
    sync.FieldSpec("ts_code", "str"),
    sync.FieldSpec("end_date", "str"),
    sync.FieldSpec("value", "int"),
]


def test_update_refusal_preserves_existing_partition(tmp_path: Path) -> None:
    fetcher = _demo_period_fetcher([10, 4])
    common = dict(
        fetcher=fetcher,
        output_dir=tmp_path,
        docs_dir=tmp_path,
        spec=_DEMO_SPEC,
        fields=_DEMO_FIELDS,
        range_start="20240101",
        range_end="20241231",
        workers=1,
        force=False,
        resume=False,
        allow_shrink=False,
    )
    sync.process_dataset(targets=[("20240331", {"period": "20240331"})], **common)
    path = sync.partition_file(tmp_path, _DEMO_SPEC, "20240331")
    assert len(pd.read_parquet(path)) == 10

    # More than half of the distinct rows disappeared: retain the original.
    with pytest.raises(sync.DatasetRunError, match="shrank"):
        sync.process_dataset(targets=[("20240331", {"period": "20240331"})], **common)
    assert len(pd.read_parquet(path)) == 10
    manifest = sync.load_json(sync.manifest_path(tmp_path, _DEMO_SPEC))
    assert manifest["partitions"]["20240331"]["rows"] == 10


def test_force_and_allow_shrink_bypass_overwrite_guard(tmp_path: Path) -> None:
    path = sync.partition_file(tmp_path, _DEMO_SPEC, "20240331")

    # seed with 10 rows
    fetcher = _demo_period_fetcher([10, 8])
    sync.process_dataset(
        fetcher=fetcher,
        output_dir=tmp_path,
        docs_dir=tmp_path,
        spec=_DEMO_SPEC,
        fields=_DEMO_FIELDS,
        targets=[("20240331", {"period": "20240331"})],
        range_start="20240101",
        range_end="20241231",
        workers=1,
        force=False,
        resume=False,
        allow_shrink=False,
    )
    assert len(pd.read_parquet(path)) == 10

    # --allow-shrink lets the 8-row refresh overwrite the 10-row partition.
    fetcher = _demo_period_fetcher([8])
    sync.process_dataset(
        fetcher=fetcher,
        output_dir=tmp_path,
        docs_dir=tmp_path,
        spec=_DEMO_SPEC,
        fields=_DEMO_FIELDS,
        targets=[("20240331", {"period": "20240331"})],
        range_start="20240101",
        range_end="20241231",
        workers=1,
        force=False,
        resume=False,
        allow_shrink=True,
    )
    assert len(pd.read_parquet(path)) == 8

    # --force also bypasses the guard even without --allow-shrink.
    fetcher = _demo_period_fetcher([5])
    sync.process_dataset(
        fetcher=fetcher,
        output_dir=tmp_path,
        docs_dir=tmp_path,
        spec=_DEMO_SPEC,
        fields=_DEMO_FIELDS,
        targets=[("20240331", {"period": "20240331"})],
        range_start="20240101",
        range_end="20241231",
        workers=1,
        force=True,
        resume=False,
        allow_shrink=False,
    )
    assert len(pd.read_parquet(path)) == 5


def test_default_200_overlap_recovers_a_shuffled_boundary_row() -> None:
    spec = sync.DatasetSpec("demo", "demo", "demo.md", "period", 250)
    fields = [sync.FieldSpec("id", "int")]
    source = list(range(510))
    calls = []

    def query(_api, _fields, params):
        offset = params["offset"]
        calls.append(offset)
        ordered = source.copy()
        # Row 249 moves across the ordinary 250-row boundary between calls.
        if offset == 0:
            ordered[249], ordered[250] = ordered[250], ordered[249]
        return pd.DataFrame({"id": ordered[offset:offset + params["limit"]]})

    plain = sync.ApiFetcher("t", "unused", query_override=query, request_interval=0, page_overlap=0)
    assert 249 not in set(plain.fetch_paginated(spec, fields, {}).frame["id"])
    calls.clear()
    overlapping = sync.ApiFetcher("t", "unused", query_override=query, request_interval=0)
    result = overlapping.fetch_paginated(spec, fields, {})
    assert calls == [0, 50, 100, 150, 200, 250, 300]
    assert set(result.frame["id"]) == set(source)


def test_duplicate_only_full_page_does_not_stop_pagination() -> None:
    spec = sync.DatasetSpec("demo", "demo", "demo.md", "period", 3)
    pages = {0: [1, 2, 3], 2: [3, 2, 1], 4: [4]}
    fetcher = sync.ApiFetcher("t", "unused", request_interval=0, page_overlap=1,
                              query_override=lambda _a, _f, p: pd.DataFrame({"id": pages[p["offset"]]}))
    result = fetcher.fetch_paginated(spec, [sync.FieldSpec("id", "int")], {})
    assert set(result.frame["id"]) == {1, 2, 3, 4}
    assert result.pages == 3


@pytest.mark.parametrize("name,date_field", [("income", "f_ann_date"), ("cashflow", "f_ann_date"),
                                             ("balancesheet", "ann_date"), ("forecast", "ann_date")])
def test_financial_discovery_uses_exact_supported_date_without_period(name, date_field) -> None:
    spec = sync.DATASET_BY_NAME[name]
    fields = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)[name]
    calls = []

    def query(_api, projection, params):
        calls.append(dict(params))
        assert "period" not in params and "start_date" not in params
        assert params[date_field] == "20240924"
        row = {"ts_code": "600764.SH", "end_date": "20160930", date_field: "20240924"}
        if spec.mode == "statement":
            row["report_type"] = params["report_type"]
        return pd.DataFrame([row])

    fetcher = sync.ApiFetcher("t", "unused", query_override=query, request_interval=0)
    periods, audit = sync.discover_financial_periods(fetcher, spec, fields,
        dt.date(2024, 9, 24), dt.date(2024, 9, 24), dt.date(2016, 1, 1))
    assert periods == {"20160930"}
    assert audit["date_field"] == date_field
    if spec.mode == "statement":
        assert {p["report_type"] for p in calls} == {str(i) for i in range(1, 13)}


def test_discovery_fails_if_proxy_ignores_predicate() -> None:
    spec = sync.DATASET_BY_NAME["income"]
    fields = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)["income"]
    fetcher = sync.ApiFetcher("t", "unused", request_interval=0,
        query_override=lambda _a, _f, p: pd.DataFrame([{
            "ts_code": "A", "end_date": "20240331", "f_ann_date": "20240401", "report_type": p["report_type"],
        }]))
    with pytest.raises(sync.SourceSchemaError, match="ignored f_ann_date"):
        sync.discover_financial_periods(fetcher, spec, fields, dt.date(2024, 9, 24),
                                        dt.date(2024, 9, 24), dt.date(2024, 1, 1))


def _sync_rows(root, rows, *, spec=_DEMO_SPEC, fields=_DEMO_FIELDS, query=None, key="20240331", **options):
    fetcher = sync.ApiFetcher("t", "unused", request_interval=0, page_overlap=0,
                              query_override=lambda _a, _f, _p: pd.DataFrame(rows))
    return sync.process_dataset(fetcher, root, root, spec, fields,
        [(key, query or {"period": key})], "20240101", "20240924", 1,
        force=False, resume=False, run_id="test", **options)


def _rows(values=(1, 2)):
    return [{"ts_code": f"A{i}", "end_date": "20240331", "value": value} for i, value in enumerate(values)]


def test_partition_backup_contains_original_bytes_and_full_set_differences(tmp_path) -> None:
    _sync_rows(tmp_path, _rows())
    target = sync.partition_file(tmp_path, _DEMO_SPEC, "20240331")
    manifest_path = sync.manifest_path(tmp_path, _DEMO_SPEC)
    before, manifest_before = target.read_bytes(), manifest_path.read_bytes()
    _sync_rows(tmp_path, _rows((9, 2, 3)))
    logs = list((tmp_path / "_history").rglob("change.json"))
    assert len(logs) == 1
    log = sync.load_json(logs[0])
    assert log["status"] == "committed"
    assert (log["old_rows"], log["new_rows"], log["added_rows"], log["removed_rows"]) == (2, 3, 2, 1)
    assert (logs[0].parent / "before.parquet").read_bytes() == before
    assert (logs[0].parent / "before_manifest.json").read_bytes() == manifest_before
    assert set(pd.read_parquet(logs[0].parent / "added.parquet")["value"]) == {9, 3}
    assert pd.read_parquet(logs[0].parent / "removed.parquet")["value"].tolist() == [1]
    assert len(pd.read_parquet(tmp_path / "demo")) == 3
    current_bytes = target.read_bytes()
    _sync_rows(tmp_path, list(reversed(_rows((9, 2, 3)))))
    assert target.read_bytes() == current_bytes
    assert len(list((tmp_path / "_history").rglob("change.json"))) == 1


def test_legacy_duplicates_are_backed_up_and_removed_without_false_shrink(tmp_path) -> None:
    _sync_rows(tmp_path, _rows())
    target = sync.partition_file(tmp_path, _DEMO_SPEC, "20240331")
    old = pd.read_parquet(target)
    checksum, size = sync.atomic_write_parquet(pd.concat([old] * 3, ignore_index=True), target)
    path = sync.manifest_path(tmp_path, _DEMO_SPEC)
    manifest = sync.load_json(path)
    manifest["partitions"]["20240331"].update(sha256=checksum, bytes=size, rows=6)
    sync.atomic_write_json(path, manifest)
    _sync_rows(tmp_path, _rows())
    log = sync.load_json(next((tmp_path / "_history").rglob("change.json")))
    assert log["old_duplicate_rows"] == 4
    assert log["added_rows"] == log["removed_rows"] == 0
    assert len(pd.read_parquet(target)) == 2


def test_backup_failure_leaves_data_and_manifest_untouched(tmp_path, monkeypatch) -> None:
    import _storage
    _sync_rows(tmp_path, _rows())
    target = sync.partition_file(tmp_path, _DEMO_SPEC, "20240331")
    manifest = sync.manifest_path(tmp_path, _DEMO_SPEC)
    before = target.read_bytes(), manifest.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("backup disk unavailable")

    monkeypatch.setattr(_storage.shutil, "copy2", fail)
    with pytest.raises(OSError, match="backup disk"):
        _sync_rows(tmp_path, _rows((9, 2)))
    assert (target.read_bytes(), manifest.read_bytes()) == before
    assert not list((tmp_path / "_transactions").glob("*/journal.json"))


def test_interrupted_commit_recovers_data_and_manifest_together(tmp_path, monkeypatch) -> None:
    import _storage
    _sync_rows(tmp_path, _rows())
    target = sync.partition_file(tmp_path, _DEMO_SPEC, "20240331")
    manifest = sync.manifest_path(tmp_path, _DEMO_SPEC)
    before = target.read_bytes()
    replace = _storage.os.replace

    def crash(source, destination):
        if Path(source).name == "manifest.json" and destination == manifest:
            raise OSError("simulated interruption")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(_storage.os, "replace", crash)
        with pytest.raises(OSError, match="interruption"):
            _sync_rows(tmp_path, _rows((9, 2, 3)))
    assert len(pd.read_parquet(target)) == 3
    assert sync.load_json(manifest)["partitions"]["20240331"]["rows"] == 2
    candidate_bytes = target.read_bytes()
    target.write_bytes(b"external modification")
    with sync.archive_lock(tmp_path), pytest.raises(sync.SyncError, match="external change"):
        sync.recover_transactions(tmp_path)
    assert target.read_bytes() == b"external modification"
    target.write_bytes(candidate_bytes)
    with sync.archive_lock(tmp_path):
        sync.recover_transactions(tmp_path)
    assert sync.load_json(manifest)["partitions"]["20240331"]["rows"] == 3
    log_path = next((tmp_path / "_history").rglob("change.json"))
    assert (log_path.parent / "before.parquet").read_bytes() == before
    assert sync.load_json(log_path)["status"] == "committed"
    assert not list((tmp_path / "_transactions").iterdir())


def test_sync_and_standalone_patch_share_the_writer_lock(tmp_path) -> None:
    import patch_industry_data as patch
    with sync.archive_lock(tmp_path):
        with pytest.raises(sync.SyncError, match="archive lock"):
            with sync.archive_lock(tmp_path):
                pass
        with pytest.raises(patch.PatchError, match="archive lock"):
            with patch.archive_lock(tmp_path):
                pass


def test_event_window_replaces_inside_and_keeps_outside(tmp_path) -> None:
    spec = sync.DATASET_BY_NAME["stk_holdernumber"]
    fields = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)[spec.name]
    rows = [{"ts_code": code, "ann_date": ann, "end_date": "20240331", "holder_num": value}
            for code, ann, value in [("A", "20240401", 1), ("B", "20240923", 2), ("C", "20240923", 3)]]
    common = dict(spec=spec, fields=fields, key="2024")
    _sync_rows(tmp_path, rows, query={"start_date": "20240101", "end_date": "20240923"}, **common)
    _sync_rows(tmp_path, [{**rows[1], "holder_num": 20}],
               query={"start_date": "20240923", "end_date": "20240924"}, **common)
    frame = pd.read_parquet(sync.partition_file(tmp_path, spec, "2024"))
    assert set(frame["ts_code"]) == {"A", "B"}
    assert set(frame["holder_num"]) == {1, 20}
    entry = sync.load_json(sync.manifest_path(tmp_path, spec))["partitions"]["2024"]
    assert entry["query"] == {"start_date": "20240101", "end_date": "20240924"}
    before = sync.partition_file(tmp_path, spec, "2024").read_bytes()
    with pytest.raises(sync.DatasetRunError, match="0 rows"):
        _sync_rows(tmp_path, [], query={"start_date": "20240923", "end_date": "20240924"}, **common)
    assert sync.partition_file(tmp_path, spec, "2024").read_bytes() == before


def test_event_targets_catch_up_and_fill_missing_years() -> None:
    manifest = {"partitions": {"2023": {}, "2024": {"query": {}}},
                "update_state": {"discovery_through": "20241230"}}
    targets = sync.update_targets(sync.DATASET_BY_NAME["stk_holdernumber"], manifest,
        dt.date(2023, 1, 1), dt.date(2025, 1, 4), 8, 2)
    assert targets == [
        ("2023", {"start_date": "20230101", "end_date": "20231231"}),
        ("2024", {"start_date": "20241229", "end_date": "20241231"}),
        ("2025", {"start_date": "20250101", "end_date": "20250104"}),
    ]


def test_history_rotation_wraps_and_includes_recent_quarters_when_not_refreshed() -> None:
    manifest = {"partitions": {p: {} for p in ("20240331", "20240630", "20240930", "20241231")},
                "update_state": {"history_cursor": "20240930"}}
    assert sync.historical_rotation(manifest, dt.date(2024, 1, 1), dt.date(2024, 12, 31), 0, 2) == ["20241231", "20240331"]
    assert sync.historical_rotation(manifest, dt.date(2024, 1, 1), dt.date(2024, 12, 31), 2, 1) == ["20240331"]


def test_industry_patch_runs_before_publish_and_blocks_unreviewed_changes(tmp_path) -> None:
    import patch_industry_data as patch
    spec = sync.DATASET_BY_NAME["index_member_all"]
    fields = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)[spec.name]
    case = patch.REVIEWED_CASES[0]
    good, bad = patch.case_row(case, patch.RETAIN_PATH), patch.case_row(case, patch.WRONG_PATH)
    bad["out_date"] = ""  # The proxy has both representations of an open interval.
    rows = [bad, good, dict(good)]

    def query(_api, _fields, params):
        return pd.DataFrame([row for row in rows if row["is_new"] == params["is_new"]])

    fetcher = sync.ApiFetcher("t", "unused", query_override=query, request_interval=0)
    common = dict(fetcher=fetcher, output_dir=tmp_path, docs_dir=tmp_path, spec=spec, fields=fields,
        targets=[("all", {"start_date": "20160101", "end_date": "20240924"})],
        range_start="20160101", range_end="20240924", workers=1, force=False, resume=False,
        run_id="industry_test")
    manifest = sync.process_dataset(**common)
    target = sync.partition_file(tmp_path, spec, "all")
    assert pd.read_parquet(target).to_dict("records") == [good]
    assert manifest["partitions"]["all"]["industry_patch_id"] == patch.PATCH_ID
    before = target.read_bytes()
    sync.process_dataset(**common)
    assert not (tmp_path / "_history").exists()
    rows[0] = {**bad, "name": "changed case"}
    with pytest.raises(sync.DatasetRunError, match="requires review"):
        sync.process_dataset(**common)
    assert target.read_bytes() == before
    report = sync.load_json(tmp_path / "_runs" / "industry_test" / "index_member_all_audit.json")
    assert report["blockers"]


def test_ci_conflicts_are_logged_without_applying_sw_patch(tmp_path) -> None:
    import patch_industry_data as patch
    spec = sync.DATASET_BY_NAME["ci_index_member"]
    fields = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)[spec.name]
    row = patch.case_row(("123456.SZ", "sample", "20240101"), patch.RETAIN_PATH)
    rows = [row, {**row, "l1_name": "another industry"}]
    fetcher = sync.ApiFetcher("t", "unused", request_interval=0,
        query_override=lambda _a, _f, p: pd.DataFrame(rows if p["is_new"] == "Y" else []))
    manifest = sync.process_dataset(fetcher, tmp_path, tmp_path, spec, fields,
        [("all", {"start_date": "20240101", "end_date": "20240924"})],
        "20240101", "20240924", 1, False, False, run_id="ci_test")
    entry = manifest["partitions"]["all"]
    assert entry["rows"] == 2
    assert any("unresolved conflicts" in warning for warning in entry["warnings"])
    assert "industry_patch_id" not in entry


def test_update_replaces_full_quarter_and_keeps_failed_cross_dataset_work(tmp_path, monkeypatch) -> None:
    """Exercise CLI backfill, discovery, version migration, failure and a later retry."""
    names = ["income", "balancesheet", "fina_indicator"]
    statement_fields = [sync.FieldSpec(name, "int" if name == "value" else "str") for name in
                        ("ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "value")]
    indicator_fields = [field for field in statement_fields if field.name not in {"f_ann_date", "report_type"}]
    fields = {name: indicator_fields if name == "fina_indicator" else statement_fields for name in names}
    data = {name: [] for name in names}
    for name in names:
        for period, ann in [("20240331", "20240401"), ("20240630", "20240701")]:
            row = {"ts_code": "A", "ann_date": ann, "end_date": period, "value": 10}
            if name != "fina_indicator":
                row.update(f_ann_date=ann, report_type="1")
            data[name].append(row)
    calls, failures = [], {"balance": False}

    def query(api, projection, params):
        name = sync.API_ALIAS_TO_DATASET[api]
        calls.append((name, dict(params)))
        if failures["balance"] and name == "balancesheet" and params.get("period") == "20240331":
            raise sync.SyncError("simulated balance failure")
        rows = data[name]
        for key, value in params.items():
            if key in {"offset", "limit"}:
                continue
            field = "end_date" if key == "period" else key
            rows = [row for row in rows if row.get(field) == value]
        return pd.DataFrame(rows, columns=projection.split(",")).iloc[params["offset"]:params["offset"] + params["limit"]]

    fetcher = sync.ApiFetcher("t", "unused", query_override=query, request_interval=0, max_retries=1)
    monkeypatch.setattr(sync, "build_fetcher", lambda _args: fetcher)
    monkeypatch.setattr(sync, "load_all_fields", lambda _path: fields)
    common = ["--output-dir", str(tmp_path), "--apis", ",".join(names)]
    assert sync.main(["backfill", *common, "--start-date", "20240101", "--end-date", "20240923"]) == 0
    update = ["update", *common, "--announcement-lookback-days", "1", "--financial-lookback-quarters", "1",
              "--history-quarters-per-run", "0"]
    assert sync.main([*update, "--as-of", "20240923"]) == 0

    old_income = dict(data["income"][0])
    data["income"][0] = {**old_income, "f_ann_date": "20240924", "value": 9}
    data["income"].append({**old_income, "report_type": "5"})
    data["balancesheet"][0]["value"] = 9
    data["fina_indicator"][0]["value"] = 9
    failures["balance"] = True
    assert sync.main([*update, "--as-of", "20240924"]) == 1
    income_path = sync.partition_file(tmp_path, sync.DATASET_BY_NAME["income"], "20240331")
    current = pd.read_parquet(income_path)
    assert set(zip(current["report_type"], current["value"])) == {("1", 9), ("5", 10)}
    balance_path = sync.manifest_path(tmp_path, sync.DATASET_BY_NAME["balancesheet"])
    state = sync.load_json(balance_path)["update_state"]
    assert state["discovery_through"] == "20240923"
    assert state["pending_periods"] == ["20240331"]
    indicator = pd.read_parquet(sync.partition_file(tmp_path, sync.DATASET_BY_NAME["fina_indicator"], "20240331"))
    assert indicator["value"].tolist() == [9]

    # Advance the source independently, so the old trigger is outside its next
    # window. The failed recipient must still consume its persisted pending set.
    income_only = ["update", "--output-dir", str(tmp_path), "--apis", "income", "--announcement-lookback-days", "1",
                   "--financial-lookback-quarters", "1", "--history-quarters-per-run", "0"]
    assert sync.main([*income_only, "--as-of", "20240925"]) == 0
    calls.clear()
    failures["balance"] = False
    assert sync.main([*update, "--as-of", "20240926"]) == 0
    assert not any(name == "income" and params.get("period") == "20240331" for name, params in calls)
    assert any(name == "balancesheet" and params.get("period") == "20240331" for name, params in calls)
    state = sync.load_json(balance_path)["update_state"]
    assert state["discovery_through"] == "20240926" and state["pending_periods"] == []
    assert sync.main(["verify", *common]) == 0
    assert list((tmp_path / "_runs").glob("*/run.log"))

    # A recipient with an unusable manifest cannot durably accept triggers.
    # Refuse the run before any source advances its cursor.
    broken_manifest = sync.load_json(balance_path)
    broken_manifest["schema_hash"] = "wrong schema"
    sync.atomic_write_json(balance_path, broken_manifest)
    data["income"][0]["f_ann_date"] = "20240927"
    assert sync.main([*update, "--as-of", "20240927"]) == 1
    income_state = sync.load_json(sync.manifest_path(tmp_path, sync.DATASET_BY_NAME["income"]))["update_state"]
    assert income_state["discovery_through"] == "20240926"


@pytest.mark.parametrize("command", ["backfill", "update", "verify", "smoke"])
def test_cli_help_renders(command, capsys) -> None:
    with pytest.raises(SystemExit) as result:
        sync.build_parser().parse_args([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_null_report_period_is_rejected() -> None:
    with pytest.raises(sync.SourceSchemaError, match="end_date"):
        sync.validate_period_frame(pd.DataFrame({"end_date": [None]}), "20240331", "demo")
