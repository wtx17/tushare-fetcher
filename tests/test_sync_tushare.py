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
}


def test_parse_all_documented_fields() -> None:
    parsed = sync.load_all_fields(sync.DEFAULT_DOCS_DIR)
    assert {name: len(fields) for name, fields in parsed.items()} == EXPECTED_FIELD_COUNTS
    assert sum(EXPECTED_FIELD_COUNTS.values()) == 599
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


def test_paginated_fetch_removes_only_cross_page_overlap() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "unused.md", "period", 2)
    fields = [sync.FieldSpec("code", "str"), sync.FieldSpec("value", "int")]

    def fake_query(_api: str, _fields: str, params: dict[str, object]) -> pd.DataFrame:
        offset = int(params["offset"])
        if offset == 0:
            # An upstream duplicate inside one page must be preserved.
            return pd.DataFrame({"code": ["A", "A"], "value": [1, 1]})
        if offset == 2:
            # A repeats the prior page and should be removed; B is new.
            return pd.DataFrame({"code": ["A", "B"], "value": [1, 2]})
        return pd.DataFrame(columns=["code", "value"])

    fetcher = sync.ApiFetcher(
        token="test-token",
        api_url="https://example.invalid",
        query_override=fake_query,
        request_interval=0,
    )
    result = fetcher.fetch_paginated(spec, fields, {"period": "20241231"})
    assert result.pages == 3
    assert result.raw_rows == 4
    assert result.last_page_rows == 0
    assert result.cross_page_duplicates == 1
    assert result.frame["code"].tolist() == ["A", "A", "B"]


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
    # empty over a non-empty existing partition is always refused...
    with pytest.raises(sync.OverwriteGuardError, match="0 rows"):
        sync.check_overwrite_safety(spec, "k", new_empty, {"rows": 5}, allow_shrink=False)
    # ...even when --allow-shrink is set.
    with pytest.raises(sync.OverwriteGuardError, match="0 rows"):
        sync.check_overwrite_safety(spec, "k", new_empty, {"rows": 5}, allow_shrink=True)


def test_overwrite_guard_refuses_row_shrink() -> None:
    spec = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
    smaller = pd.DataFrame({"ts_code": pd.Series(["A", "B"], dtype="string")})
    with pytest.raises(sync.OverwriteGuardError, match="shrank 10 -> 2"):
        sync.check_overwrite_safety(spec, "k", smaller, {"rows": 10}, allow_shrink=False)


def test_allow_shrink_permits_partial_shrink_only() -> None:
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
    )


_DEMO_SPEC = sync.DatasetSpec("demo", "demo_api", "demo.md", "period", 100)
_DEMO_FIELDS = [
    sync.FieldSpec("ts_code", "str"),
    sync.FieldSpec("end_date", "str"),
    sync.FieldSpec("value", "int"),
]


def test_update_refusal_preserves_existing_partition(tmp_path: Path) -> None:
    fetcher = _demo_period_fetcher([10, 8])
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

    # second pass returns only 8 rows -> guard refuses, old data survives
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
