#!/usr/bin/env python3
"""Mirror a Parquet directory into one ClickHouse table, preserving row multiplicity.

Recursively reads every *.parquet file. Creates a Distributed table and matching
MergeTree tables, loads and verifies ALL columns and duplicate counts,
then exchanges the Distributed entry points (atomic on each cluster node).
No append/resume, implicit deduplication, date parsing, or missing-value filling.

Requires: clickhouse-connect, pyarrow, duckdb; python-dotenv for --env-file.
The database must already exist and use Atomic on every cluster node. Each run
uses fresh physical table names: a query sees one complete old or new snapshot,
even while the cluster-wide metadata change is propagating.
The destination is an archive mirror and must not have independent writers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import socket
import tempfile
import time
import uuid

import clickhouse_connect
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


LOG = logging.getLogger("parquet_to_clickhouse")
TABLE_MARKER = "parquet_mirror_v1:"


def ch_name(name: str) -> str:
    return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"


def sql_name(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def ch_literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def split_target(target: str) -> tuple[str, str]:
    parts = target.split(".")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
        raise ValueError("target must be database.table (letters, digits and underscores)")
    return parts[0], parts[1]


def clickhouse_type(field: pa.Field) -> str:
    """Map scalar Arrow types without converting strings to dates or losing NULLs."""
    typ = field.type
    if pa.types.is_string(typ) or pa.types.is_large_string(typ) or pa.types.is_binary(typ) or pa.types.is_large_binary(typ):
        result = "String"
    elif pa.types.is_boolean(typ):
        result = "Bool"
    elif pa.types.is_integer(typ):
        result = ("UInt" if pa.types.is_unsigned_integer(typ) else "Int") + str(typ.bit_width)
    elif pa.types.is_float32(typ):
        result = "Float32"
    elif pa.types.is_float64(typ):
        result = "Float64"
    elif pa.types.is_date32(typ) or pa.types.is_date64(typ):
        result = "Date32"
    elif pa.types.is_timestamp(typ):
        if typ.tz and not re.fullmatch(r"[A-Za-z0-9_+:/-]+", typ.tz):
            raise ValueError(f"Unsupported timezone: {typ.tz}")
        precision = {"s": 0, "ms": 3, "us": 6, "ns": 9}[typ.unit]
        result = f"DateTime64({precision}, '{typ.tz or 'UTC'}')"
    elif pa.types.is_decimal128(typ) and 0 <= typ.scale <= typ.precision:
        result = f"Decimal({typ.precision}, {typ.scale})"
    else:
        raise TypeError(f"Unsupported Arrow field {field.name!r}: {typ}; no lossy conversion is attempted")
    return f"Nullable({result})" if field.nullable else result


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rows: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class Snapshot:
    root: Path
    schema: pa.Schema
    files: tuple[SourceFile, ...]

    @property
    def rows(self) -> int:
        return sum(item.rows for item in self.files)

    @property
    def digest(self) -> str:
        content = [(str(f.path.relative_to(self.root)), f.sha256) for f in self.files]
        return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def inspect_source(root: Path) -> Snapshot:
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No Parquet files in {root}; refusing to treat a wrong/empty directory as an empty dataset")
    schema = pq.ParquetFile(paths[0]).schema_arrow.remove_metadata()
    if not schema.names or len({name.casefold() for name in schema.names}) != len(schema.names):
        raise ValueError("Parquet columns must be non-empty and distinct, including case")
    for field in schema:
        clickhouse_type(field)
    records = []
    for path in paths:
        before = path.stat()
        pf = pq.ParquetFile(path)
        if not schema.equals(pf.schema_arrow.remove_metadata()):
            raise ValueError(f"Parquet schema differs: {path}")
        digest = file_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while reading: {path}")
        records.append(SourceFile(path, pf.metadata.num_rows, after.st_size, after.st_mtime_ns, digest))
    return Snapshot(root, schema, tuple(records))


def assert_source_unchanged(snapshot: Snapshot) -> None:
    if sorted(snapshot.root.rglob("*.parquet")) != [f.path for f in snapshot.files]:
        raise RuntimeError("Parquet file list changed during import")
    for item in snapshot.files:
        stat = item.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item.size, item.mtime_ns) or file_hash(item.path) != item.sha256:
            raise RuntimeError(f"Source changed during import: {item.path}")


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def archive_root(source: Path) -> Path:
    """Share the existing tushare-fetcher writer lock when inside its data tree."""
    return next((p for p in (source, *source.parents) if (p / "_catalog.json").is_file()), source)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def table_info(client, database: str, table: str):
    rows = client.query(
        "SELECT toString(uuid), engine, comment FROM system.tables WHERE database = {db:String} AND name = {table:String}",
        parameters={"db": database, "table": table},
    ).result_rows
    return tuple(rows[0]) if rows else None


def load_batches(client, table: str, snapshot: Snapshot, batch_rows: int) -> int:
    pending, rows, total = [], 0, 0
    last_log = time.monotonic()
    for item in snapshot.files:
        for batch in pq.ParquetFile(item.path).iter_batches(batch_size=batch_rows):
            pending.append(batch)
            rows += batch.num_rows
            if rows >= batch_rows:
                client.insert_arrow(table, pa.Table.from_batches(pending, schema=snapshot.schema),
                                    settings={"async_insert": 0, "insert_deduplicate": 0,
                                              "insert_distributed_sync": 1})
                total += rows
                pending, rows = [], 0
                if time.monotonic() - last_log >= 10:
                    LOG.info("Inserted %s / %s rows", f"{total:,}", f"{snapshot.rows:,}")
                    last_log = time.monotonic()
    if pending:
        client.insert_arrow(table, pa.Table.from_batches(pending, schema=snapshot.schema),
                            settings={"async_insert": 0, "insert_deduplicate": 0,
                                      "insert_distributed_sync": 1})
        total += rows
    return total


def exact_difference(source_files: list[str], exported: Path, names: list[str], work: Path, memory_limit: str) -> dict:
    """Compare full rows as a multiset; equality of count/hash alone is insufficient."""
    columns = ", ".join(sql_name(n) for n in names)
    marker = "__delta_" + uuid.uuid4().hex
    delta = sql_name(marker)
    with duckdb.connect() as db:
        db.execute("SET memory_limit = ?", [memory_limit])
        db.execute("SET temp_directory = ?", [str(work / "duckdb_spill")])
        db.execute("SET threads = 2")
        # GROUP BY compares the actual values, including NULL/NaN, and the signed
        # count preserves duplicate multiplicity. DuckDB can spill to disk.
        missing, extra = db.execute(
            f"""
            SELECT coalesce(sum(greatest({delta}, 0)), 0), coalesce(sum(greatest(-{delta}, 0)), 0)
            FROM (
                SELECT {columns}, sum({delta}) AS {delta}
                FROM (
                    SELECT {columns}, 1::BIGINT AS {delta} FROM read_parquet(?, hive_partitioning=false)
                    UNION ALL
                    SELECT {columns}, -1::BIGINT AS {delta} FROM read_parquet(?, hive_partitioning=false)
                )
                GROUP BY {columns}
                HAVING sum({delta}) != 0
            )
            """,
            [source_files, str(exported)],
        ).fetchone()
    return {"missing_rows": int(missing), "extra_rows": int(extra)}


def verify_table(client, table: str, snapshot: Snapshot, work: Path, memory_limit: str) -> dict:
    row_count = int(client.command(f"SELECT count() FROM {table}"))
    if row_count != snapshot.rows:
        raise RuntimeError(f"Row count mismatch: source={snapshot.rows}, table={row_count}")
    columns = ", ".join(ch_name(n) for n in snapshot.schema.names)
    exported = work / "clickhouse.parquet"
    downloaded = 0
    # Arrow keeps Float64 precision, NULL and non-finite values without a CSV round trip.
    with pq.ParquetWriter(exported, snapshot.schema, compression="zstd") as writer:
        with client.query_arrow_stream(f"SELECT {columns} FROM {table}", use_strings=False) as stream:
            for batch in stream:
                block = pa.Table.from_batches([batch]).cast(snapshot.schema, safe=True)
                writer.write_table(block)
                downloaded += block.num_rows
    if downloaded != snapshot.rows:
        raise RuntimeError(f"Readback incomplete: expected {snapshot.rows}, received {downloaded}")
    diff = exact_difference([str(f.path) for f in snapshot.files], exported, snapshot.schema.names, work, memory_limit)
    if diff["missing_rows"] or diff["extra_rows"]:
        raise RuntimeError(f"Full-row verification failed: {diff}")
    return {"rows": row_count, "columns": len(snapshot.schema), **diff}


def sync_directory(client, source: Path, target: str, *, cluster: str = "cb_cluster", order_by=(), lock_root: Path | None = None,
                   batch_rows: int = 100_000, temp_dir: Path | None = None,
                   memory_limit: str = "1GB", report_path: Path | None = None) -> dict:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir() or batch_rows <= 0:
        raise ValueError("source must be a directory and batch_rows must be positive")
    database, name = split_target(target)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cluster):
        raise ValueError("Invalid cluster name")
    on_cluster = f" ON CLUSTER {ch_name(cluster)}"
    destination = f"{ch_name(database)}.{ch_name(name)}"
    root = lock_root.expanduser().resolve(strict=True) if lock_root else archive_root(source)
    if not source.is_relative_to(root):
        raise ValueError("lock_root must contain the source directory")
    report_path = report_path or root / "_clickhouse_sync" / f"{target}.json"
    # All invocations on this host, even with different source directories, share
    # this target lock. Other database writers must respect the mirror's ownership.
    identity = f"{cluster}|{target}".encode()
    lock = Path(tempfile.gettempdir()) / ("parquet-clickhouse-" + hashlib.sha256(identity).hexdigest() + ".lock")
    with file_lock(lock), file_lock(root / ".sync.lock"):
        if any((root / "_transactions").glob("*/journal.json")):
            raise RuntimeError("Archive has an interrupted commit; run its normal recovery before importing")
        if not client.command(f"EXISTS DATABASE {ch_name(database)}"):
            raise ValueError(f"Destination database does not exist: {database}")
        engines = client.query("SELECT engine FROM system.databases WHERE name = {db:String}", parameters={"db": database}).result_rows
        if engines != [("Atomic",)]:
            raise ValueError("This importer requires an Atomic database on a fixed ClickHouse node")
        # An explicit existence check surfaces permission errors rather than
        # mistaking system.tables visibility filtering for a missing table.
        client.command(f"EXISTS TABLE {destination}")
        before = table_info(client, database, name)
        old_backing = None
        if before:
            if before[1] != "Distributed" or not before[2].startswith(TABLE_MARKER):
                raise ValueError("Existing target is not a Distributed mirror owned by this importer")
            metadata = json.loads(before[2][len(TABLE_MARKER):])
            old_backing = metadata['backing']
            if metadata['cluster'] != cluster or not re.fullmatch(re.escape(name) + r"__data_[0-9a-f]{32}", old_backing):
                raise ValueError("Target mirror metadata/cluster differs")
        snapshot = inspect_source(source)
        if any(col not in snapshot.schema.names for col in order_by) or len(set(order_by)) != len(order_by):
            raise ValueError("order_by must contain distinct existing column names")
        run_id = uuid.uuid4().hex
        staging_name = name + "__loading_" + run_id
        staging = f"{ch_name(database)}.{ch_name(staging_name)}"
        backing_name = name + "__data_" + run_id
        backing = f"{ch_name(database)}.{ch_name(backing_name)}"
        report = {
            "run_id": run_id, "source": str(source), "target": target,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_files": len(snapshot.files), "source_rows": snapshot.rows,
            "source_bytes": sum(f.size for f in snapshot.files), "source_sha256": snapshot.digest,
            "schema": [(f.name, clickhouse_type(f)) for f in snapshot.schema],
            "staging_table": database + "." + staging_name,
            "backing_table": database + "." + backing_name, "cluster": cluster, "status": "loading",
        }
        write_report(report_path, report)
        LOG.info("Source: %s files, %s rows; target %s", f"{len(snapshot.files):,}", f"{snapshot.rows:,}", target)
        definitions = ", ".join(f"{ch_name(f.name)} {clickhouse_type(f)}" for f in snapshot.schema)
        sorting = "(" + ", ".join(ch_name(c) for c in order_by) + ")" if order_by else "tuple()"
        created = False
        publishing = False
        start_clock = time.monotonic()
        try:
            # Unique run namespace permits safe cleanup even if a CREATE response
            # is lost after some cluster nodes have already created their table.
            created = True
            client.command(f"CREATE TABLE {backing}{on_cluster} ({definitions}) "
                           f"ENGINE = MergeTree "
                           f"ORDER BY {sorting} SETTINGS allow_nullable_key = 1")
            comment = TABLE_MARKER + json.dumps({'cluster': cluster, 'backing': backing_name})
            client.command(f"CREATE TABLE {staging}{on_cluster} ({definitions}) "
                           f"ENGINE = Distributed({ch_literal(cluster)}, {ch_literal(database)}, {ch_literal(backing_name)}, rand()) "
                           f"COMMENT {ch_literal(comment)}")
            staged_uuid = table_info(client, database, staging_name)[0]
            inserted = load_batches(client, staging, snapshot, batch_rows)
            if inserted != snapshot.rows:
                raise RuntimeError(f"Source read count changed: expected {snapshot.rows}, read {inserted}")
            report["status"] = "verifying"
            write_report(report_path, report)
            LOG.info("Reading back all columns and comparing full rows, including duplicate counts")
            with tempfile.TemporaryDirectory(prefix="parquet-ch-verify-", dir=temp_dir) as work_dir:
                report["verification"] = verify_table(client, staging, snapshot, Path(work_dir), memory_limit)
            assert_source_unchanged(snapshot)
            if table_info(client, database, name) != before:
                raise RuntimeError("Destination identity changed during import; refusing to publish")
            report.update(status="publishing", staged_uuid=staged_uuid)
            write_report(report_path, report)
            publishing = True
            if before:
                client.command(f"EXCHANGE TABLES {staging} AND {destination}{on_cluster}")
            else:
                client.command(f"RENAME TABLE {staging} TO {destination}{on_cluster}")
            if table_info(client, database, name)[0] != staged_uuid:
                raise RuntimeError("Published table UUID differs from the verified staging table")
            final_count = int(client.command(f"SELECT count() FROM {destination}"))
            if final_count != snapshot.rows:
                raise RuntimeError(f"Published row count differs: {final_count}")
            report.update(status="complete", target_rows=final_count,
                          completed_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=round(time.monotonic() - start_clock, 3))
            write_report(report_path, report)
            if before:
                # Wait for queries using every old Distributed entry point before
                # deleting the old physical snapshot they reference.
                client.command(f"DROP TABLE {staging}{on_cluster} SYNC")
                client.command(f"DROP TABLE {ch_name(database)}.{ch_name(old_backing)}{on_cluster} SYNC")
            LOG.info("Complete: %s rows; missing=0, extra=0; report %s", f"{final_count:,}", report_path)
            return report
        except BaseException as exc:
            # An EXCHANGE whose HTTP response is lost must NOT be repeated: doing
            # so could swap the old data back. Preserve both names for inspection.
            if report["status"] != "complete":
                report.update(status="publish_unconfirmed" if publishing else "failed", error=str(exc))
            else:
                report["cleanup_error"] = str(exc)
            write_report(report_path, report)
            if created and not publishing:
                try:
                    client.command(f"DROP TABLE IF EXISTS {staging}{on_cluster} SYNC")
                    client.command(f"DROP TABLE IF EXISTS {backing}{on_cluster} SYNC")
                except Exception:
                    LOG.exception("Could not remove this run's staging table %s", staging)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Recursively import every *.parquet in this directory")
    parser.add_argument("target", help="database.table; successful runs replace its entire contents")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--username-env", default="MINGHU_CLICKHOUSE_USERNAME")
    parser.add_argument("--password-env", default="MINGHU_CLICKHOUSE_PASSWORD")
    parser.add_argument("--host", help="Default: MINGHU_CLICKHOUSE_HOST or chdb.tradegdb.com")
    parser.add_argument("--port", type=int)
    parser.add_argument("--secure", action="store_true")
    parser.add_argument("--cluster", default="cb_cluster", help="Cluster used by both Distributed and local tables")
    parser.add_argument("--order-by", nargs="*", default=[])
    parser.add_argument("--lock-root", type=Path, help="Archive writer lock directory; autodetected for tushare-fetcher")
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--temp-dir", type=Path, help="Work disk for streaming readback and external comparison")
    parser.add_argument("--memory-limit", default="1GB", help="DuckDB verification memory limit; excess spills to disk")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = args.host or os.getenv("MINGHU_CLICKHOUSE_HOST", "chdb.tradegdb.com")
    # DNS has several nodes here. Resolve once so streaming readback cannot move
    # the coordinator between nodes inside the same import.
    address = socket.gethostbyname(host)
    client = clickhouse_connect.get_client(
        host=address, server_host_name=host,
        port=args.port or int(os.getenv("MINGHU_CLICKHOUSE_PORT", "8123")),
        username=os.environ[args.username_env], password=os.environ[args.password_env],
        secure=args.secure, send_receive_timeout=600, query_retries=0,
    )
    try:
        sync_directory(client, args.source, args.target, cluster=args.cluster, order_by=args.order_by, lock_root=args.lock_root,
                       batch_rows=args.batch_rows, temp_dir=args.temp_dir,
                       memory_limit=args.memory_limit, report_path=args.report)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
