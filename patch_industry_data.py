#!/usr/bin/env python3
"""Case-by-case patch for downloaded Tushare industry archives (Python >=3.11).

Default: read-only preview. No network, credentials, or sync-module imports.

    python patch_industry_data.py --data-dir ./data --report ./industry_review.json
    python patch_industry_data.py --data-dir ./data --apply
    python patch_industry_data.py --data-dir ./data --restore /path/to/backup

Only three reviewed erroneous video-media rows are removed. Their existing
oil-and-gas counterparts, all other stocks, and ci_index_member remain intact.
See industry_patch_notes.md for evidence and the unresolved cases.

Stop concurrent archive readers/sync jobs during apply/restore. Each file is
replaced atomically, but the Parquet/manifest pair is NOT one atomic transaction.
A prepared backup journal supports recovery after interruption between writes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


PATCH_ID = "sw_oil_video_media_20260909_v1"
COLUMNS = (
    "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
    "ts_code", "name", "in_date", "out_date", "is_new",
)
DATASET = "index_member_all"
FILES = ("data.parquet", "_manifest.json")

# Explicit reviewed cases, NOT a general mapping or a stock blacklist.
# Local SwClass.rar/StockClassifyUse_stock.xls places all three at old SW
# industry 210101 with exactly these entry dates. The issuer-hosted independent
# financial adviser report corroborates their oil exploration/production business:
# https://www.petrochina.com.cn/petrochina/gsgg/201403/e66ded96373f440c958a9fe41b41d810/files/083b299c9c3f47bfa5cd6f3ca702eb29.pdf
# Retain an existing oil row; never synthesize historical labels or dates.
REVIEWED_CASES = (
    ("000406.SZ", "石油大明(退市)", "19960628"),
    ("000817.SZ", "辽河油田(退市)", "19980528"),
    ("000956.SZ", "中原油气(退市)", "19991110"),
)
WRONG_PATH = ("801760.SI", "传媒", "801767.SI", "数字媒体", "857671.SI", "视频媒体")
RETAIN_PATH = ("801960.SI", "石油石化", "801961.SI", "油气开采Ⅱ", "859611.SI", "油气开采Ⅲ")


class PatchError(RuntimeError):
    """An unreviewed case, inconsistent archive, or unsafe recovery request."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Snapshot:
    root: Path
    dataset: str
    table: pa.Table
    manifest: dict
    payloads: dict[str, bytes]

    @property
    def hashes(self) -> dict[str, str]:
        return {name: sha256(data) for name, data in self.payloads.items()}


def read_snapshot(data_dir: Path, dataset: str = DATASET) -> Snapshot:
    root = Path(data_dir).expanduser().resolve()
    if (root / dataset).is_symlink():
        raise PatchError(f"Refusing a symlinked dataset directory: {root / dataset}")
    paths = {name: root / dataset / name for name in FILES}
    for path in paths.values():
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PatchError(f"Refusing a symlink or path outside the archive: {path}")
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    manifest = json.loads(payloads["_manifest.json"])
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1 or manifest.get("dataset") != dataset:
        raise PatchError(f"Unsupported manifest for {dataset}")
    if set(manifest.get("partitions", {})) != {"all"}:
        raise PatchError(f"{dataset}: this patch only supports the reviewed all partition")
    entry = manifest["partitions"]["all"]
    if entry.get("relative_path") != f"{dataset}/data.parquet":
        raise PatchError(f"{dataset}: unexpected partition path")
    table = pq.read_table(BytesIO(payloads["data.parquet"]))
    if (
        table.column_names != list(COLUMNS)
        or [item["name"] for item in manifest.get("fields", [])] != list(COLUMNS)
        or any(field.type != pa.string() for field in table.schema)
    ):
        raise PatchError(f"{dataset}: expected the existing eleven string columns in order")
    if (
        entry.get("rows") != table.num_rows
        or entry.get("bytes") != len(payloads["data.parquet"])
        or entry.get("sha256") != sha256(payloads["data.parquet"])
    ):
        raise PatchError(f"{dataset}: Parquet rows/bytes/SHA-256 do not match the manifest")
    return Snapshot(root, dataset, table, manifest, payloads)


def date_ordinal(value: str) -> int:
    if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
        raise PatchError(f"Invalid YYYYMMDD date: {value!r}")
    try:
        return datetime.strptime(value, "%Y%m%d").date().toordinal()
    except ValueError as exc:
        raise PatchError(f"Invalid YYYYMMDD date: {value!r}") from exc


def audit_membership(table: pa.Table, manifest: dict) -> dict:
    """Audit closed calendar intervals, including non-trading days.

    Sweep only interval boundaries. Latest in_date, then Y over N wins, matching
    quant_data precedence. Report ties that disagree on ANY output field.
    Multiple current rows alone are a review item, not an automatic edit.
    """
    lower = date_ordinal(manifest["range_start"])
    upper = date_ordinal(manifest["range_end"])
    if lower > upper:
        raise PatchError("Reversed archive range")
    rows = table.to_pylist()
    by_code: dict[str, list[tuple[dict, int | None, int]]] = {}
    for row in rows:
        if not row["ts_code"] or row["is_new"] not in ("Y", "N"):
            raise PatchError("Missing instrument or invalid is_new flag")
        start = None if row["in_date"] in (None, "") else date_ordinal(row["in_date"])
        end = upper if row["out_date"] in (None, "") else date_ordinal(row["out_date"])
        if start is not None and row["out_date"] not in (None, "") and start > end:
            raise PatchError(f"Reversed interval for {row['ts_code']}")
        by_code.setdefault(row["ts_code"], []).append((row, start, end))

    conflicts, multiple_current = [], []
    for code, group in sorted(by_code.items()):
        current = [row for row, _, _ in group if row["is_new"] == "Y"]
        if len(current) > 1:
            multiple_current.append({"ts_code": code, "rows": current})
        if len(group) < 2:
            continue
        intervals = [(row, max(start, lower), min(end, upper)) for row, start, end in group
                     if start is not None and start <= upper and end >= lower]
        boundaries = sorted({point for _, start, end in intervals for point in (start, end + 1)})
        for start, next_start in zip(boundaries, boundaries[1:]):
            active = [row for row, first, last in intervals if first <= start <= last]
            if len(active) < 2:
                continue
            best = max((row["in_date"], row["is_new"]) for row in active)
            tied = [row for row in active if (row["in_date"], row["is_new"]) == best]
            different = [col for col in COLUMNS if len({row[col] for row in tied}) > 1]
            if different:
                conflicts.append({
                    "ts_code": code,
                    "start": datetime.fromordinal(start).strftime("%Y%m%d"),
                    "end": datetime.fromordinal(next_start - 1).strftime("%Y%m%d"),
                    "different_fields": different,
                    "rows": tied,
                })
    counts = Counter(tuple(row[col] for col in COLUMNS) for row in rows)
    return {
        "rows": len(rows), "instruments": len(by_code),
        "null_counts": {col: sum(row[col] is None for row in rows) for col in COLUMNS},
        "missing_industry_rows": [row for row in rows if any(row[col] in (None, "")
                                  for col in ("l1_code", "l2_code", "l3_code"))],
        "exact_duplicate_excess_rows": sum(count - 1 for count in counts.values()),
        "multiple_current_instruments": len(multiple_current),
        "multiple_current": multiple_current,
        "conflicts": conflicts,
    }


def case_row(case: tuple[str, str, str], industry: tuple[str, ...]) -> dict:
    code, name, start = case
    return dict(zip(COLUMNS, (*industry, code, name, start, None, "Y")))


def patch_table(table: pa.Table, manifest: dict) -> tuple[pa.Table, dict]:
    """Apply only reviewed SW cases to a candidate, without publishing files."""
    rows = table.to_pylist()
    # The endpoint uses both null and empty text for an open interval. Keep
    # source values intact; treat these as equivalent only for case matching.
    match_rows = [{**row, "out_date": row["out_date"] or None} for row in rows]
    remove, cases, blockers = [], [], []
    for case in REVIEWED_CASES:
        bad, good = case_row(case, WRONG_PATH), case_row(case, RETAIN_PATH)
        bad_positions = [i for i, row in enumerate(match_rows) if row == bad]
        good_positions = [i for i, row in enumerate(match_rows) if row == good]
        context = [row for row in match_rows if row["ts_code"] == case[0]]
        if not context:
            status = "not_present"
        elif len(good_positions) != 1 or len(bad_positions) > 1 or len(context) != 1 + len(bad_positions):
            status = "requires_review"
            blockers.append(f"{case[0]}: complete case records differ from reviewed bad/retained rows")
        elif bad_positions:
            status = "remove_wrong_video_media_row"
            remove.extend(bad_positions)
        else:
            status = "already_clean"
        cases.append({"ts_code": case[0], "status": status, "remove": bad, "retain": good,
                      "observed_rows": context})
    removed = set(remove)
    cleaned = table.take(pa.array([i for i in range(len(rows)) if i not in removed], type=pa.int64()))
    after = audit_membership(cleaned, manifest)
    if after["conflicts"]:
        blockers.append("Unresolved equal-precedence conflicts remain; add reviewed cases before applying")
    report = {
        "patch_id": PATCH_ID, "action": "preview", "cases": cases,
        "rows_before": len(rows), "rows_after": cleaned.num_rows,
        "removed_rows": [rows[i] for i in sorted(removed)],
        "audit_before": audit_membership(table, manifest), "audit_after": after,
        "blockers": blockers,
    }
    return cleaned, report


def plan_patch(snapshot: Snapshot) -> tuple[pa.Table, dict]:
    cleaned, report = patch_table(snapshot.table, snapshot.manifest)
    report.update(data_dir=str(snapshot.root), input_sha256=snapshot.hashes)
    other = snapshot.root / "ci_index_member"
    if other.exists():
        ci = read_snapshot(snapshot.root, "ci_index_member")
        report["ci_index_member_review_only"] = audit_membership(ci.table, ci.manifest)
    return cleaned, report


@contextmanager
def archive_lock(root: Path):
    """Share the archive writer lock with sync_tushare."""
    with (root / ".sync.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PatchError("Another sync/patch process holds the archive lock") from exc
        try:
            if any((root / "_transactions").glob("*/journal.json")):
                raise PatchError("Interrupted sync commit exists; rerun sync to recover before patching")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write(path: Path, payload: bytes) -> None:
    """Fsync a same-directory temporary file, then replace one file atomically."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if path.exists():
                os.fchmod(handle.fileno(), path.stat().st_mode & 0o777)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def current_hashes(root: Path) -> dict[str, str]:
    return {name: sha256((root / DATASET / name).read_bytes()) for name in FILES}


def apply_patch(data_dir: Path, backup_dir: Path | None = None) -> dict:
    root = Path(data_dir).expanduser().resolve()
    with archive_lock(root):
        snapshot = read_snapshot(root)
        cleaned, report = plan_patch(snapshot)
        if report["blockers"]:
            raise PatchError("; ".join(report["blockers"]))
        if not report["removed_rows"]:
            report["action"] = "no_change"
            return report
        backup_root = (Path(backup_dir).expanduser().resolve() if backup_dir is not None
                       else root.parent / "industry_patch_backups")
        if backup_root.is_relative_to(root):
            raise PatchError("Backup directory must be outside the archive data directory")
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=f"{PATCH_ID}_", dir=backup_root))
        for name, payload in snapshot.payloads.items():
            atomic_write(backup / f"original_{name}", payload)
        sink = BytesIO()
        pq.write_table(cleaned, sink, compression="zstd")
        parquet = sink.getvalue()
        roundtrip = pq.read_table(BytesIO(parquet))
        if not roundtrip.equals(cleaned, check_metadata=True):
            raise PatchError(f"Candidate Parquet failed round-trip verification; backup: {backup}")
        manifest = deepcopy(snapshot.manifest)
        manifest["partitions"]["all"].update(rows=cleaned.num_rows, bytes=len(parquet), sha256=sha256(parquet))
        manifest["updated_at"] = utc_now()
        candidates = {"data.parquet": parquet, "_manifest.json": json_bytes(manifest)}
        for name, payload in candidates.items():
            atomic_write(backup / f"patched_{name}", payload)
        report.update(action="apply", backup_dir=str(backup), phase="prepared",
                      output_sha256={name: sha256(payload) for name, payload in candidates.items()})
        atomic_write(backup / "patch_log.json", json_bytes(report))
        if current_hashes(root) != snapshot.hashes:
            raise PatchError(f"Archive changed during preparation; no files installed. Backup: {backup}")
        try:
            for name, payload in candidates.items():
                atomic_write(root / DATASET / name, payload)
        except OSError as exc:
            raise PatchError(f"Apply interrupted; archive may be mixed. Use --restore {backup}") from exc
        verified = read_snapshot(root)
        if verified.hashes != report["output_sha256"]:
            raise PatchError(f"Post-apply fingerprint mismatch. Inspect archive and backup: {backup}")
        report["phase"] = "completed"
        try:
            atomic_write(backup / "patch_log.json", json_bytes(report))
        except OSError as exc:
            raise PatchError(f"Files applied, but completion log failed. Inspect backup: {backup}") from exc
        return report


def restore_backup(data_dir: Path, backup_dir: Path) -> dict:
    root, backup = Path(data_dir).expanduser().resolve(), Path(backup_dir).expanduser().resolve()
    if (root / DATASET).is_symlink() or any((root / DATASET / name).is_symlink() for name in FILES):
        raise PatchError("Refusing to restore into symlinked archive paths")
    log = json.loads((backup / "patch_log.json").read_text(encoding="utf-8"))
    if log.get("patch_id") != PATCH_ID or log.get("data_dir") != str(root):
        raise PatchError("Backup belongs to a different patch or archive")
    originals = {name: (backup / f"original_{name}").read_bytes() for name in FILES}
    for name, payload in originals.items():
        if sha256(payload) != log["input_sha256"][name]:
            raise PatchError(f"Original backup checksum mismatch: {name}")
    with archive_lock(root):
        hashes = current_hashes(root)
        for name, digest in hashes.items():
            if digest not in (log["input_sha256"][name], log["output_sha256"][name]):
                raise PatchError(f"{name} changed since this patch; refusing to overwrite later data")
        if hashes == log["input_sha256"]:
            return {"action": "already_restored", "backup_dir": str(backup)}
        try:
            for name, payload in originals.items():
                atomic_write(root / DATASET / name, payload)
        except OSError as exc:
            raise PatchError(f"Restore interrupted; retain backup and rerun --restore {backup}") from exc
        if read_snapshot(root).hashes != log["input_sha256"]:
            raise PatchError("Restored archive does not match original fingerprints")
        result = {"action": "restored", "patch_id": PATCH_ID, "backup_dir": str(backup),
                  "restored_at": utc_now(), "output_sha256": log["input_sha256"]}
        atomic_write(backup / "restore_log.json", json_bytes(result))
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Read-only preview (default)")
    action.add_argument("--apply", action="store_true", help="Apply reviewed cases with a full backup")
    action.add_argument("--restore", type=Path, metavar="BACKUP", help="Restore original files from a patch backup")
    parser.add_argument("--backup-dir", type=Path, help="Backup parent directory, outside --data-dir")
    parser.add_argument("--report", type=Path, help="Write detailed JSON to a NEW file outside --data-dir")
    args = parser.parse_args(argv)
    root = args.data_dir.expanduser().resolve()
    try:
        if args.report and (args.report.expanduser().resolve().is_relative_to(root) or args.report.expanduser().exists()):
            raise PatchError("--report must be a new file outside the archive data directory")
        if args.restore:
            report = restore_backup(root, args.restore)
        elif args.apply:
            report = apply_patch(root, args.backup_dir)
        else:
            _, report = plan_patch(read_snapshot(root))
        if args.report:
            path = args.report.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(json_bytes(report))
        print(f"{PATCH_ID}: {report['action']}")
        if "rows_before" in report:
            print(f"Rows: {report['rows_before']} -> {report['rows_after']}; reviewed removals: {len(report['removed_rows'])}")
            for case in report["cases"]:
                print(f"  {case['ts_code']}: {case['status']}")
            print(f"Remaining multiple-current instruments (review only): {report['audit_after']['multiple_current_instruments']}")
            ci = report.get("ci_index_member_review_only")
            if ci:
                print(f"ci_index_member unchanged; missing l3_code rows: {ci['null_counts']['l3_code']}")
        if "backup_dir" in report:
            print(f"Backup: {report['backup_dir']}")
        for blocker in report.get("blockers", []):
            print(f"BLOCKED: {blocker}")
        return 2 if report.get("blockers") else 0
    except (PatchError, OSError, ValueError, pa.ArrowException) as exc:
        parser.exit(2, f"industry patch failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
