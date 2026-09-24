"""Archive writer lock, exact backups, set differences and recoverable commits."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from _models import ConfigurationError, LOGGER, SyncError, now_iso


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        _sync_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        if pq.read_metadata(temporary).num_rows != len(frame):
            raise SyncError(f"Parquet row-count mismatch for {path}")
        _sync_file(temporary)
        checksum, size = sha256_file(temporary), temporary.stat().st_size
        os.replace(temporary, path)
        return checksum, size
    finally:
        temporary.unlink(missing_ok=True)


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


@contextmanager
def archive_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".sync.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SyncError(f"another sync/patch process holds the archive lock: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def frame_difference(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Exact full-row set comparison; a changed value is one removal + one addition."""
    old_unique = old.drop_duplicates(ignore_index=True)
    new_unique = new.drop_duplicates(ignore_index=True)
    schema_changed = list(old.columns) != list(new.columns) or [str(t) for t in old.dtypes] != [str(t) for t in new.dtypes]
    if schema_changed:
        removed, added = old_unique, new_unique
    else:
        both = pd.concat([old_unique, new_unique], ignore_index=True)
        exclusive = ~both.duplicated(keep=False)
        removed = old_unique.loc[exclusive.iloc[:len(old_unique)].to_numpy()].reset_index(drop=True)
        added = new_unique.loc[exclusive.iloc[len(old_unique):].to_numpy()].reset_index(drop=True)
    summary = {
        "old_rows": len(old), "new_rows": len(new),
        "old_unique_rows": len(old_unique), "new_unique_rows": len(new_unique),
        "added_rows": len(added), "removed_rows": len(removed),
        "old_duplicate_rows": len(old) - len(old_unique),
        "new_duplicate_rows": len(new) - len(new_unique),
        "schema_changed": schema_changed,
        "old_schema": {name: str(dtype) for name, dtype in old.dtypes.items()},
        "new_schema": {name: str(dtype) for name, dtype in new.dtypes.items()},
        "added_examples": json.loads(added.head(5).to_json(orient="records", force_ascii=False)),
        "removed_examples": json.loads(removed.head(5).to_json(orient="records", force_ascii=False)),
    }
    return added, removed, summary


def _hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _finish_transaction(root: Path, transaction: Path) -> None:
    journal = load_json(transaction / "journal.json")
    if journal is None:
        # No publish could have started before a complete journal was installed.
        shutil.rmtree(transaction)
        return
    for item in journal["files"]:
        target = root / item["target"]
        actual = _hash_or_none(target)
        if actual not in (item["before_sha256"], item["after_sha256"]):
            raise SyncError(f"transaction recovery refused: external change to {target}")
    for item in journal["files"]:
        target = root / item["target"]
        if _hash_or_none(target) == item["after_sha256"]:
            continue
        candidate = transaction / item["candidate"]
        if _hash_or_none(candidate) != item["after_sha256"]:
            raise SyncError(f"transaction candidate missing or corrupt: {candidate}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, target)
    if journal.get("change_log"):
        path = root / journal["change_log"]
        change = load_json(path)
        if change is None:
            raise SyncError(f"transaction change log missing: {path}")
        change.update(status="committed", committed_at=now_iso())
        atomic_write_json(path, change)
    shutil.rmtree(transaction)


def recover_transactions(root: Path) -> None:
    """Must run under archive_lock, before reading manifests or starting fetches."""
    directory = root / "_transactions"
    if directory.exists():
        for transaction in sorted(directory.iterdir()):
            if transaction.is_dir():
                LOGGER.warning("recovering interrupted partition commit: %s", transaction.name)
                _finish_transaction(root, transaction)


def commit_partition(
    root: Path,
    target: Path,
    manifest_path: Path,
    manifest: dict,
    key: str,
    frame: pd.DataFrame,
    entry: dict,
    run_id: str,
) -> dict:
    """Stage everything, preserve the old bytes, then publish data + manifest.

    The journal rolls forward a process crash between the two replacements.
    A backup or diff failure happens before either live file is touched.
    """
    previous = load_json(manifest_path) or {}
    old_entry = previous.get("partitions", {}).get(key, {})
    old_hash = _hash_or_none(target)
    if old_entry and old_hash != old_entry.get("sha256"):
        raise SyncError(f"refusing to replace missing/corrupt partition: {target}")
    frame = frame.drop_duplicates(ignore_index=True)
    entry = {**entry, "rows": len(frame), "deduplicated": True}
    updated = copy.deepcopy(manifest)
    old = pd.read_parquet(target) if target.exists() else None
    change = None
    if old is not None:
        added, removed, change = frame_difference(old, frame)
        changed = bool(len(added) or len(removed) or change["schema_changed"] or change["old_duplicate_rows"])
        if not changed:
            entry.update(sha256=old_hash, bytes=target.stat().st_size)
            if old_entry.get("last_change"):
                entry["last_change"] = old_entry["last_change"]
            updated["partitions"][key] = entry
            atomic_write_json(manifest_path, updated)
            LOGGER.info("%s partition=%s unchanged (full-row set)", manifest["dataset"], key)
            return updated

    transaction = root / "_transactions" / uuid.uuid4().hex
    transaction.mkdir(parents=True)
    try:
        checksum, size = atomic_write_parquet(frame, transaction / "data.parquet")
        entry.update(sha256=checksum, bytes=size)
        change_log = None
        if change is not None:
            history = root / "_history" / run_id / manifest["dataset"] / key / transaction.name
            history.mkdir(parents=True)
            shutil.copy2(target, history / "before.parquet")
            _sync_file(history / "before.parquet")
            if sha256_file(history / "before.parquet") != old_hash:
                raise SyncError(f"backup checksum mismatch: {target}")
            if manifest_path.exists():
                shutil.copy2(manifest_path, history / "before_manifest.json")
                _sync_file(history / "before_manifest.json")
                if sha256_file(history / "before_manifest.json") != sha256_file(manifest_path):
                    raise SyncError(f"manifest backup checksum mismatch: {manifest_path}")
            atomic_write_parquet(added, history / "added.parquet")
            atomic_write_parquet(removed, history / "removed.parquet")
            change.update(
                dataset=manifest["dataset"], partition=key, run_id=run_id,
                query=entry["query"], refresh_window=entry.get("refresh_window"),
                status="prepared", prepared_at=now_iso(),
                original_file=str(target.relative_to(root)),
                old_sha256=old_hash, new_sha256=checksum,
                backup="before.parquet", added_file="added.parquet", removed_file="removed.parquet",
            )
            change_log = str((history / "change.json").relative_to(root))
            atomic_write_json(history / "change.json", change)
            entry["last_change"] = change_log
        updated["partitions"][key] = entry
        atomic_write_json(transaction / "manifest.json", updated)
        files = [
            {"target": str(target.relative_to(root)), "candidate": "data.parquet",
             "before_sha256": old_hash, "after_sha256": checksum},
            {"target": str(manifest_path.relative_to(root)), "candidate": "manifest.json",
             "before_sha256": _hash_or_none(manifest_path),
             "after_sha256": sha256_file(transaction / "manifest.json")},
        ]
        atomic_write_json(transaction / "journal.json", {"files": files, "change_log": change_log})
        _finish_transaction(root, transaction)
    except Exception:
        if not (transaction / "journal.json").exists():
            shutil.rmtree(transaction)
        raise
    if change is not None:
        LOGGER.info("%s partition=%s changed: rows %s -> %s, added=%s removed=%s old_duplicates=%s; %s",
                    manifest["dataset"], key, change["old_rows"], change["new_rows"],
                    change["added_rows"], change["removed_rows"], change["old_duplicate_rows"], change_log)
    else:
        LOGGER.info("%s partition=%s created rows=%s", manifest["dataset"], key, len(frame))
    return updated
