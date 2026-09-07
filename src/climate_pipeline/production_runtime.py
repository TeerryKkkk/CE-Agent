"""Fail-closed durability primitives for the repaired-v2 production runner.

This module contains no scientific policy and no network client.  Its only
responsibility is to make run identity, paid-call attempt accounting, locks,
and local publication explicit and durable.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Iterable, Mapping, Sequence


class ProductionRuntimeError(RuntimeError):
    """Raised when a production durability or identity assertion fails."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync; Windows does not expose portable dir fsync."""

    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(Path(path), text)


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_json_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        if required:
            raise ProductionRuntimeError(f"missing_json:{path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionRuntimeError(f"invalid_or_partial_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise ProductionRuntimeError(f"json_object_required:{path}")
    return value


class RunLock(AbstractContextManager["RunLock"]):
    """Cross-platform non-blocking process lock held for a runner invocation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise ProductionRuntimeError(f"concurrent_run_lock_held:{self.path}") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def directory_has_entries(path: Path) -> bool:
    path = Path(path)
    return path.exists() and any(path.iterdir())


def validate_run_directories(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    resume: bool,
    expected_identity_sha256: str,
) -> None:
    output_dir = Path(output_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if output_dir == checkpoint_dir:
        raise ProductionRuntimeError("output_and_checkpoint_directories_must_differ")
    if not resume:
        collisions = [
            str(path)
            for path in (output_dir, checkpoint_dir)
            if directory_has_entries(path)
        ]
        if collisions:
            raise ProductionRuntimeError(
                "nonempty_directory_requires_identity_bound_resume:" + ",".join(collisions)
            )
        return
    for directory in (output_dir, checkpoint_dir):
        identity_path = directory / "run_identity.json"
        identity = read_json_object(identity_path)
        if identity.get("run_identity_sha256") != expected_identity_sha256:
            raise ProductionRuntimeError(
                f"resume_identity_mismatch:{identity_path}"
            )


def initialize_run_identity(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    identity: Mapping[str, Any],
    resume: bool,
) -> str:
    payload = dict(identity)
    identity_sha = canonical_json_sha256(payload)
    validate_run_directories(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        expected_identity_sha256=identity_sha,
    )
    if resume:
        return identity_sha
    record = {
        **payload,
        "run_identity_sha256": identity_sha,
        "lock_host": socket.gethostname(),
    }
    for directory in (Path(output_dir), Path(checkpoint_dir)):
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "run_identity.json", record)
    return identity_sha


@dataclass(frozen=True)
class AttemptReservation:
    should_invoke: bool
    status: str
    attempts_consumed: int
    attempt_limit: int
    record: Mapping[str, Any]


class AttemptLedger:
    """Persist intent before an external call and bound crash-window replay."""

    TERMINAL = {"complete", "technical_failure"}

    def __init__(self, root: Path, *, attempt_limit: int = 2) -> None:
        if attempt_limit < 1:
            raise ValueError("attempt_limit_must_be_positive")
        self.root = Path(root)
        self.attempt_limit = attempt_limit

    def path_for(self, stage: str, request_key: str) -> Path:
        safe_stage = stage.replace("/", "_").replace("\\", "_")
        if not request_key or any(char not in "0123456789abcdef" for char in request_key):
            raise ProductionRuntimeError("request_key_must_be_lowercase_hex")
        return self.root / "attempts" / safe_stage / f"{request_key}.json"

    def reserve(
        self,
        *,
        stage: str,
        request: Mapping[str, Any],
        timestamp: str,
    ) -> AttemptReservation:
        request_payload = dict(request)
        request_key = canonical_json_sha256(request_payload)
        path = self.path_for(stage, request_key)
        previous = read_json_object(path, required=False)
        if previous:
            if previous.get("request") != request_payload:
                raise ProductionRuntimeError(f"attempt_request_identity_mismatch:{path}")
            if previous.get("attempt_limit") != self.attempt_limit:
                raise ProductionRuntimeError(f"attempt_limit_identity_mismatch:{path}")
            status = str(previous.get("status") or "")
            if status in self.TERMINAL:
                return AttemptReservation(
                    False,
                    status,
                    int(previous.get("attempts_consumed") or 0),
                    self.attempt_limit,
                    previous,
                )
            if status != "incomplete":
                raise ProductionRuntimeError(f"invalid_attempt_status:{path}:{status}")
            consumed = int(previous.get("attempts_consumed") or 0)
            if consumed >= self.attempt_limit:
                terminal = {
                    **previous,
                    "status": "technical_failure",
                    "retryable": False,
                    "technical_failure_reason": "runner_attempt_limit_exhausted_after_crash_window",
                    "updated_at": timestamp,
                }
                atomic_write_json(path, terminal)
                return AttemptReservation(
                    False,
                    "technical_failure",
                    consumed,
                    self.attempt_limit,
                    terminal,
                )
        else:
            consumed = 0
            previous = {
                "stage": stage,
                "request_key": request_key,
                "request": request_payload,
                "attempt_limit": self.attempt_limit,
                "call_history": [],
            }
        record = {
            **previous,
            "status": "incomplete",
            "retryable": True,
            "attempts_consumed": consumed + 1,
            "last_call_intent_durable_at": timestamp,
            "updated_at": timestamp,
        }
        atomic_write_json(path, record)
        return AttemptReservation(
            True,
            "incomplete",
            consumed + 1,
            self.attempt_limit,
            record,
        )

    def finish(
        self,
        reservation: AttemptReservation,
        *,
        stage: str,
        request: Mapping[str, Any],
        success: bool,
        timestamp: str,
        raw_response: Any = None,
        response_ids: Sequence[str] = (),
        normalized_result: Any = None,
        error: str = "",
        provider_attempts: int = 1,
    ) -> dict[str, Any]:
        request_payload = dict(request)
        request_key = canonical_json_sha256(request_payload)
        path = self.path_for(stage, request_key)
        current = read_json_object(path)
        if not reservation.should_invoke or current.get("request") != request_payload:
            raise ProductionRuntimeError(f"attempt_finish_without_matching_reservation:{path}")
        history = list(current.get("call_history") or [])
        history.append(
            {
                "attempt": reservation.attempts_consumed,
                "finished_at": timestamp,
                "provider_attempts": int(provider_attempts),
                "response_ids": list(response_ids),
                "success": bool(success),
                "error": error,
            }
        )
        terminal_failure = (
            not success and reservation.attempts_consumed >= self.attempt_limit
        )
        status = "complete" if success else (
            "technical_failure" if terminal_failure else "incomplete"
        )
        record = {
            **current,
            "status": status,
            "retryable": status == "incomplete",
            "technical_failure_reason": error if terminal_failure else "",
            "raw_provider_response": raw_response,
            "response_ids": list(response_ids),
            "normalized_result": normalized_result,
            "provider_attempts_total": sum(
                int(row.get("provider_attempts") or 0) for row in history
            ),
            "call_history": history,
            "updated_at": timestamp,
            "completed_at": timestamp if status in self.TERMINAL else "",
        }
        atomic_write_json(path, record)
        return record


class DurableStageStore:
    """Identity-bound, hash-chained checkpoints for independently reusable stages.

    A stage is complete only after its payload and checksum are atomically
    durable.  The caller supplies the scientifically relevant request fields;
    the store adds the run, schema, policy, parent-stage and code identities.
    A different identity is a different path, while a corrupt record at the
    expected path is rejected instead of silently recomputed.
    """

    TERMINAL = {"complete", "technical_failure"}

    def __init__(
        self,
        root: Path,
        *,
        run_identity_sha256: str,
        schema_version: str,
        policy_version: str,
        global_identity: Mapping[str, Any],
    ) -> None:
        if not run_identity_sha256 or not schema_version or not policy_version:
            raise ValueError("durable_stage_store_identity_required")
        self.root = Path(root)
        self.run_identity_sha256 = run_identity_sha256
        self.schema_version = schema_version
        self.policy_version = policy_version
        self.global_identity = dict(global_identity)

    def request(
        self,
        *,
        stage: str,
        parent_sha256: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not stage or not parent_sha256:
            raise ProductionRuntimeError("stage_and_parent_identity_required")
        return {
            "stage": stage,
            "parent_stage_sha256": parent_sha256,
            "run_identity_sha256": self.run_identity_sha256,
            "checkpoint_schema_version": self.schema_version,
            "checkpoint_policy_version": self.policy_version,
            "global_identity": self.global_identity,
            "inputs": dict(inputs),
        }

    def path_for(self, *, stage: str, request: Mapping[str, Any]) -> Path:
        safe_stage = stage.replace("/", "_").replace("\\", "_")
        return self.root / safe_stage / f"{canonical_json_sha256(request)}.json"

    def load(
        self,
        *,
        stage: str,
        parent_sha256: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self.request(
            stage=stage, parent_sha256=parent_sha256, inputs=inputs
        )
        path = self.path_for(stage=stage, request=request)
        record = read_json_object(path, required=False)
        if not record:
            return {}
        if record.get("request") != request:
            raise ProductionRuntimeError(f"stage_request_identity_mismatch:{path}")
        status = str(record.get("status") or "")
        if status not in self.TERMINAL:
            raise ProductionRuntimeError(f"incomplete_stage_checkpoint_rejected:{path}")
        payload = record.get("payload")
        if record.get("payload_sha256") != canonical_json_sha256(payload):
            raise ProductionRuntimeError(f"stage_payload_checksum_mismatch:{path}")
        expected = canonical_json_sha256(
            {
                "request": request,
                "status": status,
                "payload_sha256": record["payload_sha256"],
            }
        )
        if record.get("stage_record_sha256") != expected:
            raise ProductionRuntimeError(f"stage_record_checksum_mismatch:{path}")
        return record

    def finish(
        self,
        *,
        stage: str,
        parent_sha256: str,
        inputs: Mapping[str, Any],
        payload: Any,
        status: str = "complete",
        completed_at: str,
    ) -> dict[str, Any]:
        if status not in self.TERMINAL:
            raise ProductionRuntimeError("durable_stage_terminal_status_required")
        request = self.request(
            stage=stage, parent_sha256=parent_sha256, inputs=inputs
        )
        path = self.path_for(stage=stage, request=request)
        existing = self.load(
            stage=stage, parent_sha256=parent_sha256, inputs=inputs
        )
        if existing:
            return existing
        payload_sha = canonical_json_sha256(payload)
        record = {
            "request": request,
            "request_sha256": canonical_json_sha256(request),
            "status": status,
            "payload": payload,
            "payload_sha256": payload_sha,
            "stage_record_sha256": canonical_json_sha256(
                {
                    "request": request,
                    "status": status,
                    "payload_sha256": payload_sha,
                }
            ),
            "completed_at": completed_at,
        }
        atomic_write_json(path, record)
        # Read-after-write is part of completion; partial/corrupt writes never
        # become a reusable checkpoint.
        return self.load(
            stage=stage, parent_sha256=parent_sha256, inputs=inputs
        )


def publish_staged_directory(*, staging_dir: Path, final_dir: Path) -> None:
    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)
    if not staging_dir.is_dir():
        raise ProductionRuntimeError(f"staged_export_missing:{staging_dir}")
    if final_dir.exists():
        raise ProductionRuntimeError(f"final_export_collision:{final_dir}")
    for path in staging_dir.rglob("*"):
        if path.is_file():
            # Windows requires a write-capable file descriptor for fsync.
            # The staging files are owned by this run and opening them in
            # update mode does not alter their contents.
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
    os.replace(staging_dir, final_dir)
    _fsync_parent(final_dir)


__all__ = [
    "AttemptLedger",
    "AttemptReservation",
    "DurableStageStore",
    "ProductionRuntimeError",
    "RunLock",
    "atomic_write_bytes",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "directory_has_entries",
    "initialize_run_identity",
    "publish_staged_directory",
    "read_json_object",
    "sha256_file",
    "validate_run_directories",
]
