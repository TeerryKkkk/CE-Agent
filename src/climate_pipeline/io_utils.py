from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from . import config


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def repo_relative_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except Exception:
        return candidate.as_posix()


def ensure_dirs() -> None:
    for path in [
        config.OUTPUT_DIR,
        config.OUTPUT_DIR / "event_dossiers",
        config.SNAPSHOT_DIR / "html",
        config.SNAPSHOT_DIR / "text",
        config.SNAPSHOT_DIR / "attachments",
        config.LIVE_ATTEMPT_DIR,
        config.LIVE_SNAPSHOT_DIR / "html",
        config.LIVE_SNAPSHOT_DIR / "text",
        config.LIVE_SNAPSHOT_DIR / "attachments",
        config.LOG_DIR,
        config.LOG_DIR / "llm",
        config.DATA_DIR / "raw",
        config.DATA_DIR / "interim",
        config.DATA_DIR / "processed",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    safe_write_text(path, payload)


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8", errors="ignore") == text:
                return
        except OSError:
            pass
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent), newline="") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if path.exists():
                try:
                    if path.read_text(encoding="utf-8", errors="ignore") == text:
                        tmp_path.unlink(missing_ok=True)
                        return
                except OSError:
                    pass
            if attempt == 4:
                tmp_path.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))


def safe_write_json(path: Path, obj: Any) -> None:
    safe_write_text(path, json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, sort_keys=True))


def load_optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def reset_output_files(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_write_text(path, "")
