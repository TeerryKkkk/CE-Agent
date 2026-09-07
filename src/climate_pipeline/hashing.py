from __future__ import annotations

import hashlib
import json
from typing import Any


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha1_text(payload)


def make_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}_{stable_json_hash(parts)[:length]}"


def row_hash(row: dict[str, Any]) -> str:
    return f"row_{stable_json_hash(row)[:16]}"
