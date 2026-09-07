"""Strict, streaming HTTP boundaries for the remaining-368 production path.

The helpers in this module intentionally reject redirects.  They never call
``read()`` without a size, and an exact byte cap is treated conservatively as
exhausted unless EOF was observed in a short read.  This makes the configured
cap an actual upper bound on bytes consumed by Python rather than a payload
limit that requires an unaccounted inspection byte.
"""

from __future__ import annotations

from dataclasses import dataclass
import urllib.error
import urllib.request
from typing import Any, BinaryIO, Mapping


class BoundedHttpError(RuntimeError):
    """A fail-closed production HTTP boundary violation."""


class RedirectRejected(urllib.error.HTTPError):
    """HTTP redirect surfaced without consuming its response body."""


class RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect and leave the body available for bounded reading."""

    def _reject(self, request: Any, response: Any, code: int, message: str, headers: Any) -> Any:
        raise RedirectRejected(request.full_url, code, message, headers, response)

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


def no_redirect_opener() -> Any:
    return urllib.request.build_opener(RejectRedirectHandler()).open


@dataclass(frozen=True)
class BoundedRead:
    data: bytes
    eof_observed: bool


def read_bounded(stream: BinaryIO, *, max_bytes: int, label: str, chunk_size: int = 64 * 1024) -> BoundedRead:
    """Read at most ``max_bytes`` and fail closed if EOF is not proven."""

    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError(f"{label}_max_bytes_must_be_positive")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size_must_be_positive")
    chunks: list[bytes] = []
    consumed = 0
    while consumed < max_bytes:
        requested = min(chunk_size, max_bytes - consumed)
        chunk = stream.read(requested)
        if not isinstance(chunk, (bytes, bytearray)):
            raise BoundedHttpError(f"{label}_returned_non_bytes")
        chunk = bytes(chunk)
        if len(chunk) > requested:
            raise BoundedHttpError(f"{label}_reader_violated_requested_size")
        if not chunk:
            return BoundedRead(b"".join(chunks), True)
        chunks.append(chunk)
        consumed += len(chunk)
        if len(chunk) < requested:
            return BoundedRead(b"".join(chunks), True)
    raise BoundedHttpError(f"{label}_exceeds_or_equals_limit:{max_bytes}")


def header_byte_count(headers: Mapping[str, Any] | Any) -> int:
    items = headers.items() if hasattr(headers, "items") else ()
    return sum(
        len(str(key).encode("utf-8")) + len(str(value).encode("utf-8")) + 4
        for key, value in items
    )


def validate_headers(
    headers: Mapping[str, Any] | Any,
    *,
    max_header_bytes: int,
    max_location_bytes: int,
) -> dict[str, str]:
    if header_byte_count(headers) > max_header_bytes:
        raise BoundedHttpError(f"response_headers_exceed_limit:{max_header_bytes}")
    normalized = {
        str(key).lower(): str(value)
        for key, value in (headers.items() if hasattr(headers, "items") else ())
    }
    location = normalized.get("location", "")
    if len(location.encode("utf-8")) > max_location_bytes:
        raise BoundedHttpError(
            f"redirect_location_exceeds_limit:{max_location_bytes}"
        )
    return normalized


def read_http_error(
    exc: urllib.error.HTTPError,
    *,
    max_error_bytes: int,
    max_redirect_bytes: int,
    max_header_bytes: int,
    max_location_bytes: int,
) -> tuple[str, str]:
    """Return ``(kind, detail)`` while enforcing distinct error/redirect caps."""

    validate_headers(
        exc.headers or {},
        max_header_bytes=max_header_bytes,
        max_location_bytes=max_location_bytes,
    )
    is_redirect = 300 <= int(exc.code) < 400
    label = "redirect_response_body" if is_redirect else "error_response_body"
    limit = max_redirect_bytes if is_redirect else max_error_bytes
    try:
        raw = read_bounded(exc, max_bytes=limit, label=label).data
    except BoundedHttpError as boundary:
        return ("redirect" if is_redirect else "error", str(boundary))
    return (
        "redirect" if is_redirect else "error",
        raw.decode("utf-8", errors="replace")[:1000],
    )


__all__ = [
    "BoundedHttpError",
    "BoundedRead",
    "RedirectRejected",
    "RejectRedirectHandler",
    "header_byte_count",
    "no_redirect_opener",
    "read_bounded",
    "read_http_error",
    "validate_headers",
]
