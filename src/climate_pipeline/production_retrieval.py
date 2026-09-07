"""Granular, bounded retrieval primitives for the remaining-368 runner.

This module deliberately separates provider transport, deterministic search
normalization/selection, raw page transport, and body normalization.  It has
no model calls and no scientific aggregation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zlib
from typing import Any, Callable, Mapping

from .bounded_http import (
    BoundedHttpError,
    no_redirect_opener,
    read_bounded,
    read_http_error,
    validate_headers,
)
from .controlled_open_retrieval import (
    LEGACY_PREFETCH_POLICY_VERSION,
    TavilyCostController,
    domain_from_url,
    normalize_url,
    schedule_fetch_decisions,
    score_prefetch_result,
)
from .production_html import (
    clean_html,
    decode_response,
    extract_title,
    looks_like_pdf_payload,
)


RETRIEVAL_PROVENANCE_KEYS = (
    "query_source",
    "found_by",
    "retrieval_round",
    "target_gap",
    "query_intent",
    "expansion_trigger",
    "generated_query",
    "validated_query",
)


class RetrievalLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawFetchResult:
    original_url: str
    final_url: str
    status: int
    headers: dict[str, str]
    compressed_bytes: bytes
    decompressed_bytes: bytes
    compressed_sha256: str
    decompressed_sha256: str
    redirect_count: int
    failure_reason: str
    redirect_chain: tuple[dict[str, Any], ...] = ()

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("compressed_bytes")
        value.pop("decompressed_bytes")
        value["compressed_byte_count"] = len(self.compressed_bytes)
        value["decompressed_byte_count"] = len(self.decompressed_bytes)
        return value


class BoundedHttpFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: int,
        max_redirects: int,
        max_compressed_bytes: int,
        max_decompressed_bytes: int,
        max_error_response_bytes: int,
        max_redirect_response_bytes: int,
        max_header_bytes: int,
        max_redirect_location_bytes: int,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        for name, value in (
            ("timeout_seconds", timeout_seconds),
            ("max_compressed_bytes", max_compressed_bytes),
            ("max_decompressed_bytes", max_decompressed_bytes),
            ("max_error_response_bytes", max_error_response_bytes),
            ("max_redirect_response_bytes", max_redirect_response_bytes),
            ("max_header_bytes", max_header_bytes),
            ("max_redirect_location_bytes", max_redirect_location_bytes),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name}_must_be_positive")
        if not 0 <= int(max_redirects) <= 1:
            raise ValueError("production_page_redirects_outside_hard_bound")
        self.timeout_seconds = int(timeout_seconds)
        self.max_redirects = int(max_redirects)
        self.max_compressed_bytes = int(max_compressed_bytes)
        self.max_decompressed_bytes = int(max_decompressed_bytes)
        self.max_error_response_bytes = int(max_error_response_bytes)
        self.max_redirect_response_bytes = int(max_redirect_response_bytes)
        self.max_header_bytes = int(max_header_bytes)
        self.max_redirect_location_bytes = int(max_redirect_location_bytes)
        self.opener = opener or no_redirect_opener()

    def fetch(self, url: str) -> RawFetchResult:
        original_url = url
        current_url = url
        visited = {self._redirect_identity(url)}
        chain: list[dict[str, Any]] = []
        while True:
            request = urllib.request.Request(
                current_url,
                headers={
                    "User-Agent": "CE-Agent-remaining368/2.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate, identity",
                },
            )
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    headers = validate_headers(
                        getattr(response, "headers", {}),
                        max_header_bytes=self.max_header_bytes,
                        max_location_bytes=self.max_redirect_location_bytes,
                    )
                    raw = read_bounded(
                        response,
                        max_bytes=self.max_compressed_bytes,
                        label="compressed_fetch_body",
                    ).data
                    decompressed = self._decompress(raw, headers.get("content-encoding", ""))
                    final_url = response.geturl() if hasattr(response, "geturl") else current_url
                    status = int(getattr(response, "status", 200))
                    return self._result(
                        original_url,
                        final_url,
                        status,
                        headers,
                        raw,
                        decompressed,
                        len(chain),
                        "",
                        tuple(chain),
                    )
            except urllib.error.HTTPError as exc:
                try:
                    headers = validate_headers(
                        exc.headers or {},
                        max_header_bytes=self.max_header_bytes,
                        max_location_bytes=self.max_redirect_location_bytes,
                    )
                    kind, detail = read_http_error(
                        exc,
                        max_error_bytes=self.max_error_response_bytes,
                        max_redirect_bytes=self.max_redirect_response_bytes,
                        max_header_bytes=self.max_header_bytes,
                        max_location_bytes=self.max_redirect_location_bytes,
                    )
                    reason = f"{kind}_http_{exc.code}:{detail[:200]}"
                except Exception as boundary:
                    headers = {}
                    reason = f"http_boundary:{type(boundary).__name__}:{str(boundary)[:200]}"
                if 300 <= int(exc.code) < 400:
                    location = headers.get("location", "")
                    resolved = urllib.parse.urljoin(current_url, location) if location else ""
                    entry = {
                        "status": int(exc.code),
                        "url": current_url,
                        "location": location,
                        "resolved_url": resolved,
                    }
                    chain.append(entry)
                    redirect_failure = self._redirect_failure(
                        current_url=current_url,
                        resolved_url=resolved,
                        detail=detail,
                        visited=visited,
                        redirects_followed=len(chain) - 1,
                    )
                    if redirect_failure:
                        return self._result(
                            original_url,
                            current_url,
                            int(exc.code),
                            headers,
                            b"",
                            b"",
                            len(chain) - 1,
                            redirect_failure,
                            tuple(chain),
                        )
                    visited.add(self._redirect_identity(resolved))
                    current_url = resolved
                    continue
                return self._result(
                    original_url,
                    current_url,
                    int(exc.code),
                    headers,
                    b"",
                    b"",
                    len(chain),
                    reason,
                    tuple(chain),
                )
            except (TimeoutError, socket.timeout) as exc:
                return self._result(
                    original_url, current_url, 0, {}, b"", b"", len(chain),
                    f"timeout:{type(exc).__name__}", tuple(chain)
                )
            except Exception as exc:
                return self._result(
                    original_url, current_url, 0, {}, b"", b"", len(chain),
                    f"transport:{type(exc).__name__}:{str(exc)[:300]}", tuple(chain)
                )

    @staticmethod
    def _redirect_identity(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, "")
        )

    def _redirect_failure(
        self,
        *,
        current_url: str,
        resolved_url: str,
        detail: str,
        visited: set[str],
        redirects_followed: int,
    ) -> str:
        if detail.startswith("redirect_response_body_"):
            return f"redirect_boundary:{detail}"
        if not resolved_url:
            return "redirect_missing_location"
        current = urllib.parse.urlsplit(current_url)
        resolved = urllib.parse.urlsplit(resolved_url)
        if resolved.scheme.lower() not in {"http", "https"}:
            return f"redirect_disallowed_scheme:{resolved.scheme.lower()}"
        if (
            resolved.scheme.lower() != current.scheme.lower()
            or resolved.netloc.lower() != current.netloc.lower()
        ):
            return "redirect_cross_origin_rejected"
        identity = self._redirect_identity(resolved_url)
        if identity in visited:
            return "redirect_loop_rejected"
        if redirects_followed >= self.max_redirects:
            return f"redirect_limit_exhausted:{self.max_redirects}"
        return ""

    def _decompress(self, raw: bytes, encoding: str) -> bytes:
        lowered = encoding.strip().lower()
        if lowered in {"", "identity"}:
            if len(raw) > self.max_decompressed_bytes:
                raise RetrievalLimitError(
                    f"decompressed_fetch_exceeds_limit:{self.max_decompressed_bytes}"
                )
            return raw
        elif lowered == "gzip":
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif lowered == "deflate":
            decoder = zlib.decompressobj()
        else:
            raise RetrievalLimitError(f"unsupported_content_encoding:{lowered}")
        output: list[bytes] = []
        produced = 0
        for offset in range(0, len(raw), 64 * 1024):
            chunk = raw[offset : offset + 64 * 1024]
            remaining = self.max_decompressed_bytes - produced
            if remaining <= 0:
                raise RetrievalLimitError(
                    f"decompressed_fetch_exceeds_limit:{self.max_decompressed_bytes}"
                )
            piece = decoder.decompress(chunk, remaining)
            output.append(piece)
            produced += len(piece)
            if decoder.unconsumed_tail:
                raise RetrievalLimitError(
                    f"decompressed_fetch_exceeds_limit:{self.max_decompressed_bytes}"
                )
        remaining = self.max_decompressed_bytes - produced
        if remaining == 0:
            if decoder.eof:
                return b"".join(output)
            raise RetrievalLimitError(
                f"decompressed_fetch_exceeds_limit:{self.max_decompressed_bytes}"
            )
        tail = decoder.flush(remaining)
        if produced + len(tail) > self.max_decompressed_bytes or not decoder.eof:
            raise RetrievalLimitError(
                f"decompressed_fetch_exceeds_limit_or_incomplete:{self.max_decompressed_bytes}"
            )
        output.append(tail)
        return b"".join(output)

    @staticmethod
    def _result(
        original_url: str,
        final_url: str,
        status: int,
        headers: dict[str, str],
        compressed: bytes,
        decompressed: bytes,
        redirects: int,
        failure: str,
        redirect_chain: tuple[dict[str, Any], ...] = (),
    ) -> RawFetchResult:
        return RawFetchResult(
            original_url=original_url,
            final_url=final_url,
            status=status,
            headers=headers,
            compressed_bytes=compressed,
            decompressed_bytes=decompressed,
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            decompressed_sha256=hashlib.sha256(decompressed).hexdigest(),
            redirect_count=redirects,
            failure_reason=failure,
            redirect_chain=redirect_chain,
        )


def normalize_search_envelope(
    *,
    case: Mapping[str, Any],
    query: Mapping[str, Any],
    envelope: Mapping[str, Any],
    metadata: Mapping[str, Any],
    max_results: int,
    prefetch_policy_version: str = LEGACY_PREFETCH_POLICY_VERSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = envelope.get("payload") if isinstance(envelope, Mapping) else None
    raw_results = payload.get("results") if isinstance(payload, Mapping) else []
    raw_results = raw_results if isinstance(raw_results, list) else []
    raw_results = [row for row in raw_results[:max_results] if isinstance(row, dict)]
    base = {key: str(query.get(key) or "") for key in RETRIEVAL_PROVENANCE_KEYS}
    search_rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    raw_by_url: dict[str, dict[str, Any]] = {}
    error = str(metadata.get("error") or "")
    if error:
        search_rows.append(
            {
                "candidate_id": case["candidate_id"],
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "lane": query["lane"],
                "result_rank": "",
                "url": "",
                "title": "",
                "snippet": "",
                "raw_content_available": "false",
                "tavily_error": error,
                **base,
            }
        )
        return search_rows, scores, raw_by_url
    for rank, raw in enumerate(raw_results, start=1):
        url = str(raw.get("url") or "")
        title = str(raw.get("title") or url)
        snippet = str(raw.get("content") or raw.get("snippet") or "")
        raw_content = str(raw.get("raw_content") or "")
        row = {
            "candidate_id": case["candidate_id"],
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "lane": query["lane"],
            "missing_component": query["missing_component"],
            "result_rank": rank,
            "url": url,
            "title": title,
            "snippet": snippet,
            "raw_content_available": str(bool(raw_content)).lower(),
            "tavily_error": "",
            **base,
        }
        search_rows.append(row)
        if url:
            raw_by_url[normalize_url(url)] = raw
        score = score_prefetch_result(
            sample=dict(case),
            query_id=str(query["query_id"]),
            lane=str(query["lane"]),
            missing_component=str(query["missing_component"]),
            title=title,
            snippet=snippet,
            url=url,
            prefetch_policy_version=prefetch_policy_version,
        )
        score.update(
            {
                "query_text": query["query_text"],
                "retrieval_tier": query.get("retrieval_tier", ""),
                "fallback_reason": query.get("fallback_reason", ""),
                "unmet_gap_before_search": query.get("unmet_gap_before_search", ""),
                "domains_or_source_families_targeted": query.get("domains_or_source_families_targeted", ""),
                **base,
            }
        )
        scores.append(score)
    return search_rows, scores, raw_by_url


def select_urls(
    *,
    case: Mapping[str, Any],
    queries: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    controls: TavilyCostController,
    round_number: int,
    already_fetched_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    query_by_id = {str(row.get("query_id") or ""): row for row in queries}
    decisions = schedule_fetch_decisions(
        sample=dict(case),
        scorecards=score_rows,
        controls=controls,
        round_number=round_number,
        fetched_urls_global=set(already_fetched_urls or ()),
        domain_counts=Counter(),
        candidate_pdf_counts=Counter(),
        global_counts=Counter(),
        candidate_fetch_count=0,
    )
    for decision in decisions:
        query = query_by_id.get(str(decision.get("query_id") or ""), {})
        for key in RETRIEVAL_PROVENANCE_KEYS:
            decision[key] = query.get(key, decision.get(key, ""))
        decision["retrieval_round"] = query.get("retrieval_round", round_number)
        decision["validated_query"] = query.get("validated_query", query.get("query_text", ""))
    return decisions


def normalize_fetched_body(
    *,
    raw: RawFetchResult,
    title_hint: str,
    max_normalized_bytes: int,
    min_cleaned_text_chars: int,
) -> dict[str, Any]:
    content_type = raw.headers.get("content-type", "")
    if raw.failure_reason:
        return {"status": "technical_failure", "body_text": "", "title": title_hint, "reason": raw.failure_reason}
    if looks_like_pdf_payload(raw.decompressed_bytes, content_type):
        return {
            "status": "unsupported_body_type",
            "body_type": "application/pdf",
            "body_text": "",
            "title": title_hint,
            "reason": "pdf_requires_disabled_fallback",
        }
    media_type = content_type.partition(";")[0].strip().lower()
    if (
        media_type.startswith(("image/", "audio/", "video/", "font/"))
        or media_type
        in {
            "application/octet-stream",
            "application/zip",
            "application/x-7z-compressed",
            "application/x-rar-compressed",
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
    ):
        return {
            "status": "unsupported_body_type",
            "body_type": media_type or "binary",
            "body_text": "",
            "title": title_hint,
            "reason": f"unsupported_body_type:{media_type or 'binary'}",
        }
    text, encoding = decode_response(raw.decompressed_bytes, raw.headers)
    cleaned = clean_html(text, raw.final_url)
    encoded = cleaned.encode("utf-8")
    if len(encoded) > max_normalized_bytes:
        return {"status": "technical_failure", "body_text": "", "title": title_hint, "reason": f"normalized_body_exceeds_limit:{max_normalized_bytes}"}
    if len(cleaned.strip()) < min_cleaned_text_chars:
        return {"status": "technical_failure", "body_text": "", "title": extract_title(text) or title_hint, "reason": "normalized_body_too_short"}
    return {
        "status": "complete",
        "body_text": cleaned,
        "body_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_byte_count": len(encoded),
        "title": extract_title(text) or title_hint,
        "encoding": encoding,
        "reason": "",
    }


def page_record(
    *,
    case: Mapping[str, Any],
    query: Mapping[str, Any],
    decision: Mapping[str, Any],
    raw_search_row: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    body = str(normalized.get("body_text") or "")
    normalized_status = str(normalized.get("status") or "")
    unsupported = normalized_status == "unsupported_body_type"
    body_source = (
        "direct_http_normalized"
        if body
        else "none_unsupported_body_type"
        if unsupported
        else "none_technical_failure"
    )
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    url = str(decision.get("url") or "")
    diagnostic_reason = "" if body.strip() else str(normalized.get("reason") or "body_unavailable")
    return {
        "phase2_3_sample_id": case["candidate_id"],
        "raw_candidate_id": case["candidate_id"],
        "candidate_id": case["candidate_id"],
        "county": case["county"],
        "state": "California",
        "state_abbrev": "CA",
        "FIPS": case["FIPS"],
        "county_fips": case["FIPS"],
        "source_url": url,
        "source_title": str(normalized.get("title") or raw_search_row.get("title") or url),
        "source_family": str(raw_search_row.get("source_family") or ""),
        "discovered_source_family": "",
        "source_lane": decision.get("lane", ""),
        "query_id": decision.get("query_id", ""),
        "query_text": query.get("query_text", ""),
        **{key: str(query.get(key) or "") for key in RETRIEVAL_PROVENANCE_KEYS},
        "body_text_or_archived_body_text": body,
        "body_text_source": body_source,
        "accepted_from_snippet": False,
        "fetch_status": (
            "success"
            if body.strip()
            else "unsupported_body_type"
            if unsupported
            else "technical_failure"
        ),
        "diagnostic_reason": diagnostic_reason,
        "diagnostic_class": (
            ""
            if body.strip()
            else "unsupported_body_type"
            if unsupported
            else "technical_failure"
        ),
        "counts_toward_case_terminal_technical_failure": bool(
            not body.strip() and not unsupported
        ),
        "technical_failure_reason": (
            diagnostic_reason if not body.strip() and not unsupported else ""
        ),
        "search_raw_content_available": bool(raw_search_row.get("raw_content")),
        "search_raw_content_policy": "diagnostic_trace_only",
        "search_raw_content_eligible_for_judge": False,
        "search_raw_content_eligible_for_evidence": False,
        "search_raw_content_eligible_for_truth": False,
        "eligible_for_judge": bool(body.strip()),
        "eligible_for_evidence": bool(body.strip()),
        "eligible_for_truth": bool(body.strip()),
        "frozen_body_sha256": body_hash,
        "page_id": "capage_" + hashlib.sha256(
            json.dumps(
                {"candidate_id": case["candidate_id"], "url": url, "body_sha256": body_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20],
        "reader_result": {
            "fetch_method": body_source,
            "failure_reason": normalized.get("reason", ""),
        },
    }


__all__ = [
    "BoundedHttpFetcher",
    "RawFetchResult",
    "RetrievalLimitError",
    "normalize_fetched_body",
    "normalize_search_envelope",
    "page_record",
    "select_urls",
]
