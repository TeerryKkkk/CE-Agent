from __future__ import annotations

import json
import hashlib
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import config
from .schema_models import SearchResult

from .bounded_http import (
    BoundedHttpError,
    no_redirect_opener,
    read_bounded,
    read_http_error,
    validate_headers,
)


def read_tavily_api_key(path: Path | None = None) -> str:
    if config.TAVILY_API_KEY:
        return config.TAVILY_API_KEY
    key_paths = [path] if path else [
        config.PROJECT_ROOT / "tavily_apikey.txt",
        config.PROJECT_ROOT / "apikeys" / "tavily_apikey.txt",
    ]
    for key_path in key_paths:
        if not key_path or not key_path.exists():
            continue
        with key_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    return value
    return ""


def _safe_error(exc: Exception, api_key: str) -> str:
    text = str(exc)
    if api_key:
        text = text.replace(api_key, "[REDACTED_TAVILY_KEY]")
    return text[:500]


def parse_tavily_results(payload: dict[str, Any], max_results: int = 5) -> list[SearchResult]:
    raw_results = payload.get("results") if isinstance(payload.get("results"), list) else []
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(str(url))
        results.append(
            SearchResult(
                title=str(item.get("title") or url),
                url=str(url),
                snippet=str(item.get("content") or item.get("snippet") or ""),
                rank=len(results) + 1,
            )
        )
        if len(results) >= max_results:
            break
    return results


class TavilySearchClient:
    def __init__(
        self,
        api_key: str | None = None,
        search_url: str | None = None,
        opener=None,
        search_depth: str = "advanced",
        timeout_seconds: int | None = None,
        allow_auth_body_fallback: bool = True,
        max_request_bytes: int = 65536,
        max_response_bytes: int = 2_000_000,
        max_error_response_bytes: int = 65_536,
        max_redirects: int = 0,
        max_redirect_response_bytes: int = 16_384,
        max_header_bytes: int = 65_536,
        max_redirect_location_bytes: int = 4_096,
    ) -> None:
        self.api_key = api_key if api_key is not None else read_tavily_api_key()
        self.search_url = search_url or config.TAVILY_SEARCH_URL
        if int(max_redirects) != 0:
            raise ValueError("tavily_redirects_must_be_disabled")
        self.opener = opener or no_redirect_opener()
        self.search_depth = search_depth
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.TAVILY_TIMEOUT_SECONDS
        )
        if self.timeout_seconds <= 0:
            raise ValueError("tavily_timeout_must_be_positive")
        self.allow_auth_body_fallback = bool(allow_auth_body_fallback)
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_response_bytes", max_response_bytes),
            ("max_error_response_bytes", max_error_response_bytes),
            ("max_redirect_response_bytes", max_redirect_response_bytes),
            ("max_header_bytes", max_header_bytes),
            ("max_redirect_location_bytes", max_redirect_location_bytes),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name}_must_be_positive")
        self.max_request_bytes = int(max_request_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.max_error_response_bytes = int(max_error_response_bytes)
        self.max_redirects = 0
        self.max_redirect_response_bytes = int(max_redirect_response_bytes)
        self.max_header_bytes = int(max_header_bytes)
        self.max_redirect_location_bytes = int(max_redirect_location_bytes)

    def _request(
        self,
        query: str,
        max_results: int,
        include_key_in_body: bool = False,
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        return self._request_envelope(
            query,
            max_results,
            include_key_in_body=include_key_in_body,
            include_raw_content=include_raw_content,
        )["payload"]

    def _request_envelope(
        self,
        query: str,
        max_results: int,
        include_key_in_body: bool = False,
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
        }
        if include_key_in_body:
            body["api_key"] = self.api_key
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        wire = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(wire) > self.max_request_bytes:
            raise ValueError(
                f"tavily_request_body_exceeds_limit:{self.max_request_bytes}"
            )
        request = urllib.request.Request(
            self.search_url,
            data=wire,
            headers=headers,
            method="POST",
        )
        with self.opener(request, timeout=self.timeout_seconds) as response:
            validate_headers(
                getattr(response, "headers", {}),
                max_header_bytes=self.max_header_bytes,
                max_location_bytes=self.max_redirect_location_bytes,
            )
            raw_bytes = read_bounded(
                response,
                max_bytes=self.max_response_bytes,
                label="tavily_response_body",
            ).data
        raw = raw_bytes.decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("tavily_response_not_object")
        return {
            "payload": parsed,
            "raw_text": raw,
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_byte_count": len(raw_bytes),
            "request_sha256": hashlib.sha256(wire).hexdigest(),
            "request_byte_count": len(wire),
        }

    def search_raw_envelope(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_raw_content: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the bounded provider envelope for durable raw checkpointing."""

        if not self.api_key:
            return {}, {
                "client": "tavily",
                "query": query,
                "result_count": 0,
                "error": "missing_tavily_api_key",
            }
        try:
            envelope = self._request_envelope(
                query,
                max_results,
                include_raw_content=include_raw_content,
            )
        except urllib.error.HTTPError as exc:
            # No compatibility/body-auth fallback is allowed in production.
            kind, suffix = read_http_error(
                exc,
                max_error_bytes=self.max_error_response_bytes,
                max_redirect_bytes=self.max_redirect_response_bytes,
                max_header_bytes=self.max_header_bytes,
                max_location_bytes=self.max_redirect_location_bytes,
            )
            return {}, {
                "client": "tavily",
                "query": query,
                "result_count": 0,
                "error": f"tavily_{kind}_http_{exc.code}: {_safe_error(Exception(suffix), self.api_key)}",
            }
        except (TimeoutError, socket.timeout) as exc:
            return {}, {
                "client": "tavily",
                "query": query,
                "result_count": 0,
                "error": f"tavily_timeout_after_{self.timeout_seconds}s: {_safe_error(exc, self.api_key)}",
            }
        except Exception as exc:
            return {}, {
                "client": "tavily",
                "query": query,
                "result_count": 0,
                "error": _safe_error(exc, self.api_key),
            }
        results = envelope["payload"].get("results")
        result_count = len(results[:max_results]) if isinstance(results, list) else 0
        return envelope, {
            "client": "tavily",
            "query": query,
            "result_count": result_count,
            "error": None,
            "search_depth": self.search_depth,
            "include_raw_content": include_raw_content,
        }

    def search(self, query: str, max_results: int = 5) -> tuple[list[SearchResult], dict[str, Any]]:
        if not self.api_key:
            return [], {"client": "tavily", "query": query, "result_count": 0, "error": "missing_tavily_api_key"}
        try:
            payload = self._request(query, max_results)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403} and self.allow_auth_body_fallback:
                try:
                    payload = self._request(query, max_results, include_key_in_body=True)
                except Exception as fallback_exc:
                    return [], {
                        "client": "tavily",
                        "query": query,
                        "result_count": 0,
                        "error": f"tavily_http_{exc.code}; fallback_failed: {_safe_error(fallback_exc, self.api_key)}",
                    }
            else:
                return [], {
                    "client": "tavily",
                    "query": query,
                    "result_count": 0,
                    "error": f"tavily_http_{exc.code}: {_safe_error(exc, self.api_key)}",
                }
        except (TimeoutError, socket.timeout) as exc:
            return [], {
                "client": "tavily",
                "query": query,
                "result_count": 0,
                "error": f"tavily_timeout_after_{self.timeout_seconds}s: {_safe_error(exc, self.api_key)}",
            }
        except Exception as exc:
            return [], {"client": "tavily", "query": query, "result_count": 0, "error": _safe_error(exc, self.api_key)}
        results = parse_tavily_results(payload, max_results=max_results)
        return results, {
            "client": "tavily",
            "query": query,
            "result_count": len(results),
            "error": None,
            "search_depth": self.search_depth,
        }

    def search_raw(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_raw_content: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.api_key:
            return [], {"client": "tavily", "query": query, "result_count": 0, "error": "missing_tavily_api_key"}
        try:
            payload = self._request(query, max_results, include_raw_content=include_raw_content)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403} and self.allow_auth_body_fallback:
                try:
                    payload = self._request(
                        query,
                        max_results,
                        include_key_in_body=True,
                        include_raw_content=include_raw_content,
                    )
                except Exception as fallback_exc:
                    return [], {
                        "client": "tavily",
                        "query": query,
                        "result_count": 0,
                        "error": f"tavily_http_{exc.code}; fallback_failed: {_safe_error(fallback_exc, self.api_key)}",
                    }
            else:
                return [], {
                    "client": "tavily",
                    "query": query,
                    "result_count": 0,
                    "error": f"tavily_http_{exc.code}: {_safe_error(exc, self.api_key)}",
                }
        except (TimeoutError, socket.timeout) as exc:
            return [], {
                "client": "tavily",
                "query": query,
                "result_count": 0,
                "error": f"tavily_timeout_after_{self.timeout_seconds}s: {_safe_error(exc, self.api_key)}",
            }
        except Exception as exc:
            return [], {"client": "tavily", "query": query, "result_count": 0, "error": _safe_error(exc, self.api_key)}
        raw_results = payload.get("results") if isinstance(payload.get("results"), list) else []
        rows = [row for row in raw_results if isinstance(row, dict)]
        return rows[:max_results], {
            "client": "tavily",
            "query": query,
            "result_count": len(rows[:max_results]),
            "error": None,
            "search_depth": self.search_depth,
            "include_raw_content": include_raw_content,
        }
