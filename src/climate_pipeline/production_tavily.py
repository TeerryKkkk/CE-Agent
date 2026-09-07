"""Fail-closed Tavily transport used only by the formal remaining-368 runner."""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
import urllib.request
from typing import Any

from .bounded_http import (
    no_redirect_opener,
    read_bounded,
    read_http_error,
    validate_headers,
)


class TavilySearchClient:
    def __init__(
        self,
        *,
        api_key: str,
        search_url: str,
        search_depth: str,
        timeout_seconds: int,
        allow_auth_body_fallback: bool,
        max_request_bytes: int,
        max_response_bytes: int,
        max_error_response_bytes: int,
        max_redirects: int,
        max_redirect_response_bytes: int,
        max_header_bytes: int,
        max_redirect_location_bytes: int,
        opener: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("missing_tavily_api_key")
        if allow_auth_body_fallback:
            raise ValueError("tavily_auth_body_fallback_forbidden")
        if int(max_redirects) != 0:
            raise ValueError("tavily_redirects_must_be_disabled")
        limits = {
            "timeout_seconds": timeout_seconds,
            "max_request_bytes": max_request_bytes,
            "max_response_bytes": max_response_bytes,
            "max_error_response_bytes": max_error_response_bytes,
            "max_redirect_response_bytes": max_redirect_response_bytes,
            "max_header_bytes": max_header_bytes,
            "max_redirect_location_bytes": max_redirect_location_bytes,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits.values()):
            raise ValueError(f"invalid_tavily_limits:{limits}")
        self.api_key = api_key
        self.search_url = search_url
        self.search_depth = search_depth
        self.timeout_seconds = timeout_seconds
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.max_error_response_bytes = max_error_response_bytes
        self.max_redirect_response_bytes = max_redirect_response_bytes
        self.max_header_bytes = max_header_bytes
        self.max_redirect_location_bytes = max_redirect_location_bytes
        self.opener = opener or no_redirect_opener()

    def _request_envelope(
        self, query: str, max_results: int, *, include_raw_content: bool
    ) -> dict[str, Any]:
        body = {
            "query": query,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
        }
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
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with self.opener(request, timeout=self.timeout_seconds) as response:
            validate_headers(
                getattr(response, "headers", {}),
                max_header_bytes=self.max_header_bytes,
                max_location_bytes=self.max_redirect_location_bytes,
            )
            raw = read_bounded(
                response,
                max_bytes=self.max_response_bytes,
                label="tavily_response_body",
            ).data
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(parsed, dict):
            raise ValueError("tavily_response_not_object")
        return {
            "payload": parsed,
            "raw_text": raw.decode("utf-8", errors="replace"),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_byte_count": len(raw),
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
        try:
            envelope = self._request_envelope(
                query, max_results, include_raw_content=include_raw_content
            )
        except urllib.error.HTTPError as exc:
            kind, detail = read_http_error(
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
                "error": f"tavily_{kind}_http_{exc.code}:{detail}",
            }
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            return {}, {
                "client": "tavily", "query": query, "result_count": 0,
                "error": f"tavily_transport:{type(exc).__name__}:{str(exc)[:300]}",
            }
        except Exception as exc:
            return {}, {
                "client": "tavily", "query": query, "result_count": 0,
                "error": f"tavily_boundary:{type(exc).__name__}:{str(exc)[:300]}",
            }
        results = envelope["payload"].get("results")
        count = len(results[:max_results]) if isinstance(results, list) else 0
        return envelope, {
            "client": "tavily",
            "query": query,
            "result_count": count,
            "error": None,
            "search_depth": self.search_depth,
            "include_raw_content": include_raw_content,
        }


__all__ = ["TavilySearchClient"]
