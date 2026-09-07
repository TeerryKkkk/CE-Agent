from __future__ import annotations

import html
import importlib.util
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import config
from .hashing import sha256_bytes
from .io_utils import repo_relative_path, safe_write_text


@dataclass
class WebReaderResult:
    original_url: str
    final_url: str
    fetch_method: str
    http_status: int
    raw_html_path: str
    cleaned_text_path: str
    raw_html_len: int
    cleaned_text_len: int
    content_hash: str
    encoding: str | None
    failure_reason: str | None
    fallback_attempted: bool
    fallback_used: bool
    reader_url: str | None = None
    content_type: str = ""
    title: str = ""

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def extract_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", re.sub("<.*?>", "", match.group(1))).strip())


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>|</div>|</li>|</tr>|</h[1-6]>", "\n", text)
    text = re.sub(r"<.*?>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_failure(value: str) -> str:
    value = re.sub(r"(?i)(authorization|api[-_]?key|token)=([^&\s]+)", r"\1=[REDACTED]", value)
    return value[:500]


def _looks_like_document_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    target = f"{parsed.path}?{parsed.query}".lower()
    return bool(
        re.search(r"\.(pdf|doc|docx|wps|xls|xlsx)(?:$|[?&#])", target)
        or "file-download" in target
        or ("fileurl=" in target and ".pdf" in target)
    )


def _looks_like_pdf_payload(data: bytes, content_type: str) -> bool:
    return "application/pdf" in content_type.lower() or data.lstrip().startswith(b"%PDF")


def _is_reader_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "warning: target url returned error",
            "# 403 forbidden",
            "# 404 not found",
            "target url returned error 403",
            "target url returned error 404",
        )
    )


def _decode_response(data: bytes, headers: Any) -> tuple[str, str]:
    encoding = None
    get_charset = getattr(headers, "get_content_charset", None)
    if callable(get_charset):
        encoding = get_charset()
    if not encoding:
        content_type = headers.get("content-type", "") if hasattr(headers, "get") else ""
        match = re.search(r"charset=([\w-]+)", content_type, re.I)
        if match:
            encoding = match.group(1)
    candidates = [encoding, "utf-8", "gb18030"]
    for candidate in [item for item in candidates if item]:
        try:
            return data.decode(candidate), candidate
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def _clean_html(raw_html: str, url: str) -> str:
    trafilatura_text = ""
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            raw_html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=False,
        )
        if extracted:
            trafilatura_text = extracted.strip()
    except Exception:
        trafilatura_text = ""
    fallback_text = html_to_text(raw_html)
    if len(fallback_text) > len(trafilatura_text):
        return fallback_text
    return trafilatura_text


def build_jina_reader_url(original_url: str) -> str:
    if original_url.startswith("https://r.jina.ai/http://"):
        original_url = original_url.replace("https://r.jina.ai/http://", "http://", 1)
    parsed = urllib.parse.urlparse(original_url)
    if parsed.netloc:
        target = urllib.parse.urlunparse(("http", parsed.netloc, parsed.path or "/", "", parsed.query, ""))
    else:
        stripped = original_url.replace("https://r.jina.ai/http://", "", 1)
        stripped = stripped.removeprefix("https://").removeprefix("http://")
        target = f"http://{stripped}"
    return f"https://r.jina.ai/{target}"


class WebReader:
    def __init__(
        self,
        snapshot_dir: Path | None = None,
        opener=None,
        min_cleaned_text_chars: int | None = None,
        timeout_seconds: int | None = None,
        retries: int | None = None,
        *,
        allow_jina_fallback: bool = True,
        allow_playwright_fallback: bool = True,
        max_response_bytes: int | None = None,
    ) -> None:
        self.snapshot_dir = snapshot_dir or config.LIVE_SNAPSHOT_DIR
        self.opener = opener or urllib.request.urlopen
        self.min_cleaned_text_chars = min_cleaned_text_chars or config.WEB_READER_MIN_CLEANED_TEXT_CHARS
        self.timeout_seconds = timeout_seconds or config.WEB_READER_TIMEOUT_SECONDS
        self.retries = retries if retries is not None else config.WEB_READER_HTTP_RETRIES
        self.allow_jina_fallback = bool(allow_jina_fallback)
        self.allow_playwright_fallback = bool(allow_playwright_fallback)
        if max_response_bytes is not None and max_response_bytes <= 0:
            raise ValueError("max_response_bytes_must_be_positive")
        self.max_response_bytes = max_response_bytes

    def read(self, url: str, title_hint: str = "") -> WebReaderResult:
        if self.allow_jina_fallback and _looks_like_document_url(url):
            jina_result = self._read_jina(url, title_hint)
            if self._is_good_enough(jina_result):
                jina_result.fallback_attempted = True
                jina_result.fallback_used = True
                return jina_result

        http_result = self._read_http(url, title_hint)
        if self._is_good_enough(http_result):
            return http_result

        failure_parts = [http_result.failure_reason or "http_empty_or_too_short"]
        http_result.fallback_attempted = True
        if self.allow_jina_fallback:
            jina_result = self._read_jina(url, title_hint)
            if self._is_good_enough(jina_result):
                jina_result.fallback_attempted = True
                jina_result.fallback_used = True
                return jina_result
            failure_parts.append(jina_result.failure_reason or "jina_reader_empty_or_too_short")

        if self.allow_playwright_fallback:
            playwright_result, playwright_failure = self._read_playwright_if_available(url, title_hint)
            if playwright_result and self._is_good_enough(playwright_result):
                playwright_result.fallback_attempted = True
                playwright_result.fallback_used = True
                return playwright_result
            if playwright_failure:
                failure_parts.append(playwright_failure)

        http_result.failure_reason = "; ".join(part for part in failure_parts if part)
        http_result.fallback_attempted = True
        http_result.fallback_used = False
        return http_result

    def _is_good_enough(self, result: WebReaderResult) -> bool:
        if result.failure_reason:
            return False
        text_path = config.PROJECT_ROOT / result.cleaned_text_path
        if _is_reader_error_text((text_path.read_text(encoding="utf-8", errors="ignore") if text_path.exists() else "")):
            return False
        return 200 <= result.http_status < 400 and result.cleaned_text_len >= self.min_cleaned_text_chars

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ERA5EvidenceMVP/0.1",
            "Accept": accept,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def _read_http(self, url: str, title_hint: str) -> WebReaderResult:
        raw_html = ""
        cleaned_text = ""
        final_url = url
        status = 0
        content_type = ""
        encoding = None
        failure = None
        attempts = max(1, self.retries + 1)
        for _ in range(attempts):
            try:
                request = urllib.request.Request(url, headers=self._headers("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"))
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    status = getattr(response, "status", 200)
                    final_url = response.geturl() if hasattr(response, "geturl") else url
                    content_type = response.headers.get("content-type", "") if hasattr(response, "headers") else ""
                    data = response.read(
                        self.max_response_bytes + 1
                        if self.max_response_bytes is not None
                        else -1
                    )
                    if self.max_response_bytes is not None and len(data) > self.max_response_bytes:
                        raise ValueError(
                            f"response_body_exceeds_limit:{self.max_response_bytes}"
                        )
                    if _looks_like_pdf_payload(data, content_type):
                        raw_html = ""
                        cleaned_text = ""
                        encoding = None
                        failure = "http_pdf_requires_text_reader"
                        break
                    raw_html, encoding = _decode_response(data, response.headers if hasattr(response, "headers") else {})
                cleaned_text = _clean_html(raw_html, final_url)
                failure = None
                break
            except urllib.error.HTTPError as exc:
                status = exc.code
                final_url = exc.geturl() if hasattr(exc, "geturl") else url
                content_type = exc.headers.get("content-type", "") if exc.headers else ""
                data = exc.read() if hasattr(exc, "read") else b""
                raw_html, encoding = _decode_response(data, exc.headers or {})
                cleaned_text = _clean_html(raw_html, final_url) if raw_html else ""
                failure = f"http_error_{status}"
                break
            except (TimeoutError, socket.timeout) as exc:
                failure = f"http_timeout_after_{self.timeout_seconds}s: {_safe_failure(str(exc))}"
            except Exception as exc:
                failure = _safe_failure(str(exc))
        if status == 0 and not failure:
            failure = "http_fetch_failed"
        if status and len(cleaned_text) < self.min_cleaned_text_chars and not failure:
            failure = "http_cleaned_text_too_short"
        return self._save_result(
            original_url=url,
            final_url=final_url,
            fetch_method="http",
            http_status=status,
            raw_html=raw_html,
            cleaned_text=cleaned_text,
            encoding=encoding,
            failure_reason=failure,
            fallback_attempted=False,
            fallback_used=False,
            reader_url=None,
            content_type=content_type,
            title=extract_title(raw_html) or title_hint,
        )

    def _read_jina(self, url: str, title_hint: str) -> WebReaderResult:
        reader_url = build_jina_reader_url(url)
        text = ""
        status = 0
        content_type = ""
        encoding = None
        failure = None
        try:
            request = urllib.request.Request(reader_url, headers=self._headers("text/plain, text/markdown, */*"))
            with self.opener(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 200)
                content_type = response.headers.get("content-type", "") if hasattr(response, "headers") else ""
                data = response.read(
                    self.max_response_bytes + 1
                    if self.max_response_bytes is not None
                    else -1
                )
                if self.max_response_bytes is not None and len(data) > self.max_response_bytes:
                    raise ValueError(
                        f"response_body_exceeds_limit:{self.max_response_bytes}"
                    )
                text, encoding = _decode_response(data, response.headers if hasattr(response, "headers") else {})
        except urllib.error.HTTPError as exc:
            status = exc.code
            data = exc.read() if hasattr(exc, "read") else b""
            text, encoding = _decode_response(data, exc.headers or {})
            failure = f"jina_reader_http_error_{status}"
        except (TimeoutError, socket.timeout) as exc:
            failure = f"jina_reader_timeout_after_{self.timeout_seconds}s: {_safe_failure(str(exc))}"
        except Exception as exc:
            failure = f"jina_reader_failed: {_safe_failure(str(exc))}"
        text = text.strip()
        if status == 0 and not failure:
            failure = "jina_reader_fetch_failed"
        if status and _is_reader_error_text(text) and not failure:
            failure = "jina_reader_target_error"
        if status and len(text) < self.min_cleaned_text_chars and not failure:
            failure = "jina_reader_cleaned_text_too_short"
        title = title_hint
        match = re.search(r"(?im)^title:\s*(.+)$", text)
        if match:
            title = match.group(1).strip()
        return self._save_result(
            original_url=url,
            final_url=url,
            fetch_method="jina_reader",
            http_status=status,
            raw_html="",
            cleaned_text=text,
            encoding=encoding,
            failure_reason=failure,
            fallback_attempted=True,
            fallback_used=False,
            reader_url=reader_url,
            content_type=content_type,
            title=title,
        )

    def _read_playwright_if_available(self, url: str, title_hint: str) -> tuple[WebReaderResult | None, str | None]:
        if importlib.util.find_spec("playwright") is None:
            return None, "playwright_unavailable"
        try:
            from playwright.sync_api import sync_playwright  # type: ignore

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(extra_http_headers={"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})
                response = page.goto(url, wait_until="networkidle", timeout=self.timeout_seconds * 1000)
                raw_html = page.content()
                status = response.status if response else 0
                final_url = page.url
                title = page.title() or title_hint
                browser.close()
            text = _clean_html(raw_html, final_url)
            failure = None
            if status == 0:
                failure = "playwright_fetch_failed"
            elif len(text) < self.min_cleaned_text_chars:
                failure = "playwright_cleaned_text_too_short"
            return (
                self._save_result(
                    original_url=url,
                    final_url=final_url,
                    fetch_method="playwright",
                    http_status=status,
                    raw_html=raw_html,
                    cleaned_text=text,
                    encoding="browser",
                    failure_reason=failure,
                    fallback_attempted=True,
                    fallback_used=False,
                    reader_url=None,
                    content_type="text/html; playwright",
                    title=title,
                ),
                None,
            )
        except Exception as exc:
            return None, f"playwright_failed: {_safe_failure(str(exc))}"

    def _save_result(
        self,
        original_url: str,
        final_url: str,
        fetch_method: str,
        http_status: int,
        raw_html: str,
        cleaned_text: str,
        encoding: str | None,
        failure_reason: str | None,
        fallback_attempted: bool,
        fallback_used: bool,
        reader_url: str | None,
        content_type: str,
        title: str,
    ) -> WebReaderResult:
        payload = raw_html or cleaned_text or final_url or original_url
        content_hash = sha256_bytes(payload.encode("utf-8"))
        html_path = self.snapshot_dir / "html" / f"{content_hash}.html"
        text_path = self.snapshot_dir / "text" / f"{content_hash}.txt"
        safe_write_text(html_path, raw_html)
        safe_write_text(text_path, cleaned_text)
        return WebReaderResult(
            original_url=original_url,
            final_url=final_url,
            fetch_method=fetch_method,
            http_status=http_status,
            raw_html_path=repo_relative_path(html_path),
            cleaned_text_path=repo_relative_path(text_path),
            raw_html_len=len(raw_html),
            cleaned_text_len=len(cleaned_text),
            content_hash=content_hash,
            encoding=encoding,
            failure_reason=failure_reason,
            fallback_attempted=fallback_attempted,
            fallback_used=fallback_used,
            reader_url=reader_url,
            content_type=content_type,
            title=title,
        )
