"""Frozen body-normalization helpers under the canonical package namespace."""

from __future__ import annotations

import html
import re
from typing import Any


def extract_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    if not match:
        return ""
    return html.unescape(
        re.sub(r"\s+", " ", re.sub("<.*?>", "", match.group(1))).strip()
    )


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


def looks_like_pdf_payload(data: bytes, content_type: str) -> bool:
    return "application/pdf" in content_type.lower() or data.lstrip().startswith(b"%PDF")


def decode_response(data: bytes, headers: Any) -> tuple[str, str]:
    encoding = None
    get_charset = getattr(headers, "get_content_charset", None)
    if callable(get_charset):
        encoding = get_charset()
    if not encoding:
        content_type = headers.get("content-type", "") if hasattr(headers, "get") else ""
        match = re.search(r"charset=([\w-]+)", content_type, re.I)
        if match:
            encoding = match.group(1)
    for candidate in [item for item in (encoding, "utf-8", "gb18030") if item]:
        try:
            return data.decode(candidate), candidate
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def clean_html(raw_html: str, url: str) -> str:
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
    return fallback_text if len(fallback_text) > len(trafilatura_text) else trafilatura_text


__all__ = [
    "clean_html",
    "decode_response",
    "extract_title",
    "html_to_text",
    "looks_like_pdf_payload",
]
