from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Iterable


HAZARD_TERMS = [
    "atmospheric river",
    "winter storm",
    "flash flood",
    "flood warning",
    "flood watch",
    "flooding",
    "flood",
    "storm",
    "mudslide",
    "landslide",
    "debris flow",
    "rockslide",
    "river",
    "heavy rain",
    "rainfall",
]

IMPACT_TERMS = [
    "evacuation order",
    "evacuation warning",
    "evacuation center",
    "evacuation",
    "access warning",
    "return home",
    "road closure",
    "roadway flooding",
    "road washed out",
    "bridge closure",
    "closure",
    "closed",
    "damage",
    "property damage",
    "rescue",
    "shelter",
    "public works",
    "emergency response",
    "local proclamation",
    "disaster declaration",
    "assistance center",
    "power outage",
]

GENERIC_PAGE_SIGNALS = [
    "emergency plan",
    "emergency plans",
    "preparedness",
    "hazard mitigation plan",
    "resource page",
    "resources",
    "assistance resources",
    "index page",
    "standing plan",
    "congresswoman",
    "congressman",
    "house.gov",
]

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

MONTH_ABBREVIATIONS = {
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec",
}

RAW_DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2}(?:,\s*\d{4})?\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
)


@dataclass(frozen=True)
class CandidateContext:
    candidate_id: str
    county: str
    county_fips: str = ""
    state: str = "California"
    drought_window: str = ""
    wet_window: str = ""
    candidate_stratum: str = ""
    locality_hints: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSpan:
    candidate_id: str
    source_url: str
    span_id: str
    span_text: str
    location_terms_found: list[str]
    time_terms_found: list[str]
    hazard_terms_found: list[str]
    impact_terms_found: list[str]
    generic_page_signals: list[str]
    selected_for_llm: bool
    notes: str = ""

    def to_row(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "source_url": self.source_url,
            "span_id": self.span_id,
            "span_text": self.span_text,
            "location_terms_found": ";".join(self.location_terms_found),
            "time_terms_found": ";".join(self.time_terms_found),
            "hazard_terms_found": ";".join(self.hazard_terms_found),
            "impact_terms_found": ";".join(self.impact_terms_found),
            "generic_page_signals": ";".join(self.generic_page_signals),
            "selected_for_llm": str(self.selected_for_llm).lower(),
            "notes": self.notes,
        }

    def to_prompt_dict(self) -> dict[str, object]:
        return self.to_row()


def build_location_terms(context: CandidateContext) -> list[str]:
    terms = [context.county, context.state, *context.locality_hints]
    county_base = re.sub(r"\s+county$", "", context.county, flags=re.IGNORECASE).strip()
    if county_base and county_base != context.county:
        terms.append(county_base)
    if context.county_fips:
        terms.append(context.county_fips)
    return _unique_terms(terms)


def build_time_terms(wet_window: str) -> list[str]:
    start, end = parse_window_dates(wet_window)
    terms: list[str] = []
    if start:
        terms.extend(_date_terms(start))
    if end and end != start:
        terms.extend(_date_terms(end))
    if start and end:
        for current_year, current_month in _months_between(start, end):
            month_name = MONTH_NAMES[current_month - 1]
            abbr = MONTH_ABBREVIATIONS[month_name]
            terms.append(f"{month_name} {current_year}")
            terms.append(f"{abbr}. {current_year}")
            terms.append(f"{abbr} {current_year}")
        if start.year != end.year:
            terms.append("late December through January")
            terms.append("late December through Jan")
        if start.month == 12 and end.month == 1:
            terms.append("late December through January")
        if start.year == 2022 and end.year == 2023:
            terms.append("winter storms between Dec. 27, 2022 - Jan. 31, 2023")
            terms.append("Dec. 27, 2022 - Jan. 31, 2023")
    if start:
        terms.append(str(start.year))
    if end and end.year != start.year:
        terms.append(str(end.year))
    terms.extend(["January 2023 atmospheric river", "winter storm", "atmospheric river event"])
    return _unique_terms(terms)


def parse_window_dates(window: str) -> tuple[date | None, date | None]:
    parts = [part.strip() for part in re.split(r"\s+to\s+", window or "", maxsplit=1)]
    if len(parts) == 1:
        parts.append(parts[0])
    parsed = [_parse_date_part(part) for part in parts[:2]]
    return parsed[0], parsed[1]


def extract_candidate_spans(
    *,
    candidate: CandidateContext,
    source_url: str,
    body_text: str,
    max_spans: int = 10,
    max_span_chars: int = 1100,
) -> list[EvidenceSpan]:
    location_terms = build_location_terms(candidate)
    time_terms = build_time_terms(candidate.wet_window)
    scored: list[tuple[int, int, EvidenceSpan]] = []

    for index, segment in enumerate(_candidate_segments(body_text, max_span_chars=max_span_chars)):
        found_location = find_terms(segment, location_terms)
        found_time = _find_time_terms(segment, time_terms)
        found_hazard = find_terms(segment, HAZARD_TERMS)
        found_impact = find_terms(segment, IMPACT_TERMS)
        found_generic = find_terms(segment, GENERIC_PAGE_SIGNALS)
        score = _span_score(found_location, found_time, found_hazard, found_impact, found_generic)
        selected = _selected_for_llm(found_location, found_time, found_hazard, found_impact, found_generic, score)
        if not selected and not found_generic:
            continue
        span = EvidenceSpan(
            candidate_id=candidate.candidate_id,
            source_url=source_url,
            span_id=f"{candidate.candidate_id}_span_{index + 1:03d}",
            span_text=segment,
            location_terms_found=found_location,
            time_terms_found=found_time,
            hazard_terms_found=found_hazard,
            impact_terms_found=found_impact,
            generic_page_signals=found_generic,
            selected_for_llm=selected,
            notes=_span_notes(found_location, found_time, found_hazard, found_impact, found_generic),
        )
        scored.append((score, -index, span))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    spans = [span for _, _, span in scored[:max_spans]]
    if spans:
        return spans

    fallback_text = _clip(clean_text(body_text), max_span_chars)
    return [
        EvidenceSpan(
            candidate_id=candidate.candidate_id,
            source_url=source_url,
            span_id=f"{candidate.candidate_id}_span_001",
            span_text=fallback_text,
            location_terms_found=find_terms(fallback_text, location_terms),
            time_terms_found=_find_time_terms(fallback_text, time_terms),
            hazard_terms_found=find_terms(fallback_text, HAZARD_TERMS),
            impact_terms_found=find_terms(fallback_text, IMPACT_TERMS),
            generic_page_signals=find_terms(fallback_text, GENERIC_PAGE_SIGNALS),
            selected_for_llm=True,
            notes="fallback_first_page_chunk",
        )
    ]


def clean_text(text: str) -> str:
    unescaped = html.unescape(text or "")
    unescaped = re.sub(r"<[^>]+>", " ", unescaped)
    unescaped = re.sub(r"\s+", " ", unescaped)
    return unescaped.strip()


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    lower = text.lower()
    found = []
    for term in terms:
        norm = str(term).strip()
        if not norm:
            continue
        if norm.lower() in lower and norm not in found:
            found.append(norm)
    return found


def _candidate_segments(body_text: str, *, max_span_chars: int) -> list[str]:
    prepared = _insert_soft_breaks(body_text)
    raw_parts = [part.strip() for part in re.split(r"\n{1,}", prepared) if part.strip()]
    segments: list[str] = []
    for part in raw_parts:
        cleaned = clean_text(part)
        if not cleaned:
            continue
        if len(cleaned) <= max_span_chars:
            segments.append(cleaned)
        else:
            segments.extend(_sentence_windows(cleaned, max_span_chars=max_span_chars))
    return _unique_terms(segments)


def _insert_soft_breaks(text: str) -> str:
    value = text or ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+(?=#{2,6}\s)", "\n", value)
    value = re.sub(r"\s+(?=\*{0,4}\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2})", "\n", value)
    value = re.sub(r"\s+(?=\*{0,4}\d{1,2}/\d{1,2}/\d{2,4})", "\n", value)
    value = re.sub(r"\s+(?=(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})", "\n", value)
    value = re.sub(r"\s+(?=Title:\s)", "\n", value)
    return value


def _sentence_windows(text: str, *, max_span_chars: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return [_clip(text, max_span_chars)]
    windows: list[str] = []
    for index in range(0, len(sentences), 2):
        window = " ".join(sentences[index : index + 3]).strip()
        if window:
            windows.append(_clip(window, max_span_chars))
    return windows


def _span_score(
    found_location: list[str],
    found_time: list[str],
    found_hazard: list[str],
    found_impact: list[str],
    found_generic: list[str],
) -> int:
    return (
        5 * bool(found_location)
        + 5 * bool(found_time)
        + 4 * bool(found_impact)
        + 3 * bool(found_hazard)
        + 2 * bool(found_generic)
        + min(len(found_location), 3)
        + min(len(found_time), 3)
        + min(len(found_impact), 3)
    )


def _selected_for_llm(
    found_location: list[str],
    found_time: list[str],
    found_hazard: list[str],
    found_impact: list[str],
    found_generic: list[str],
    score: int,
) -> bool:
    if found_generic and (found_time or found_hazard or found_impact):
        return True
    if found_location and found_time and (found_hazard or found_impact):
        return True
    if found_location and found_hazard and found_impact:
        return True
    return score >= 12


def _span_notes(
    found_location: list[str],
    found_time: list[str],
    found_hazard: list[str],
    found_impact: list[str],
    found_generic: list[str],
) -> str:
    notes: list[str] = []
    if found_location and found_time and found_impact:
        notes.append("compact_location_time_impact_candidate")
    if found_generic:
        notes.append("generic_or_resource_signal")
    if found_time == [term for term in found_time if re.fullmatch(r"\d{4}", term)]:
        notes.append("year_only_or_weak_time")
    return ";".join(notes)


def _find_time_terms(text: str, generated_terms: Iterable[str]) -> list[str]:
    found = find_terms(text, generated_terms)
    for match in RAW_DATE_PATTERN.findall(text):
        value = match.strip()
        if value and value not in found:
            found.append(value)
    return found


def _parse_date_part(value: str) -> date | None:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%m/%d/%Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m":
                return date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def _date_terms(value: date) -> list[str]:
    month = MONTH_NAMES[value.month - 1]
    abbr = MONTH_ABBREVIATIONS[month]
    return [
        value.isoformat(),
        f"{value.month}/{value.day}/{value.year}",
        f"{value.month:02d}/{value.day:02d}/{str(value.year)[2:]}",
        f"{month} {value.day}, {value.year}",
        f"{month} {value.day}",
        f"{abbr}. {value.day}, {value.year}",
        f"{abbr}. {value.day}",
        f"{abbr} {value.day}",
    ]


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        values.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return values


def _clip(text: str, max_chars: int) -> str:
    value = text.strip()
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped or value[:max_chars].strip()


def _unique_terms(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique
