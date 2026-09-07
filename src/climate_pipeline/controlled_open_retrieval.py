from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


RETRIEVAL_SCHEMA_VERSION = "phase2_3_2_calibrated_controlled_open_retrieval_v1"
LEGACY_PREFETCH_POLICY_VERSION = "legacy_prefetch_scoring_v1"
TARGET_JURISDICTION_PREFETCH_POLICY_VERSION = (
    "target_jurisdiction_specificity_gate_v1"
)

LANE_PRIORITY = {
    "local_government_emergency": 10,
    "county_city_official": 11,
    "sheriff_oes_emergency": 12,
    "transportation_roads": 20,
    "public_works_flood_control": 21,
    "caloes_state_emergency": 22,
    "caltrans_transportation": 23,
    "local_news": 30,
    "regional_news": 31,
    "water_river_reservoir": 40,
    "california_dwr_water": 41,
    "agriculture_drought_impact": 50,
    "agriculture_official": 51,
    "general_web": 90,
}

REQUIRED_LANES = tuple(LANE_PRIORITY)

FLOOD_RAIN_TERMS = (
    "flood",
    "flooding",
    "flash flood",
    "heavy rain",
    "rainfall",
    "storm",
    "thunderstorm",
    "severe weather",
    "high water",
    "atmospheric river",
    "winter storm",
)
DROUGHT_TERMS = (
    "drought",
    "water restriction",
    "water shortage",
    "crop loss",
    "livestock",
    "pasture",
    "burn ban",
)
IMPACT_TERMS = (
    "road closure",
    "closed road",
    "high water",
    "damage",
    "rescue",
    "evacuation",
    "emergency declaration",
    "disaster declaration",
    "power outage",
    "crop loss",
    "livestock loss",
    "water restriction",
    "water supply",
    "reservoir",
    "evacuation order",
    "access warning",
    "return home",
    "road washed out",
    "bridge closure",
    "mudslide",
    "landslide",
    "debris flow",
    "rockslide",
    "public works",
    "evacuation center",
)

NOISE_PATH_MARKERS = (
    "/tag/",
    "/tags/",
    "/category/",
    "/categories/",
    "/archive/",
    "/archives/",
    "/search",
    "/?s=",
    "?q=",
)
NOISE_TEXT_MARKERS = (
    "search results",
    "tag archive",
    "category archive",
    "articles tagged",
    "hazard mitigation plan",
    "mitigation action plan",
    "climate resilience",
    "preparedness plan",
)

GENERIC_PLANNING_DOCUMENT_PATTERNS = (
    r"\b(?:statewide|state|county|local|regional)?\s*emergency\s+(?:operations\s+)?plan\b",
    r"\b(?:all[\s-]?hazards?\s+)?preparedness\s+(?:plan|guide|framework)\b",
    r"\bhazard\s+mitigation\s+(?:action\s+)?plan\b",
    r"\bmitigation\s+action\s+plan\b",
    r"\bcontinuity\s+of\s+operations\s+plan\b",
    r"\bclimate\s+(?:adaptation|resilience)\s+(?:plan|guide|framework)\b",
)
SEO_SPAM_MARKERS = (
    "coupon",
    "casino",
    "slot",
    "betting",
    "seo",
    "best places to live",
    "homes for sale",
)
STATE_NAMES = (
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "florida",
    "georgia",
    "illinois",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "mississippi",
    "missouri",
    "new mexico",
    "oklahoma",
    "tennessee",
)

STATE_ABBREVIATIONS = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "florida": "FL",
    "georgia": "GA",
    "illinois": "IL",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "mississippi": "MS",
    "missouri": "MO",
    "new mexico": "NM",
    "oklahoma": "OK",
    "tennessee": "TN",
    "texas": "TX",
}

FIPS_STATE_PREFIXES = {
    "06": ("California", "CA"),
    "48": ("Texas", "TX"),
}

@dataclass(frozen=True)
class TavilyCostController:
    enable_structured_official_first: bool = True
    enable_targeted_official_search: bool = True
    enable_tavily_fallback: bool = True
    max_open_queries_per_candidate_round1: int = 1
    max_open_queries_per_candidate_total: int = 2
    max_results_per_query: int = 3
    max_open_fetches_per_candidate_round1: int = 2
    max_open_fetches_per_candidate_total: int = 4
    max_open_fetches_per_candidate_round2: int = 3
    max_queries_per_lane_per_candidate: int = 1
    max_fetches_per_lane_per_candidate: int = 2
    max_open_fetches_total: int = 80
    enable_general_web_lane: bool = False
    enable_advanced_search_by_default: bool = False
    cache_search_results: bool = True
    cache_url_fetches: bool = True
    deduplicate_queries_before_search: bool = True
    deduplicate_urls_before_fetch: bool = True
    per_domain_global_cap: int = 8
    per_candidate_pdf_cap: int = 1
    global_pdf_cap: int = 8
    general_web_fetch_cap: int = 4
    round3_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_public_id(sample: dict[str, Any]) -> str:
    return (
        str(sample.get("raw_candidate_id") or "")
        or str(sample.get("dter_event_id") or "")
        or str(sample.get("phase2_3_sample_id") or "")
    )


def sample_id(sample: dict[str, Any]) -> str:
    return str(sample.get("phase2_3_sample_id") or candidate_public_id(sample))


def normalize_county(county: str) -> str:
    return re.sub(r"\s+county$", "", str(county or ""), flags=re.I).strip()


def _word_text(value: str) -> str:
    """Normalize visible provider metadata without reading provider raw_content."""

    return " ".join(re.findall(r"[a-z0-9]+", urllib.parse.unquote(str(value or "")).lower()))


def _county_name_variants(county: str) -> tuple[str, ...]:
    base = _word_text(normalize_county(county))
    if not base:
        return ()
    return (f"{base} county", f"county of {base}")


def target_jurisdiction_binding(
    *,
    sample: dict[str, Any],
    title: str,
    snippet: str,
    url: str,
    structured_jurisdiction: str = "",
    structured_jurisdiction_reliable: bool = False,
) -> dict[str, Any]:
    """Return a conservative, candidate-relative county binding decision.

    Bare county-name substrings are intentionally insufficient: ``Sierra``
    must not match ``Sierra Nevada``, just as ``Lake`` and ``Orange`` must not
    match ordinary words.  Search-provider ``raw_content`` is not an input.
    """

    county = normalize_county(str(sample.get("county") or ""))
    variants = _county_name_variants(county)
    visible_text = _word_text(f"{title} {snippet}")
    for variant in variants:
        if re.search(rf"\b{re.escape(variant)}\b", visible_text):
            return {
                "bound": True,
                "reason": "canonical_county_phrase_in_title_or_snippet",
                "canonical_county": county,
            }

    parsed = urllib.parse.urlparse(str(url or ""))
    host = (parsed.hostname or "").lower().rstrip(".")
    path = urllib.parse.unquote(parsed.path or "").lower()
    compact_county = re.sub(r"[^a-z0-9]", "", county.lower())
    compact_url = re.sub(r"[^a-z0-9]", "", f"{host}/{path}")
    if compact_county and any(
        marker in compact_url
        for marker in (f"{compact_county}county", f"countyof{compact_county}")
    ):
        return {
            "bound": True,
            "reason": "explicit_county_identity_in_url",
            "canonical_county": county,
        }

    if structured_jurisdiction_reliable:
        structured_text = _word_text(structured_jurisdiction)
        if structured_text in variants:
            return {
                "bound": True,
                "reason": "reliable_structured_county_field",
                "canonical_county": county,
            }

    return {
        "bound": False,
        "reason": "target_county_not_bound",
        "canonical_county": county,
    }


def is_generic_planning_document(*, title: str, snippet: str, url: str) -> bool:
    text = _word_text(f"{title} {snippet} {url}")
    return any(re.search(pattern, text) for pattern in GENERIC_PLANNING_DOCUMENT_PATTERNS)


def _target_event_time_in_text(
    text: str, *, target_year: str, target_month_year: str
) -> bool:
    lowered = _word_text(text)
    month_year = _word_text(target_month_year)
    if month_year:
        month, _, year = month_year.partition(" ")
        if month and year and re.search(
            rf"\b{re.escape(month)}(?:\s+\d{{1,2}})?\s+{re.escape(year)}\b",
            lowered,
        ):
            return True
    return bool(target_year and re.search(rf"\b{re.escape(target_year)}[-/]\d{{1,2}}\b", str(text or "").lower()))


def _fips_prefix(sample: dict[str, Any]) -> str:
    fips = str(sample.get("FIPS") or sample.get("county_fips") or "").strip()
    digits = re.sub(r"\D", "", fips)
    return digits[:2]


def _state_name_from_sample(sample: dict[str, Any]) -> str:
    raw_state = str(sample.get("state") or sample.get("state_name") or "").strip()
    if raw_state:
        lowered = raw_state.lower()
        if len(raw_state) == 2:
            for name, abbrev in STATE_ABBREVIATIONS.items():
                if raw_state.upper() == abbrev:
                    return name.title()
        return lowered.title()
    fips_state = FIPS_STATE_PREFIXES.get(_fips_prefix(sample))
    return fips_state[0] if fips_state else ""


def _state_abbrev_from_sample(sample: dict[str, Any]) -> str:
    raw_abbrev = str(sample.get("state_abbrev") or sample.get("state_abbreviation") or "").strip()
    if raw_abbrev:
        return raw_abbrev.upper()
    state_name = _state_name_from_sample(sample).lower()
    if state_name in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[state_name]
    fips_state = FIPS_STATE_PREFIXES.get(_fips_prefix(sample))
    return fips_state[1] if fips_state else ""


def _state_aliases_from_sample(sample: dict[str, Any]) -> set[str]:
    aliases = {value.lower() for value in (_state_name_from_sample(sample), _state_abbrev_from_sample(sample)) if value}
    return aliases


def _is_california_sample(sample: dict[str, Any]) -> bool:
    return "california" in _state_aliases_from_sample(sample) or "ca" in _state_aliases_from_sample(sample) or _fips_prefix(sample) == "06"


def _is_california_gap(gap_state: dict[str, Any]) -> bool:
    state = str(gap_state.get("state") or "").strip().lower()
    county_fips = str(gap_state.get("FIPS") or gap_state.get("county_fips") or "").strip()
    return state in {"california", "ca"} or county_fips.startswith("06")


def _county_state_phrase(sample: dict[str, Any]) -> str:
    county = normalize_county(str(sample.get("county") or ""))
    state_name = _state_name_from_sample(sample)
    if county and state_name:
        return f"{county} County {state_name}"
    if county:
        return f"{county} County"
    return state_name


def normalize_query(query: str) -> str:
    lowered = query.lower()
    lowered = re.sub(r"[\"'`]", "", lowered)
    lowered = re.sub(r"\bcounty\b", "", lowered)
    lowered = re.sub(r"\btx\b", "texas", lowered)
    lowered = re.sub(r"\bca\b", "california", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+$", "", parsed.path or "/")
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [
        (k, v)
        for k, v in query_pairs
        if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}
    ]
    query = urllib.parse.urlencode(sorted(filtered))
    return urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))


def domain_from_url(url: str) -> str:
    return urllib.parse.urlparse(str(url or "")).netloc.lower().removeprefix("www.")


def is_pdf_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(str(url or ""))
    target = f"{parsed.path}?{parsed.query}".lower()
    return bool(re.search(r"\.pdf(?:$|[?&#])", target))


def _sample_month_year(sample: dict[str, Any]) -> tuple[str, str]:
    value = str(sample.get("rain_start") or "")
    match = re.match(r"(\d{4})-(\d{2})", value)
    if not match:
        year = value[:4] if value[:4].isdigit() else ""
        return year, year
    year, month = match.groups()
    month_name = [
        "",
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
    ][int(month)]
    return f"{month_name} {year}", year


def compute_evidence_gap_state(
    sample: dict[str, Any],
    has_drought_evidence: bool,
    has_rain_flood_evidence: bool,
    has_impact_evidence: bool,
) -> dict[str, Any]:
    missing: list[str] = []
    if not has_drought_evidence:
        missing.append("drought")
    if not has_rain_flood_evidence:
        missing.append("rain_flood")
    if not has_impact_evidence:
        missing.append("impact")
    if not missing:
        reason = "all_components_supported_after_reference_retrieval"
    else:
        reason_parts = []
        if "drought" not in missing:
            reason_parts.append("drought_supported_by_reference_skip_generic_drought_open_search")
        if "rain_flood" in missing:
            reason_parts.append("rain_flood_evidence_missing")
        if "impact" in missing:
            reason_parts.append("impact_evidence_missing")
        if "drought" in missing:
            reason_parts.append("drought_evidence_missing")
        reason = ";".join(reason_parts)
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "candidate_id": candidate_public_id(sample),
        "phase2_3_sample_id": sample_id(sample),
        "county": sample.get("county", ""),
        "state": sample.get("state", ""),
        "FIPS": sample.get("FIPS", sample.get("county_fips", "")),
        "event_date_or_event_window": f"{sample.get('drought_start', '')} to {sample.get('rain_end', '')}".strip(),
        "drought_window": f"{sample.get('drought_start', '')} to {sample.get('drought_end', '')}".strip(),
        "rain_event_window": f"{sample.get('rain_start', '')} to {sample.get('rain_end', '')}".strip(),
        "has_drought_evidence": has_drought_evidence,
        "has_rain_flood_evidence": has_rain_flood_evidence,
        "has_impact_evidence": has_impact_evidence,
        "missing_components": missing,
        "open_search_needed": bool(missing),
        "open_search_reason": reason,
    }


def plan_search_lanes(gap_state: dict[str, Any], controls: TavilyCostController | None = None) -> list[dict[str, Any]]:
    controls = controls or TavilyCostController()
    missing = set(gap_state.get("missing_components") or [])
    if not gap_state.get("open_search_needed"):
        return []
    california_gap = _is_california_gap(gap_state)
    lanes: list[tuple[str, str]] = []
    if "rain_flood" in missing:
        lanes.extend(
            [
                ("local_government_emergency", "rain_flood"),
                ("transportation_roads", "rain_flood"),
                ("local_news", "rain_flood"),
                ("water_river_reservoir", "rain_flood"),
            ]
        )
        if california_gap:
            lanes.extend(
                [
                    ("county_city_official", "rain_flood"),
                    ("sheriff_oes_emergency", "rain_flood"),
                    ("public_works_flood_control", "rain_flood"),
                    ("caloes_state_emergency", "rain_flood"),
                    ("caltrans_transportation", "rain_flood"),
                    ("california_dwr_water", "rain_flood"),
                    ("regional_news", "rain_flood"),
                ]
            )
    if "impact" in missing:
        lanes.extend(
            [
                ("transportation_roads", "impact"),
                ("local_government_emergency", "impact"),
                ("local_news", "impact"),
                ("water_river_reservoir", "impact"),
                ("agriculture_drought_impact", "impact"),
            ]
        )
        if california_gap:
            lanes.extend(
                [
                    ("county_city_official", "impact"),
                    ("sheriff_oes_emergency", "impact"),
                    ("public_works_flood_control", "impact"),
                    ("caloes_state_emergency", "impact"),
                    ("caltrans_transportation", "impact"),
                    ("california_dwr_water", "impact"),
                    ("agriculture_official", "impact"),
                    ("regional_news", "impact"),
                ]
            )
    if "drought" in missing:
        lanes.extend(
            [
                ("agriculture_drought_impact", "drought"),
                ("water_river_reservoir", "drought"),
                ("local_government_emergency", "drought"),
            ]
        )
        if california_gap:
            lanes.extend(
                [
                    ("agriculture_official", "drought"),
                    ("county_city_official", "drought"),
                    ("california_dwr_water", "drought"),
                ]
            )
    if controls.enable_general_web_lane:
        for component in ("rain_flood", "impact", "drought"):
            if component in missing:
                lanes.append(("general_web", component))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for lane, component in sorted(lanes, key=lambda item: (LANE_PRIORITY[item[0]], item[1])):
        key = (lane, component)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "lane": lane,
                "missing_component": component,
                "priority": LANE_PRIORITY[lane],
                "enabled": lane != "general_web" or controls.enable_general_web_lane,
            }
        )
    return deduped


def _query_templates(
    *,
    sample: dict[str, Any],
    lane: str,
    missing_component: str,
    round_number: int,
) -> list[str]:
    county = normalize_county(str(sample.get("county") or ""))
    month_year, year = _sample_month_year(sample)
    state_name = _state_name_from_sample(sample)
    state_abbrev = _state_abbrev_from_sample(sample)
    county_state = _county_state_phrase(sample)
    county_state_short = f"{county} County {state_abbrev}".strip() if county and state_abbrev else county_state
    # Forward requests are grounded only by candidate data. Static
    # county-to-place dictionaries are intentionally absent; bounded LLM
    # expansion may independently propose a place, subject to validation.
    primary_locality = county
    secondary_locality = county
    if missing_component == "rain_flood":
        templates = {
            "local_government_emergency": [
                f'{county_state} emergency management flooding "{year}"',
                f'{county_state} storm flooding "{month_year}"',
            ],
            "county_city_official": [
                f'{county_state} county official flooding evacuation "{month_year}"',
                f'{primary_locality} {state_name} city official atmospheric river flooding "{month_year}"',
            ],
            "sheriff_oes_emergency": [
                f'{county_state} sheriff OES evacuation order flooding "{month_year}"',
                f'{county_state} emergency management access warning storm "{month_year}"',
            ],
            "transportation_roads": [
                f'{county_state} flooding "road closure" "{month_year}"',
                f'{county_state} "high water" roads "{month_year}"',
            ],
            "public_works_flood_control": [
                f'{county_state} public works flood control road closure "{month_year}"',
                f'{county_state} road washed out debris flow "{month_year}"',
            ],
            "caloes_state_emergency": [
                f'{county_state} CalOES storm flooding evacuation "{month_year}"',
                f'site:caloes.ca.gov "{county}" flood emergency "{year}"',
            ],
            "caltrans_transportation": [
                f'{county_state} Caltrans road closure storm damage "{month_year}"',
                f'site:dot.ca.gov "{county}" flooding road closure "{year}"',
            ],
            "local_news": [
                f'"{county}" {state_name} flooding "{month_year}"',
                f'"{county}" {state_name} "heavy rain" "{month_year}"',
            ],
            "regional_news": [
                f'"{primary_locality}" "{month_year}" atmospheric river flooding impact',
                f'"{county}" {state_abbrev} storm evacuation road closure "{month_year}"',
            ],
            "water_river_reservoir": [
                f'{county_state} flooding river "{month_year}"',
                f'{county_state} reservoir flooding "{year}"',
            ],
            "california_dwr_water": [
                f'{county_state} California DWR river flooding "{month_year}"',
                f'site:water.ca.gov "{county}" flooding river "{year}"',
            ],
            "general_web": [
                f'{county_state} flooding "{month_year}"',
                f'{county_state} "flash flood" "{year}"',
            ],
        }
    elif missing_component == "impact":
        templates = {
            "local_government_emergency": [
                f'{county_state} emergency management flooding "{year}"',
                f'{county_state} storm damage "{month_year}"',
            ],
            "county_city_official": [
                f'{county_state} county official evacuation return home flooding "{month_year}"',
                f'{primary_locality} {state_name} city official flash report atmospheric river "{month_year}"',
            ],
            "sheriff_oes_emergency": [
                f'{county_state} sheriff evacuation order storm damage "{month_year}"',
                f'{county_state} OES emergency response evacuation center "{month_year}"',
            ],
            "transportation_roads": [
                f'{county_state} flooding "road closure" "{month_year}"',
                f'{county_state} "high water" roads "{month_year}"',
            ],
            "public_works_flood_control": [
                f'{county_state} public works road closure mudslide debris flow "{month_year}"',
                f'{county_state} bridge closure road washed out storm damage "{month_year}"',
            ],
            "caloes_state_emergency": [
                f'{county_state} CalOES evacuation storm damage "{month_year}"',
                f'site:caloes.ca.gov "{county}" storm damage evacuation "{year}"',
            ],
            "caltrans_transportation": [
                f'{county_state} Caltrans road closure mudslide "{month_year}"',
                f'site:dot.ca.gov "{county}" storm damage road closure "{year}"',
            ],
            "local_news": [
                f'"{county}" {state_name} storm damage "{month_year}"',
                f'"{county}" {state_name} flooding roads "{month_year}"',
            ],
            "regional_news": [
                f'"{primary_locality}" atmospheric river emergency response "{month_year}"',
                f'"{secondary_locality}" flooding road closure evacuation "{month_year}"',
            ],
            "water_river_reservoir": [
                f'{county_state} water rescue flooding "{year}"',
                f'{county_state} reservoir river flooding damage "{year}"',
            ],
            "california_dwr_water": [
                f'{county_state} California DWR flood control damage "{month_year}"',
                f'site:water.ca.gov "{county}" flood control storm damage "{year}"',
            ],
            "agriculture_drought_impact": [
                f'{county_state} drought crop loss "{year}"',
                f'{county_state} drought livestock "{year}"',
                f'{county_state} water restriction drought "{year}"',
            ],
            "agriculture_official": [
                f'{county_state} agricultural commissioner storm damage crop loss "{year}"',
                f'{county_state} RMA crop loss drought flood "{year}"',
            ],
            "general_web": [
                f'{county_state} flooding damage roads "{month_year}"',
                f'{county_state} storm impacts "{year}"',
            ],
        }
    else:
        drought_year = str(sample.get("drought_end_month_end") or sample.get("drought_end") or "")[:4] or year
        templates = {
            "agriculture_drought_impact": [
                f'{county_state} drought crop loss "{drought_year}"',
                f'{county_state} drought livestock "{drought_year}"',
            ],
            "agriculture_official": [
                f'{county_state} agricultural commissioner drought crop loss "{drought_year}"',
                f'{county_state} RMA drought crop loss "{drought_year}"',
            ],
            "water_river_reservoir": [
                f'{county_state} water restriction drought "{drought_year}"',
                f'{county_state} reservoir drought "{drought_year}"',
            ],
            "california_dwr_water": [
                f'{county_state} California DWR drought water restriction "{drought_year}"',
                f'site:water.ca.gov "{county}" drought water supply "{drought_year}"',
            ],
            "local_government_emergency": [
                f'{county_state} drought declaration "{drought_year}"',
                f'{county_state} burn ban drought "{drought_year}"',
            ],
            "county_city_official": [
                f'{county_state} county official drought water restriction "{drought_year}"',
                f'{county_state_short} drought emergency proclamation "{drought_year}"',
            ],
            "general_web": [
                f'{county_state} drought "{drought_year}"',
            ],
        }
    values = templates.get(lane, [])
    if round_number <= 1:
        return values[:1]
    return values[1:] or values[:1]


def _near_duplicate_key(query: str) -> str:
    normalized = normalize_query(query)
    tokens = [
        token
        for token in normalized.split()
        if token
        and token
        not in {
            "texas",
            "california",
            "county",
            "storm",
            "flooding",
            "flood",
            "heavy",
            "rain",
            "rainfall",
            "roads",
            "road",
        }
    ]
    return " ".join(tokens)


def build_lane_queries(
    *,
    sample: dict[str, Any],
    gap_state: dict[str, Any],
    controls: TavilyCostController,
    round_number: int,
    existing_query_texts: Iterable[str] = (),
    existing_query_rows: Iterable[dict[str, Any]] = (),
    search_backend: str = "Tavily",
) -> list[dict[str, Any]]:
    if not gap_state.get("open_search_needed"):
        return []
    existing_norm = {normalize_query(query) for query in existing_query_texts}
    existing_near = {_near_duplicate_key(query) for query in existing_query_texts}
    per_candidate_limit = (
        controls.max_open_queries_per_candidate_round1
        if round_number == 1
        else controls.max_open_queries_per_candidate_total
    )
    existing_count = len(existing_norm)
    remaining = max(0, per_candidate_limit - existing_count)
    if remaining <= 0:
        return []
    lane_counts: Counter[str] = Counter(
        str(row.get("lane") or "")
        for row in existing_query_rows
        if row.get("lane")
    )
    rows: list[dict[str, Any]] = []
    for plan in plan_search_lanes(gap_state, controls):
        lane = plan["lane"]
        if not plan["enabled"]:
            continue
        if lane_counts[lane] >= controls.max_queries_per_lane_per_candidate:
            continue
        for query_text in _query_templates(
            sample=sample,
            lane=lane,
            missing_component=plan["missing_component"],
            round_number=round_number,
        ):
            normalized = normalize_query(query_text)
            near_key = _near_duplicate_key(query_text)
            if controls.deduplicate_queries_before_search and (
                normalized in existing_norm or near_key in existing_near
            ):
                continue
            row = {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "candidate_id": candidate_public_id(sample),
                "phase2_3_sample_id": sample_id(sample),
                "lane": lane,
                "missing_component": plan["missing_component"],
                "query_text": query_text,
                "round": round_number,
                "query_source": "template",
                "search_backend": search_backend,
            }
            rows.append(row)
            existing_norm.add(normalized)
            existing_near.add(near_key)
            lane_counts[lane] += 1
            break
        if len(rows) >= remaining:
            break
    return rows[:remaining]


def classify_source_type(domain: str, url: str, title: str = "") -> str:
    text = f"{domain} {url} {title}".lower()
    if "dot.ca.gov" in domain or "roads.dot.ca.gov" in domain or "caltrans" in text:
        return "transportation_official"
    if "water.ca.gov" in domain or "dwr" in text or "floodcontrol" in domain or "flood-control" in text:
        return "water_authority"
    if "caloes.ca.gov" in domain or "cal oes" in text or "caloes" in text:
        return "local_or_state_government"
    if domain.endswith(".ca.gov") or domain.endswith("ca.gov"):
        if "transportation" in text or "road" in text:
            return "transportation_official"
        if "water" in text or "river" in text or "flood" in text:
            return "water_authority"
        return "local_or_state_government"
    if any(
        token in domain
        for token in (
            "countyofmonterey.gov",
            "sanjoseca.gov",
            "maderacounty.com",
            "maderacounty.gov",
            "lacounty.gov",
            "modoccounty.us",
        )
    ):
        return "local_or_state_government"
    if (
        domain.endswith(".gov")
        and any(token in text for token in ("california", " ca ", "county", "city", "sheriff", "public works", "oes"))
    ):
        if "transportation" in text or "road" in text:
            return "transportation_official"
        if "water" in text or "river" in text or "flood control" in text:
            return "water_authority"
        return "local_or_state_government"
    if domain.endswith(".gov") or ".tx.us" in domain:
        if "txdot" in domain or "transportation" in text or "road" in text:
            return "transportation_official"
        if "water" in text or "riverauthority" in domain:
            return "water_authority"
        if "weather.gov" in domain or "noaa.gov" in domain:
            return "nws_noaa"
        return "local_or_state_government"
    if (
        domain.endswith(".tx.us")
        or domain.endswith("tx.gov")
        or re.search(r"\bco\.", domain)
        or "angelinacounty.net" in domain
        or "nuecesco.com" in domain
        or ("revize.com" in domain and ("county" in text or "lufkin" in text or "tx" in text))
    ):
        if "txdot" in domain or "transportation" in text or "road" in text:
            return "transportation_official"
        if "water" in text or "riverauthority" in domain:
            return "water_authority"
        return "local_or_state_government"
    if any(token in domain for token in ("tamu.edu", "agrilife", "usda", "fsa", "ucanr.edu", "cdfa.ca.gov")):
        return "agriculture_extension_or_federal"
    if any(
        token in domain
        for token in (
            "news",
            "times",
            "tribune",
            "chron",
            "kltv",
            "kxan",
            "abc",
            "cbs",
            "nbc",
            "mercurynews",
            "lookout",
            "ksbw",
            "ktla",
            "latimes",
        )
    ):
        return "credible_local_news"
    return "other"


def _has_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _state_alias_in_text(text: str, aliases: Iterable[str]) -> bool:
    lowered = text.lower()
    for alias in aliases:
        alias = alias.lower()
        if not alias:
            continue
        if len(alias) == 2:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return True
        elif alias in lowered:
            return True
    return False


def _explicit_years(text: str) -> set[str]:
    return set(re.findall(r"\b(20[0-3]\d)\b", text))


def hard_reject_reason(
    *,
    title: str,
    snippet: str,
    url: str,
    candidate_county: str,
    state: str,
    target_year: str,
    target_month_year: str,
    missing_component: str,
) -> str:
    domain = domain_from_url(url)
    text = f"{title} {snippet} {url}".lower()
    county = normalize_county(candidate_county).lower()
    state_text = str(state or "").lower()
    state_aliases = {state_text}
    if state_text in STATE_ABBREVIATIONS:
        state_aliases.add(STATE_ABBREVIATIONS[state_text].lower())
    if len(state_text) == 2:
        for name, abbrev in STATE_ABBREVIATIONS.items():
            if state_text.upper() == abbrev:
                state_aliases.add(name)
    state_aliases.discard("")
    has_location = bool(county and county in text) or _state_alias_in_text(text, state_aliases)
    has_time = bool(target_year and target_year in text) or bool(target_month_year and target_month_year.lower() in text)
    has_hazard_or_impact = _has_any(text, FLOOD_RAIN_TERMS + DROUGHT_TERMS + IMPACT_TERMS)
    source_type = classify_source_type(domain, url, title)
    trusted_source = source_type in {
        "local_or_state_government",
        "transportation_official",
        "water_authority",
        "nws_noaa",
        "agriculture_extension_or_federal",
        "credible_local_news",
    }
    event_specific_signal = (has_location and has_time) or (has_location and has_hazard_or_impact) or (has_time and has_hazard_or_impact)
    years = _explicit_years(text)
    if years and target_year and target_year not in years:
        return "wrong_year_explicit_target_year_absent"
    if county and any(state_name in text for state_name in STATE_NAMES) and not _state_alias_in_text(text, state_aliases):
        return "wrong_location_explicit"
    if (
        any(marker in text for marker in NOISE_PATH_MARKERS)
        or any(marker in text for marker in NOISE_TEXT_MARKERS[:3])
    ) and not event_specific_signal:
        return "generic_archive_list_tag_or_search_page"
    if ("hazard mitigation plan" in text or "mitigation action plan" in text) and not event_specific_signal:
        return "generic_hazard_mitigation_plan"
    if is_pdf_url(url) and not (event_specific_signal or trusted_source):
        return "old_or_weak_background_pdf"
    if not has_location and not has_time and not has_hazard_or_impact:
        return "no_target_location_or_time_signal"
    if missing_component in {"rain_flood", "impact"} and _has_any(text, ("drought monitor", "climate outlook", "drought background")) and not _has_any(text, FLOOD_RAIN_TERMS + IMPACT_TERMS):
        return "generic_drought_or_climate_background_for_non_drought_gap"
    if (
        domain
        and any(state_name in text for state_name in STATE_NAMES)
        and not _state_alias_in_text(text, state_aliases)
        and county
        and county not in text
        and not trusted_source
    ):
        return "unrelated_to_target_region"
    if any(marker in text for marker in SEO_SPAM_MARKERS):
        return "seo_scraper_or_spam_like_page"
    return ""


def score_prefetch_result(
    *,
    sample: dict[str, Any],
    query_id: str,
    lane: str,
    missing_component: str,
    title: str,
    snippet: str,
    url: str,
    prefetch_policy_version: str = LEGACY_PREFETCH_POLICY_VERSION,
) -> dict[str, Any]:
    domain = domain_from_url(url)
    source_type = classify_source_type(domain, url, title)
    month_year, year = _sample_month_year(sample)
    text = f"{title} {snippet} {url}".lower()
    county = normalize_county(str(sample.get("county") or ""))
    county_lower = county.lower()
    state_aliases = _state_aliases_from_sample(sample)
    jurisdiction = target_jurisdiction_binding(
        sample=sample,
        title=title,
        snippet=snippet,
        url=url,
    )
    jurisdiction_gate_enabled = (
        prefetch_policy_version == TARGET_JURISDICTION_PREFETCH_POLICY_VERSION
    )
    if prefetch_policy_version not in {
        LEGACY_PREFETCH_POLICY_VERSION,
        TARGET_JURISDICTION_PREFETCH_POLICY_VERSION,
    }:
        raise ValueError(f"unsupported_prefetch_policy_version:{prefetch_policy_version}")
    hard_reason = hard_reject_reason(
        title=title,
        snippet=snippet,
        url=url,
        candidate_county=str(sample.get("county") or ""),
        state=_state_name_from_sample(sample),
        target_year=year,
        target_month_year=month_year,
        missing_component=missing_component,
    )
    generic_planning_document = is_generic_planning_document(
        title=title,
        snippet=snippet,
        url=url,
    )
    candidate_event_specific = bool(
        jurisdiction["bound"]
        and _target_event_time_in_text(
            f"{title} {snippet} {url}",
            target_year=year,
            target_month_year=month_year,
        )
        and _has_any(
            f"{title} {snippet} {url}",
            FLOOD_RAIN_TERMS + DROUGHT_TERMS + IMPACT_TERMS,
        )
    )
    if (
        jurisdiction_gate_enabled
        and generic_planning_document
        and not candidate_event_specific
    ):
        hard_reason = "generic_planning_document_without_candidate_event_binding"
    elif jurisdiction_gate_enabled and not jurisdiction["bound"]:
        hard_reason = "target_jurisdiction_binding_missing"
    components: dict[str, int] = {}
    positive: list[str] = []
    negative: list[str] = []

    trusted_source_types = {
        "local_or_state_government",
        "transportation_official",
        "water_authority",
        "nws_noaa",
        "agriculture_extension_or_federal",
        "credible_local_news",
    }
    trusted_source = source_type in trusted_source_types

    county_match = bool(jurisdiction["bound"]) if jurisdiction_gate_enabled else bool(
        county_lower and county_lower in text
    )
    if county_match:
        components["county_signal"] = 30
        positive.append(
            "target_county_bound_by_jurisdiction_predicate"
            if jurisdiction_gate_enabled
            else "target_county_in_title_snippet_or_url"
        )
    else:
        components["county_signal"] = -10 if trusted_source else -18
        negative.append("missing_target_county_signal")
    if _state_alias_in_text(text, state_aliases):
        components["state_signal"] = 8
        positive.append("target_state_signal")
    else:
        components["state_signal"] = -5
    if year and year in text:
        components["year_signal"] = 25
        positive.append("target_year_signal")
    elif month_year and month_year.lower() in text:
        components["year_signal"] = 25
        positive.append("target_month_year_signal")
    else:
        components["year_signal"] = -8 if trusted_source else -16
        negative.append("missing_target_time_signal")
    if missing_component == "rain_flood":
        terms = FLOOD_RAIN_TERMS
    elif missing_component == "impact":
        terms = IMPACT_TERMS + FLOOD_RAIN_TERMS
    else:
        terms = DROUGHT_TERMS
    hits = [term for term in terms if term in text]
    components["hazard_or_impact_terms"] = min(30, 8 * len(hits))
    if hits:
        positive.append("component_terms:" + ",".join(hits[:5]))
    else:
        components["hazard_or_impact_terms"] = -15
        negative.append("missing_component_terms")
    source_bonus = {
        "local_or_state_government": 22,
        "transportation_official": 25,
        "water_authority": 22,
        "nws_noaa": 22,
        "agriculture_extension_or_federal": 20,
        "credible_local_news": 18,
    }.get(source_type, 0)
    components["source_type"] = source_bonus
    if source_bonus:
        positive.append(f"preferred_source_type:{source_type}")
    if lane == "transportation_roads" and _has_any(text, ("road", "high water", "closure")):
        components["lane_alignment"] = 12
        positive.append("lane_term_alignment")
    elif lane == "water_river_reservoir" and _has_any(text, ("river", "reservoir", "water")):
        components["lane_alignment"] = 12
        positive.append("lane_term_alignment")
    elif lane == "agriculture_drought_impact" and _has_any(text, ("crop", "livestock", "drought", "pasture")):
        components["lane_alignment"] = 12
        positive.append("lane_term_alignment")
    elif lane == "local_government_emergency" and _has_any(text, ("emergency", "declaration", "county", ".gov")):
        components["lane_alignment"] = 12
        positive.append("lane_term_alignment")
    elif lane == "local_news" and source_type == "credible_local_news":
        components["lane_alignment"] = 10
        positive.append("lane_source_alignment")
    else:
        components["lane_alignment"] = 0
    has_event_specific_signal = bool(
        (county_match and (year in text or bool(hits)))
        or (trusted_source and (year in text or bool(hits)))
    )
    if any(marker in text for marker in NOISE_PATH_MARKERS) or any(marker in text for marker in NOISE_TEXT_MARKERS):
        components["noise_penalty"] = -12 if has_event_specific_signal else -28
        negative.append("archive_list_or_generic_planning_signal")
    if is_pdf_url(url):
        pdf_penalty = -8
        if county_match and year in text and hits:
            pdf_penalty = 0
        elif has_event_specific_signal:
            pdf_penalty = -3
        components["pdf_penalty"] = pdf_penalty
        negative.append("pdf_result")
    if missing_component in {"rain_flood", "impact"} and _has_any(text, ("drought monitor", "climate outlook")) and not _has_any(text, FLOOD_RAIN_TERMS + IMPACT_TERMS):
        components["generic_drought_penalty"] = -25
        negative.append("generic_drought_background_for_non_drought_gap")
    if any(marker in text for marker in SEO_SPAM_MARKERS):
        components["spam_penalty"] = -60
        negative.append("seo_scraper_or_spam_like")
    if trusted_source and (county_lower in text or _state_alias_in_text(text, state_aliases)) and hits:
        components["trusted_ambiguous_source_floor"] = max(
            0,
            45 - sum(components.values()),
        )
        if components["trusted_ambiguous_source_floor"]:
            positive.append("trusted_source_kept_fetchable_when_ambiguous")
    score = sum(components.values())
    if hard_reason:
        decision = "skip_prefetch"
        score = min(score, 0)
        negative.append(hard_reason)
    elif score >= 70 and not (jurisdiction_gate_enabled and not county_match):
        decision = "fetch_round1"
    elif score >= 40:
        decision = "fetch_round2_candidate"
    elif score >= 20:
        decision = "hold_low_priority"
    else:
        decision = "skip_prefetch"
    result = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "candidate_id": candidate_public_id(sample),
        "phase2_3_sample_id": sample_id(sample),
        "query_id": query_id,
        "lane": lane,
        "url": url,
        "title": title,
        "snippet": snippet,
        "normalized_url": normalize_url(url),
        "domain": domain,
        "source_type": source_type,
        "missing_component": missing_component,
        "score": score,
        "decision": decision,
        "score_components": components,
        "positive_reasons": positive,
        "negative_reasons": negative,
        "hard_reject_reason": hard_reason,
        "document_type": "pdf" if is_pdf_url(url) else "html",
    }
    if jurisdiction_gate_enabled:
        result.update(
            {
                "prefetch_policy_version": prefetch_policy_version,
                "target_jurisdiction_bound": bool(jurisdiction["bound"]),
                "target_jurisdiction_binding_reason": jurisdiction["reason"],
                "generic_planning_document": generic_planning_document,
                "candidate_event_specific": candidate_event_specific,
            }
        )
    return result


def schedule_fetch_decisions(
    *,
    sample: dict[str, Any],
    scorecards: list[dict[str, Any]],
    controls: TavilyCostController,
    round_number: int,
    fetched_urls_global: set[str],
    domain_counts: Counter[str],
    candidate_pdf_counts: Counter[str],
    global_counts: Counter[str],
    candidate_fetch_count: int = 0,
) -> list[dict[str, Any]]:
    sid = sample_id(sample)
    candidates = [
        row
        for row in scorecards
        if row["phase2_3_sample_id"] == sid
        and row["decision"] in (["fetch_round1"] if round_number == 1 else ["fetch_round1", "fetch_round2_candidate"])
    ]
    candidates.sort(key=lambda row: (row["score"], -LANE_PRIORITY.get(row["lane"], 99)), reverse=True)
    round_cap = (
        controls.max_open_fetches_per_candidate_round1
        if round_number == 1
        else controls.max_open_fetches_per_candidate_round2
    )
    selected = 0
    lane_counts: Counter[str] = Counter()
    planned_urls = set(fetched_urls_global)
    planned_domain_counts = Counter(domain_counts)
    planned_candidate_pdf_counts = Counter(candidate_pdf_counts)
    planned_global_counts = Counter(global_counts)
    decisions: list[dict[str, Any]] = []
    for row in candidates:
        url = row["normalized_url"]
        domain = row["domain"]
        lane = row["lane"]
        reason = ""
        action = "fetch"
        if selected >= round_cap:
            action = "defer"
            reason = "round_candidate_fetch_cap_reached"
        elif candidate_fetch_count + selected >= controls.max_open_fetches_per_candidate_total:
            action = "skip_budget"
            reason = "candidate_total_fetch_cap_reached"
        elif planned_global_counts["open_fetches_total"] >= controls.max_open_fetches_total:
            action = "skip_budget"
            reason = "global_open_fetch_cap_reached"
        elif lane_counts[lane] >= controls.max_fetches_per_lane_per_candidate:
            action = "skip_budget"
            reason = "candidate_lane_fetch_cap_reached"
        elif controls.deduplicate_urls_before_fetch and url in planned_urls:
            action = "skip_duplicate"
            reason = "duplicate_url_already_fetched"
        elif planned_domain_counts[domain] >= controls.per_domain_global_cap:
            action = "skip_budget"
            reason = "per_domain_global_cap_reached"
        elif is_pdf_url(url) and planned_candidate_pdf_counts[sid] >= controls.per_candidate_pdf_cap:
            action = "skip_budget"
            reason = "per_candidate_pdf_cap_reached"
        elif is_pdf_url(url) and planned_global_counts["pdf_fetches"] >= controls.global_pdf_cap:
            action = "skip_budget"
            reason = "global_pdf_cap_reached"
        elif lane == "general_web" and planned_global_counts["general_web_fetches"] >= controls.general_web_fetch_cap:
            action = "skip_budget"
            reason = "general_web_fetch_cap_reached"
        if action == "fetch":
            selected += 1
            planned_urls.add(url)
            lane_counts[lane] += 1
            planned_domain_counts[domain] += 1
            planned_global_counts["open_fetches_total"] += 1
            if is_pdf_url(url):
                planned_candidate_pdf_counts[sid] += 1
                planned_global_counts["pdf_fetches"] += 1
            if lane == "general_web":
                planned_global_counts["general_web_fetches"] += 1
        decisions.append(
            {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "candidate_id": row["candidate_id"],
                "phase2_3_sample_id": sid,
                "query_id": row["query_id"],
                "lane": lane,
                "round": round_number,
                "retrieval_tier": row.get("retrieval_tier", ""),
                "fallback_reason": row.get("fallback_reason", ""),
                "official_sources_attempted": row.get("official_sources_attempted", ""),
                "unmet_gap_before_fallback": row.get("unmet_gap_before_fallback", ""),
                "unmet_gap_before_search": row.get("unmet_gap_before_search", ""),
                "domains_or_source_families_targeted": row.get("domains_or_source_families_targeted", ""),
                "url": row["url"],
                "normalized_url": url,
                "domain": domain,
                "score": row["score"],
                "prefetch_decision": row["decision"],
                "fetch_decision": action,
                "decision_reason": reason or "within_budget",
            }
        )
    return decisions


def build_search_result_cache(rows: Iterable[dict[str, Any]], max_results: int = 3) -> dict[str, list[dict[str, Any]]]:
    cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_urls_by_query: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        query_text = str(row.get("query_text") or "")
        if not query_text:
            continue
        key = normalize_query(query_text)
        url_key = normalize_url(str(row.get("url") or row.get("source_url") or ""))
        if not url_key or url_key in seen_urls_by_query[key]:
            continue
        if len(cache[key]) >= max_results:
            continue
        seen_urls_by_query[key].add(url_key)
        cache[key].append(row)
    return dict(cache)


def build_url_body_cache(pages: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in pages:
        url = str(row.get("source_url") or "")
        body = str(row.get("body_text_or_archived_body_text") or "")
        if not url or not body:
            continue
        cache[normalize_url(url)] = row
    return cache


def initial_cost_ledger(controls: TavilyCostController) -> dict[str, Any]:
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "controls": controls.to_dict(),
        "tavily_search_calls": 0,
        "tavily_search_calls_by_lane": {},
        "unique_queries": 0,
        "cached_query_hits": 0,
        "new_tavily_calls_after_cache": 0,
        "search_results_returned": 0,
        "search_results_scored": 0,
        "search_results_skipped_before_fetch": 0,
        "open_urls_fetched": 0,
        "cached_url_hits": 0,
        "duplicate_urls_skipped": 0,
        "pdf_fetches": 0,
        "general_web_queries": 0,
        "general_web_fetches": 0,
        "open_pages_accepted": 0,
        "open_pages_rejected": 0,
        "cost_per_open_accepted_page": None,
        "cost_per_supported_candidate": None,
        "planned_queries": 0,
        "executed_queries": 0,
        "skipped_queries_due_to_stop_condition": 0,
        "skipped_queries_due_to_cache": 0,
        "skipped_queries_due_to_budget": 0,
        "skipped_queries_due_to_replay_gate": 0,
        "unique_live_tavily_queries": 0,
        "round1_queries": 0,
        "round2_queries": 0,
        "round3_queries": 0,
        "queries_by_lane": {},
        "open_fetches_by_round": {},
        "open_fetches_by_lane": {},
        "accepted_open_pages_by_round": {},
        "accepted_open_pages_by_lane": {},
        "prior_accepted_urls_preserved": 0,
        "prior_accepted_urls_lost": 0,
    }


def finalize_cost_ledger(
    ledger: dict[str, Any],
    *,
    open_supported_candidate_count: int,
) -> dict[str, Any]:
    accepted = int(ledger.get("open_pages_accepted") or 0)
    calls = int(ledger.get("tavily_search_calls") or 0)
    ledger["cost_per_open_accepted_page"] = None if accepted == 0 else calls / accepted
    ledger["cost_per_supported_candidate"] = (
        None if open_supported_candidate_count == 0 else calls / open_supported_candidate_count
    )
    live_query_norms = ledger.pop("_live_tavily_query_norms", [])
    ledger["unique_live_tavily_queries"] = len(live_query_norms)
    return ledger


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def offline_replay_diagnostics(
    *,
    samples: list[dict[str, Any]],
    prior_pages: list[dict[str, Any]],
    controls: TavilyCostController,
) -> dict[str, Any]:
    sample_by_id = {sample_id(sample): sample for sample in samples}
    open_pages = [row for row in prior_pages if row.get("search_mode") == "open_web_fallback"]
    scored: list[dict[str, Any]] = []
    for page in open_pages:
        sample = sample_by_id.get(str(page.get("phase2_3_sample_id") or ""))
        if not sample:
            continue
        accepted = bool(page.get("accepted"))
        body = str(page.get("body_text_or_archived_body_text") or "")
        title = str(page.get("source_title") or "")
        missing_component = "impact" if accepted and _has_any(body, IMPACT_TERMS) else "rain_flood"
        scorecard = score_prefetch_result(
            sample=sample,
            query_id=str(page.get("query_id") or ""),
            lane=str(page.get("source_lane") or "local_news"),
            missing_component=missing_component,
            title=title,
            snippet=body[:500],
            url=str(page.get("source_url") or ""),
        )
        scorecard["baseline_accepted"] = accepted
        scored.append(scorecard)
    accepted_scores = [float(row["score"]) for row in scored if row["baseline_accepted"]]
    rejected_scores = [float(row["score"]) for row in scored if not row["baseline_accepted"]]
    simulations: dict[str, dict[str, Any]] = {}
    for top_k in (1, 2, 3):
        retained = 0
        avoided_rejected = 0
        fetched = 0
        for sid in sorted({row["phase2_3_sample_id"] for row in scored}):
            rows = [row for row in scored if row["phase2_3_sample_id"] == sid]
            rows.sort(key=lambda row: row["score"], reverse=True)
            selected = rows[:top_k]
            fetched += len(selected)
            retained += sum(1 for row in selected if row["baseline_accepted"])
            avoided_rejected += sum(1 for row in rows[top_k:] if not row["baseline_accepted"])
        simulations[f"top_{top_k}"] = {
            "simulated_fetches": fetched,
            "accepted_open_pages_retained": retained,
            "rejected_fetches_avoided": avoided_rejected,
        }
    accepted_mean = _safe_mean(accepted_scores)
    rejected_mean = _safe_mean(rejected_scores)
    replay_passed = bool(
        accepted_scores
        and (accepted_mean or 0) >= (rejected_mean or -math.inf)
        and simulations["top_2"]["accepted_open_pages_retained"] > 0
    )
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "prior_open_pages_scored": len(scored),
        "prior_accepted_open_pages": len(accepted_scores),
        "prior_rejected_open_pages": len(rejected_scores),
        "accepted_score_mean": accepted_mean,
        "rejected_score_mean": rejected_mean,
        "accepted_scores_above_rejected_mean": (
            sum(1 for value in accepted_scores if rejected_mean is not None and value > rejected_mean)
            if rejected_mean is not None
            else 0
        ),
        "simulated_fetch_policies": simulations,
        "replay_passed": replay_passed,
        "controls": controls.to_dict(),
    }


def prior_accepted_urls_from_baseline(baseline_summary: dict[str, Any]) -> list[str]:
    baseline_open = baseline_summary.get("open_retrieval_baseline", {})
    urls = baseline_open.get("accepted_open_urls", []) if isinstance(baseline_open, dict) else []
    return [str(url) for url in urls if str(url or "").strip()]


def _rows_by_normalized_url(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        url = str(row.get("normalized_url") or row.get("url") or row.get("source_url") or "")
        if not url:
            continue
        grouped[normalize_url(url)].append(row)
    return dict(grouped)


def _best_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row.get("score") or 0), reverse=True)[0]


def _prior_url_audit_action(url: str, loss_cause: str, scorecard: dict[str, Any] | None) -> tuple[str, str]:
    normalized = normalize_url(url)
    if "nps.gov/yell/" in normalized:
        return (
            "intentionally_dropped",
            "The retained body snapshot is a Yellowstone National Park page outside Texas and outside the frozen Texas candidate locations.",
        )
    if "nuecesco.com/" in normalized:
        return (
            "intentionally_dropped",
            "The URL is a valid Nueces County local-government flood page, but the frozen 37-candidate manifest contains no Nueces County candidate.",
        )
    if "weather.gov/lub/" in normalized and not scorecard:
        return (
            "intentionally_dropped",
            "The page is a credible NWS regional event page, but the retained page has no Phase 2.3.1 candidate/query association and does not provide a target-county match for the frozen 37 sample.",
        )
    if "drought.gov/news/" in normalized and not scorecard:
        return (
            "intentionally_dropped",
            "The page is a credible statewide drought-to-deluge article, but it lacks a retained Phase 2.3.1 candidate/query association and is not county-specific enough for strict body validation.",
        )
    if loss_cause == "hard_reject":
        return (
            "restored",
            "The old accepted URL had enough county/date/hazard or impact signal to be scored and validated by body text instead of prefetch hard-rejected.",
        )
    if scorecard and scorecard.get("decision") in {"fetch_round1", "fetch_round2_candidate"}:
        return "restored", "The calibrated scorer keeps this prior positive fetchable."
    if loss_cause in {"score_threshold", "query_omission"}:
        return "downranked", "The URL should remain a lower-priority replay positive unless body validation fails."
    return "downranked", "The URL remains a calibration positive but is not automatically accepted."


def build_prior_accepted_open_url_audit(
    *,
    prior_accepted_urls: list[str],
    phase231_search_results: list[dict[str, Any]],
    phase231_scorecards: list[dict[str, Any]],
    phase231_fetch_decisions: list[dict[str, Any]],
    phase231_pages: list[dict[str, Any]],
    phase231_claims: list[dict[str, Any]],
    phase231_impacts: list[dict[str, Any]],
    phase231_rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    search_by_url = _rows_by_normalized_url(phase231_search_results)
    score_by_url = _rows_by_normalized_url(phase231_scorecards)
    fetch_by_url = _rows_by_normalized_url(phase231_fetch_decisions)
    pages_by_url = _rows_by_normalized_url(phase231_pages)
    claims_by_url = _rows_by_normalized_url(phase231_claims)
    impacts_by_url = _rows_by_normalized_url(phase231_impacts)
    rejected_by_url = _rows_by_normalized_url(phase231_rejected)
    audit_rows: list[dict[str, Any]] = []
    for url in prior_accepted_urls:
        normalized = normalize_url(url)
        scorecard = _best_scorecard(score_by_url.get(normalized, []))
        fetch_rows = fetch_by_url.get(normalized, [])
        page_rows = pages_by_url.get(normalized, [])
        search_rows = search_by_url.get(normalized, [])
        rejected_rows = rejected_by_url.get(normalized, [])
        claim_rows = claims_by_url.get(normalized, [])
        impact_rows = impacts_by_url.get(normalized, [])
        fetched = bool(page_rows)
        fetched_decision = next((row for row in fetch_rows if row.get("fetch_decision") == "fetch"), fetch_rows[0] if fetch_rows else None)
        candidate_id = (
            (scorecard or {}).get("candidate_id")
            or (fetched_decision or {}).get("candidate_id")
            or (page_rows[0].get("raw_candidate_id") if page_rows else "")
            or (search_rows[0].get("candidate_id") if search_rows else "")
            or ""
        )
        phase231_decision = (scorecard or {}).get("decision") or (fetched_decision or {}).get("prefetch_decision") or "not_scored"
        hard_reason = str((scorecard or {}).get("hard_reject_reason") or "")
        fetch_reason = str((fetched_decision or {}).get("decision_reason") or "")
        if fetched and any(row.get("accepted") for row in page_rows):
            loss_cause = "preserved"
        elif hard_reason:
            loss_cause = "hard_reject"
        elif scorecard and phase231_decision in {"skip_prefetch", "hold_low_priority"}:
            loss_cause = "score_threshold"
        elif fetched_decision and fetched_decision.get("fetch_decision") in {"skip_budget", "defer"}:
            loss_cause = "fetch_budget"
        elif fetched_decision and fetched_decision.get("fetch_decision") == "skip_duplicate":
            loss_cause = "dedup/cache_issue"
        elif not search_rows and not scorecard:
            loss_cause = "query_omission"
        else:
            loss_cause = "other"
        action, justification = _prior_url_audit_action(url, loss_cause, scorecard)
        prior_evidence_types = sorted(
            {
                *(str(row.get("component") or "") for row in claim_rows if row.get("component")),
                *(str(row.get("impact_type") or "impact") for row in impact_rows),
            }
        )
        prior_body_records = [
            str(row.get("evidence_quote_or_body_span") or row.get("impact_description_normalized") or "")[:500]
            for row in [*claim_rows, *impact_rows]
            if row.get("validation_status") == "accepted"
        ]
        audit_rows.append(
            {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "candidate_id": candidate_id or "unknown_from_retained_phase2_3_1_artifacts",
                "phase2_3_sample_id": (
                    (scorecard or {}).get("phase2_3_sample_id")
                    or (fetched_decision or {}).get("phase2_3_sample_id")
                    or (page_rows[0].get("phase2_3_sample_id") if page_rows else "")
                    or (search_rows[0].get("phase2_3_sample_id") if search_rows else "")
                    or ""
                ),
                "url": url,
                "normalized_url": normalized,
                "prior_source_lane_if_available": (
                    (scorecard or {}).get("lane")
                    or (fetched_decision or {}).get("lane")
                    or (page_rows[0].get("source_lane") if page_rows else "")
                    or "unknown"
                ),
                "prior_accepted_evidence_type": prior_evidence_types or ["prior_open_accepted_url_from_baseline_snapshot"],
                "prior_body_supported_claim_or_impact_record": prior_body_records
                or ["not_available_in_retained_baseline_snapshot"],
                "fetched_in_phase2_3_1": fetched,
                "phase2_3_1_lane": (scorecard or {}).get("lane") or (fetched_decision or {}).get("lane") or "",
                "phase2_3_1_score": (scorecard or {}).get("score"),
                "phase2_3_1_decision": phase231_decision,
                "score_components": (scorecard or {}).get("score_components", {}),
                "positive_reasons": (scorecard or {}).get("positive_reasons", []),
                "negative_reasons": (scorecard or {}).get("negative_reasons", []),
                "hard_reject_reason": hard_reason,
                "phase2_3_1_fetch_decision": (fetched_decision or {}).get("fetch_decision", ""),
                "phase2_3_1_fetch_decision_reason": fetch_reason,
                "phase2_3_1_rejection_reasons": sorted(
                    {str(row.get("rejection_reason") or "") for row in rejected_rows if row.get("rejection_reason")}
                ),
                "loss_cause": loss_cause,
                "loss_flags": {
                    "hard_reject": loss_cause == "hard_reject",
                    "score_threshold": loss_cause == "score_threshold",
                    "lane_routing": loss_cause == "query_omission" and bool(search_rows),
                    "general_web_disabled": False,
                    "query_omission": loss_cause == "query_omission",
                    "fetch_budget": loss_cause == "fetch_budget",
                    "url_normalization_mismatch": False,
                    "dedup_cache_issue": loss_cause == "dedup/cache_issue",
                    "other": loss_cause == "other",
                },
                "calibration_decision": action,
                "justification": justification,
                "phase2_3_1_search_result_seen": bool(search_rows),
            }
        )
    return audit_rows


def build_prior_accepted_open_url_audit_markdown(audit_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2.3.2 Prior Accepted Open URL Audit",
        "",
        "| Candidate | URL | 2.3.1 fetched | 2.3.1 lane | Score | Decision | Loss cause | Calibration decision |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in audit_rows:
        lines.append(
            "| {candidate} | {url} | {fetched} | {lane} | {score} | {decision} | {loss} | {calibration} |".format(
                candidate=row.get("candidate_id", ""),
                url=row.get("url", ""),
                fetched=str(row.get("fetched_in_phase2_3_1", False)).lower(),
                lane=row.get("phase2_3_1_lane", ""),
                score="" if row.get("phase2_3_1_score") is None else row.get("phase2_3_1_score"),
                decision=row.get("phase2_3_1_decision", ""),
                loss=row.get("loss_cause", ""),
                calibration=row.get("calibration_decision", ""),
            )
        )
    lines.extend(["", "## URL-Level Findings", ""])
    for row in audit_rows:
        lines.extend(
            [
                f"### {row.get('url')}",
                "",
                f"- Candidate: {row.get('candidate_id')}",
                f"- Phase 2.3.1 decision: {row.get('phase2_3_1_decision')} (score: {row.get('phase2_3_1_score')})",
                f"- Hard reject: {row.get('hard_reject_reason') or 'none'}",
                f"- Loss cause: {row.get('loss_cause')}",
                f"- Calibration decision: {row.get('calibration_decision')}",
                f"- Justification: {row.get('justification')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def phase232_offline_replay_diagnostics(
    *,
    samples: list[dict[str, Any]],
    phase231_search_results: list[dict[str, Any]],
    prior_audit_rows: list[dict[str, Any]],
    controls: TavilyCostController,
) -> dict[str, Any]:
    sample_by_id = {str(sample.get("phase2_3_sample_id") or ""): sample for sample in samples}
    prior_positive_urls = {str(row.get("normalized_url") or "") for row in prior_audit_rows}
    intentionally_dropped = {
        str(row.get("normalized_url") or "")
        for row in prior_audit_rows
        if row.get("calibration_decision") == "intentionally_dropped"
    }
    eligible_prior_positive_urls = prior_positive_urls - intentionally_dropped
    scored: list[dict[str, Any]] = []
    for result in phase231_search_results:
        sid = str(result.get("phase2_3_sample_id") or "")
        sample = sample_by_id.get(sid)
        if not sample:
            continue
        scorecard = score_prefetch_result(
            sample=sample,
            query_id=str(result.get("query_id") or ""),
            lane=str(result.get("lane") or "local_news"),
            missing_component=str(result.get("missing_component") or "impact"),
            title=str(result.get("title") or ""),
            snippet=str(result.get("snippet") or ""),
            url=str(result.get("url") or ""),
        )
        scorecard["prior_positive"] = scorecard["normalized_url"] in prior_positive_urls
        scorecard["eligible_prior_positive"] = scorecard["normalized_url"] in eligible_prior_positive_urls
        scored.append(scorecard)
    fetchable_decisions = {"fetch_round1", "fetch_round2_candidate"}
    fetchable_prior_urls = {
        row["normalized_url"]
        for row in scored
        if row.get("eligible_prior_positive") and row.get("decision") in fetchable_decisions
    }
    simulations: dict[str, dict[str, Any]] = {}
    for top_k in (1, 2, 3):
        retained: set[str] = set()
        fetched = 0
        avoided_rejected = 0
        for sid in sorted({row["phase2_3_sample_id"] for row in scored}):
            rows = [row for row in scored if row["phase2_3_sample_id"] == sid]
            rows.sort(key=lambda row: (float(row.get("score") or 0), -LANE_PRIORITY.get(row.get("lane", ""), 99)), reverse=True)
            selected = rows[:top_k]
            fetched += len(selected)
            retained.update(row["normalized_url"] for row in selected if row.get("eligible_prior_positive"))
            avoided_rejected += sum(1 for row in rows[top_k:] if not row.get("prior_positive"))
        simulations[f"top_{top_k}"] = {
            "simulated_fetches": fetched,
            "prior_accepted_urls_retained": len(retained),
            "old_rejected_pages_avoided": avoided_rejected,
            "expected_open_acceptance_rate": None if fetched == 0 else len(retained) / fetched,
        }
    false_negative_causes = Counter(
        str(row.get("loss_cause") or "unknown")
        for row in prior_audit_rows
        if row.get("loss_cause") != "preserved"
    )
    replay_passed = len(fetchable_prior_urls) >= min(5, len(eligible_prior_positive_urls))
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "prior_accepted_url_count": len(prior_positive_urls),
        "eligible_prior_accepted_url_count": len(eligible_prior_positive_urls),
        "intentionally_dropped_prior_urls": sorted(intentionally_dropped),
        "prior_accepted_urls_fetchable_after_calibration": len(fetchable_prior_urls),
        "fetchable_prior_urls": sorted(fetchable_prior_urls),
        "prior_accepted_urls_not_fetchable_after_calibration": sorted(eligible_prior_positive_urls - fetchable_prior_urls),
        "old_rejected_pages_avoided": simulations["top_2"]["old_rejected_pages_avoided"],
        "simulated_top_k_fetch_policies": simulations,
        "expected_supported_candidate_preservation": "estimated_from_prior_positive_fetchability",
        "expected_impact_evidence_preservation": "estimated_from_prior_positive_fetchability",
        "false_negative_causes": dict(false_negative_causes),
        "replay_passed": replay_passed,
        "controls": controls.to_dict(),
    }


def build_phase232_replay_report(replay: dict[str, Any], audit_rows: list[dict[str, Any]]) -> str:
    dropped = replay.get("intentionally_dropped_prior_urls", [])
    false_causes = Counter(replay.get("false_negative_causes", {}))
    cause_lines = "\n".join(f"- {key}: {value}" for key, value in false_causes.most_common()) or "- None."
    return f"""# Phase 2.3.2 Offline Replay Report

## Summary

- Prior accepted URLs: {replay.get('prior_accepted_url_count')}
- Eligible prior accepted URLs after justified drops: {replay.get('eligible_prior_accepted_url_count')}
- Fetchable after calibrated scoring: {replay.get('prior_accepted_urls_fetchable_after_calibration')}
- Replay passed: {str(replay.get('replay_passed')).lower()}
- Intentionally dropped URLs: {len(dropped)}

## Simulated Fetch Policies

```json
{json.dumps(replay.get('simulated_top_k_fetch_policies'), indent=2, sort_keys=True)}
```

## False-Negative Causes

{cause_lines}

## Audit Basis

The replay treats the seven Phase 2.3 baseline accepted open URLs as calibration positives, except URLs explicitly
marked `intentionally_dropped` in `prior_accepted_open_url_audit.jsonl` because they are outside the frozen Texas
candidate scope or lack a retained candidate/query association.
"""


def stable_query_fingerprint(candidate_id: str, lane: str, missing_component: str, query_text: str) -> str:
    payload = "|".join([candidate_id, lane, missing_component, normalize_query(query_text)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
