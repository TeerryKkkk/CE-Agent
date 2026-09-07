from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


CE_STATUSES = {
    "ce_supported",
    "ce_partial",
    "ce_public_drought_weak",
    "ce_unsupported",
    "needs_review",
}

IMPACT_STATUSES = {
    "impact_material",
    "impact_weak",
    "impact_context_only",
    "impact_not_found",
    "needs_review",
}
DIRECT_OBSERVED_IMPACT_STATUSES = {
    "direct_material",
    "direct_weak",
    "direct_context_only",
    "direct_not_found",
    "direct_needs_review",
}
ADMIN_RESPONSE_STATUSES = {
    "admin_material_proxy",
    "admin_context_only",
    "admin_not_found",
    "admin_rejected",
    "admin_lane_failed",
}
COMBINED_IMPACT_STATUSES = {
    "material_direct",
    "material_admin_proxy",
    "weak_only",
    "context_only",
    "not_found",
    "needs_review",
}
MATERIAL_IMPACT_PATTERNS = {
    "compound_transition_material_impact",
    "drought_and_wet_material_impact",
    "wet_material_impact_only",
    "drought_material_impact_only",
    "no_material_impact",
}

CASE_USE_LABELS = {
    "strong_ce_with_material_impact",
    "ce_with_weak_impact",
    "ce_without_public_impact",
    "wet_impact_drought_public_weak",
    "impact_only_not_ce",
    "not_supported_or_review",
}

IMPACT_NEEDS_REVIEW_REASONS = {
    "impact_needs_review_channel",
    "impact_needs_review_materiality",
    "impact_needs_review_locality",
    "impact_needs_review_time_alignment",
    "impact_needs_review_source_quality",
}
MANUAL_REVIEW_REASONS = {
    "manual_review_due_to_output_taxonomy",
    "manual_review_due_to_evidence_quality",
    "manual_review_due_to_gate_uncertainty",
    "manual_review_due_to_retrieval_miss",
}
CHANNEL_ALIASES = {
    "transportation": "roads / transport",
    "roads": "roads / transport",
    "road closures": "roads / transport",
    "public_services": "public services / water",
    "public services": "public services / water",
    "utility": "energy / utility",
    "utilities": "energy / utility",
    "crop_loss": "agriculture",
    "crop loss": "agriculture",
    "agriculture_loss": "agriculture",
    "agriculture loss": "agriculture",
}
MATERIAL_DAMAGE_THRESHOLD_USD = 100_000.0
WEAK_ROAD_TERMS = (
    "road",
    "roadway",
    "highway",
    "bridge",
    "mudslide",
    "debris",
    "washout",
    "washed out",
    "impassable",
    "closure",
    "closed",
)
DIRECT_IMPACT_CHANNELS = {
    "agriculture",
    "energy / utility",
    "human impact",
    "property / housing",
    "public services / water",
    "roads / transport",
}
WET_IMPACT_COMPONENT_VALUES = {
    "wet",
    "rain",
    "rainfall",
    "rain_flood",
    "rain_flood_component",
    "wet_component",
    "extreme_rainfall",
    "flood",
    "storm",
}
DROUGHT_IMPACT_COMPONENT_VALUES = {
    "drought",
    "drought_component",
}
COMPOUND_IMPACT_COMPONENT_VALUES = {
    "both",
    "compound",
    "compound_sequence",
    "drought_then_rainfall",
    "drought_then_flood",
}
ADMIN_RESPONSE_SOURCE_LANES = {"openfema_admin_response"}
ADMIN_RESPONSE_SOURCE_FAMILIES = {"fema_openfema_admin_response_structured"}
MATERIAL_ACTION_TERMS = {
    "rescued",
    "rescue",
    "evacuat",
    "shelter",
    "levee break",
    "levee breaks",
    "levee breach",
    "levee breached",
    "washout",
    "washed out",
    "infrastructure damage",
    "infrastructure damaged",
    "utility outage",
    "power outage",
    "water outage",
    "service outage",
    "crop loss",
    "crop damage",
    "crops damaged",
    "agricultural loss",
    "school closure",
    "public service disruption",
    "emergency response",
    "public storm damage",
}


@dataclass(frozen=True)
class SplitLabelResult:
    ce_status: str
    impact_status: str
    case_use_label: str
    ce_status_reason: str
    impact_materiality_reason: str
    case_use_reason: str
    impact_needs_review_reason: str = ""
    manual_review_reason: str = ""
    ce_material_impact_gate_passed: bool = False
    drought_impact_status: str = "impact_not_found"
    wet_impact_status: str = "impact_not_found"
    compound_impact_status: str = "impact_not_found"
    material_impact_pattern: str = "no_material_impact"

    def as_dict(self) -> dict[str, str]:
        return {
            "ce_status": self.ce_status,
            "impact_status": self.impact_status,
            "case_use_label": self.case_use_label,
            "ce_status_reason": self.ce_status_reason,
            "impact_materiality_reason": self.impact_materiality_reason,
            "case_use_reason": self.case_use_reason,
            "impact_needs_review_reason": self.impact_needs_review_reason,
            "manual_review_reason": self.manual_review_reason,
            "ce_material_impact_gate_passed": str(self.ce_material_impact_gate_passed).lower(),
            "drought_impact_status": self.drought_impact_status,
            "wet_impact_status": self.wet_impact_status,
            "compound_impact_status": self.compound_impact_status,
            "material_impact_pattern": self.material_impact_pattern,
        }


@dataclass(frozen=True)
class EvidenceImpactClassification:
    evidence_impact_status: str
    impact_materiality_reason: str
    impact_needs_review_reason: str = ""


def _get(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def _as_text(value: Any) -> str:
    value = "" if value is None else value
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _decision(row: Mapping[str, Any]) -> str:
    for key in ("pipeline_decision_label", "final_status", "integrated_llm_status", "status"):
        value = _as_text(_get(row, key))
        if value:
            return value.lower()
    return ""


def _is_accepted(row: Mapping[str, Any]) -> bool:
    return _decision(row) == "accepted"


def _is_context_only(row: Mapping[str, Any]) -> bool:
    return _decision(row) == "context_only"


def _is_aligned(row: Mapping[str, Any]) -> bool:
    return _same_county_or_local(row) and _same_window_or_close(row)


def _split_channels(value: Any) -> set[str]:
    text = _as_text(value).replace(",", ";")
    return {_normalize_channel(part) for part in text.split(";") if part.strip()}


def _normalize_channel(value: str) -> str:
    raw = value.strip().lower()
    compact = re.sub(r"\s+", " ", raw.replace("_", " ")).strip()
    return CHANNEL_ALIASES.get(raw) or CHANNEL_ALIASES.get(compact) or compact


def _has_candidate_metadata(case_row: Mapping[str, Any]) -> bool:
    drought_window = _as_text(_get(case_row, "drought_window"))
    wet_window = _as_text(_get(case_row, "wet_event_window"))
    event_window = _as_text(_get(case_row, "event_window")).lower()
    return bool(drought_window and wet_window) or ("drought" in event_window and "wet" in event_window)


def _same_county_or_local(row: Mapping[str, Any]) -> bool:
    if "same_county_or_local" in row:
        return _as_bool(_get(row, "same_county_or_local"))
    location_match = _as_text(_get(row, "location_match")).lower()
    if location_match in {"same_county", "mapped_city", "mapped_city_or_place", "same_locality"}:
        return True
    if "county_match" in row:
        return _as_bool(_get(row, "county_match"))
    return False


def _same_window_or_close(row: Mapping[str, Any]) -> bool:
    if "same_window_or_close" in row:
        return _as_bool(_get(row, "same_window_or_close"))
    time_match = _as_text(_get(row, "time_match")).lower()
    if time_match in {"exact_window", "same_storm_sequence", "same_month", "drought_end_week"}:
        return True
    time_alignment = _as_text(_get(row, "time_alignment")).lower()
    return time_alignment in {"aligned", "exact_window", "close_storm_sequence", "drought_end_week"}


def _row_supports_component(row: Mapping[str, Any], component_name: str) -> bool:
    component = _as_text(_get(row, "component_supported")).lower()
    if component in {component_name, "both"}:
        return True
    support_field = {
        "drought": "supports_drought",
        "wet": "supports_wet_event",
    }.get(component_name)
    return bool(support_field and _as_bool(_get(row, support_field)))


def _row_supports_impact(row: Mapping[str, Any]) -> bool:
    return _as_bool(_get(row, "supports_impact")) or _as_bool(_get(row, "impact_supported"))


def _impact_component_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in (
        "impact_component",
        "component_attribution",
        "hazard_context",
        "component_supported",
        "impact_component_supported",
    ):
        value = _as_text(_get(row, key)).lower()
        if value:
            normalized = re.sub(r"\s+", "_", value.replace("-", "_").replace("/", "_")).strip("_")
            tokens.add(normalized)
    return tokens


def _row_supports_drought_impact(row: Mapping[str, Any]) -> bool:
    if not _row_supports_impact(row):
        return False
    if _as_bool(_get(row, "drought_impact_support")) or _as_bool(_get(row, "drought_impact_supported")):
        return True
    tokens = _impact_component_tokens(row)
    return bool(tokens & DROUGHT_IMPACT_COMPONENT_VALUES)


def _row_supports_wet_impact(row: Mapping[str, Any]) -> bool:
    if not _row_supports_impact(row):
        return False
    if (
        _as_bool(_get(row, "wet_impact_support"))
        or _as_bool(_get(row, "wet_impact_supported"))
        or _as_bool(_get(row, "rain_flood_impact_support"))
        or _as_bool(_get(row, "rain_flood_impact_supported"))
    ):
        return True
    tokens = _impact_component_tokens(row)
    if tokens & WET_IMPACT_COMPONENT_VALUES:
        return True
    return _row_supports_component(row, "wet")


def _row_supports_compound_impact(row: Mapping[str, Any]) -> bool:
    if not _row_supports_impact(row):
        return False
    if _as_bool(_get(row, "explicit_transition_support")) or _as_bool(_get(row, "explicit_linkage_support")):
        return True
    tokens = _impact_component_tokens(row)
    return bool(tokens & COMPOUND_IMPACT_COMPONENT_VALUES)


def _impact_status_rank(status: str) -> int:
    return {
        "impact_material": 4,
        "impact_weak": 3,
        "needs_review": 2,
        "impact_context_only": 1,
        "impact_not_found": 0,
    }.get(status, 0)


def _best_component_impact_status(
    classified_rows: Iterable[tuple[Mapping[str, Any], EvidenceImpactClassification]],
    predicate: Any,
) -> str:
    best = "impact_not_found"
    for row, item in classified_rows:
        if not predicate(row):
            continue
        status = item.evidence_impact_status
        if _is_accepted(row) and _is_aligned(row) and status in {"impact_material", "impact_weak"}:
            candidate = status
        elif status == "needs_review":
            candidate = "needs_review"
        elif status == "impact_context_only":
            candidate = "impact_context_only"
        else:
            candidate = "impact_not_found"
        if _impact_status_rank(candidate) > _impact_status_rank(best):
            best = candidate
    return best


def _is_admin_response_row(row: Mapping[str, Any]) -> bool:
    lane = _as_text(_get(row, "source_lane")).lower()
    family = _as_text(_get(row, "source_family")).lower()
    admin_status = _as_text(_get(row, "admin_response_status")).lower()
    return (
        lane in ADMIN_RESPONSE_SOURCE_LANES
        or family in ADMIN_RESPONSE_SOURCE_FAMILIES
        or admin_status.startswith("admin_")
    )


INFRA_DROUGHT_CHECK_STATUSES = {
    "api_failed",
    "infra_failed",
    "lane_failed",
    "live_api_fallback_failed",
    "parse_failed",
}


def _is_usdm_row(row: Mapping[str, Any]) -> bool:
    joined = " ".join(
        _as_text(_get(row, key)).lower()
        for key in ("source_family", "source_lane", "source_dataset", "source_title")
    )
    return "usdm" in joined or "drought monitor" in joined


def _public_drought_check_status(rows: Iterable[Mapping[str, Any]], public_drought: bool) -> str:
    row_list = list(rows)
    if public_drought:
        if any(
            _is_usdm_row(row)
            and _as_text(_get(row, "fetch_status")).lower() == "live_api_fallback_success"
            for row in row_list
        ):
            return "live_api_fallback_success"
        return "success"

    usdm_rows = [row for row in row_list if _is_usdm_row(row)]
    if not usdm_rows:
        return "unknown"

    for row in usdm_rows:
        explicit = _as_text(_get(row, "public_drought_check_status")).lower()
        if explicit in INFRA_DROUGHT_CHECK_STATUSES:
            return explicit
        decision = _decision(row)
        if decision in {"lane_failed", "infra_failed"}:
            return decision
        fetch_status = _as_text(_get(row, "fetch_status")).lower()
        if fetch_status in INFRA_DROUGHT_CHECK_STATUSES:
            return fetch_status
        if fetch_status == "live_api_fallback_failed":
            return "api_failed"
        if fetch_status in {"local_cache_miss", "not_attempted", "parse_failed"}:
            return "lane_failed"
        reason_text = " ".join(
            _as_text(_get(row, key)).lower()
            for key in (
                "decision_reason",
                "final_reason",
                "quoted_snippet_or_short_paraphrase",
                "quoted_supporting_spans",
            )
        )
        if "cache" in reason_text and (
            "miss" in reason_text
            or "missing" in reason_text
            or "could not find" in reason_text
            or "no local cached" in reason_text
        ):
            return "lane_failed"

    if any(_decision(row) in {"accepted", "context_only", "rejected"} for row in usdm_rows):
        return "success"
    return "unknown"


def derive_accepted_gate_flags(evidence_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive case gates from accepted, aligned evidence rows only."""
    rows = list(evidence_rows)
    accepted_rows = [row for row in rows if _is_accepted(row)]
    aligned_rows = [row for row in accepted_rows if _is_aligned(row)]
    classified_aligned_rows = [(row, classify_evidence_impact(row)) for row in aligned_rows]

    public_drought = any(_row_supports_component(row, "drought") for row in aligned_rows)
    wet_event = any(_row_supports_component(row, "wet") for row in aligned_rows)
    impact = any(_row_supports_impact(row) for row in aligned_rows)
    material_impact = any(
        item.evidence_impact_status == "impact_material"
        for _row, item in classified_aligned_rows
    )
    drought_impact = any(_row_supports_drought_impact(row) for row in aligned_rows)
    wet_impact = any(_row_supports_wet_impact(row) for row in aligned_rows)
    compound_impact = any(_row_supports_compound_impact(row) for row in aligned_rows)
    drought_material_impact = any(
        _row_supports_drought_impact(row) and item.evidence_impact_status == "impact_material"
        for row, item in classified_aligned_rows
    )
    wet_material_impact = any(
        _row_supports_wet_impact(row) and item.evidence_impact_status == "impact_material"
        for row, item in classified_aligned_rows
    )
    compound_material_impact = any(
        _row_supports_compound_impact(row) and item.evidence_impact_status == "impact_material"
        for row, item in classified_aligned_rows
    )
    same_county = any(
        (_row_supports_component(row, "drought") or _row_supports_component(row, "wet") or _row_supports_impact(row))
        and _same_county_or_local(row)
        for row in accepted_rows
    )
    same_window = any(
        (_row_supports_component(row, "drought") or _row_supports_component(row, "wet") or _row_supports_impact(row))
        and _same_window_or_close(row)
        for row in accepted_rows
    )

    result = {
        "public_drought_found": public_drought,
        "wet_event_found": wet_event,
        "impact_found": impact,
        "same_county_match": same_county,
        "same_window_match": same_window,
        "drought_gate_passed": public_drought,
        "wet_event_gate_passed": wet_event,
        "impact_gate_passed": impact,
        "material_impact_gate_passed": material_impact,
        "drought_impact_found": drought_impact,
        "wet_impact_found": wet_impact,
        "compound_impact_found": compound_impact,
        "drought_material_impact_gate_passed": drought_material_impact,
        "wet_material_impact_gate_passed": wet_material_impact,
        "compound_material_impact_gate_passed": compound_material_impact,
        "ce_material_impact_gate_passed": wet_material_impact or compound_material_impact,
        "public_drought_check_status": _public_drought_check_status(rows, public_drought),
    }
    result["material_impact_pattern"] = derive_material_impact_pattern(result)
    return result


def derive_material_impact_pattern(gate_flags: Mapping[str, Any]) -> str:
    """Classify case-level material impact attribution from side-specific gates."""
    drought = _as_bool(_get(gate_flags, "drought_material_impact_gate_passed"))
    wet = _as_bool(_get(gate_flags, "wet_material_impact_gate_passed"))
    compound = _as_bool(_get(gate_flags, "compound_material_impact_gate_passed"))
    if compound:
        return "compound_transition_material_impact"
    if drought and wet:
        return "drought_and_wet_material_impact"
    if wet:
        return "wet_material_impact_only"
    if drought:
        return "drought_material_impact_only"
    return "no_material_impact"


def _public_drought_supported(
    case_row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> bool:
    del case_row
    return derive_accepted_gate_flags(evidence_rows)["public_drought_found"]


def _wet_event_supported(
    case_row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> bool:
    del case_row
    return derive_accepted_gate_flags(evidence_rows)["wet_event_found"]


def _same_county_supported(
    case_row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> bool:
    del case_row
    return derive_accepted_gate_flags(evidence_rows)["same_county_match"]


def _same_window_supported(
    case_row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> bool:
    del case_row
    return derive_accepted_gate_flags(evidence_rows)["same_window_match"]


def _derive_ce_status_with_reason(
    case_row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> tuple[str, str]:
    rows = list(evidence_rows)
    if _as_bool(_get(case_row, "run_completed", True)) is False:
        return "needs_review", "Run did not complete, so CE support cannot be safely classified."
    if _as_bool(_get(case_row, "reached_evidence_judgment", True)) is False:
        return "needs_review", "Evidence judgment was not reached for this case."

    has_metadata = _has_candidate_metadata(case_row)
    gate_flags = derive_accepted_gate_flags(rows)
    drought_supported = gate_flags["public_drought_found"]
    wet_supported = gate_flags["wet_event_found"]
    same_county = gate_flags["same_county_match"]
    same_window = gate_flags["same_window_match"]
    public_drought_check_status = str(gate_flags.get("public_drought_check_status") or "unknown")
    drought_check_failed = public_drought_check_status in INFRA_DROUGHT_CHECK_STATUSES

    if has_metadata and drought_supported and wet_supported and same_county and same_window:
        return (
            "ce_supported",
            "Candidate metadata, public/structured drought support, wet-event support, county alignment, and window alignment are all present.",
        )
    if has_metadata and wet_supported and same_county and same_window and not drought_supported and drought_check_failed:
        return (
            "needs_review",
            "Wet-event evidence aligns by county/window, but the structured public drought check failed and cannot be treated as negative drought evidence.",
        )
    if has_metadata and wet_supported and same_county and same_window and not drought_supported:
        return (
            "ce_public_drought_weak",
            "Wet-event evidence aligns by county/window, but public/structured drought evidence is missing or weak.",
        )
    if has_metadata and (drought_supported or wet_supported):
        gaps = []
        if not drought_supported:
            gaps.append("public drought support")
        if not wet_supported:
            gaps.append("wet-event support")
        if not same_county:
            gaps.append("county alignment")
        if not same_window:
            gaps.append("window alignment")
        return (
            "ce_partial",
            "Some CE-side support exists, but these gates remain weak or missing: "
            + "; ".join(gaps)
            + ".",
        )
    if has_metadata:
        return (
            "ce_unsupported",
            "Candidate metadata exists, but retrieved/structured public evidence does not align enough to support the CE side.",
        )
    return "ce_unsupported", "No usable drought-to-wet candidate metadata or CE-side support is present."


def derive_ce_status(
    row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> str:
    status, _ = _derive_ce_status_with_reason(row, evidence_rows)
    return status


def _parse_damage_values(text: str, name: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(rf"{re.escape(name)}=([0-9]+(?:\.[0-9]+)?)([KMB]?)", text, re.IGNORECASE):
        amount = float(match.group(1))
        suffix = match.group(2).upper()
        multiplier = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]
        values.append(amount * multiplier)
    return values


def _parse_casualty_count(text: str, name: str) -> int:
    match = re.search(rf"{re.escape(name)}=([0-9]+)/([0-9]+)", text, re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1)) + int(match.group(2))


def _max_damage_usd(text: str) -> float:
    values = _parse_damage_values(text, "property") + _parse_damage_values(text, "crops")
    return max(values) if values else 0.0


def _material_action_term_found(text: str) -> str | None:
    lower = text.lower()
    for term in MATERIAL_ACTION_TERMS:
        if term in lower:
            return term
    return None


def _road_only_or_low_damage(row: Mapping[str, Any], text: str, max_damage: float) -> bool:
    channels = _split_channels(_get(row, "impact_channel"))
    if not channels:
        return False
    roadish = channels <= {"roads / transport", "property / housing", "energy / utility", "human impact"}
    lower = text.lower()
    has_road_term = any(term in lower for term in WEAK_ROAD_TERMS)
    return roadish and has_road_term and max_damage < MATERIAL_DAMAGE_THRESHOLD_USD


def _impact_review_reason(row: Mapping[str, Any]) -> str:
    if not _same_county_or_local(row):
        return "impact_needs_review_locality"
    if not _same_window_or_close(row):
        return "impact_needs_review_time_alignment"
    if not _is_accepted(row):
        return "impact_needs_review_source_quality"
    return "impact_needs_review_channel"


def classify_evidence_impact(row: Mapping[str, Any]) -> EvidenceImpactClassification:
    if _is_context_only(row):
        return EvidenceImpactClassification(
            "impact_context_only",
            "Evidence item is context-only and does not supply direct event-level impact support.",
        )
    if not _row_supports_impact(row):
        return EvidenceImpactClassification(
            "impact_not_found",
            "Evidence item does not support a public impact observation.",
        )

    if not _is_accepted(row):
        return EvidenceImpactClassification(
            "needs_review",
            "Impact-like evidence was present but the row was not accepted.",
            _impact_review_reason(row),
        )
    if not _is_aligned(row):
        return EvidenceImpactClassification(
            "needs_review",
            "Impact evidence is not safely aligned to the same county/locality and event window.",
            _impact_review_reason(row),
        )

    text = (
        _as_text(_get(row, "quoted_snippet_or_short_paraphrase"))
        or _as_text(_get(row, "quoted_supporting_spans"))
        or _as_text(_get(row, "final_reason"))
    )
    channels = _split_channels(_get(row, "impact_channel") or _get(row, "impact_types"))
    if not (channels & DIRECT_IMPACT_CHANNELS):
        return EvidenceImpactClassification(
            "needs_review",
            "Accepted impact evidence has an unresolved or unsupported impact channel after normalization.",
            "impact_needs_review_channel",
        )

    deaths = _parse_casualty_count(text, "deaths")
    injuries = _parse_casualty_count(text, "injuries")
    max_damage = _max_damage_usd(text)
    material_term = _material_action_term_found(text)

    if deaths + injuries > 0:
        return EvidenceImpactClassification(
            "impact_material",
            f"Direct local same-window impact includes reported casualties: deaths={deaths}, injuries={injuries}.",
        )
    if max_damage >= MATERIAL_DAMAGE_THRESHOLD_USD:
        return EvidenceImpactClassification(
            "impact_material",
            f"Direct local same-window impact includes damage estimate >= ${MATERIAL_DAMAGE_THRESHOLD_USD:,.0f}.",
        )
    if material_term in MATERIAL_ACTION_TERMS:
        return EvidenceImpactClassification(
            "impact_material",
            f"Direct local same-window impact includes material public-consequence term: {material_term}.",
        )
    if _road_only_or_low_damage(row, text, max_damage):
        return EvidenceImpactClassification(
            "impact_weak",
            "Impact is mainly road/transport or low-dollar NOAA narrative support, so it is kept as weak.",
        )
    if "noaa_ncei_storm_events" in _as_text(_get(row, "source_family")).lower() and max_damage < MATERIAL_DAMAGE_THRESHOLD_USD:
        return EvidenceImpactClassification(
            "impact_weak",
            "NOAA structured row has impact cues, but no casualties, high damage estimate, evacuation, rescue, or other strong materiality signal.",
        )
    return EvidenceImpactClassification(
        "impact_weak",
        "Public impact is observed, but materiality is not strong enough for case-study use.",
    )


def _derive_impact_status_with_reason(evidence_rows: Iterable[Mapping[str, Any]]) -> tuple[str, str, str]:
    classified_rows = [(row, classify_evidence_impact(row)) for row in evidence_rows]
    accepted_impact_rows = [
        (row, item)
        for row, item in classified_rows
        if _is_accepted(row) and _row_supports_impact(row)
    ]
    accepted_aligned_impact_rows = [
        (row, item)
        for row, item in accepted_impact_rows
        if _is_aligned(row)
    ]

    for _row, item in accepted_aligned_impact_rows:
        if item.evidence_impact_status == "impact_material":
            return "impact_material", item.impact_materiality_reason, ""
    for _row, item in accepted_aligned_impact_rows:
        if item.evidence_impact_status == "impact_weak":
            return "impact_weak", item.impact_materiality_reason, ""

    for _row, item in accepted_impact_rows:
        if item.evidence_impact_status == "needs_review":
            return (
                "needs_review",
                item.impact_materiality_reason,
                item.impact_needs_review_reason,
            )

    if any(item.evidence_impact_status == "impact_context_only" for _row, item in classified_rows):
        return (
            "impact_context_only",
            "Only context/background rows are available; no direct event-level public impact observation was accepted.",
            "",
        )
    return "impact_not_found", "No accepted public impact evidence row was found.", ""


def derive_component_impact_statuses(evidence_rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    classified_rows = [(row, classify_evidence_impact(row)) for row in evidence_rows]
    return {
        "drought_impact_status": _best_component_impact_status(classified_rows, _row_supports_drought_impact),
        "wet_impact_status": _best_component_impact_status(classified_rows, _row_supports_wet_impact),
        "compound_impact_status": _best_component_impact_status(classified_rows, _row_supports_compound_impact),
    }


def derive_impact_status(
    row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> str:
    del row
    status, _reason, _review_reason = _derive_impact_status_with_reason(evidence_rows)
    return status


def derive_direct_observed_impact_status(evidence_rows: Iterable[Mapping[str, Any]]) -> str:
    direct_rows = [row for row in evidence_rows if not _is_admin_response_row(row)]
    status, _reason, _review_reason = _derive_impact_status_with_reason(direct_rows)
    return {
        "impact_material": "direct_material",
        "impact_weak": "direct_weak",
        "impact_context_only": "direct_context_only",
        "impact_not_found": "direct_not_found",
        "needs_review": "direct_needs_review",
    }.get(status, "direct_needs_review")


def derive_admin_response_status(evidence_rows: Iterable[Mapping[str, Any]]) -> str:
    admin_rows = [row for row in evidence_rows if _is_admin_response_row(row)]
    if not admin_rows:
        return "admin_not_found"
    if any(
        _is_accepted(row)
        and _as_text(_get(row, "admin_response_status")).lower() == "admin_material_proxy"
        for row in admin_rows
    ):
        return "admin_material_proxy"
    if any(_decision(row) == "lane_failed" for row in admin_rows):
        return "admin_lane_failed"
    if any(
        _decision(row) == "context_only"
        or _as_text(_get(row, "admin_response_status")).lower() == "admin_context_only"
        for row in admin_rows
    ):
        return "admin_context_only"
    if any(
        _decision(row) == "rejected"
        or _as_text(_get(row, "admin_response_status")).lower() == "admin_rejected"
        for row in admin_rows
    ):
        return "admin_rejected"
    return "admin_not_found"


def derive_combined_impact_status(
    direct_observed_impact_status: str,
    admin_response_status: str,
) -> str:
    direct = _as_text(direct_observed_impact_status).lower()
    admin = _as_text(admin_response_status).lower()
    if direct == "direct_material":
        return "material_direct"
    if admin == "admin_material_proxy":
        return "material_admin_proxy"
    if direct == "direct_weak":
        return "weak_only"
    if direct == "direct_needs_review" or admin == "admin_lane_failed":
        return "needs_review"
    if direct == "direct_context_only" or admin == "admin_context_only":
        return "context_only"
    return "not_found"


def derive_case_use_label(
    ce_status: str,
    impact_status: str,
    ce_material_impact_gate_passed: bool | None = None,
) -> str:
    if ce_status == "ce_supported" and impact_status == "impact_material":
        if ce_material_impact_gate_passed is False:
            return "ce_with_weak_impact"
        return "strong_ce_with_material_impact"
    if ce_status in {"ce_supported", "ce_partial"} and impact_status == "impact_weak":
        return "ce_with_weak_impact"
    if ce_status in {"ce_supported", "ce_partial"} and impact_status in {
        "impact_context_only",
        "impact_not_found",
    }:
        return "ce_without_public_impact"
    if ce_status == "ce_public_drought_weak" and impact_status in {
        "impact_material",
        "impact_weak",
    }:
        return "wet_impact_drought_public_weak"
    if ce_status in {"ce_unsupported", "needs_review"} and impact_status in {
        "impact_material",
        "impact_weak",
    }:
        return "impact_only_not_ce"
    return "not_supported_or_review"


def _case_use_reason(ce_status: str, impact_status: str, case_use_label: str) -> str:
    if case_use_label == "strong_ce_with_material_impact":
        return "CE is fully supported and wet/compound impact evidence is material enough for strong case-study use."
    if case_use_label == "ce_with_weak_impact":
        return "CE is supported or partial, but impact evidence is weak, context-only, or not attributed to the wet/compound component."
    if case_use_label == "ce_without_public_impact":
        return "CE is supported or partial, but no public impact observation was found in the existing outputs."
    if case_use_label == "wet_impact_drought_public_weak":
        return "Wet event and impact are observed, but public drought support is missing or weak."
    if case_use_label == "impact_only_not_ce":
        return "Impact evidence exists, but CE-side support is not adequate."
    return f"Combined status is not suitable for a supported case label: ce_status={ce_status}, impact_status={impact_status}."


def _manual_review_reason(ce_status: str, impact_status: str, impact_review_reason: str) -> str:
    if ce_status == "needs_review":
        return "manual_review_due_to_gate_uncertainty"
    if impact_status != "needs_review":
        return ""
    if impact_review_reason in {
        "impact_needs_review_locality",
        "impact_needs_review_time_alignment",
    }:
        return "manual_review_due_to_gate_uncertainty"
    if impact_review_reason in IMPACT_NEEDS_REVIEW_REASONS:
        return "manual_review_due_to_evidence_quality"
    return "manual_review_due_to_evidence_quality"


def derive_case_level_split_labels(
    case_row: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]
) -> SplitLabelResult:
    rows = list(evidence_rows)
    ce_status, ce_reason = _derive_ce_status_with_reason(case_row, rows)
    impact_status, impact_reason, impact_review_reason = _derive_impact_status_with_reason(rows)
    gate_flags = derive_accepted_gate_flags(rows)
    component_impact_statuses = derive_component_impact_statuses(rows)
    case_use_label = derive_case_use_label(
        ce_status,
        impact_status,
        bool(gate_flags["ce_material_impact_gate_passed"]),
    )
    return SplitLabelResult(
        ce_status=ce_status,
        impact_status=impact_status,
        case_use_label=case_use_label,
        ce_status_reason=ce_reason,
        impact_materiality_reason=impact_reason,
        case_use_reason=_case_use_reason(ce_status, impact_status, case_use_label),
        impact_needs_review_reason=impact_review_reason,
        manual_review_reason=_manual_review_reason(ce_status, impact_status, impact_review_reason),
        ce_material_impact_gate_passed=bool(gate_flags["ce_material_impact_gate_passed"]),
        drought_impact_status=component_impact_statuses["drought_impact_status"],
        wet_impact_status=component_impact_statuses["wet_impact_status"],
        compound_impact_status=component_impact_statuses["compound_impact_status"],
        material_impact_pattern=str(gate_flags["material_impact_pattern"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _as_text(row.get(field, "")) for field in fieldnames})


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "无"
    return "；".join(f"{key}: {counter[key]}" for key in sorted(counter))


def write_ce_impact_label_split_outputs(
    *,
    case_results_path: str | Path = "outputs/california_patched_22case_rerun/pipeline_reproduction_results_patched_22case.csv",
    evidence_items_path: str | Path = "outputs/california_patched_22case_rerun/pipeline_evidence_items_patched_22case.csv",
    comparison_path: str | Path = "outputs/california_patched_22case_rerun/independent_vs_pipeline_comparison_patched_22case.csv",
    output_dir: str | Path = "outputs/california_ce_impact_label_split",
) -> dict[str, Any]:
    case_path = Path(case_results_path)
    evidence_path = Path(evidence_items_path)
    comparison_csv_path = Path(comparison_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows = _read_csv(case_path)
    evidence_rows = _read_csv(evidence_path)
    comparison_rows = _read_csv(comparison_csv_path) if comparison_csv_path.exists() else []

    evidence_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_case[_as_text(_get(row, "independent_case_id"))].append(row)

    split_rows: list[dict[str, str]] = []
    for case in case_rows:
        case_id = _as_text(_get(case, "independent_case_id"))
        case_evidence = evidence_by_case.get(case_id, [])
        result = derive_case_level_split_labels(case, case_evidence)
        gate_flags = derive_accepted_gate_flags(case_evidence)
        split_rows.append(
            {
                "case_id": case_id,
                "county": _as_text(_get(case, "county")),
                "prior_run_label": _as_text(_get(case, "pipeline_label")),
                "ce_status": result.ce_status,
                "impact_status": result.impact_status,
                "case_use_label": result.case_use_label,
                "public_drought_evidence_found": str(gate_flags["public_drought_found"]).lower(),
                "public_drought_check_status": str(gate_flags["public_drought_check_status"]),
                "wet_event_evidence_found": str(gate_flags["wet_event_found"]).lower(),
                "impact_evidence_found": str(gate_flags["impact_found"]).lower(),
                "drought_impact_found": str(gate_flags["drought_impact_found"]).lower(),
                "wet_impact_found": str(gate_flags["wet_impact_found"]).lower(),
                "compound_impact_found": str(gate_flags["compound_impact_found"]).lower(),
                "same_county_match": str(gate_flags["same_county_match"]).lower(),
                "same_window_match": str(gate_flags["same_window_match"]).lower(),
                "material_impact_gate_passed": str(gate_flags["material_impact_gate_passed"]).lower(),
                "drought_material_impact_gate_passed": str(gate_flags["drought_material_impact_gate_passed"]).lower(),
                "wet_material_impact_gate_passed": str(gate_flags["wet_material_impact_gate_passed"]).lower(),
                "compound_material_impact_gate_passed": str(gate_flags["compound_material_impact_gate_passed"]).lower(),
                "ce_material_impact_gate_passed": str(gate_flags["ce_material_impact_gate_passed"]).lower(),
                "material_impact_pattern": str(gate_flags["material_impact_pattern"]),
                "drought_impact_status": result.drought_impact_status,
                "wet_impact_status": result.wet_impact_status,
                "compound_impact_status": result.compound_impact_status,
                "impact_channels_found": _as_text(_get(case, "pipeline_impact_channels_found")),
                "impact_materiality_reason": result.impact_materiality_reason,
                "impact_needs_review_reason": result.impact_needs_review_reason,
                "manual_review_reason": result.manual_review_reason,
                "ce_status_reason": result.ce_status_reason,
                "case_use_reason": result.case_use_reason,
            }
        )

    evidence_class_rows: list[dict[str, str]] = []
    for row in evidence_rows:
        classification = classify_evidence_impact(row)
        evidence_class_rows.append(
            {
                "case_id": _as_text(_get(row, "independent_case_id")),
                "source_title": _as_text(_get(row, "source_title")),
                "source_url": _as_text(_get(row, "source_url")),
                "source_family": _as_text(_get(row, "source_family")),
                "retrieval_tier": _as_text(_get(row, "retrieval_tier")),
                "source_lane": _as_text(_get(row, "source_lane")),
                "supports_drought": str(_as_bool(_get(row, "supports_drought"))).lower(),
                "supports_wet_event": str(_as_bool(_get(row, "supports_wet_event"))).lower(),
                "supports_impact": str(_as_bool(_get(row, "supports_impact"))).lower(),
                "drought_impact_support": str(_row_supports_drought_impact(row)).lower(),
                "wet_impact_support": str(_row_supports_wet_impact(row)).lower(),
                "compound_impact_support": str(_row_supports_compound_impact(row)).lower(),
                "impact_channel": _as_text(_get(row, "impact_channel")),
                "evidence_impact_status": classification.evidence_impact_status,
                "impact_materiality_reason": classification.impact_materiality_reason,
                "impact_needs_review_reason": classification.impact_needs_review_reason,
            }
        )

    case_fields = [
        "case_id",
        "county",
        "prior_run_label",
        "ce_status",
        "impact_status",
        "case_use_label",
        "public_drought_evidence_found",
        "wet_event_evidence_found",
        "impact_evidence_found",
        "drought_impact_found",
        "wet_impact_found",
        "compound_impact_found",
        "same_county_match",
        "same_window_match",
        "material_impact_gate_passed",
        "drought_material_impact_gate_passed",
        "wet_material_impact_gate_passed",
        "compound_material_impact_gate_passed",
        "ce_material_impact_gate_passed",
        "material_impact_pattern",
        "drought_impact_status",
        "wet_impact_status",
        "compound_impact_status",
        "impact_channels_found",
        "impact_materiality_reason",
        "impact_needs_review_reason",
        "manual_review_reason",
        "ce_status_reason",
        "case_use_reason",
    ]
    evidence_fields = [
        "case_id",
        "source_title",
        "source_url",
        "source_family",
        "retrieval_tier",
        "source_lane",
        "supports_drought",
        "supports_wet_event",
        "supports_impact",
        "drought_impact_support",
        "wet_impact_support",
        "compound_impact_support",
        "impact_channel",
        "evidence_impact_status",
        "impact_materiality_reason",
        "impact_needs_review_reason",
    ]

    _write_csv(out_dir / "case_level_ce_impact_split.csv", case_fields, split_rows)
    _write_csv(out_dir / "evidence_level_impact_classification.csv", evidence_fields, evidence_class_rows)

    old_label_counts = Counter(row["prior_run_label"] for row in split_rows)
    ce_counts = Counter(row["ce_status"] for row in split_rows)
    impact_counts = Counter(row["impact_status"] for row in split_rows)
    case_use_counts = Counter(row["case_use_label"] for row in split_rows)
    comparison_status_counts = Counter(_as_text(_get(row, "reproduction_status")) for row in comparison_rows)

    cases_postprocessed = len(split_rows)
    public_impact_cases = sum(row["impact_status"] in {"impact_material", "impact_weak"} for row in split_rows)
    material_impact_cases = impact_counts["impact_material"]
    weak_impact_cases = impact_counts["impact_weak"]
    context_only_impact_cases = impact_counts["impact_context_only"]
    old_strong_now_weak = sum(
        row["prior_run_label"] == "strong_ce_with_impact"
        and row["impact_status"] == "impact_weak"
        for row in split_rows
    )
    strong_case_studies = case_use_counts["strong_ce_with_material_impact"]
    ce_weak_impact_cases = case_use_counts["ce_with_weak_impact"]
    wet_impact_drought_weak_cases = case_use_counts["wet_impact_drought_public_weak"]

    report = f"""# CE/Impact 标签拆分报告

本次只对既有 patched 22-case California rerun 输出做确定性后处理，没有重跑 discovery、retrieval、Tavily、NOAA/USDM structured lanes、LLM judge 或 audit agent，也没有改写历史输出。

## 改动内容

- 新增 `ce_status`：只描述 drought-to-wet / drought-to-flood / drought-to-storm 的 CE/event 支持。
- 新增 `impact_status`：只描述公开 impact 观测的强弱和材料性。
- 新增 `case_use_label`：把 CE 支持和 impact 强度组合成下游可用标签。
- 旧 `pipeline_label` 被保留为 `prior_run_label`，不再作为主最终标签。

拆分原因是旧 `strong_ce_with_impact` 把 CE 证据和 impact 观测混在一起。Impact 可以作为 case-study 强度和公共观测维度记录，但不应反过来升级 CE 支持。

## 运行范围

- 输入 case 表：`{case_path.as_posix()}`
- 输入 evidence 表：`{evidence_path.as_posix()}`
- 输入 comparison 表仅用于统计对照：`{comparison_csv_path.as_posix()}`
- 输出目录：`{out_dir.as_posix()}`

## 主要结果

- postprocessed cases：{cases_postprocessed}
- CE supported：{ce_counts["ce_supported"]}
- CE public drought weak：{ce_counts["ce_public_drought_weak"]}
- CE partial：{ce_counts["ce_partial"]}
- CE unsupported / needs review：{ce_counts["ce_unsupported"] + ce_counts["needs_review"]}
- 有公开 impact 观测的 cases：{public_impact_cases}
- material impact cases：{material_impact_cases}
- weak impact cases：{weak_impact_cases}
- context-only impact/background cases：{context_only_impact_cases}
- 旧 `strong_ce_with_impact` 现在仅为 weak-impact 的 cases：{old_strong_now_weak}
- 可作为 strong case study 的 cases：{strong_case_studies}
- CE cases with weak impact：{ce_weak_impact_cases}
- wet-impact 但 public drought weak 的 cases：{wet_impact_drought_weak_cases}

## 分布

- 旧 pipeline label：{_format_counter(old_label_counts)}
- 新 CE status：{_format_counter(ce_counts)}
- 新 impact status：{_format_counter(impact_counts)}
- 新 case-use label：{_format_counter(case_use_counts)}
- 独立 baseline comparison status（仅评估用）：{_format_counter(comparison_status_counts)}

## 保守处理

新规则不会让 impact evidence 升级 `ce_status`。如果 public drought evidence 缺失，即使 wet event 和 impact 都存在，也标为 `ce_public_drought_weak`，case-use label 为 `wet_impact_drought_public_weak`。

Impact materiality 也比旧聚合更保守。NOAA structured row 中的 road-only、zero/low damage、宽 storm sequence narrative、或缺少明确伤亡/救援/撤离/高额损失的 impact cues 被保留为 `impact_weak`，而不是自动成为 strong/material。

## 剩余不确定性

- 部分 NOAA structured narrative 会在同一 storm sequence 中复用宽区域描述，仍需要人工或 audit layer 判断具体 locality/materiality。
- `impact_material` 不是完整损失估计，只表示当前公开证据足以支持 strong case-study 用途。
- `impact_weak` 不表示没有影响，只表示当前输出中的公开证据偏低强度、偏 road-only、偏叙述或材料性不足。
"""
    (out_dir / "CE_IMPACT_LABEL_SPLIT_REPORT.md").write_text(report, encoding="utf-8")

    old_vs = f"""# 旧标签与 CE/Impact 拆分对比

本文件比较 patched rerun 的旧 `pipeline_label` 与新的三轴输出。该步骤是 label/output split，不是新 retrieval run。

## 旧 pipeline labels

{_format_counter(old_label_counts)}

旧标签的问题是 `strong_ce_with_impact` 同时表达了两件事：CE/event 支持，以及 public impact observation。这样会让 road-only、低损失或 NOAA broad narrative 的 impact cue 看起来像完整 CE case-study 支持。

## 新 CE status

{_format_counter(ce_counts)}

`ce_status` 只看 drought/wet/local/window 侧证据。Impact evidence 不参与 CE status 升级。

## 新 impact status

{_format_counter(impact_counts)}

`impact_status` 单独记录公开 impact 观测强度。material impact 需要更明确的伤亡、救援/撤离、较高损失或直接公共服务/基础设施后果；弱 impact 仍作为公开观测保存。

## 新 case-use labels

{_format_counter(case_use_counts)}

今后建议报告：

- CE 支持数量：`ce_status=ce_supported` 的数量。
- 公开 impact 观测数量：`impact_status` 为 material/weak/context-only 的数量。
- 强案例数量：`case_use_label=strong_ce_with_material_impact`。
- CE 但 impact weak 的数量：`case_use_label=ce_with_weak_impact`。
- wet/impact 支持但 drought public evidence weak 的数量：`case_use_label=wet_impact_drought_public_weak`。

这比继续报告单一 `strong_ce_with_impact` 更清楚，也更符合证据对象应该拆分 support axis 和 impact axis 的设计。
"""
    (out_dir / "old_vs_split_label_summary.md").write_text(old_vs, encoding="utf-8")

    changed_files = """Source/test files changed:
src/climate_pipeline/ce_impact_labeling.py
tests/test_ce_impact_label_split.py

Generated derived output files:
outputs/california_ce_impact_label_split/CE_IMPACT_LABEL_SPLIT_REPORT.md
outputs/california_ce_impact_label_split/case_level_ce_impact_split.csv
outputs/california_ce_impact_label_split/evidence_level_impact_classification.csv
outputs/california_ce_impact_label_split/old_vs_split_label_summary.md
outputs/california_ce_impact_label_split/changed_files.txt

Historical output files modified: none.
"""
    (out_dir / "changed_files.txt").write_text(changed_files, encoding="utf-8")

    return {
        "cases_postprocessed": cases_postprocessed,
        "ce_status_distribution": dict(sorted(ce_counts.items())),
        "impact_status_distribution": dict(sorted(impact_counts.items())),
        "case_use_label_distribution": dict(sorted(case_use_counts.items())),
        "strong_case_study_candidates": strong_case_studies,
        "ce_cases_with_weak_impact": ce_weak_impact_cases,
        "wet_impact_cases_with_weak_public_drought": wet_impact_drought_weak_cases,
        "old_strong_now_weak_impact": old_strong_now_weak,
        "files_created": [
            str(out_dir / "CE_IMPACT_LABEL_SPLIT_REPORT.md"),
            str(out_dir / "case_level_ce_impact_split.csv"),
            str(out_dir / "evidence_level_impact_classification.csv"),
            str(out_dir / "old_vs_split_label_summary.md"),
            str(out_dir / "changed_files.txt"),
        ],
    }
