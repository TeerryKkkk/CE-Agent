"""Historical compatibility aggregation helpers.

Repaired-v2 production truth is calculated only by
``pipeline.canonical_aggregation.aggregate_repaired_v2_case``.  The legacy
mapping helpers below remain for frozen historical table compatibility and are
not production truth authorities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .pipeline.canonical_aggregation import aggregate_repaired_v2_case


def _value(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value).strip().lower()
    return ""


def _axis(any_yes: bool, any_unresolved: bool = False) -> str:
    return "yes" if any_yes else ("unresolved" if any_unresolved else "no")


def aggregate_case_axes(
    webpage_rows: Iterable[Mapping[str, Any]],
    structured_rows: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    """Aggregate each evidence axis without cross-axis uncertainty leakage."""
    webpages = list(webpage_rows)
    accepted_structured = [
        row
        for row in structured_rows
        if _value(row, "deterministic_match_accepted") in {"true", "1", "yes"}
    ]

    drought_yes = any(
        _value(row, "drought_hazard_support") == "yes"
        for row in webpages
    ) or any(
        _value(row, "drought_hazard_support") == "yes"
        for row in accepted_structured
    )
    drought_unresolved = any(
        _value(row, "drought_hazard_support") == "unresolved"
        for row in webpages
    )
    wet_yes = any(
        _value(
            row,
            "wet_hazard_support",
            "candidate_hazard_support",
            "candidate_hazard_support",
        )
        == "yes"
        for row in webpages
    ) or any(_value(row, "wet_hazard_support") == "yes" for row in accepted_structured)
    wet_unresolved = any(
        _value(
            row,
            "wet_hazard_support",
            "candidate_hazard_support",
            "candidate_hazard_support",
        )
        == "unresolved"
        for row in webpages
    )

    impact_yes = any(
        _value(row, "realized_impact_support", "realized_impact_support") == "yes"
        for row in webpages
    ) or any(_value(row, "impact_support") == "yes" for row in accepted_structured)
    impact_unresolved = any(
        _value(row, "realized_impact_support", "realized_impact_support") == "unresolved"
        for row in webpages
    )

    attribution_yes = any(
        _value(row, "attribution_support", "explicit_attribution_support") == "yes"
        for row in webpages
    )
    attribution_unresolved = any(
        _value(row, "attribution_support", "explicit_attribution_support") == "unresolved"
        for row in webpages
    )

    linkage_yes = any(
        _value(
            row,
            "transition_support",
            "explicit_drought_to_wet_transition_support",
        )
        == "yes"
        for row in webpages
    )
    linkage_unresolved = any(
        _value(
            row,
            "transition_support",
            "explicit_drought_to_wet_transition_support",
        )
        == "unresolved"
        for row in webpages
    )

    return {
        "drought_hazard_support": _axis(
            drought_yes,
            drought_unresolved and not drought_yes,
        ),
        "wet_hazard_support": _axis(wet_yes, wet_unresolved and not wet_yes),
        "impact_support": _axis(impact_yes, impact_unresolved and not impact_yes),
        "explicit_hazard_to_impact_attribution_support": _axis(
            attribution_yes,
            attribution_unresolved and not attribution_yes,
        ),
        "explicit_linkage_support": _axis(linkage_yes, linkage_unresolved and not linkage_yes),
    }


def support_tier(axes: Mapping[str, str]) -> str:
    drought = axes["drought_hazard_support"]
    wet = axes["wet_hazard_support"]
    impact = axes["impact_support"]
    linkage = axes["explicit_linkage_support"]
    if drought == wet == impact == linkage == "yes":
        return "all_four_axes_supported"
    if drought == wet == impact == "yes":
        return "hazards_and_impact_supported_no_explicit_linkage"
    if drought == wet == "yes":
        return "both_hazard_axes_supported"
    if "unresolved" in {drought, wet, impact, linkage}:
        return "needs_review"
    return "partial_or_no_public_support"


def recompute_case_results(
    *,
    page_rows: Iterable[Mapping[str, Any]],
    structured_rows: Iterable[Mapping[str, Any]],
    manifest_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically produce case results in frozen manifest order."""
    pages_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    structured_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in page_rows:
        pages_by_case[str(row.get("candidate_id") or "")].append(row)
    for row in structured_rows:
        structured_by_case[str(row.get("candidate_id") or "")].append(row)

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_row in manifest_rows:
        candidate_id = str(manifest_row.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"invalid_or_duplicate_candidate_id:{candidate_id}")
        seen.add(candidate_id)
        webpages = pages_by_case[candidate_id]
        axes = aggregate_case_axes(webpages, structured_by_case[candidate_id])
        results.append(
            {
                "candidate_id": candidate_id,
                "county": str(manifest_row.get("county") or manifest_row.get("county_name") or ""),
                **axes,
                "support_tier": support_tier(axes),
                "webpage_count": len(webpages),
                "webpage_supported_count": sum(
                    _value(row, "page_result", "page_result") == "supports"
                    for row in webpages
                ),
                "webpage_unresolved_count": sum(
                    _value(row, "page_result", "page_result") == "unresolved"
                    for row in webpages
                ),
                "webpage_insufficient_count": sum(
                    _value(row, "page_result", "page_result")
                    == "insufficient_source_content"
                    for row in webpages
                ),
            }
        )
    return results
