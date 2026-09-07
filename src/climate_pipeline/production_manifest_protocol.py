"""Canonical imports for the frozen post-40 manifest protocol.

These functions are a namespace-only extraction of the production-reachable
helpers from ``scripts.retrieve``.  Their query, evidence,
and guard semantics are unchanged; placing them below ``climate_pipeline``
lets the isolated launcher use one module namespace and only ``<repo>/src``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from climate_pipeline.controlled_open_retrieval import (
    TavilyCostController,
    build_lane_queries,
    normalize_url,
    stable_query_fingerprint,
)
from climate_pipeline.llm_evidence_judge import quoted_spans_are_body_grounded
from climate_pipeline.llm_evidence_validation import target_hazard_axis_for_page


EVIDENCE_FIELDS = [
    "candidate_id", "county", "fips", "evidence_origin", "retrieval_tier",
    "query_id", "found_by", "query_source", "retrieval_round", "target_gap",
    "query_intent", "expansion_trigger", "generated_query", "validated_query",
    "source_lane", "source_family", "source_url", "source_title", "final_status",
    "component_supported", "impact_supported", "impact_types", "wet_impact_support",
    "drought_impact_support", "explicit_transition_support", "page_result",
    "target_hazard_axis", "drought_hazard_support", "wet_hazard_support",
    "candidate_hazard_support", "realized_impact_support",
    "explicit_attribution_support", "explicit_drought_to_wet_transition_support",
    "location_match", "time_match", "page_type", "quoted_supporting_spans",
    "quote_body_grounded", "accepted_from_snippet", "failure_reason_if_rejected",
    "final_reason",
]

RETRIEVAL_PROVENANCE_KEYS = [
    "query_source", "found_by", "retrieval_round", "target_gap", "query_intent",
    "expansion_trigger", "generated_query", "validated_query",
]


def new_retrieval_state() -> dict[str, Any]:
    """Compatibility state for offline baseline orchestration tests only."""

    return {
        "fetched_urls_global": set(),
        "domain_counts": Counter(),
        "candidate_pdf_counts": Counter(),
        "global_counts": Counter(),
        "global_counts_by_round": {},
        "candidate_fetch_counts_by_round": {},
    }


def execute_open_query_rows(*_args: Any, **_kwargs: Any) -> Any:
    """Fail-closed compatibility seam for the superseded baseline runner.

    The formal repaired-v2 runner overrides expansion orchestration and cannot
    reach this function.  The symbol lets offline baseline tests inject a
    fixture without importing the repository-root demo under a second package
    namespace.
    """

    raise RuntimeError("legacy_open_query_executor_not_available_in_production_namespace")


def split_join(values: Iterable[Any]) -> str:
    return "; ".join(str(value) for value in values if str(value))


def current_case_from_manifest(
    row: Mapping[str, str], inventory_row: Mapping[str, str]
) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    return {
        "candidate_id": candidate_id,
        "phase2_3_sample_id": candidate_id,
        "raw_candidate_id": candidate_id,
        "county": inventory_row.get("county_name", row.get("county", "")),
        "state": "California",
        "state_abbrev": "CA",
        "FIPS": str(inventory_row.get("county_fips", row.get("fips", ""))).zfill(5),
        "county_fips": str(inventory_row.get("county_fips", row.get("fips", ""))).zfill(5),
        "drought_start": inventory_row.get("drought_start_month", row.get("drought_start", "")),
        "drought_end": inventory_row.get("drought_end_month", ""),
        "drought_end_month_end": inventory_row.get("drought_end_month_end", row.get("drought_end_month_end", "")),
        "rain_start": inventory_row.get("rainfall_window_start", row.get("rain_start", "")),
        "rain_end": inventory_row.get("rainfall_window_end", row.get("rain_end", "")),
        "rainfall_year": inventory_row.get("rainfall_year", row.get("year", "")),
        "lag_days": inventory_row.get("lag_days", row.get("lag_days", "")),
        "lag_bin": inventory_row.get("lag_bin", row.get("lag_bin", "")),
        "spatial_match_level": inventory_row.get("spatial_match_level", row.get("spatial_match_strength", "")),
        "drought_severity": inventory_row.get("drought_wmo_severity_max", row.get("drought_severity", "")),
        "min_spei": inventory_row.get("drought_min_spei_min", row.get("min_spei", "")),
        "rainfall_days": inventory_row.get("rainfall_duration_days", row.get("rainfall_days", "")),
        "rainfall_grid_count": inventory_row.get("rainfall_source_grid_count", row.get("rainfall_grid_count", "")),
        "drought_window": f"{inventory_row.get('drought_start_month', row.get('drought_start', ''))} to {inventory_row.get('drought_end_month_end', row.get('drought_end_month_end', ''))}",
        "wet_event_window": f"{inventory_row.get('rainfall_window_start', row.get('rain_start', ''))} to {inventory_row.get('rainfall_window_end', row.get('rain_end', ''))}",
        "event_window": f"drought {inventory_row.get('drought_start_month', row.get('drought_start', ''))} to {inventory_row.get('drought_end_month_end', row.get('drought_end_month_end', ''))}; wet {inventory_row.get('rainfall_window_start', row.get('rain_start', ''))} to {inventory_row.get('rainfall_window_end', row.get('rain_end', ''))}",
        "demo_group": row.get("demo_group", ""),
        "old_positive_label_if_any": row.get("old_positive_label_if_any", ""),
        "retrieval_allowed": "true",
        "source_inventory_policy_version": inventory_row.get("policy_version", ""),
    }


def case_card_context(cases: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        str(case["candidate_id"]): {
            "candidate_id": str(case["candidate_id"]),
            "county": str(case["county"]),
            "county_fips": str(case["FIPS"]),
            "drought_window": str(case["drought_window"]),
            "wet_window": str(case["wet_event_window"]),
            "candidate_stratum": str(case["demo_group"]),
        }
        for case in cases
    }


def query_rows_for_case(
    case: dict[str, Any], gap: dict[str, Any], controls: TavilyCostController
) -> list[dict[str, Any]]:
    rows = build_lane_queries(
        sample=case,
        gap_state=gap,
        controls=controls,
        round_number=1,
        search_backend="tavily",
    )
    for row in rows:
        row["query_id"] = stable_query_fingerprint(
            str(row["candidate_id"]),
            str(row["lane"]),
            str(row["missing_component"]),
            str(row["query_text"]),
        )
        row["retrieval_tier"] = "controlled_tavily_open_web"
        row["retrieval_round"] = 1
        row["found_by"] = "fixed_template_open_web"
        row["target_gap"] = str(row.get("missing_component") or "")
        row["query_intent"] = "fixed_template_retrieval"
        row["expansion_trigger"] = ""
        row["generated_query"] = ""
        row["validated_query"] = str(row.get("query_text") or "")
        row["fallback_reason"] = str(gap.get("open_search_reason") or "")
        row["unmet_gap_before_search"] = split_join(gap.get("missing_components") or [])
        row["domains_or_source_families_targeted"] = str(row.get("lane") or "")
    return rows


def validation_prefilter_rows(
    pages: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for page in pages:
        case = cases_by_id[str(page["candidate_id"])]
        rows.append(
            {
                "candidate_id": str(page["candidate_id"]),
                "source_url": str(page["source_url"]),
                "source_family": str(page.get("discovered_source_family") or page.get("source_family") or ""),
                "source_lane": str(page.get("source_lane") or ""),
                "event_window": str(case["wet_event_window"]),
                "county": str(case["county"]),
            }
        )
    return rows


def web_evidence_rows_from_validation(
    result: Any,
    pages: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    page_by_key = {
        (str(page["candidate_id"]), normalize_url(str(page["source_url"]))): page
        for page in pages
    }
    rows: list[dict[str, Any]] = []
    for normalized in result.normalized_rows:
        candidate_id = str(normalized["candidate_id"])
        source_url = str(normalized["source_url"])
        case = cases_by_id[candidate_id]
        page = page_by_key.get((candidate_id, normalize_url(source_url)), {})
        direct_final = normalized.get("guarded_local_result")
        primary = normalized.get("raw_semantic_judgment") or normalized.get("primary_judgment") or {}
        final = normalized.get("final_judgment") or (
            direct_final if isinstance(direct_final, dict) and "page_result" not in direct_final else {}
        )
        source_metadata = normalized["source_metadata"]
        candidate_metadata = normalized.get("candidate_metadata") or {}
        target_hazard_axis = str(
            candidate_metadata.get("target_hazard_axis")
            or target_hazard_axis_for_page(page)
        )
        if direct_final and "page_result" in direct_final:
            candidate_event = primary.get("candidate_event") or {}
            quoted_spans = [str(value) for value in candidate_event.get("supporting_quotes") or []]
            page_result = str(direct_final.get("page_result") or "unresolved")
            final_status = {
                "supports": "accepted", "does_not_support": "rejected",
                "unresolved": "needs_review", "insufficient_source_content": "needs_review",
            }[page_result]
            candidate_hazard_support = str(direct_final.get("candidate_hazard_support") or "unresolved")
            realized_impact_support = str(direct_final.get("realized_impact_support") or "unresolved")
            explicit_attribution_support = str(direct_final.get("explicit_attribution_support") or "unresolved")
            explicit_transition_support = str(direct_final.get("explicit_drought_to_wet_transition_support") or "no")
            component = (
                target_hazard_axis
                if candidate_hazard_support == "yes"
                else "none"
            )
            drought_hazard_support = (
                candidate_hazard_support
                if target_hazard_axis == "drought"
                else "unresolved"
            )
            wet_hazard_support = (
                candidate_hazard_support
                if target_hazard_axis == "wet"
                else "unresolved"
            )
            impact_supported = realized_impact_support == "yes"
            impact_types = [
                str(value.get("claim") or "")
                for value in candidate_event.get("realized_impacts") or []
                if isinstance(value, dict)
            ]
            location_match = str(candidate_event.get("location_relation") or "unknown")
            time_match = str(candidate_event.get("date_relation") or "unknown")
            page_type = str(primary.get("event_identity") or "unresolved_event_identity")
            failure_reason = next(
                (
                    str(action.get("reason") or "")
                    for action in direct_final.get("guard_actions") or []
                    if action.get("action") in {"blocked", "rejected"}
                ),
                "none",
            )
            final_reason = "; ".join(
                f"{action.get('guard')}:{action.get('action')}:{action.get('reason')}"
                for action in direct_final.get("guard_actions") or []
            )
        else:
            quoted_spans = [str(value) for value in final.get("final_quoted_spans") or []]
            page_result = {
                "accepted": "supports", "rejected": "does_not_support",
                "needs_review": "unresolved",
            }.get(str(final.get("final_status") or ""), "unresolved")
            final_status = str(final.get("final_status") or "")
            component = str(final.get("final_component_supported") or "none")
            impact_supported = bool(final.get("final_impact_supported"))
            impact_types = [str(value) for value in final.get("final_impact_types") or []]
            drought_hazard_support = "yes" if component in {"drought", "both"} else "no"
            wet_hazard_support = "yes" if component in {"wet", "both"} else "no"
            candidate_hazard_support = (
                "yes"
                if component in {"drought", "wet", "both"}
                else "no"
            )
            if component == "drought":
                target_hazard_axis = "drought"
            realized_impact_support = "yes" if impact_supported else "no"
            explicit_attribution_support = "yes" if impact_supported and final_status == "accepted" else "no"
            explicit_transition_support = "yes" if bool(primary.get("explicit_transition_support")) and final_status == "accepted" else "no"
            location_match = str(primary.get("location_match") or "")
            time_match = str(primary.get("time_match") or "")
            page_type = str(primary.get("page_type") or "")
            failure_reason = str(final.get("failure_reason_if_rejected") or primary.get("failure_reason_if_rejected") or "none")
            final_reason = str(final.get("final_reason") or "")
        body_text = str(page.get("body_text_or_archived_body_text") or "")
        grounded = quoted_spans_are_body_grounded(quoted_spans, body_text=body_text) if final_status == "accepted" else False
        rows.append(
            {
                "candidate_id": candidate_id,
                "county": case["county"],
                "fips": case["FIPS"],
                "evidence_origin": "validated_open_web_body",
                "retrieval_tier": "controlled_tavily_open_web",
                "query_id": str(page.get("query_id") or source_metadata.get("query_id") or ""),
                "found_by": str(page.get("found_by") or "fixed_template_open_web"),
                "query_source": str(page.get("query_source") or "template"),
                "retrieval_round": str(page.get("retrieval_round") or "1"),
                "target_gap": str(page.get("target_gap") or ""),
                "query_intent": str(page.get("query_intent") or ""),
                "expansion_trigger": str(page.get("expansion_trigger") or ""),
                "generated_query": str(page.get("generated_query") or ""),
                "validated_query": str(page.get("validated_query") or page.get("query_text") or ""),
                "source_lane": str(source_metadata.get("source_lane") or page.get("source_lane") or ""),
                "source_family": str(source_metadata.get("source_family") or source_metadata.get("raw_source_family") or ""),
                "source_url": source_url,
                "source_title": str(page.get("source_title") or source_metadata.get("source_title") or ""),
                "final_status": final_status,
                "component_supported": component,
                "impact_supported": str(impact_supported).lower(),
                "impact_types": split_join(impact_types),
                "wet_impact_support": str(component in {"wet", "both"} and impact_supported).lower(),
                "drought_impact_support": str(component in {"drought", "both"} and impact_supported).lower(),
                "explicit_transition_support": str(explicit_transition_support == "yes").lower(),
                "page_result": page_result,
                "target_hazard_axis": target_hazard_axis,
                "drought_hazard_support": drought_hazard_support,
                "wet_hazard_support": wet_hazard_support,
                "candidate_hazard_support": candidate_hazard_support,
                "realized_impact_support": realized_impact_support,
                "explicit_attribution_support": explicit_attribution_support,
                "explicit_drought_to_wet_transition_support": explicit_transition_support,
                "location_match": location_match,
                "time_match": time_match,
                "page_type": page_type,
                "quoted_supporting_spans": " || ".join(quoted_spans),
                "quote_body_grounded": str(grounded).lower(),
                "accepted_from_snippet": str(bool(page.get("accepted_from_snippet"))).lower(),
                "failure_reason_if_rejected": failure_reason,
                "final_reason": final_reason,
            }
        )
    return rows


__all__ = [
    "EVIDENCE_FIELDS",
    "RETRIEVAL_PROVENANCE_KEYS",
    "case_card_context",
    "current_case_from_manifest",
    "query_rows_for_case",
    "validation_prefilter_rows",
    "web_evidence_rows_from_validation",
]
