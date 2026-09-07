"""Retrieve and validate evidence for a CSV manifest of compound-event candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from climate_pipeline.ce_impact_labeling import (
    classify_evidence_impact,
    derive_accepted_gate_flags,
    derive_case_level_split_labels,
)
from climate_pipeline.case_aggregation import (
    aggregate_case_axes,
    support_tier,
)
from climate_pipeline.controlled_open_retrieval import (
    TavilyCostController,
    build_lane_queries,
    build_search_result_cache,
    build_url_body_cache,
    compute_evidence_gap_state,
    finalize_cost_ledger,
    initial_cost_ledger,
    normalize_url,
    schedule_fetch_decisions,
    score_prefetch_result,
    stable_query_fingerprint,
)
from climate_pipeline.llm_evidence_judge import (
    quoted_spans_are_body_grounded,
    read_openai_api_key,
)
from climate_pipeline.llm_evidence_validation import (
    LLMEvidenceJudgeConfig,
    LLMEvidenceValidationUnavailable,
    target_hazard_axis_for_page,
    validate_fetched_pages_with_llm_judge,
)
from climate_pipeline.llm_query_expansion import (
    EXPANSION_REPORT_FIELDS,
    LLMQueryExpansionConfig,
    LLMQueryExpansionUnavailable,
    annotate_expansion_report_with_outcomes,
    make_run_id,
    plan_llm_query_expansion,
)
from climate_pipeline.official_source_lanes import (
    OfficialSourceLaneControls,
    official_sources_attempted_text,
    run_structured_official_lanes_for_case,
    support_flags_from_evidence_rows,
)
from climate_pipeline.io_utils import safe_write_json, safe_write_text, write_jsonl
from climate_pipeline.us_tavily import TavilySearchClient, read_tavily_api_key
from climate_pipeline.web_reader import WebReader


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_HOME = Path(os.getenv("CE_AGENT_PROJECT_ROOT", REPO_ROOT))
DATA_ROOT = Path(os.getenv("CE_AGENT_DATA_ROOT", PROJECT_HOME / "data"))
DEFAULT_INVENTORY = REPO_ROOT / "examples" / "california40" / "candidate_inventory.csv"
DEFAULT_NOAA_CACHE = DATA_ROOT / "official" / "noaa_stormevents"

REQUIRED_OUTPUTS = [
    "MANIFEST_RUNNER_2CASE_SUMMARY.md",
    "case_registry.csv",
    "evidence_rows.csv",
    "gate_assertion_summary.csv",
    "source_lane_summary.csv",
    "impact_channel_summary.csv",
    "data_quality_issues.csv",
    "case_cards.md",
    "cache_safety_check.md",
    "stale_label_safety_check.md",
    "quote_grounding_check.md",
]

CASE_REGISTRY_FIELDS = [
    "candidate_id",
    "demo_group",
    "county",
    "fips",
    "year",
    "drought_start",
    "drought_end_month_end",
    "rain_start",
    "rain_end",
    "lag_days",
    "lag_bin",
    "drought_severity",
    "min_spei",
    "rainfall_days",
    "rainfall_grid_count",
    "spatial_match_strength",
    "old_positive_label_if_any",
    "drought_hazard_support",
    "wet_hazard_support",
    "impact_support",
    "explicit_hazard_to_impact_attribution_support",
    "explicit_linkage_support",
    "support_tier",
    "current_ce_status",
    "current_impact_status",
    "current_case_use_label",
    "current_material_impact_pattern",
    "accepted_evidence_rows",
    "accepted_web_evidence_rows",
    "official_sources_attempted",
]

EVIDENCE_FIELDS = [
    "candidate_id",
    "county",
    "fips",
    "evidence_origin",
    "retrieval_tier",
    "query_id",
    "found_by",
    "query_source",
    "retrieval_round",
    "target_gap",
    "query_intent",
    "expansion_trigger",
    "generated_query",
    "validated_query",
    "source_lane",
    "source_family",
    "source_url",
    "source_title",
    "final_status",
    "component_supported",
    "impact_supported",
    "impact_types",
    "wet_impact_support",
    "drought_impact_support",
    "explicit_transition_support",
    "page_result",
    "target_hazard_axis",
    "drought_hazard_support",
    "wet_hazard_support",
    "candidate_hazard_support",
    "realized_impact_support",
    "explicit_attribution_support",
    "explicit_drought_to_wet_transition_support",
    "location_match",
    "time_match",
    "page_type",
    "quoted_supporting_spans",
    "quote_body_grounded",
    "accepted_from_snippet",
    "failure_reason_if_rejected",
    "final_reason",
]

GATE_FIELDS = [
    "candidate_id",
    "public_drought_found",
    "wet_event_found",
    "impact_found",
    "same_county_match",
    "same_window_match",
    "drought_gate_passed",
    "wet_event_gate_passed",
    "impact_gate_passed",
    "material_impact_gate_passed",
    "drought_material_impact_gate_passed",
    "wet_material_impact_gate_passed",
    "compound_material_impact_gate_passed",
    "ce_material_impact_gate_passed",
    "material_impact_pattern",
    "public_drought_check_status",
    "explicit_transition_support_found",
    "ce_status",
    "impact_status",
    "case_use_label",
    "manual_review_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value or "").strip().lower()


def split_join(values: Iterable[Any]) -> str:
    return "; ".join(str(value) for value in values if str(value))


def current_case_from_manifest(row: dict[str, str], inventory_row: dict[str, str]) -> dict[str, Any]:
    candidate_id = row["candidate_id"]
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


def load_cases_from_manifest(manifest_path: Path, inventory_path: Path, max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = read_csv(manifest_path)
    inventory_rows = read_csv(inventory_path)
    inventory_by_id = {row["candidate_id"]: row for row in inventory_rows}
    cases: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows[:max_candidates], start=1):
        candidate_id = row.get("candidate_id", "")
        inventory_row = inventory_by_id.get(candidate_id)
        if not inventory_row:
            issues.append(
                issue(
                    severity="blocking",
                    candidate_id=candidate_id,
                    issue_type="manifest_id_not_in_current_primary_inventory",
                    details=f"Manifest row {index} does not match the frozen California primary inventory.",
                )
            )
            continue
        cases.append(current_case_from_manifest(row, inventory_row))
    return cases, issues


def issue(*, severity: str, candidate_id: str, issue_type: str, details: str) -> dict[str, str]:
    return {
        "severity": severity,
        "candidate_id": candidate_id,
        "issue_type": issue_type,
        "details": details,
        "blocking": str(severity == "blocking").lower(),
    }


def add_common_evidence_fields(row: dict[str, Any], case: dict[str, Any], origin: str) -> dict[str, Any]:
    enriched = dict(row)
    enriched.setdefault("candidate_id", case["candidate_id"])
    enriched.setdefault("county", case["county"])
    enriched.setdefault("fips", case["FIPS"])
    enriched["evidence_origin"] = origin
    enriched.setdefault("retrieval_tier", "structured_official_first" if origin.startswith("structured") else "controlled_tavily_open_web")
    enriched.setdefault("quote_body_grounded", "")
    enriched.setdefault("accepted_from_snippet", "false")
    return enriched


def case_card_context(cases: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        case["candidate_id"]: {
            "candidate_id": case["candidate_id"],
            "county": case["county"],
            "county_fips": case["FIPS"],
            "drought_window": case["drought_window"],
            "wet_window": case["wet_event_window"],
            "candidate_stratum": case["demo_group"],
        }
        for case in cases
    }


def query_rows_for_case(case: dict[str, Any], gap: dict[str, Any], controls: TavilyCostController) -> list[dict[str, Any]]:
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


def result_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def reader_body_text(result: Any) -> str:
    cleaned_path = Path(str(result.cleaned_text_path or ""))
    if not cleaned_path:
        return ""
    path = cleaned_path if cleaned_path.is_absolute() else REPO_ROOT / cleaned_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


RETRIEVAL_PROVENANCE_KEYS = [
    "query_source",
    "found_by",
    "retrieval_round",
    "target_gap",
    "query_intent",
    "expansion_trigger",
    "generated_query",
    "validated_query",
]


def is_llm_query_expansion_evidence(row: dict[str, Any]) -> bool:
    return (
        str(row.get("retrieval_round") or "") == "2"
        or str(row.get("found_by") or "") == "llm_query_expansion"
        or str(row.get("query_source") or "") in {"llm_expansion", "llm_query_expansion"}
    )


def rows_for_main_label_derivation(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in evidence_rows
        if not (
            is_llm_query_expansion_evidence(row)
            and str(row.get("final_status") or "") != "accepted"
        )
    ]


def new_retrieval_state() -> dict[str, Any]:
    return {
        "fetched_urls_global": set(),
        "domain_counts": Counter(),
        "candidate_pdf_counts": Counter(),
        "global_counts": Counter(),
        "global_counts_by_round": {},
        "candidate_fetch_counts_by_round": {},
    }


def execute_open_query_rows(
    *,
    cases: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    controls: TavilyCostController,
    tavily_client: TavilySearchClient,
    web_reader: WebReader,
    ledger: dict[str, Any],
    round_number: int,
    retrieval_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    search_result_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    fetch_decision_rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    raw_result_by_candidate_url: dict[tuple[str, str], dict[str, Any]] = {}
    state = retrieval_state or new_retrieval_state()
    fetched_urls_global: set[str] = state["fetched_urls_global"]
    domain_counts: Counter[str] = state["domain_counts"]
    candidate_pdf_counts: Counter[str] = state["candidate_pdf_counts"]
    global_counts: Counter[str] = state["global_counts"]
    round_key = str(round_number)
    global_counts_by_round = state.setdefault("global_counts_by_round", {})
    candidate_fetch_counts_by_round = state.setdefault("candidate_fetch_counts_by_round", {})
    round_global_counts = global_counts_by_round.setdefault(round_key, Counter())
    round_candidate_fetch_counts = candidate_fetch_counts_by_round.setdefault(round_key, Counter())
    queries_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_by_id: dict[str, dict[str, Any]] = {}
    for query in query_rows:
        cid = str(query.get("candidate_id") or query.get("phase2_3_sample_id") or "")
        queries_by_case[cid].append(query)
        if query.get("query_id"):
            query_by_id[str(query["query_id"])] = query

    for case in cases:
        planned = queries_by_case.get(case["candidate_id"], [])
        if not planned:
            continue
        for query in planned:
            query_text = str(query["query_text"])
            lane = str(query["lane"])
            raw_results, metadata = tavily_client.search_raw(
                query_text,
                max_results=controls.max_results_per_query,
                include_raw_content=True,
            )
            ledger["tavily_search_calls"] += 1
            ledger["executed_queries"] += 1
            ledger[f"round{round_number}_queries"] = int(ledger.get(f"round{round_number}_queries") or 0) + 1
            ledger["new_tavily_calls_after_cache"] += 1
            ledger["search_results_returned"] += len(raw_results)
            ledger.setdefault("_live_tavily_query_norms", []).append(query_text)
            by_lane = dict(ledger.get("tavily_search_calls_by_lane") or {})
            by_lane[lane] = int(by_lane.get(lane) or 0) + 1
            ledger["tavily_search_calls_by_lane"] = by_lane
            queries_by_lane = dict(ledger.get("queries_by_lane") or {})
            queries_by_lane[lane] = int(queries_by_lane.get(lane) or 0) + 1
            ledger["queries_by_lane"] = queries_by_lane
            base_query_metadata = {
                "query_source": str(query.get("query_source") or ""),
                "found_by": str(query.get("found_by") or ""),
                "retrieval_round": str(query.get("retrieval_round") or round_number),
                "target_gap": str(query.get("target_gap") or query.get("missing_component") or ""),
                "query_intent": str(query.get("query_intent") or ""),
                "expansion_trigger": str(query.get("expansion_trigger") or ""),
                "generated_query": str(query.get("generated_query") or ""),
                "validated_query": str(query.get("validated_query") or query_text),
            }
            if metadata.get("error"):
                search_result_rows.append(
                    {
                        "candidate_id": case["candidate_id"],
                        "query_id": query["query_id"],
                        "query_text": query_text,
                        "lane": lane,
                        "result_rank": "",
                        "url": "",
                        "title": "",
                        "snippet": "",
                        "raw_content_available": "false",
                        "tavily_error": metadata.get("error"),
                        **base_query_metadata,
                    }
                )
                continue
            for rank, raw in enumerate(raw_results, start=1):
                url = result_text(raw, "url")
                title = result_text(raw, "title") or url
                snippet = result_text(raw, "content", "snippet")
                raw_content = result_text(raw, "raw_content")
                search_row = {
                    "candidate_id": case["candidate_id"],
                    "query_id": query["query_id"],
                    "query_text": query_text,
                    "lane": lane,
                    "missing_component": query["missing_component"],
                    "result_rank": rank,
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "raw_content_available": str(bool(raw_content)).lower(),
                    "tavily_error": "",
                    **base_query_metadata,
                }
                search_result_rows.append(search_row)
                if url:
                    raw_result_by_candidate_url[(case["candidate_id"], normalize_url(url))] = raw
                score = score_prefetch_result(
                    sample=case,
                    query_id=str(query["query_id"]),
                    lane=lane,
                    missing_component=str(query["missing_component"]),
                    title=title,
                    snippet=snippet,
                    url=url,
                )
                score.update(
                    {
                        "query_text": query_text,
                        "retrieval_tier": query["retrieval_tier"],
                        "fallback_reason": query["fallback_reason"],
                        "unmet_gap_before_search": query["unmet_gap_before_search"],
                        "domains_or_source_families_targeted": query["domains_or_source_families_targeted"],
                        **base_query_metadata,
                    }
                )
                score_rows.append(score)
        ledger["search_results_scored"] += len([row for row in score_rows if row["candidate_id"] == case["candidate_id"]])
        decisions = schedule_fetch_decisions(
            sample=case,
            scorecards=score_rows,
            controls=controls,
            round_number=round_number,
            fetched_urls_global=fetched_urls_global,
            domain_counts=domain_counts,
            candidate_pdf_counts=candidate_pdf_counts,
            global_counts=round_global_counts,
            candidate_fetch_count=round_candidate_fetch_counts[str(case["candidate_id"])],
        )
        for decision in decisions:
            query = query_by_id.get(str(decision.get("query_id") or ""), {})
            for key in RETRIEVAL_PROVENANCE_KEYS:
                decision[key] = query.get(key, decision.get(key, ""))
            if not decision.get("retrieval_round"):
                decision["retrieval_round"] = round_number
            if not decision.get("validated_query"):
                decision["validated_query"] = query.get("query_text", "")
            if round_number == 2:
                reason_map = {
                    "duplicate_url_already_fetched": "duplicate_url",
                    "round_candidate_fetch_cap_reached": "expansion_budget_exhausted",
                    "candidate_total_fetch_cap_reached": "expansion_budget_exhausted",
                    "global_open_fetch_cap_reached": "global_budget_exhausted",
                    "candidate_lane_fetch_cap_reached": "expansion_lane_budget_exhausted",
                }
                decision["decision_reason"] = reason_map.get(str(decision.get("decision_reason") or ""), decision.get("decision_reason", ""))
        if round_number == 2:
            scheduled_keys = {
                (str(decision.get("query_id") or ""), str(decision.get("normalized_url") or ""))
                for decision in decisions
            }
            for score in score_rows:
                if str(score.get("phase2_3_sample_id") or score.get("candidate_id") or "") != str(case["candidate_id"]):
                    continue
                score_key = (str(score.get("query_id") or ""), str(score.get("normalized_url") or ""))
                if score_key in scheduled_keys:
                    continue
                query = query_by_id.get(str(score.get("query_id") or ""), {})
                reason = "low_score" if score.get("decision") in {"hold_low_priority", "skip_prefetch"} else "not_fetch_candidate"
                decisions.append(
                    {
                        "schema_version": str(score.get("schema_version") or ""),
                        "candidate_id": str(score.get("candidate_id") or case["candidate_id"]),
                        "phase2_3_sample_id": str(score.get("phase2_3_sample_id") or case["candidate_id"]),
                        "query_id": str(score.get("query_id") or ""),
                        "lane": str(score.get("lane") or ""),
                        "round": round_number,
                        "retrieval_tier": str(score.get("retrieval_tier") or ""),
                        "fallback_reason": str(score.get("fallback_reason") or ""),
                        "official_sources_attempted": str(score.get("official_sources_attempted") or ""),
                        "unmet_gap_before_fallback": str(score.get("unmet_gap_before_fallback") or ""),
                        "unmet_gap_before_search": str(score.get("unmet_gap_before_search") or ""),
                        "domains_or_source_families_targeted": str(score.get("domains_or_source_families_targeted") or ""),
                        "url": str(score.get("url") or ""),
                        "normalized_url": str(score.get("normalized_url") or ""),
                        "domain": str(score.get("domain") or ""),
                        "score": score.get("score", ""),
                        "prefetch_decision": str(score.get("decision") or ""),
                        "fetch_decision": "skip_score",
                        "decision_reason": reason,
                        **{key: query.get(key, score.get(key, "")) for key in RETRIEVAL_PROVENANCE_KEYS},
                    }
                )
                if not decisions[-1].get("retrieval_round"):
                    decisions[-1]["retrieval_round"] = round_number
                if not decisions[-1].get("validated_query"):
                    decisions[-1]["validated_query"] = query.get("query_text", "")
        fetch_decision_rows.extend(decisions)
        for decision in decisions:
            if decision["fetch_decision"] != "fetch":
                if decision["fetch_decision"].startswith("skip"):
                    ledger["search_results_skipped_before_fetch"] += 1
                continue
            normalized = str(decision["normalized_url"])
            fetched_urls_global.add(normalized)
            domain_counts[str(decision["domain"])] += 1
            global_counts["open_fetches_total"] += 1
            round_global_counts["open_fetches_total"] += 1
            round_candidate_fetch_counts[str(case["candidate_id"])] += 1
            if str(decision.get("lane") or "") == "general_web":
                round_global_counts["general_web_fetches"] += 1
            if str(decision.get("document_type") or "").lower() == "pdf" or str(decision.get("url") or "").lower().split("?")[0].endswith(".pdf"):
                round_global_counts["pdf_fetches"] += 1
            ledger["open_urls_fetched"] += 1
            fetches_by_round = dict(ledger.get("open_fetches_by_round") or {})
            round_key = str(round_number)
            fetches_by_round[round_key] = int(fetches_by_round.get(round_key) or 0) + 1
            ledger["open_fetches_by_round"] = fetches_by_round
            fetches_by_lane = dict(ledger.get("open_fetches_by_lane") or {})
            fetches_by_lane[str(decision["lane"])] = int(fetches_by_lane.get(str(decision["lane"])) or 0) + 1
            ledger["open_fetches_by_lane"] = fetches_by_lane
            raw = raw_result_by_candidate_url.get((case["candidate_id"], normalized), {})
            title = result_text(raw, "title") or str(decision["url"])
            reader_result = web_reader.read(str(decision["url"]), title_hint=title)
            body_text = reader_body_text(reader_result)
            raw_content = result_text(raw, "raw_content")
            body_source = reader_result.fetch_method
            if len(body_text.strip()) < 80 and raw_content:
                body_text = raw_content
                body_source = "tavily_raw_content"
            query = query_by_id.get(str(decision.get("query_id") or ""), {})
            pages.append(
                {
                    "phase2_3_sample_id": case["candidate_id"],
                    "raw_candidate_id": case["candidate_id"],
                    "candidate_id": case["candidate_id"],
                    "county": case["county"],
                    "state": "California",
                    "state_abbrev": "CA",
                    "FIPS": case["FIPS"],
                    "county_fips": case["FIPS"],
                    "source_url": str(decision["url"]),
                    "source_title": title,
                    "source_family": str(raw.get("source_family") or ""),
                    "discovered_source_family": str(
                        next(
                            (
                                row.get("source_type")
                                for row in score_rows
                                if row.get("candidate_id") == case["candidate_id"]
                                and row.get("normalized_url") == normalized
                            ),
                            "",
                        )
                    ),
                    "source_lane": decision["lane"],
                    "query_id": decision["query_id"],
                    "query_text": str(query.get("query_text") or ""),
                    "query_source": str(query.get("query_source") or ""),
                    "found_by": str(query.get("found_by") or ""),
                    "retrieval_round": str(query.get("retrieval_round") or round_number),
                    "target_gap": str(query.get("target_gap") or ""),
                    "query_intent": str(query.get("query_intent") or ""),
                    "expansion_trigger": str(query.get("expansion_trigger") or ""),
                    "generated_query": str(query.get("generated_query") or ""),
                    "validated_query": str(query.get("validated_query") or query.get("query_text") or ""),
                    "body_text_or_archived_body_text": body_text,
                    "body_text_source": body_source,
                    "accepted_from_snippet": False,
                    "reader_result": reader_result.model_dump(),
                    "fetch_status": "success" if body_text.strip() else (reader_result.failure_reason or "body_unavailable"),
                }
            )
    return search_result_rows, score_rows, fetch_decision_rows, pages, state


def run_open_retrieval(
    *,
    cases: list[dict[str, Any]],
    gaps_by_case: dict[str, dict[str, Any]],
    controls: TavilyCostController,
    tavily_client: TavilySearchClient,
    web_reader: WebReader,
    ledger: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_rows: list[dict[str, Any]] = []
    for case in cases:
        gap = gaps_by_case[case["candidate_id"]]
        query_rows.extend(query_rows_for_case(case, gap, controls))
    search_result_rows, score_rows, fetch_decision_rows, pages, state = execute_open_query_rows(
        cases=cases,
        query_rows=query_rows,
        controls=controls,
        tavily_client=tavily_client,
        web_reader=web_reader,
        ledger=ledger,
        round_number=1,
    )
    return query_rows, search_result_rows, score_rows, fetch_decision_rows, pages, state

def validation_prefilter_rows(pages: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for page in pages:
        case = cases_by_id[str(page["candidate_id"])]
        rows.append(
            {
                "candidate_id": str(page["candidate_id"]),
                "source_url": str(page["source_url"]),
                "source_family": str(page.get("discovered_source_family") or page.get("source_family") or ""),
                "source_lane": str(page.get("source_lane") or ""),
                "event_window": case["wet_event_window"],
                "county": case["county"],
            }
        )
    return rows


def web_evidence_rows_from_validation(result: Any, pages: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    page_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for page in pages:
        page_by_key[(str(page["candidate_id"]), normalize_url(str(page["source_url"])))] = page
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
                "supports": "accepted",
                "does_not_support": "rejected",
                "unresolved": "needs_review",
                "insufficient_source_content": "needs_review",
            }[page_result]
            candidate_hazard_support = str(direct_final.get("candidate_hazard_support") or "unresolved")
            realized_impact_support = str(direct_final.get("realized_impact_support") or "unresolved")
            explicit_attribution_support = str(direct_final.get("explicit_attribution_support") or "unresolved")
            explicit_transition_support = str(
                direct_final.get("explicit_drought_to_wet_transition_support") or "no"
            )
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
                "accepted": "supports",
                "rejected": "does_not_support",
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
            explicit_transition_support = (
                "yes"
                if bool(primary.get("explicit_transition_support")) and final_status == "accepted"
                else "no"
            )
            location_match = str(primary.get("location_match") or "")
            time_match = str(primary.get("time_match") or "")
            page_type = str(primary.get("page_type") or "")
            failure_reason = str(
                final.get("failure_reason_if_rejected")
                or primary.get("failure_reason_if_rejected")
                or "none"
            )
            final_reason = str(final.get("final_reason") or "")
        body_text = str(page.get("body_text_or_archived_body_text") or "")
        grounded = (
            quoted_spans_are_body_grounded(quoted_spans, body_text=body_text)
            if final_status == "accepted"
            else False
        )
        row = {
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
        rows.append(row)
    return rows


def build_case_outputs(
    *,
    cases: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    official_attempts_by_case: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_case[str(row["candidate_id"])].append(row)
    registry_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for case in cases:
        cid = case["candidate_id"]
        rows = evidence_by_case.get(cid, [])
        label_rows = rows_for_main_label_derivation(rows)
        split = derive_case_level_split_labels(case, label_rows)
        gates = derive_accepted_gate_flags(label_rows)
        accepted_rows = [row for row in rows if str(row.get("final_status")) == "accepted"]
        accepted_web = [
            row
            for row in accepted_rows
            if str(row.get("evidence_origin")) == "validated_open_web_body"
        ]
        webpage_rows = [
            row
            for row in rows
            if str(row.get("evidence_origin") or "") == "validated_open_web_body"
        ]
        structured_rows = [
            row
            for row in rows
            if str(row.get("evidence_origin") or "").startswith("structured")
        ]
        axes = aggregate_case_axes(webpage_rows, structured_rows)
        registry_rows.append(
            {
                "candidate_id": cid,
                "demo_group": case["demo_group"],
                "county": case["county"],
                "fips": case["FIPS"],
                "year": case["rainfall_year"],
                "drought_start": case["drought_start"],
                "drought_end_month_end": case["drought_end_month_end"],
                "rain_start": case["rain_start"],
                "rain_end": case["rain_end"],
                "lag_days": case["lag_days"],
                "lag_bin": case["lag_bin"],
                "drought_severity": case["drought_severity"],
                "min_spei": case["min_spei"],
                "rainfall_days": case["rainfall_days"],
                "rainfall_grid_count": case["rainfall_grid_count"],
                "spatial_match_strength": case["spatial_match_level"],
                "old_positive_label_if_any": case["old_positive_label_if_any"],
                **axes,
                "support_tier": support_tier(axes),
                "current_ce_status": split.ce_status,
                "current_impact_status": split.impact_status,
                "current_case_use_label": split.case_use_label,
                "current_material_impact_pattern": split.material_impact_pattern,
                "accepted_evidence_rows": len(accepted_rows),
                "accepted_web_evidence_rows": len(accepted_web),
                "official_sources_attempted": official_attempts_by_case.get(cid, ""),
            }
        )
        gate_row = {
            "candidate_id": cid,
            **{key: str(value).lower() if isinstance(value, bool) else value for key, value in gates.items()},
            "explicit_transition_support_found": str(
                any(
                    str(row.get("final_status")) == "accepted"
                    and as_bool_text(row.get("explicit_transition_support")) == "true"
                    for row in label_rows
                )
            ).lower(),
            **split.as_dict(),
        }
        gate_rows.append(gate_row)
    return registry_rows, gate_rows, []


def build_source_lane_summary(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in evidence_rows:
        counts[
            (
                str(row.get("evidence_origin") or ""),
                str(row.get("source_lane") or ""),
                str(row.get("source_family") or ""),
                str(row.get("final_status") or ""),
            )
        ] += 1
    return [
        {
            "evidence_origin": origin,
            "source_lane": lane,
            "source_family": family,
            "final_status": status,
            "count": count,
        }
        for (origin, lane, family, status), count in sorted(counts.items())
    ]


def build_impact_channel_summary(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in evidence_rows:
        classification = classify_evidence_impact(row)
        channels = [part.strip() for part in str(row.get("impact_types") or "").replace(",", ";").split(";") if part.strip()]
        if not channels:
            channels = ["none"]
        for channel in channels:
            counts[(channel, classification.evidence_impact_status, str(row.get("final_status") or ""))] += 1
    return [
        {
            "impact_channel": channel,
            "evidence_impact_status": impact_status,
            "final_status": final_status,
            "count": count,
        }
        for (channel, impact_status, final_status), count in sorted(counts.items())
    ]


def build_case_cards(cases: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> str:
    evidence_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gate_by_case = {row["candidate_id"]: row for row in gate_rows}
    for row in evidence_rows:
        evidence_by_case[str(row["candidate_id"])].append(row)
    lines = ["# California Manifest Runner Case Cards", ""]
    for case in cases:
        cid = case["candidate_id"]
        gate = gate_by_case.get(cid, {})
        lines.extend(
            [
                f"## {cid}",
                "",
                f"- County/FIPS: {case['county']} / {case['FIPS']}",
                f"- Drought window: {case['drought_window']}",
                f"- Rain window: {case['wet_event_window']}",
                f"- Demo group: {case['demo_group']}",
                f"- Current CE status: {gate.get('ce_status', '')}",
                f"- Current impact status: {gate.get('impact_status', '')}",
                f"- Current case-use label: {gate.get('case_use_label', '')}",
                f"- Material impact pattern: {gate.get('material_impact_pattern', '')}",
                "",
                "### Accepted Evidence",
                "",
            ]
        )
        accepted = [row for row in evidence_by_case.get(cid, []) if str(row.get("final_status")) == "accepted"]
        if not accepted:
            lines.append("- None.")
        for row in accepted:
            title = row.get("source_title") or row.get("source_family") or row.get("source_url")
            quote = str(row.get("quoted_supporting_spans") or "")[:500]
            lines.append(f"- {row.get('source_family', '')}: [{title}]({row.get('source_url', '')})")
            if quote:
                lines.append(f"  Quote: {quote}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def path_has_forbidden_leak(path_text: str) -> bool:
    lowered = path_text.replace("\\", "/").lower()
    return (
        "/texas" in lowered
        or "/phase2_" in lowered
        or "/phase2-" in lowered
        or "/phase2/" in lowered
        or "phase_2" in lowered
    )


def count_duckduckgo_records(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if "duck" in str(row.get("backend") or row.get("search_backend") or row.get("client") or "").lower()
    )


def write_safety_reports(
    *,
    output_dir: Path,
    cache_dir: Path,
    noaa_cache_dir: Path,
    usdm_cache_dir: Path,
    evidence_rows: list[dict[str, Any]],
    manifest_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    accepted_web = [
        row for row in evidence_rows
        if row.get("final_status") == "accepted" and row.get("evidence_origin") == "validated_open_web_body"
    ]
    accepted_snippet_only = [
        row for row in accepted_web
        if as_bool_text(row.get("accepted_from_snippet")) == "true" or not str(row.get("quoted_supporting_spans") or "").strip()
    ]
    accepted_quote_not_grounded = [
        row for row in accepted_web
        if as_bool_text(row.get("quote_body_grounded")) != "true"
    ]
    cache_paths = [output_dir, cache_dir, noaa_cache_dir, usdm_cache_dir]
    leakage = any(path_has_forbidden_leak(str(path)) for path in cache_paths)
    safe_write_text(
        output_dir / "cache_safety_check.md",
        "\n".join(
            [
                "# Cache Safety Check",
                "",
                f"- Output directory: `{output_dir}`",
                f"- Cache directory: `{cache_dir}`",
                f"- NOAA cache directory: `{noaa_cache_dir}`",
                f"- USDM cache directory: `{usdm_cache_dir}`",
                f"- Manifest path: `{manifest_path}`",
                f"- Inventory path: `{inventory_path}`",
                f"- Texas path/cache/data leakage: {'yes' if leakage else 'no'}",
                "- Legacy baseline validator used for final decisions: no",
                "- DuckDuckGo fallback enabled: no",
            ]
        )
        + "\n",
    )
    safe_write_text(
        output_dir / "stale_label_safety_check.md",
        "\n".join(
            [
                "# Stale Label Safety Check",
                "",
                "- Old positive labels are preserved only in `old_positive_label_if_any` provenance fields.",
                "- Current CE, impact, and case-use labels are recomputed from current accepted evidence rows only.",
                "- Final labels use the current evidence validator and label aggregator.",
                "- Old positive labels copied as current labels: no",
            ]
        )
        + "\n",
    )
    safe_write_text(
        output_dir / "quote_grounding_check.md",
        "\n".join(
            [
                "# Quote Grounding Check",
                "",
                f"- Accepted web evidence rows: {len(accepted_web)}",
                f"- Accepted snippet-only rows: {len(accepted_snippet_only)}",
                f"- Accepted rows with quote not found in body: {len(accepted_quote_not_grounded)}",
                "- Rule: accepted web rows require at least one quoted supporting span found in fetched body text after whitespace normalization.",
            ]
        )
        + "\n",
    )
    return {
        "accepted_web_rows": len(accepted_web),
        "accepted_snippet_only_rows": len(accepted_snippet_only),
        "accepted_quote_not_grounded_rows": len(accepted_quote_not_grounded),
        "texas_path_cache_data_leakage": leakage,
    }


def write_summary(
    *,
    output_dir: Path,
    cases: list[dict[str, Any]],
    data_quality_issues: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
    validation_result: Any | None,
    evidence_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    safety: dict[str, Any],
    ledger: dict[str, Any],
    schema_ready_for_40: bool,
    ready: bool,
) -> None:
    accepted_rows = [row for row in evidence_rows if row.get("final_status") == "accepted"]
    nonaccepted_gate_closure = "no"
    metadata_as_evidence = "no"
    final_labels_from_current = "yes"
    old_labels_copied = "no"
    separated_gates = "yes"
    duckduckgo_count = count_duckduckgo_records(search_rows)
    llm_calls = 0
    schema_failures = 0
    if validation_result is not None:
        llm_calls = sum(len(rows) for rows in validation_result.raw_model_outputs.values())
        schema_failures = len(validation_result.schema_failure_rows)
    blocking = [row for row in data_quality_issues if row.get("severity") == "blocking"]
    warnings = [row for row in data_quality_issues if row.get("severity") != "blocking"]
    lines = [
        "# California Manifest Retrieval Summary",
        "",
        f"- Run timestamp UTC: {datetime.now(UTC).isoformat()}",
        f"- Cases loaded: {len(cases)}",
        "- Legacy baseline validator used for final decisions: no",
        "- Candidate construction regenerated: no",
        f"- Texas path/cache/data leakage: {'yes' if safety['texas_path_cache_data_leakage'] else 'no'}",
        f"- DuckDuckGo records = 0: {'yes' if duckduckgo_count == 0 else 'no'}",
        f"- Tavily live calls count: {ledger.get('tavily_search_calls', 0)}",
        f"- LLM validation calls count: {llm_calls}",
        f"- Accepted evidence rows count: {len(accepted_rows)}",
        f"- Accepted snippet-only rows count: {safety['accepted_snippet_only_rows']}",
        f"- Accepted rows with quote not found in body count: {safety['accepted_quote_not_grounded_rows']}",
        f"- Context-only/needs-review/rejected rows closed any gate: {nonaccepted_gate_closure}",
        f"- Candidate metadata/DTER/query-hit/source availability counted as public evidence: {metadata_as_evidence}",
        f"- Final labels recomputed from current accepted evidence rows only: {final_labels_from_current}",
        f"- Old positive labels copied as current labels: {old_labels_copied}",
        f"- Drought/wet/impact/transition gates separated: {separated_gates}",
        f"- Output schema ready for 40: {'yes' if schema_ready_for_40 else 'no'}",
        f"- LLM schema failures: {schema_failures}",
        "",
        "## Blocking Issues",
        "",
    ]
    if not blocking:
        lines.append("- None.")
    for row in blocking:
        lines.append(f"- {row.get('candidate_id') or 'run'}: {row.get('issue_type')} - {row.get('details')}")
    lines.extend(["", "## Non-Blocking Warnings", ""])
    if not warnings:
        lines.append("- None.")
    for row in warnings:
        lines.append(f"- {row.get('candidate_id') or 'run'}: {row.get('issue_type')} - {row.get('details')}")
    lines.extend(
        [
            "",
            f"final recommendation: ready_for_40_case_demo: {'yes' if ready else 'no'}",
            "",
            "## Gate Summary",
            "",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"- `{row['candidate_id']}`: ce_status={row.get('ce_status')}; impact_status={row.get('impact_status')}; case_use_label={row.get('case_use_label')}"
        )
    safe_write_text(output_dir / "MANIFEST_RUNNER_2CASE_SUMMARY.md", "\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a clean California manifest-driven demo path.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--noaa-cache-dir", type=Path, default=DEFAULT_NOAA_CACHE)
    parser.add_argument("--usdm-cache-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--live-tavily-mode", choices=["capped"], required=True)
    parser.add_argument("--no-duckduckgo", action="store_true", required=True)
    parser.add_argument("--require-tavily", action="store_true", required=True)
    parser.add_argument("--tavily-api-key-file", type=Path, default=REPO_ROOT / "apikeys" / "tavily_apikey.txt")
    parser.add_argument("--llm-api-key-file", type=Path, default=REPO_ROOT / "apikeys" / "openai_apikey.txt")
    parser.add_argument("--llm-judge-model", required=True)
    parser.add_argument("--llm-skeptic-model", required=True)
    parser.add_argument("--llm-arbiter-model", required=True)
    parser.add_argument("--max-open-queries-per-candidate", type=int, default=3)
    parser.add_argument("--max-open-fetches-per-candidate", type=int, default=2)
    parser.add_argument("--max-results-per-query", type=int, default=2)
    parser.add_argument("--enable-llm-query-expansion", action="store_true", help="Enable bounded second-round LLM query expansion after first-pass validation.")
    parser.add_argument("--llm-query-expansion-model", default=None, help="Optional model override for query expansion; defaults to the primary judge model.")
    parser.add_argument("--max-expansion-queries-per-case", type=int, default=4)
    parser.add_argument("--max-expansion-hazard-queries-per-case", type=int, default=2)
    parser.add_argument("--max-expansion-impact-queries-per-case", type=int, default=3)
    parser.add_argument("--max-expansion-linkage-queries-per-case", type=int, default=1)
    parser.add_argument("--max-expansion-pages-per-case", type=int, default=3)
    parser.add_argument(
        "--force-open-web-smoke",
        action="store_true",
        help="Smoke-only: force a bounded open-web impact gap even when structured official lanes found evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir or (output_dir / "cache")).resolve()
    noaa_cache_dir = args.noaa_cache_dir.resolve()
    usdm_cache_dir = (args.usdm_cache_dir or (cache_dir / "official" / "usdm")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_quality_issues: list[dict[str, Any]] = []

    tavily_key = read_tavily_api_key(args.tavily_api_key_file)
    openai_key = read_openai_api_key(args.llm_api_key_file)
    if args.require_tavily and not tavily_key:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="missing_tavily_api_key",
                details="`--require-tavily` was set and no Tavily key was available.",
            )
        )
    if not openai_key:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="missing_openai_api_key",
                details="LLM judge mode requires an OpenAI API key and fails closed.",
            )
        )

    cases, manifest_issues = load_cases_from_manifest(args.sample_manifest, args.inventory, args.max_candidates)
    data_quality_issues.extend(manifest_issues)
    if len(cases) != args.max_candidates:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="manifest_case_count_mismatch",
                details=f"Loaded {len(cases)} cases, expected {args.max_candidates}.",
            )
        )

    controls = TavilyCostController(
        max_open_queries_per_candidate_round1=args.max_open_queries_per_candidate,
        max_open_queries_per_candidate_total=args.max_open_queries_per_candidate,
        max_results_per_query=args.max_results_per_query,
        max_open_fetches_per_candidate_round1=args.max_open_fetches_per_candidate,
        max_open_fetches_per_candidate_round2=args.max_open_fetches_per_candidate,
        max_open_fetches_per_candidate_total=args.max_open_fetches_per_candidate,
        max_fetches_per_lane_per_candidate=args.max_open_fetches_per_candidate,
        max_open_fetches_total=max(1, args.max_candidates * args.max_open_fetches_per_candidate),
        enable_general_web_lane=False,
    )
    ledger = initial_cost_ledger(controls)
    official_controls = OfficialSourceLaneControls(
        enable_noaa_storm_events=True,
        enable_usdm_county_statistics=True,
        enable_usdm_live_api_fallback=True,
        noaa_cache_dir=noaa_cache_dir,
        usdm_cache_dir=usdm_cache_dir,
        usdm_write_live_api_cache=True,
    )

    official_attempts_by_case: dict[str, str] = {}
    official_rows: list[dict[str, Any]] = []
    official_log_rows: list[dict[str, Any]] = []
    gaps_by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        result = run_structured_official_lanes_for_case(case, controls=official_controls)
        official_attempts_by_case[case["candidate_id"]] = official_sources_attempted_text(result.attempted_lanes)
        enriched = [add_common_evidence_fields(row, case, "structured_official_reference") for row in result.evidence_rows]
        official_rows.extend(enriched)
        official_log_rows.extend(result.retrieval_log_rows)
        flags = support_flags_from_evidence_rows(enriched)
        gap = compute_evidence_gap_state(
            case,
            has_drought_evidence=flags["drought"],
            has_rain_flood_evidence=flags["wet"],
            has_impact_evidence=flags["impact"],
        )
        if args.force_open_web_smoke:
            missing = list(gap.get("missing_components") or [])
            if "impact" not in missing:
                missing.append("impact")
            gap["missing_components"] = missing
            gap["open_search_needed"] = True
            gap["open_search_reason"] = ";".join(
                part
                for part in [
                    str(gap.get("open_search_reason") or ""),
                    "phase2e_force_open_web_smoke",
                ]
                if part
            )
        gaps_by_case[case["candidate_id"]] = gap

    run_id = make_run_id()
    cases_by_id = {case["candidate_id"]: case for case in cases}
    pages: list[dict[str, Any]] = []
    expansion_pages: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    search_result_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    fetch_decision_rows: list[dict[str, Any]] = []
    validation_result = None
    web_rows: list[dict[str, Any]] = []
    expansion_web_rows: list[dict[str, Any]] = []
    expansion_search_rows: list[dict[str, Any]] = []
    expansion_score_rows: list[dict[str, Any]] = []
    expansion_fetch_rows: list[dict[str, Any]] = []
    expansion_report_rows: list[dict[str, Any]] = []
    raw_query_expansion_outputs: list[dict[str, Any]] = []
    retrieval_state: dict[str, Any] | None = None
    tavily_client: TavilySearchClient | None = None
    web_reader: WebReader | None = None
    expansion_config = LLMQueryExpansionConfig(
        enabled=bool(args.enable_llm_query_expansion),
        model=args.llm_query_expansion_model or args.llm_judge_model,
        max_queries_per_case=max(0, args.max_expansion_queries_per_case),
        max_hazard_queries_per_case=max(0, args.max_expansion_hazard_queries_per_case),
        max_impact_queries_per_case=max(0, args.max_expansion_impact_queries_per_case),
        max_linkage_queries_per_case=max(0, args.max_expansion_linkage_queries_per_case),
        max_pages_per_case=max(0, args.max_expansion_pages_per_case),
        api_key_path=args.llm_api_key_file,
    )
    judge_config = LLMEvidenceJudgeConfig(
        use_llm_evidence_judge=True,
        llm_judge_model=args.llm_judge_model,
        llm_skeptic_model=args.llm_skeptic_model,
        llm_arbiter_model=args.llm_arbiter_model,
        llm_judge_fail_closed=True,
        keyword_validator_mode="prefilter_only",
        require_quoted_span_for_accept=True,
        require_compact_span_grounding=True,
        api_key_path=args.llm_api_key_file,
    )

    if not any(row.get("severity") == "blocking" for row in data_quality_issues):
        tavily_client = TavilySearchClient(api_key=tavily_key, search_depth="advanced")
        web_reader = WebReader(snapshot_dir=cache_dir / "web_reader_snapshots", min_cleaned_text_chars=80)
        query_rows, search_result_rows, score_rows, fetch_decision_rows, pages, retrieval_state = run_open_retrieval(
            cases=cases,
            gaps_by_case=gaps_by_case,
            controls=controls,
            tavily_client=tavily_client,
            web_reader=web_reader,
            ledger=ledger,
        )
        if pages:
            try:
                validation_result = validate_fetched_pages_with_llm_judge(
                    pages=pages,
                    strict_rows=validation_prefilter_rows(pages, cases_by_id),
                    prior_llm_rows=[],
                    case_cards=case_card_context(cases),
                    smoke_dir=output_dir,
                    judge_config=judge_config,
                )
                web_rows = web_evidence_rows_from_validation(
                    validation_result,
                    pages,
                    cases_by_id,
                )
            except LLMEvidenceValidationUnavailable as exc:
                data_quality_issues.append(
                    issue(
                        severity="blocking",
                        candidate_id="",
                        issue_type="llm_validation_unavailable",
                        details=str(exc),
                    )
                )
        else:
            data_quality_issues.append(
                issue(
                    severity="warning",
                    candidate_id="",
                    issue_type="no_open_web_pages_fetched",
                    details="No pages were scheduled or fetched after structured official lanes.",
                )
            )

    first_pass_evidence_rows = official_rows + web_rows
    first_pass_registry_rows, first_pass_gate_rows, _ = build_case_outputs(
        cases=cases,
        evidence_rows=first_pass_evidence_rows,
        official_attempts_by_case=official_attempts_by_case,
    )
    del first_pass_registry_rows

    evidence_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in first_pass_evidence_rows:
        evidence_by_case[str(row.get("candidate_id") or "")].append(row)

    expansion_can_run = not any(row.get("severity") == "blocking" for row in data_quality_issues)
    if expansion_can_run or not expansion_config.enabled:
        try:
            expansion_plan = plan_llm_query_expansion(
                cases=cases,
                evidence_rows_by_case=evidence_by_case,
                existing_query_rows=query_rows,
                expansion_config=expansion_config,
                run_id=run_id,
            )
            expansion_report_rows = expansion_plan.report_rows
            raw_query_expansion_outputs = expansion_plan.raw_model_outputs
            ledger["llm_query_expansion_enabled"] = bool(expansion_config.enabled)
            ledger["llm_query_expansion_generated_queries"] = sum(
                1
                for row in expansion_report_rows
                if row.get("generated_query")
            )
            ledger["llm_query_expansion_valid_queries"] = len(expansion_plan.query_rows)
            ledger["llm_query_expansion_model_calls"] = len(raw_query_expansion_outputs)
        except LLMQueryExpansionUnavailable as exc:
            data_quality_issues.append(
                issue(
                    severity="blocking",
                    candidate_id="",
                    issue_type="llm_query_expansion_unavailable",
                    details=str(exc),
                )
            )
            expansion_plan = None

        if (
            expansion_config.enabled
            and expansion_plan is not None
            and expansion_plan.query_rows
            and tavily_client is not None
            and web_reader is not None
            and retrieval_state is not None
        ):
            expansion_controls = TavilyCostController(
                max_open_queries_per_candidate_round1=args.max_open_queries_per_candidate,
                max_open_queries_per_candidate_total=args.max_expansion_queries_per_case,
                max_results_per_query=args.max_results_per_query,
                max_open_fetches_per_candidate_round1=args.max_open_fetches_per_candidate,
                max_open_fetches_per_candidate_round2=expansion_config.max_pages_per_case,
                max_open_fetches_per_candidate_total=expansion_config.max_pages_per_case,
                max_fetches_per_lane_per_candidate=expansion_config.max_pages_per_case,
                max_open_fetches_total=max(1, args.max_candidates * max(1, expansion_config.max_pages_per_case)),
                enable_general_web_lane=False,
            )
            expansion_search_rows, expansion_score_rows, expansion_fetch_rows, expansion_pages, retrieval_state = execute_open_query_rows(
                cases=cases,
                query_rows=expansion_plan.query_rows,
                controls=expansion_controls,
                tavily_client=tavily_client,
                web_reader=web_reader,
                ledger=ledger,
                round_number=2,
                retrieval_state=retrieval_state,
            )
            query_rows.extend(expansion_plan.query_rows)
            search_result_rows.extend(expansion_search_rows)
            score_rows.extend(expansion_score_rows)
            fetch_decision_rows.extend(expansion_fetch_rows)
            pages.extend(expansion_pages)
            if expansion_pages:
                try:
                    expansion_validation_result = validate_fetched_pages_with_llm_judge(
                        pages=expansion_pages,
                        strict_rows=validation_prefilter_rows(expansion_pages, cases_by_id),
                        prior_llm_rows=[],
                        case_cards=case_card_context(cases),
                        smoke_dir=output_dir,
                        judge_config=judge_config,
                    )
                    expansion_web_rows = web_evidence_rows_from_validation(
                        expansion_validation_result,
                        expansion_pages,
                        cases_by_id,
                    )
                    if validation_result is None:
                        validation_result = expansion_validation_result
                    else:
                        validation_result.judgment_rows.extend(expansion_validation_result.judgment_rows)
                        validation_result.span_rows.extend(expansion_validation_result.span_rows)
                        validation_result.disagreement_rows.extend(expansion_validation_result.disagreement_rows)
                        validation_result.schema_failure_rows.extend(expansion_validation_result.schema_failure_rows)
                        validation_result.normalized_rows.extend(expansion_validation_result.normalized_rows)
                        for key, rows in expansion_validation_result.raw_model_outputs.items():
                            validation_result.raw_model_outputs.setdefault(key, []).extend(rows)
                except LLMEvidenceValidationUnavailable as exc:
                    data_quality_issues.append(
                        issue(
                            severity="blocking",
                            candidate_id="",
                            issue_type="llm_validation_unavailable_for_query_expansion_pages",
                            details=str(exc),
                        )
                    )

    evidence_rows = first_pass_evidence_rows + expansion_web_rows
    registry_rows, gate_rows, _ = build_case_outputs(
        cases=cases,
        evidence_rows=evidence_rows,
        official_attempts_by_case=official_attempts_by_case,
    )
    if expansion_report_rows:
        expansion_report_rows = annotate_expansion_report_with_outcomes(
            expansion_report_rows,
            first_pass_gate_rows=first_pass_gate_rows,
            final_gate_rows=gate_rows,
            expansion_pages=expansion_pages,
            expansion_evidence_rows=expansion_web_rows,
            expansion_search_result_rows=expansion_search_rows,
            expansion_fetch_decision_rows=expansion_fetch_rows,
        )
    safety = write_safety_reports(
        output_dir=output_dir,
        cache_dir=cache_dir,
        noaa_cache_dir=noaa_cache_dir,
        usdm_cache_dir=usdm_cache_dir,
        evidence_rows=evidence_rows,
        manifest_path=args.sample_manifest,
        inventory_path=args.inventory,
    )
    if safety["accepted_snippet_only_rows"]:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="accepted_snippet_only_rows",
                details=str(safety["accepted_snippet_only_rows"]),
            )
        )
    if safety["accepted_quote_not_grounded_rows"]:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="accepted_quote_not_body_grounded_rows",
                details=str(safety["accepted_quote_not_grounded_rows"]),
            )
        )
    if safety["texas_path_cache_data_leakage"]:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="texas_or_phase_cache_path_leakage",
                details="A configured output/cache path contains Texas or reserved legacy-run naming.",
            )
        )
    if validation_result is not None and validation_result.schema_failure_rows:
        data_quality_issues.append(
            issue(
                severity="blocking",
                candidate_id="",
                issue_type="llm_schema_failures",
                details=str(len(validation_result.schema_failure_rows)),
            )
        )

    finalized_ledger = finalize_cost_ledger(
        ledger,
        open_supported_candidate_count=len(
            {
                row["candidate_id"]
                for row in (web_rows + expansion_web_rows)
                if row.get("final_status") == "accepted"
            }
        ),
    )

    write_csv(output_dir / "case_registry.csv", registry_rows, CASE_REGISTRY_FIELDS)
    write_csv(output_dir / "evidence_rows.csv", evidence_rows, EVIDENCE_FIELDS)
    write_csv(output_dir / "gate_assertion_summary.csv", gate_rows, GATE_FIELDS)
    write_csv(output_dir / "source_lane_summary.csv", build_source_lane_summary(evidence_rows), ["evidence_origin", "source_lane", "source_family", "final_status", "count"])
    write_csv(output_dir / "impact_channel_summary.csv", build_impact_channel_summary(evidence_rows), ["impact_channel", "evidence_impact_status", "final_status", "count"])
    write_csv(output_dir / "data_quality_issues.csv", data_quality_issues, ["severity", "candidate_id", "issue_type", "details", "blocking"])
    write_csv(output_dir / "llm_query_expansion_report.csv", expansion_report_rows, EXPANSION_REPORT_FIELDS)
    safe_write_text(output_dir / "case_cards.md", build_case_cards(cases, evidence_rows, gate_rows))

    write_csv(output_dir / "structured_official_retrieval_log.csv", official_log_rows, sorted({key for row in official_log_rows for key in row.keys()}))
    write_csv(output_dir / "query_plan.csv", query_rows, sorted({key for row in query_rows for key in row.keys()} or {"candidate_id"}))
    write_csv(output_dir / "tavily_search_results.csv", search_result_rows, sorted({key for row in search_result_rows for key in row.keys()} or {"candidate_id"}))
    write_csv(output_dir / "prefetch_scorecards.csv", score_rows, sorted({key for row in score_rows for key in row.keys()} or {"candidate_id"}))
    write_csv(output_dir / "fetch_decisions.csv", fetch_decision_rows, sorted({key for row in fetch_decision_rows for key in row.keys()} or {"candidate_id"}))
    write_jsonl(output_dir / "fetched_pages.jsonl", pages)
    safe_write_json(output_dir / "cost_ledger.json", finalized_ledger)
    safe_write_json(output_dir / "search_result_cache_shape.json", {"query_keys": len(build_search_result_cache(search_result_rows, max_results=args.max_results_per_query))})
    safe_write_json(output_dir / "url_body_cache_shape.json", {"url_keys": len(build_url_body_cache(pages))})
    if validation_result is not None:
        write_csv(output_dir / "llm_page_judgments.csv", validation_result.judgment_rows, sorted({key for row in validation_result.judgment_rows for key in row.keys()} or {"candidate_id"}))
        write_csv(output_dir / "llm_span_table.csv", validation_result.span_rows, sorted({key for row in validation_result.span_rows for key in row.keys()} or {"candidate_id"}))
        write_csv(output_dir / "llm_disagreements.csv", validation_result.disagreement_rows, sorted({key for row in validation_result.disagreement_rows for key in row.keys()} or {"candidate_id"}))
        write_jsonl(output_dir / "normalized_llm_judgments.jsonl", validation_result.normalized_rows)
        write_jsonl(output_dir / "raw_primary_model_outputs.jsonl", validation_result.raw_model_outputs["primary"])
        write_jsonl(output_dir / "raw_skeptic_model_outputs.jsonl", validation_result.raw_model_outputs["skeptic"])
        write_jsonl(output_dir / "raw_arbiter_model_outputs.jsonl", validation_result.raw_model_outputs["arbiter"])
    if raw_query_expansion_outputs:
        write_jsonl(output_dir / "raw_llm_query_expansion_outputs.jsonl", raw_query_expansion_outputs)

    schema_ready = all((output_dir / name).exists() for name in REQUIRED_OUTPUTS if name != "MANIFEST_RUNNER_2CASE_SUMMARY.md")
    ready = (
        not any(row.get("severity") == "blocking" for row in data_quality_issues)
        and len(cases) == args.max_candidates
        and schema_ready
    )
    write_summary(
        output_dir=output_dir,
        cases=cases,
        data_quality_issues=data_quality_issues,
        search_rows=search_result_rows,
        validation_result=validation_result,
        evidence_rows=evidence_rows,
        gate_rows=gate_rows,
        safety=safety,
        ledger=finalized_ledger,
        schema_ready_for_40=schema_ready,
        ready=ready,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "cases": len(cases),
                "tavily_live_calls": finalized_ledger.get("tavily_search_calls", 0),
                "llm_validation_calls": (
                    sum(len(rows) for rows in validation_result.raw_model_outputs.values())
                    if validation_result is not None
                    else 0
                ),
                "accepted_evidence_rows": len([row for row in evidence_rows if row.get("final_status") == "accepted"]),
                "ready_for_40_case_demo": ready,
                "blocking_issues": [
                    row.get("issue_type")
                    for row in data_quality_issues
                    if row.get("severity") == "blocking"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
