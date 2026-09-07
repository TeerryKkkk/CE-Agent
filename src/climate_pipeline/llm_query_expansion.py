from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from . import config
from .ce_impact_labeling import classify_evidence_impact
from .controlled_open_retrieval import (
    _near_duplicate_key,
    normalize_county,
    normalize_query,
    stable_query_fingerprint,
)
from .llm_evidence_judge import JudgeSchemaError, ModelCallResult, OpenAIEvidenceJudgeClient, read_openai_api_key


FOUND_BY = "llm_query_expansion"
QUERY_SOURCE = "llm_expansion"
PROMPT_VERSION = "llm_query_expansion_v1"

QUERY_INTENT_HAZARD = "hazard_public_corroboration"
QUERY_INTENT_IMPACT = "impact_enrichment"
QUERY_INTENT_LINKAGE = "linkage_search"

RESEARCH_ONLY_TERMS = (
    "spei",
    "dter",
    "p99",
    "rawce",
    "ce-agent",
    "ce agent",
    "compound event",
    "compound-event",
    "candidate event",
    "candidate-event",
)

RESTRICTED_NON_LINKAGE_QUERY_TERMS = (
    "drought-to-flood transition",
    "drought to flood transition",
    "drought-to-wet transition",
    "drought to wet transition",
    "linkage",
)

PREFERRED_IMPACT_QUERY_TERMS = (
    "flooding",
    "flash flood",
    "storm damage",
    "road closure",
    "high water",
    "washout",
    "flood damage",
    "crop loss",
    "drought crop loss",
    "drought disaster",
    "water restrictions",
    "irrigation",
    "livestock drought",
    "power outage",
    "utility outage",
    "emergency management",
    "evacuation",
    "rescue",
    "public works",
    "local government",
    "transportation agency",
    "agriculture agency",
    "local news",
)

PREFERRED_LINKAGE_QUERY_TERMS = (
    "drought followed by heavy rain",
    "drought followed by flooding",
    "after months of drought flooding",
    "drought then storm damage",
    "drought conditions before flooding",
)

HAZARD_OR_IMPACT_TERMS = (
    "drought",
    "dry",
    "water restriction",
    "water restrictions",
    "water shortage",
    "crop loss",
    "livestock",
    "flood",
    "flooding",
    "flash flood",
    "rain",
    "rainfall",
    "storm",
    "atmospheric river",
    "high water",
    "debris flow",
    "mudslide",
    "road closure",
    "road closures",
    "storm damage",
    "flood damage",
    "washout",
    "washed out",
    "damage",
    "evacuation",
    "rescue",
    "emergency",
    "declaration",
    "power outage",
    "utility outage",
    "utility",
    "irrigation",
    "local government",
    "transportation agency",
    "agriculture agency",
    "local news",
    "public works",
    "caltrans",
)

APPROVED_SOURCE_ANCHORS = (
    "caltrans",
    "cal oes",
    "caloes",
    "dwr",
    "department of water resources",
    "sheriff",
    "oes",
    "emergency management",
    "public works",
    "flood control",
    "agricultural commissioner",
    "county",
)

COMMON_ALLOWED_PROPER_NOUNS = {
    "California",
    "CA",
    "Caltrans",
    "Cal OES",
    "CalOES",
    "DWR",
    "FEMA",
    "OpenFEMA",
    "NOAA",
    "NCEI",
    "USDM",
    "USDA",
    "RMA",
    "National Weather Service",
    "NWS",
}

MONTH_NAMES = (
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
)

SEASON_NAMES = {"Winter", "Spring", "Summer", "Fall", "Autumn"}

EXPANSION_REPORT_FIELDS = [
    "run_id",
    "candidate_id",
    "county",
    "state",
    "expansion_enabled",
    "candidate_triggered",
    "retrieval_round",
    "target_gap",
    "query_intent",
    "expansion_trigger",
    "evidence_snapshot_hash",
    "llm_model",
    "prompt_version",
    "generated_query",
    "normalized_generated_query",
    "validated_query",
    "query_id",
    "query_source",
    "found_by",
    "validation_status",
    "validation_reasons",
    "location_anchor_ok",
    "state_anchor_ok",
    "time_anchor_ok",
    "hazard_or_impact_anchor_ok",
    "duplicate_fixed_query",
    "research_only_terms_found",
    "restricted_public_query_terms",
    "ungrounded_proper_nouns",
    "broad_state_only",
    "accepted_for_search",
    "planned_lane",
    "search_backend",
    "expansion_search_results_returned",
    "expansion_unique_result_urls",
    "expansion_result_urls_deduplicated",
    "expansion_fetch_candidates_considered",
    "expansion_pages_skipped_before_fetch",
    "expansion_fetch_skip_reasons",
    "generated_query_count",
    "valid_query_count",
    "rejected_query_count",
    "expansion_pages_fetched",
    "expansion_accepted_rows",
    "expansion_needs_review_rows",
    "expansion_context_only_rows",
    "expansion_rejected_rows",
    "expansion_hazard_support_found",
    "expansion_impact_support_found",
    "expansion_linkage_support_found",
    "first_pass_ce_status",
    "first_pass_impact_status",
    "first_pass_case_use_label",
    "first_pass_material_impact_pattern",
    "final_ce_status",
    "final_impact_status",
    "final_case_use_label",
    "final_material_impact_pattern",
    "case_label_changed",
    "changed_label_supported_by_accepted_evidence",
    "created_at",
]


class QueryExpansionGenerator(Protocol):
    def generate_queries(self, payload: Mapping[str, Any]) -> ModelCallResult:
        ...


@dataclass(frozen=True)
class LLMQueryExpansionConfig:
    enabled: bool = False
    model: str = config.OPENAI_MODEL
    max_queries_per_case: int = 3
    max_hazard_queries_per_case: int = 2
    max_impact_queries_per_case: int = 2
    max_linkage_queries_per_case: int = 1
    max_pages_per_case: int = 1
    api_key_path: Path | None = None
    prompt_version: str = PROMPT_VERSION
    fail_closed: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key_path"] = str(self.api_key_path) if self.api_key_path else ""
        return data


@dataclass(frozen=True)
class ExpansionTrigger:
    target_gap: str
    query_intent: str
    expansion_trigger: str
    max_queries: int


@dataclass(frozen=True)
class QueryValidationResult:
    accepted: bool
    validated_query: str
    reasons: tuple[str, ...]
    location_anchor_ok: bool
    state_anchor_ok: bool
    time_anchor_ok: bool
    hazard_or_impact_anchor_ok: bool
    duplicate_fixed_query: bool
    research_only_terms_found: tuple[str, ...]
    restricted_public_query_terms: tuple[str, ...]
    ungrounded_proper_nouns: tuple[str, ...]
    broad_state_only: bool


@dataclass
class QueryExpansionPlan:
    query_rows: list[dict[str, Any]]
    report_rows: list[dict[str, Any]]
    raw_model_outputs: list[dict[str, Any]]
    triggers_by_case: dict[str, list[ExpansionTrigger]]


class LLMQueryExpansionUnavailable(RuntimeError):
    pass


class OpenAIQueryExpansionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        responses_url: str | None = None,
        timeout_seconds: int | None = None,
        max_attempts: int = 2,
        max_input_tokens: int = 32_768,
        max_output_tokens: int = 2_048,
        max_request_bytes: int = 262_144,
        max_response_bytes: int = 262_144,
        max_error_response_bytes: int = 65_536,
        max_redirects: int = 0,
        max_redirect_response_bytes: int = 16_384,
        max_header_bytes: int = 65_536,
        max_redirect_location_bytes: int = 4_096,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("query_expansion_max_attempts_must_be_positive")
        self.max_attempts = max_attempts
        self.client = OpenAIEvidenceJudgeClient(
            api_key=api_key,
            model=model or config.OPENAI_MODEL,
            responses_url=responses_url,
            timeout_seconds=timeout_seconds,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
            max_error_response_bytes=max_error_response_bytes,
            max_redirects=max_redirects,
            max_redirect_response_bytes=max_redirect_response_bytes,
            max_header_bytes=max_header_bytes,
            max_redirect_location_bytes=max_redirect_location_bytes,
        )

    def generate_queries(self, payload: Mapping[str, Any]) -> ModelCallResult:
        instructions = (
            "You are a constrained search-query assistant for a climate/disaster evidence pipeline. "
            "Generate only additional web search queries. Do not decide hazard, impact, linkage, or case labels. "
            "Use ordinary public-source search language, not research/internal phrasing. "
            "For impact enrichment, prefer concrete terms such as flooding, flash flood, storm damage, road closure, "
            "high water, washout, flood damage, crop loss, drought crop loss, water restrictions, irrigation, "
            "livestock drought, power outage, utility outage, emergency management, evacuation, rescue, public works, "
            "local government, transportation agency, agriculture agency, and local news. "
            "Avoid SPEI, DTER, p99, rawce, CE-Agent, compound event, candidate event, drought-to-flood transition, "
            "drought-to-wet transition, transition, and linkage in ordinary hazard or impact queries. "
            "Only optional linkage-search queries may use linkage-oriented ideas, and even then prefer public wording such as "
            "drought followed by heavy rain, drought followed by flooding, after months of drought flooding, "
            "drought then storm damage, or drought conditions before flooding. "
            "Do not invent storm names, roads, casualties, damages, crop losses, dollar losses, disaster IDs, "
            "declarations, or agency actions unless they are already present in the supplied metadata or accepted evidence. "
            "Each query must be bounded by local place, candidate time window, and a hazard or impact term. "
            "Return JSON only."
        )
        return self.client._call_json(
            role="llm_query_expansion",
            instructions=instructions,
            user_payload=dict(payload),
            schema=QUERY_EXPANSION_SCHEMA,
            schema_name="llm_query_expansion",
            validator=validate_query_expansion_payload,
            max_attempts=self.max_attempts,
        )


QUERY_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string"},
                    "intended_purpose": {"type": "string"},
                    "target_gap": {"type": "string"},
                    "location_anchor_used": {"type": "string"},
                    "time_anchor_used": {"type": "string"},
                    "hazard_or_impact_anchor_used": {"type": "string"},
                    "risk_note": {"type": "string"},
                },
                "required": [
                    "query_text",
                    "intended_purpose",
                    "target_gap",
                    "location_anchor_used",
                    "time_anchor_used",
                    "hazard_or_impact_anchor_used",
                    "risk_note",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}


def validate_query_expansion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JudgeSchemaError("query expansion response must be an object")
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise JudgeSchemaError("query expansion response must contain a queries array")
    normalized_queries: list[dict[str, str]] = []
    for item in queries:
        if not isinstance(item, dict):
            raise JudgeSchemaError("query expansion item must be an object")
        query_text = str(item.get("query_text") or "").strip()
        if not query_text:
            continue
        normalized_queries.append(
            {
                "query_text": query_text,
                "intended_purpose": str(item.get("intended_purpose") or "").strip(),
                "target_gap": str(item.get("target_gap") or "").strip(),
                "location_anchor_used": str(item.get("location_anchor_used") or "").strip(),
                "time_anchor_used": str(item.get("time_anchor_used") or "").strip(),
                "hazard_or_impact_anchor_used": str(item.get("hazard_or_impact_anchor_used") or "").strip(),
                "risk_note": str(item.get("risk_note") or "").strip(),
            }
        )
    return {"queries": normalized_queries}


def make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def evidence_snapshot_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {
            "candidate_id": _text(row, "candidate_id"),
            "source_url": _text(row, "source_url"),
            "source_lane": _text(row, "source_lane"),
            "final_status": _text(row, "final_status"),
            "component_supported": _text(row, "component_supported"),
            "impact_supported": _text(row, "impact_supported"),
            "explicit_transition_support": _text(row, "explicit_transition_support"),
        }
        for row in rows
    ]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def diagnose_expansion_gaps(
    case: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    expansion_config: LLMQueryExpansionConfig,
) -> list[ExpansionTrigger]:
    rows = list(evidence_rows)
    triggers: list[ExpansionTrigger] = []

    structured_drought = any(_is_structured_official(row) and _row_supports_component(row, "drought") for row in rows)
    structured_wet = any(_is_structured_official(row) and _row_supports_component(row, "wet") for row in rows)
    accepted_drought = any(_row_supports_component(row, "drought") for row in rows)
    accepted_wet = any(_row_supports_component(row, "wet") for row in rows)

    if expansion_config.max_hazard_queries_per_case > 0:
        if not structured_drought and not accepted_drought:
            triggers.append(
                ExpansionTrigger(
                    target_gap="drought_hazard_public_corroboration",
                    query_intent=QUERY_INTENT_HAZARD,
                    expansion_trigger="official_or_structured_drought_support_missing_after_first_pass",
                    max_queries=1,
                )
            )
        if not structured_wet and not accepted_wet:
            triggers.append(
                ExpansionTrigger(
                    target_gap="wet_hazard_public_corroboration",
                    query_intent=QUERY_INTENT_HAZARD,
                    expansion_trigger="official_or_structured_wet_support_missing_after_first_pass",
                    max_queries=1,
                )
            )

    impact_reasons = _impact_gap_reasons(rows)
    if expansion_config.max_impact_queries_per_case > 0 and impact_reasons:
        triggers.append(
            ExpansionTrigger(
                target_gap="impact_enrichment",
                query_intent=QUERY_INTENT_IMPACT,
                expansion_trigger=";".join(impact_reasons),
                max_queries=expansion_config.max_impact_queries_per_case,
            )
        )

    explicit_linkage = any(
        _is_accepted(row)
        and (_as_bool(row.get("explicit_transition_support")) or _as_bool(row.get("explicit_linkage_support")))
        for row in rows
    )
    if (
        expansion_config.max_linkage_queries_per_case > 0
        and accepted_drought
        and accepted_wet
        and not explicit_linkage
    ):
        triggers.append(
            ExpansionTrigger(
                target_gap="explicit_linkage_absent",
                query_intent=QUERY_INTENT_LINKAGE,
                expansion_trigger="both_hazard_legs_supported_but_explicit_transition_or_linkage_absent",
                max_queries=1,
            )
        )

    total_budget = max(0, expansion_config.max_queries_per_case)
    selected: list[ExpansionTrigger] = []
    used = 0
    hazard_used = 0
    for trigger in triggers:
        if used >= total_budget:
            break
        if trigger.query_intent == QUERY_INTENT_HAZARD:
            if hazard_used >= expansion_config.max_hazard_queries_per_case:
                continue
            allowed = min(trigger.max_queries, expansion_config.max_hazard_queries_per_case - hazard_used, total_budget - used)
            hazard_used += allowed
        elif trigger.query_intent == QUERY_INTENT_LINKAGE:
            allowed = min(trigger.max_queries, expansion_config.max_linkage_queries_per_case, total_budget - used)
        else:
            allowed = min(trigger.max_queries, expansion_config.max_impact_queries_per_case, total_budget - used)
        if allowed <= 0:
            continue
        selected.append(
            ExpansionTrigger(
                target_gap=trigger.target_gap,
                query_intent=trigger.query_intent,
                expansion_trigger=trigger.expansion_trigger,
                max_queries=allowed,
            )
        )
        used += allowed
    return selected


def plan_llm_query_expansion(
    *,
    cases: Iterable[Mapping[str, Any]],
    evidence_rows_by_case: Mapping[str, list[Mapping[str, Any]]],
    existing_query_rows: Iterable[Mapping[str, Any]],
    expansion_config: LLMQueryExpansionConfig,
    run_id: str | None = None,
    generator: QueryExpansionGenerator | None = None,
    generated_at: str | None = None,
) -> QueryExpansionPlan:
    run_id = run_id or make_run_id()
    generated_at = generated_at or datetime.now(UTC).isoformat()
    existing_rows = list(existing_query_rows)
    existing_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in existing_rows:
        existing_by_case.setdefault(_text(row, "candidate_id"), []).append(row)

    query_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    triggers_by_case: dict[str, list[ExpansionTrigger]] = {}

    if not expansion_config.enabled:
        for case in cases:
            cid = _candidate_id(case)
            report_rows.append(
                _base_report_row(
                    run_id=run_id,
                    case=case,
                    expansion_config=expansion_config,
                    evidence_rows=evidence_rows_by_case.get(cid, []),
                    generated_at=generated_at,
                    candidate_triggered=False,
                    validation_status="disabled",
                    validation_reasons="llm_query_expansion_disabled",
                )
            )
        return QueryExpansionPlan(query_rows, report_rows, raw_outputs, triggers_by_case)

    active_cases: list[Mapping[str, Any]] = []
    for case in cases:
        cid = _candidate_id(case)
        rows = evidence_rows_by_case.get(cid, [])
        triggers = diagnose_expansion_gaps(case, rows, expansion_config)
        triggers_by_case[cid] = triggers
        if triggers:
            active_cases.append(case)
        else:
            report_rows.append(
                _base_report_row(
                    run_id=run_id,
                    case=case,
                    expansion_config=expansion_config,
                    evidence_rows=rows,
                    generated_at=generated_at,
                    candidate_triggered=False,
                    validation_status="no_trigger",
                    validation_reasons="first_pass_evidence_has_no_clear_expansion_gap",
                )
            )

    if not active_cases:
        return QueryExpansionPlan(query_rows, report_rows, raw_outputs, triggers_by_case)

    if generator is None:
        api_key = read_openai_api_key(expansion_config.api_key_path)
        if not api_key:
            if expansion_config.fail_closed:
                raise LLMQueryExpansionUnavailable("missing_openai_api_key_for_query_expansion")
            for case in active_cases:
                cid = _candidate_id(case)
                report_rows.append(
                    _base_report_row(
                        run_id=run_id,
                        case=case,
                        expansion_config=expansion_config,
                        evidence_rows=evidence_rows_by_case.get(cid, []),
                        generated_at=generated_at,
                        candidate_triggered=True,
                        validation_status="generation_unavailable",
                        validation_reasons="missing_openai_api_key_for_query_expansion",
                    )
                )
            return QueryExpansionPlan(query_rows, report_rows, raw_outputs, triggers_by_case)
        generator = OpenAIQueryExpansionClient(api_key=api_key, model=expansion_config.model)

    for case in active_cases:
        cid = _candidate_id(case)
        rows = evidence_rows_by_case.get(cid, [])
        triggers = triggers_by_case.get(cid, [])
        payload = build_query_expansion_payload(
            case=case,
            evidence_rows=rows,
            existing_query_rows=existing_by_case.get(cid, []),
            triggers=triggers,
            expansion_config=expansion_config,
        )
        call = generator.generate_queries(payload)
        raw_outputs.append(_raw_generation_record(call, cid))
        if call.parsed is None:
            report_rows.append(
                _base_report_row(
                    run_id=run_id,
                    case=case,
                    expansion_config=expansion_config,
                    evidence_rows=rows,
                    generated_at=generated_at,
                    candidate_triggered=True,
                    validation_status="generation_failed",
                    validation_reasons=call.error or "llm_query_generation_failed",
                )
            )
            continue

        generated = _select_generated_items(call.parsed.get("queries") or [], triggers, expansion_config)
        if not generated:
            report_rows.append(
                _base_report_row(
                    run_id=run_id,
                    case=case,
                    expansion_config=expansion_config,
                    evidence_rows=rows,
                    generated_at=generated_at,
                    candidate_triggered=True,
                    validation_status="no_queries_generated",
                    validation_reasons="llm_returned_no_query_candidates",
                )
            )
            continue

        existing_texts = [_text(row, "query_text") for row in existing_by_case.get(cid, [])]
        valid_count = 0
        rejected_count = 0
        generated_norms: set[str] = set()
        generated_near: set[str] = set()
        for item in generated:
            generated_query = _text(item, "query_text")
            item_target_gap = _text(item, "target_gap") or _text(item, "_target_gap")
            item_intent = _text(item, "intended_purpose") or _text(item, "_query_intent")
            trigger = _matching_trigger(item, triggers)
            validation = validate_generated_query(
                query=generated_query,
                case=case,
                evidence_rows=rows,
                existing_query_texts=existing_texts,
                query_intent=trigger.query_intent if trigger else item_intent,
                target_gap=trigger.target_gap if trigger else item_target_gap,
                generated_query_norms=generated_norms,
                generated_query_near_keys=generated_near,
            )
            if validation.accepted:
                generated_norms.add(normalize_query(generated_query))
                generated_near.add(_near_duplicate_key(generated_query))
                valid_count += 1
                lane = infer_expansion_lane(generated_query, trigger.target_gap if trigger else item_target_gap)
                missing_component = target_gap_to_missing_component(trigger.target_gap if trigger else item_target_gap)
                query_id = stable_query_fingerprint(cid, lane, missing_component, validation.validated_query)
                query_row = {
                    "schema_version": "llm_query_expansion_v1",
                    "candidate_id": cid,
                    "phase2_3_sample_id": cid,
                    "lane": lane,
                    "missing_component": missing_component,
                    "query_text": validation.validated_query,
                    "round": 2,
                    "retrieval_round": 2,
                    "query_source": QUERY_SOURCE,
                    "found_by": FOUND_BY,
                    "search_backend": "tavily",
                    "query_id": query_id,
                    "target_gap": trigger.target_gap if trigger else item_target_gap,
                    "query_intent": trigger.query_intent if trigger else item_intent,
                    "expansion_trigger": trigger.expansion_trigger if trigger else "",
                    "generated_query": generated_query,
                    "validated_query": validation.validated_query,
                    "retrieval_tier": "controlled_tavily_open_web",
                    "fallback_reason": trigger.expansion_trigger if trigger else "",
                    "unmet_gap_before_search": trigger.target_gap if trigger else item_target_gap,
                    "domains_or_source_families_targeted": lane,
                }
                query_rows.append(query_row)
            else:
                rejected_count += 1
                query_id = ""
                lane = infer_expansion_lane(generated_query, trigger.target_gap if trigger else item_target_gap)

            report_rows.append(
                _report_row_for_generated_query(
                    run_id=run_id,
                    case=case,
                    expansion_config=expansion_config,
                    evidence_rows=rows,
                    generated_at=generated_at,
                    generated_item=item,
                    trigger=trigger,
                    validation=validation,
                    query_id=query_id,
                    lane=lane,
                )
            )
        _fill_generation_counts(report_rows, cid, len(generated), valid_count, rejected_count)

    return QueryExpansionPlan(query_rows, report_rows, raw_outputs, triggers_by_case)


def build_query_expansion_payload(
    *,
    case: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    existing_query_rows: Iterable[Mapping[str, Any]],
    triggers: Iterable[ExpansionTrigger],
    expansion_config: LLMQueryExpansionConfig,
) -> dict[str, Any]:
    rows = list(evidence_rows)
    existing_queries = [_text(row, "query_text") for row in existing_query_rows if _text(row, "query_text")]
    accepted_summary: list[dict[str, str]] = []
    other_summary: list[dict[str, str]] = []
    for row in rows:
        summary = {
            "source_lane": _text(row, "source_lane"),
            "source_family": _text(row, "source_family"),
            "final_status": _text(row, "final_status"),
            "component_supported": _text(row, "component_supported"),
            "impact_supported": _text(row, "impact_supported"),
            "impact_types": _text(row, "impact_types"),
            "explicit_transition_support": _text(row, "explicit_transition_support") or _text(row, "explicit_linkage_support"),
            "quoted_supporting_spans": _text(row, "quoted_supporting_spans")[:300],
            "source_url": _text(row, "source_url"),
        }
        if _is_accepted(row):
            accepted_summary.append(summary)
        elif _text(row, "final_status") in {"needs_review", "context_only"} or _as_bool(row.get("impact_supported")):
            other_summary.append(summary)

    return {
        "candidate": {
            "candidate_id": _candidate_id(case),
            "county": _text(case, "county"),
            "state": _state_name(case),
            "state_abbrev": _state_abbrev(case),
            "time_window": _text(case, "event_window"),
            "drought_window": _text(case, "drought_window")
            or f"{_text(case, 'drought_start')} to {_text(case, 'drought_end_month_end') or _text(case, 'drought_end')}",
            "wet_window": _text(case, "wet_event_window")
            or f"{_text(case, 'rain_start')} to {_text(case, 'rain_end')}",
            "hazard_type": "drought followed by wet/rainfall/flood window",
        },
        "fixed_queries_already_used": existing_queries,
        "first_pass_evidence_summary": {
            "accepted_rows": accepted_summary[:10],
            "needs_review_or_context_rows": other_summary[:6],
        },
        "gap_diagnosis": [
            {
                "target_gap": trigger.target_gap,
                "query_intent": trigger.query_intent,
                "expansion_trigger": trigger.expansion_trigger,
                "query_budget": trigger.max_queries,
            }
            for trigger in triggers
        ],
        "useful_source_categories": [
            "local government",
            "emergency management",
            "sheriff/OES",
            "public works",
            "Caltrans or relevant transportation agency",
            "DWR or relevant water agency",
            "agriculture agencies",
            "utilities",
            "local news",
        ],
        "preferred_impact_query_terms": list(PREFERRED_IMPACT_QUERY_TERMS),
        "preferred_linkage_query_terms": list(PREFERRED_LINKAGE_QUERY_TERMS),
        "avoid_terms_for_ordinary_hazard_or_impact_queries": list(RESEARCH_ONLY_TERMS) + list(RESTRICTED_NON_LINKAGE_QUERY_TERMS),
        "query_budget": min(expansion_config.max_queries_per_case, sum(trigger.max_queries for trigger in triggers)),
        "rules": [
            "Generate search queries only; generated queries are not evidence.",
            "Do not generate hazard, impact, linkage, or case labels.",
            "Use ordinary public-source search language with concrete terms like flooding, storm damage, road closure, high water, flood damage, crop loss, water restrictions, power outage, emergency management, public works, agriculture agency, or local news.",
            "Do not invent specific roads, storm names, declarations, disaster IDs, damages, casualties, crop losses, or agency actions.",
            "Every query must include a local place anchor, a time anchor from the candidate windows, and a hazard or impact anchor.",
            "Avoid internal research terms such as SPEI, DTER, p99, rawce, CE-Agent, compound event, and candidate event.",
            "Avoid drought-to-flood transition, drought-to-wet transition, transition, or linkage wording except when the active gap is explicit_linkage_absent.",
            "For linkage searches, prefer public wording like drought followed by heavy rain, drought followed by flooding, drought then storm damage, or drought conditions before flooding.",
        ],
    }


def validate_generated_query(
    *,
    query: str,
    case: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
    existing_query_texts: Iterable[str],
    query_intent: str = "",
    target_gap: str = "",
    generated_query_norms: Iterable[str] = (),
    generated_query_near_keys: Iterable[str] = (),
) -> QueryValidationResult:
    query = str(query or "").strip()
    normalized = normalize_query(query)
    existing_norms = {normalize_query(text) for text in existing_query_texts if str(text or "").strip()}
    existing_near = {_near_duplicate_key(text) for text in existing_query_texts if str(text or "").strip()}
    existing_norms.update(generated_query_norms)
    existing_near.update(generated_query_near_keys)

    location_anchor_ok = _has_location_anchor(query, case)
    state_anchor_ok = _has_state_anchor_or_unambiguous_local_anchor(query, case)
    time_anchor_ok = _has_time_anchor(query, case)
    hazard_anchor_ok = _has_hazard_or_impact_anchor(query)
    target_gap_mismatch = _target_gap_anchor_mismatch(query, target_gap, query_intent)
    duplicate = normalized in existing_norms or _near_duplicate_key(query) in existing_near
    research_terms = tuple(term for term in RESEARCH_ONLY_TERMS if term in query.lower())
    restricted_public_terms = _restricted_public_query_terms(query, query_intent)
    candidate_id = _candidate_id(case).lower()
    if candidate_id and candidate_id in query.lower():
        research_terms = tuple(dict.fromkeys((*research_terms, "candidate_id")))
    ungrounded = tuple(_ungrounded_proper_nouns(query, case, evidence_rows))
    broad_state_only = _is_broad_state_only_query(query, case)

    reasons: list[str] = []
    if not location_anchor_ok:
        reasons.append("missing_county_or_locality_anchor")
    if not state_anchor_ok:
        reasons.append("missing_state_or_unambiguous_local_anchor")
    if not time_anchor_ok:
        reasons.append("missing_candidate_window_time_anchor")
    if not hazard_anchor_ok:
        reasons.append("missing_hazard_or_impact_anchor")
    if target_gap_mismatch:
        reasons.append(f"target_gap_anchor_mismatch:{target_gap_mismatch}")
    if duplicate:
        reasons.append("duplicate_or_near_duplicate_fixed_query")
    if research_terms:
        reasons.append("internal_research_terms:" + ",".join(research_terms))
    if restricted_public_terms:
        reasons.append("linkage_language_without_linkage_trigger:" + ",".join(restricted_public_terms))
    if ungrounded:
        reasons.append("ungrounded_proper_nouns:" + ",".join(ungrounded))
    if broad_state_only:
        reasons.append("broad_state_only_query")

    return QueryValidationResult(
        accepted=not reasons and bool(query),
        validated_query=query if not reasons else "",
        reasons=tuple(reasons),
        location_anchor_ok=location_anchor_ok,
        state_anchor_ok=state_anchor_ok,
        time_anchor_ok=time_anchor_ok,
        hazard_or_impact_anchor_ok=hazard_anchor_ok,
        duplicate_fixed_query=duplicate,
        research_only_terms_found=research_terms,
        restricted_public_query_terms=restricted_public_terms,
        ungrounded_proper_nouns=ungrounded,
        broad_state_only=broad_state_only,
    )


def infer_expansion_lane(query: str, target_gap: str) -> str:
    text = query.lower()
    if "caltrans" in text or "dot.ca.gov" in text:
        return "caltrans_transportation"
    if "dwr" in text or "water.ca.gov" in text or "water agency" in text:
        return "california_dwr_water"
    if "sheriff" in text or " oes" in f" {text}" or "emergency management" in text:
        return "sheriff_oes_emergency"
    if "public works" in text or "flood control" in text:
        return "public_works_flood_control"
    if "agricultur" in text or "crop" in text or "livestock" in text:
        return "agriculture_official"
    if "road" in text or "highway" in text or "transport" in text:
        return "transportation_roads"
    if "news" in text:
        return "local_news"
    if "drought" in target_gap:
        return "agriculture_drought_impact"
    if "linkage" in target_gap or "transition" in target_gap:
        return "local_news"
    return "local_government_emergency"


def target_gap_to_missing_component(target_gap: str) -> str:
    text = str(target_gap or "").lower()
    if "drought" in text:
        return "drought"
    if "wet" in text or "rain" in text or "flood" in text:
        return "rain_flood"
    return "impact"


def annotate_expansion_report_with_outcomes(
    report_rows: list[dict[str, Any]],
    *,
    first_pass_gate_rows: Iterable[Mapping[str, Any]],
    final_gate_rows: Iterable[Mapping[str, Any]],
    expansion_pages: Iterable[Mapping[str, Any]],
    expansion_evidence_rows: Iterable[Mapping[str, Any]],
    expansion_search_result_rows: Iterable[Mapping[str, Any]] = (),
    expansion_fetch_decision_rows: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    first_by_case = {_text(row, "candidate_id"): row for row in first_pass_gate_rows}
    final_by_case = {_text(row, "candidate_id"): row for row in final_gate_rows}
    pages_by_case = _count_by_case(expansion_pages)
    pages_by_query = _count_by_query(expansion_pages)
    search_rows_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for search_row in expansion_search_result_rows:
        if _text(search_row, "query_id"):
            search_rows_by_query.setdefault(_text(search_row, "query_id"), []).append(search_row)
    fetch_rows_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for fetch_row in expansion_fetch_decision_rows:
        if _text(fetch_row, "query_id"):
            fetch_rows_by_query.setdefault(_text(fetch_row, "query_id"), []).append(fetch_row)
    evidence_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in expansion_evidence_rows:
        evidence_by_case.setdefault(_text(row, "candidate_id"), []).append(row)

    for row in report_rows:
        cid = _text(row, "candidate_id")
        first = first_by_case.get(cid, {})
        final = final_by_case.get(cid, {})
        candidate_expansion_rows = evidence_by_case.get(cid, [])
        query_id = _text(row, "query_id")
        if query_id:
            expansion_rows = [evidence for evidence in candidate_expansion_rows if _text(evidence, "query_id") == query_id]
            search_rows = search_rows_by_query.get(query_id, [])
            fetch_rows = fetch_rows_by_query.get(query_id, [])
            pages_fetched = pages_by_query.get(query_id, 0)
        elif _text(row, "generated_query"):
            expansion_rows = []
            search_rows = []
            fetch_rows = []
            pages_fetched = 0
        else:
            expansion_rows = candidate_expansion_rows
            search_rows = []
            fetch_rows = []
            pages_fetched = pages_by_case.get(cid, 0)
        unique_result_urls = {
            _text(search_row, "url")
            for search_row in search_rows
            if _text(search_row, "url")
        }
        duplicate_result_count = max(0, len([row for row in search_rows if _text(row, "url")]) - len(unique_result_urls))
        skipped_fetch_rows = [row for row in fetch_rows if _text(row, "fetch_decision") != "fetch"]
        skip_reason_counts: dict[str, int] = {}
        for fetch_row in skipped_fetch_rows:
            reason = _text(fetch_row, "decision_reason") or "not_fetched"
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
        status_counts = {status: 0 for status in ("accepted", "needs_review", "context_only", "rejected")}
        for evidence in expansion_rows:
            status = _text(evidence, "final_status")
            if status in status_counts:
                status_counts[status] += 1
        row["expansion_search_results_returned"] = len(search_rows)
        row["expansion_unique_result_urls"] = len(unique_result_urls)
        row["expansion_result_urls_deduplicated"] = duplicate_result_count
        row["expansion_fetch_candidates_considered"] = len(fetch_rows)
        row["expansion_pages_skipped_before_fetch"] = len(skipped_fetch_rows)
        row["expansion_fetch_skip_reasons"] = ";".join(
            f"{reason}:{count}" for reason, count in sorted(skip_reason_counts.items())
        )
        row["expansion_pages_fetched"] = pages_fetched
        row["expansion_accepted_rows"] = status_counts["accepted"]
        row["expansion_needs_review_rows"] = status_counts["needs_review"]
        row["expansion_context_only_rows"] = status_counts["context_only"]
        row["expansion_rejected_rows"] = status_counts["rejected"]
        row["expansion_hazard_support_found"] = str(any(_row_supports_component(evidence, "drought") or _row_supports_component(evidence, "wet") for evidence in expansion_rows)).lower()
        row["expansion_impact_support_found"] = str(any(_row_supports_impact(evidence) for evidence in expansion_rows)).lower()
        row["expansion_linkage_support_found"] = str(
            any(
                _is_accepted(evidence)
                and (_as_bool(evidence.get("explicit_transition_support")) or _as_bool(evidence.get("explicit_linkage_support")))
                for evidence in expansion_rows
            )
        ).lower()
        row["first_pass_ce_status"] = _text(first, "ce_status")
        row["first_pass_impact_status"] = _text(first, "impact_status")
        row["first_pass_case_use_label"] = _text(first, "case_use_label")
        row["first_pass_material_impact_pattern"] = _text(first, "material_impact_pattern")
        row["final_ce_status"] = _text(final, "ce_status")
        row["final_impact_status"] = _text(final, "impact_status")
        row["final_case_use_label"] = _text(final, "case_use_label")
        row["final_material_impact_pattern"] = _text(final, "material_impact_pattern")
        changed = any(
            row.get(first_key) != row.get(final_key)
            for first_key, final_key in (
                ("first_pass_ce_status", "final_ce_status"),
                ("first_pass_impact_status", "final_impact_status"),
                ("first_pass_case_use_label", "final_case_use_label"),
                ("first_pass_material_impact_pattern", "final_material_impact_pattern"),
            )
        )
        row["case_label_changed"] = str(changed).lower()
        row["changed_label_supported_by_accepted_evidence"] = str(
            changed and any(_is_accepted(evidence) for evidence in expansion_rows)
        ).lower()
    return report_rows


def _impact_gap_reasons(rows: list[Mapping[str, Any]]) -> list[str]:
    accepted_rows = [row for row in rows if _is_accepted(row)]
    accepted_impact = [row for row in accepted_rows if _row_supports_impact(row)]
    nonaccepted_impact_like = [
        row
        for row in rows
        if not _is_accepted(row)
        and (_as_bool(row.get("impact_supported")) or _text(row, "final_status") in {"needs_review", "context_only"})
    ]
    reasons: list[str] = []
    if not accepted_impact:
        reasons.append("no_accepted_impact_evidence")
        if nonaccepted_impact_like:
            reasons.append("only_context_or_needs_review_impact_like_evidence")
        return reasons

    classifications = [classify_evidence_impact(row).evidence_impact_status for row in accepted_impact]
    if "impact_material" not in classifications:
        reasons.append("only_weak_or_nonmaterial_impact_evidence")

    channels = _impact_channels(accepted_impact)
    if len(channels) <= 1:
        if not channels or channels <= {"transportation", "roads / transport"}:
            reasons.append("single_impact_channel_or_road_only")
        else:
            reasons.append("single_impact_channel")

    if accepted_impact and all(_is_noaa_structured(row) for row in accepted_impact):
        reasons.append("only_noaa_structured_impact_without_local_open_web_narrative")

    if not any(_is_local_narrative_source(row) for row in accepted_impact):
        reasons.append("no_local_government_emergency_transport_agriculture_utility_or_news_impact_source")

    wet_impact = any(_row_supports_wet_impact(row) for row in accepted_impact)
    drought_impact = any(_row_supports_drought_impact(row) for row in accepted_impact)
    if wet_impact and not drought_impact:
        reasons.append("wet_impact_exists_but_drought_impact_missing")
    elif drought_impact and not wet_impact:
        reasons.append("drought_impact_exists_but_wet_impact_missing")

    return list(dict.fromkeys(reasons))


def _select_generated_items(
    generated: list[Mapping[str, Any]],
    triggers: list[ExpansionTrigger],
    expansion_config: LLMQueryExpansionConfig,
) -> list[dict[str, Any]]:
    budgets = {trigger.target_gap: trigger.max_queries for trigger in triggers}
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    total = 0
    for item in generated:
        if total >= expansion_config.max_queries_per_case:
            break
        trigger = _matching_trigger(item, triggers)
        if trigger is None:
            continue
        count = counts.get(trigger.target_gap, 0)
        if count >= budgets.get(trigger.target_gap, 0):
            continue
        enriched = dict(item)
        enriched["_target_gap"] = trigger.target_gap
        enriched["_query_intent"] = trigger.query_intent
        selected.append(enriched)
        counts[trigger.target_gap] = count + 1
        total += 1
    return selected


def _matching_trigger(item: Mapping[str, Any], triggers: list[ExpansionTrigger]) -> ExpansionTrigger | None:
    target_gap = _text(item, "target_gap").lower()
    purpose = _text(item, "intended_purpose").lower()
    for trigger in triggers:
        if trigger.target_gap.lower() == target_gap:
            return trigger
    for trigger in triggers:
        if trigger.query_intent.lower() in purpose or trigger.target_gap.lower() in purpose:
            return trigger
    if len(triggers) == 1:
        return triggers[0]
    return None


def _report_row_for_generated_query(
    *,
    run_id: str,
    case: Mapping[str, Any],
    expansion_config: LLMQueryExpansionConfig,
    evidence_rows: Iterable[Mapping[str, Any]],
    generated_at: str,
    generated_item: Mapping[str, Any],
    trigger: ExpansionTrigger | None,
    validation: QueryValidationResult,
    query_id: str,
    lane: str,
) -> dict[str, Any]:
    row = _base_report_row(
        run_id=run_id,
        case=case,
        expansion_config=expansion_config,
        evidence_rows=evidence_rows,
        generated_at=generated_at,
        candidate_triggered=True,
        validation_status="accepted_for_search" if validation.accepted else "rejected",
        validation_reasons=";".join(validation.reasons) if validation.reasons else "passed_deterministic_validation",
    )
    generated_query = _text(generated_item, "query_text")
    row.update(
        {
            "retrieval_round": "2",
            "target_gap": trigger.target_gap if trigger else _text(generated_item, "target_gap"),
            "query_intent": trigger.query_intent if trigger else _text(generated_item, "intended_purpose"),
            "expansion_trigger": trigger.expansion_trigger if trigger else "",
            "generated_query": generated_query,
            "normalized_generated_query": normalize_query(generated_query),
            "validated_query": validation.validated_query,
            "query_id": query_id,
            "query_source": QUERY_SOURCE,
            "found_by": FOUND_BY,
            "location_anchor_ok": str(validation.location_anchor_ok).lower(),
            "state_anchor_ok": str(validation.state_anchor_ok).lower(),
            "time_anchor_ok": str(validation.time_anchor_ok).lower(),
            "hazard_or_impact_anchor_ok": str(validation.hazard_or_impact_anchor_ok).lower(),
            "duplicate_fixed_query": str(validation.duplicate_fixed_query).lower(),
            "research_only_terms_found": ";".join(validation.research_only_terms_found),
            "restricted_public_query_terms": ";".join(validation.restricted_public_query_terms),
            "ungrounded_proper_nouns": ";".join(validation.ungrounded_proper_nouns),
            "broad_state_only": str(validation.broad_state_only).lower(),
            "accepted_for_search": str(validation.accepted).lower(),
            "planned_lane": lane,
            "search_backend": "tavily",
        }
    )
    return row


def _base_report_row(
    *,
    run_id: str,
    case: Mapping[str, Any],
    expansion_config: LLMQueryExpansionConfig,
    evidence_rows: Iterable[Mapping[str, Any]],
    generated_at: str,
    candidate_triggered: bool,
    validation_status: str,
    validation_reasons: str,
) -> dict[str, Any]:
    row = {field: "" for field in EXPANSION_REPORT_FIELDS}
    row.update(
        {
            "run_id": run_id,
            "candidate_id": _candidate_id(case),
            "county": _text(case, "county"),
            "state": _state_name(case),
            "expansion_enabled": str(expansion_config.enabled).lower(),
            "candidate_triggered": str(candidate_triggered).lower(),
            "retrieval_round": "2",
            "evidence_snapshot_hash": evidence_snapshot_hash(evidence_rows),
            "llm_model": expansion_config.model,
            "prompt_version": expansion_config.prompt_version,
            "validation_status": validation_status,
            "validation_reasons": validation_reasons,
            "query_source": QUERY_SOURCE if candidate_triggered else "",
            "found_by": FOUND_BY if candidate_triggered else "",
            "accepted_for_search": "false",
            "created_at": generated_at,
        }
    )
    return row


def _fill_generation_counts(report_rows: list[dict[str, Any]], candidate_id: str, generated: int, valid: int, rejected: int) -> None:
    for row in report_rows:
        if _text(row, "candidate_id") != candidate_id:
            continue
        row["generated_query_count"] = generated
        row["valid_query_count"] = valid
        row["rejected_query_count"] = rejected


def _raw_generation_record(call: ModelCallResult, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "role": call.role,
        "model": call.model,
        "status": call.status,
        "raw_output_text": call.raw_output_text,
        "attempts": call.attempts,
        "usage": call.usage,
        "error": call.error,
    }


def _has_location_anchor(query: str, case: Mapping[str, Any]) -> bool:
    text = query.lower()
    county = normalize_county(_text(case, "county")).lower()
    if county and (county in text or f"{county} county" in text):
        return True
    for hint in _locality_hints(case):
        if hint.lower() in text:
            return True
    return False


def _has_state_anchor_or_unambiguous_local_anchor(query: str, case: Mapping[str, Any]) -> bool:
    text = query.lower()
    state = _state_name(case).lower()
    abbrev = _state_abbrev(case).lower()
    if state and state in text:
        return True
    if abbrev and re.search(rf"\b{re.escape(abbrev)}\b", text):
        return True
    county = normalize_county(_text(case, "county")).lower()
    if county and f"{county} county" in text:
        return True
    return any(anchor in text for anchor in APPROVED_SOURCE_ANCHORS)


def _has_time_anchor(query: str, case: Mapping[str, Any]) -> bool:
    text = query.lower()
    anchors = _time_anchors(case)
    return any(anchor and anchor.lower() in text for anchor in anchors)


def _has_hazard_or_impact_anchor(query: str) -> bool:
    text = query.lower()
    return any(term in text for term in HAZARD_OR_IMPACT_TERMS)


def _restricted_public_query_terms(query: str, query_intent: str) -> tuple[str, ...]:
    if str(query_intent or "").lower() == QUERY_INTENT_LINKAGE:
        return ()
    lowered = query.lower()
    return tuple(term for term in RESTRICTED_NON_LINKAGE_QUERY_TERMS if term in lowered)


def _target_gap_anchor_mismatch(query: str, target_gap: str, query_intent: str) -> str:
    if str(query_intent or "").lower() != QUERY_INTENT_HAZARD:
        return ""
    lowered = query.lower()
    gap = str(target_gap or "").lower()
    if "drought" in gap and not any(term in lowered for term in ("drought", "dry", "water restriction", "water restrictions", "water shortage")):
        return "drought"
    if any(term in gap for term in ("wet", "rain", "flood")) and not any(
        term in lowered for term in ("wet", "rain", "rainfall", "flood", "flooding", "storm", "high water")
    ):
        return "wet"
    return ""


def _is_broad_state_only_query(query: str, case: Mapping[str, Any]) -> bool:
    text = query.lower()
    state = _state_name(case).lower()
    abbrev = _state_abbrev(case).lower()
    has_state = bool(state and state in text) or bool(abbrev and re.search(rf"\b{re.escape(abbrev)}\b", text))
    return has_state and not _has_location_anchor(query, case)


def _ungrounded_proper_nouns(
    query: str,
    case: Mapping[str, Any],
    evidence_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    grounded = _grounded_text(case, evidence_rows).lower()
    allowed = {value.lower() for value in COMMON_ALLOWED_PROPER_NOUNS}
    allowed.update(name.lower() for name in MONTH_NAMES)
    allowed.update(name.lower() for name in SEASON_NAMES)
    allowed.update(_state_name(case).lower().split())
    allowed.add(_state_abbrev(case).lower())
    county = normalize_county(_text(case, "county"))
    if county:
        allowed.update(token.lower() for token in county.split())
        allowed.add(f"{county} County".lower())
    allowed.update(hint.lower() for hint in _locality_hints(case))

    phrases: list[str] = []
    phrases.extend(match.group(0).strip() for match in re.finditer(r"\b(?:Highway|Route|Interstate|I-|SR)\s*[-]?\s*\d+\b", query))
    phrases.extend(
        match.group(0).strip()
        for match in re.finditer(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|\d+))*\b", query)
    )
    ungrounded: list[str] = []
    for phrase in dict.fromkeys(phrases):
        lowered = phrase.lower()
        if not lowered or lowered in allowed:
            continue
        if any(token in allowed for token in lowered.split()):
            remainder = " ".join(token for token in lowered.split() if token not in allowed and not token.isdigit())
            if not remainder:
                continue
        if lowered in grounded:
            continue
        if any(source_anchor in lowered for source_anchor in APPROVED_SOURCE_ANCHORS):
            continue
        if phrase.isdigit():
            continue
        ungrounded.append(phrase)
    return ungrounded


def _grounded_text(case: Mapping[str, Any], evidence_rows: Iterable[Mapping[str, Any]]) -> str:
    parts = [str(value) for value in case.values() if value is not None]
    parts.extend(_locality_hints(case))
    for row in evidence_rows:
        parts.extend(
            _text(row, key)
            for key in (
                "source_title",
                "source_lane",
                "source_family",
                "quoted_supporting_spans",
                "final_reason",
            )
        )
    return " ".join(parts)


def _time_anchors(case: Mapping[str, Any]) -> set[str]:
    raw_values = [
        _text(case, "drought_start"),
        _text(case, "drought_end"),
        _text(case, "drought_end_month_end"),
        _text(case, "rain_start"),
        _text(case, "rain_end"),
        _text(case, "rainfall_year"),
        _text(case, "year"),
        _text(case, "drought_window"),
        _text(case, "wet_event_window"),
        _text(case, "event_window"),
    ]
    anchors: set[str] = set()
    for value in raw_values:
        anchors.update(re.findall(r"\b(?:19|20)\d{2}\b", value))
        for year, month in re.findall(r"\b((?:19|20)\d{2})-(\d{2})\b", value):
            anchors.add(f"{year}-{month}")
            month_int = int(month)
            if 1 <= month_int <= 12:
                month_name = list(MONTH_NAMES)[month_int - 1]
                anchors.add(f"{month_name} {year}")
        for month_name in MONTH_NAMES:
            for year in re.findall(rf"\b{month_name}\s+((?:19|20)\d{{2}})\b", value, flags=re.I):
                anchors.add(f"{month_name} {year}")
    return anchors


def _locality_hints(case: Mapping[str, Any]) -> list[str]:
    hints: list[str] = []
    for key in ("locality_hints", "approved_locality_hints"):
        raw = case.get(key)
        if isinstance(raw, (list, tuple, set)):
            hints.extend(str(value) for value in raw if str(value).strip())
        elif raw:
            hints.extend(part.strip() for part in re.split(r"[;,|]", str(raw)) if part.strip())
    return list(dict.fromkeys(hints))


def _impact_channels(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    channels: set[str] = set()
    for row in rows:
        raw = _text(row, "impact_types") or _text(row, "impact_channel")
        for part in re.split(r"[;,|]", raw):
            value = part.strip().lower()
            if value:
                channels.add(value)
    return channels


def _row_supports_component(row: Mapping[str, Any], component: str) -> bool:
    if not _is_accepted(row):
        return False
    raw = _text(row, "component_supported").lower()
    if component == "drought":
        return raw in {"drought", "both"} or _as_bool(row.get("supports_drought"))
    if component == "wet":
        return raw in {"wet", "both", "rain_flood"} or _as_bool(row.get("supports_wet_event"))
    return False


def _row_supports_impact(row: Mapping[str, Any]) -> bool:
    return _is_accepted(row) and (_as_bool(row.get("impact_supported")) or _as_bool(row.get("supports_impact")))


def _row_supports_wet_impact(row: Mapping[str, Any]) -> bool:
    component = _text(row, "component_supported").lower()
    return _row_supports_impact(row) and (
        _as_bool(row.get("wet_impact_support"))
        or _as_bool(row.get("wet_impact_supported"))
        or component in {"wet", "both", "rain_flood"}
    )


def _row_supports_drought_impact(row: Mapping[str, Any]) -> bool:
    component = _text(row, "component_supported").lower()
    return _row_supports_impact(row) and (
        _as_bool(row.get("drought_impact_support"))
        or _as_bool(row.get("drought_impact_supported"))
        or component in {"drought", "both"}
    )


def _is_structured_official(row: Mapping[str, Any]) -> bool:
    if not _is_accepted(row):
        return False
    origin = _text(row, "evidence_origin").lower()
    lane = _text(row, "source_lane").lower()
    family = _text(row, "source_family").lower()
    return (
        "structured_official" in origin
        or lane in {"usdm_county_statistics", "noaa_storm_events", "openfema_admin_response"}
        or any(token in family for token in ("usdm", "noaa", "ncei", "openfema"))
    )


def _is_noaa_structured(row: Mapping[str, Any]) -> bool:
    lane = _text(row, "source_lane").lower()
    family = _text(row, "source_family").lower()
    return lane == "noaa_storm_events" or "noaa" in family or "ncei" in family


def _is_local_narrative_source(row: Mapping[str, Any]) -> bool:
    if _text(row, "evidence_origin") != "validated_open_web_body":
        return False
    text = f"{_text(row, 'source_lane')} {_text(row, 'source_family')}".lower()
    return any(
        token in text
        for token in (
            "county",
            "city",
            "sheriff",
            "oes",
            "emergency",
            "public_works",
            "public works",
            "transport",
            "caltrans",
            "agriculture",
            "utility",
            "news",
            "water",
            "dwr",
        )
    )


def _is_accepted(row: Mapping[str, Any]) -> bool:
    return _text(row, "final_status").lower() == "accepted"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def _text(row: Mapping[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def _candidate_id(case: Mapping[str, Any]) -> str:
    return _text(case, "candidate_id") or _text(case, "phase2_3_sample_id") or _text(case, "raw_candidate_id")


def _state_name(case: Mapping[str, Any]) -> str:
    state = _text(case, "state") or _text(case, "state_name")
    if state.upper() == "CA":
        return "California"
    return state or "California"


def _state_abbrev(case: Mapping[str, Any]) -> str:
    return (_text(case, "state_abbrev") or _text(case, "state_abbreviation") or "CA").upper()


def _count_by_case(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        cid = _text(row, "candidate_id")
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _count_by_query(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        query_id = _text(row, "query_id")
        if not query_id:
            continue
        counts[query_id] = counts.get(query_id, 0) + 1
    return counts
