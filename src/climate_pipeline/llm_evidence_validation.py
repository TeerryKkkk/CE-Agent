from __future__ import annotations

import calendar
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from . import config
from .evidence_span_extractor import (
    CandidateContext,
    EvidenceSpan,
)
from .full_body_direct_judge import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    EXACT_MODEL,
    RESPONSES_URL,
    DirectPageJudgeClient,
    derive_local_page_result,
    preflight_source_readability,
)
from .llm_evidence_judge import (
    deterministic_arbiter,
    read_openai_api_key,
)


class LLMEvidenceValidationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMEvidenceJudgeConfig:
    use_llm_evidence_judge: bool = False
    llm_judge_model: str = EXACT_MODEL
    # Retained as no-op compatibility fields for historical callers. The direct
    # webpage path never instantiates or calls skeptic/arbiter clients.
    llm_skeptic_model: str = EXACT_MODEL
    llm_arbiter_model: str = EXACT_MODEL
    llm_judge_fail_closed: bool = True
    keyword_validator_mode: str = "baseline_only"
    require_quoted_span_for_accept: bool = True
    require_compact_span_grounding: bool = True
    max_spans_per_page: int = 10
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_input_tokens: int = 131_072
    max_request_bytes: int = 1_000_000
    max_response_bytes: int = 262_144
    max_error_response_bytes: int = 65_536
    responses_url: str = RESPONSES_URL
    timeout_seconds: int = 360
    api_key_path: Path | None = None
    api_key_value: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key_path"] = str(self.api_key_path) if self.api_key_path else ""
        data["api_key_value"] = "[REDACTED]" if self.api_key_value else ""
        return data


@dataclass
class IntegratedValidationResult:
    judgment_rows: list[dict[str, str]]
    span_rows: list[dict[str, str]]
    disagreement_rows: list[dict[str, str]]
    schema_failure_rows: list[dict[str, str]]
    normalized_rows: list[dict[str, Any]]
    raw_model_outputs: dict[str, list[dict[str, Any]]]
    config: LLMEvidenceJudgeConfig


def validate_fetched_pages_with_llm_judge(
    *,
    pages: list[dict[str, Any]],
    strict_rows: list[dict[str, str]] | None = None,
    prior_llm_rows: list[dict[str, str]] | None = None,
    case_cards: dict[str, dict[str, str]] | None = None,
    smoke_dir: Path | None = None,
    judge_config: LLMEvidenceJudgeConfig,
    saved_direct_calls: Mapping[str, Any] | None = None,
) -> IntegratedValidationResult:
    if judge_config.use_llm_evidence_judge:
        api_key = judge_config.api_key_value or read_openai_api_key(
            judge_config.api_key_path
        )
        if not api_key:
            if judge_config.llm_judge_fail_closed:
                raise LLMEvidenceValidationUnavailable("missing_openai_api_key")
            api_key = ""
    else:
        api_key = ""

    strict_index = index_rows_by_candidate_url_occurrence(strict_rows or [])
    prior_llm_index = index_rows_by_candidate_url_occurrence(prior_llm_rows or [])
    case_cards = case_cards or {}
    duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
    direct_client = (
        DirectPageJudgeClient(
            api_key=api_key,
            model=judge_config.llm_judge_model,
            max_output_tokens=judge_config.max_output_tokens,
            responses_url=judge_config.responses_url,
            timeout_seconds=judge_config.timeout_seconds,
            max_input_tokens=judge_config.max_input_tokens,
            max_request_bytes=judge_config.max_request_bytes,
            max_response_bytes=judge_config.max_response_bytes,
            max_error_response_bytes=judge_config.max_error_response_bytes,
        )
        if judge_config.use_llm_evidence_judge and saved_direct_calls is None
        else None
    )

    judgment_rows: list[dict[str, str]] = []
    span_rows: list[dict[str, str]] = []
    disagreement_rows: list[dict[str, str]] = []
    schema_failures: list[dict[str, str]] = []
    normalized_rows: list[dict[str, Any]] = []
    raw_outputs: dict[str, list[dict[str, Any]]] = {"primary": [], "skeptic": [], "arbiter": []}

    for page in pages:
        candidate_id = page_candidate_id(page)
        source_url = str(page.get("source_url") or "")
        normalized_url = normalize_url(source_url)
        duplicate_counts[(candidate_id, normalized_url)] += 1
        occurrence = duplicate_counts[(candidate_id, normalized_url)]
        strict_row = strict_index.get((candidate_id, normalized_url, occurrence)) or {}
        prior_llm_row = prior_llm_index.get((candidate_id, normalized_url, occurrence)) or {}
        context = candidate_context_from_sources(candidate_id, page, strict_row, case_cards)
        body_text = load_body_text(page, smoke_dir=smoke_dir)
        # The upgraded semantic path receives the full canonical body. Span
        # extraction remains available for historical replays but is not an
        # input to this judgment.
        spans: list[EvidenceSpan] = []
        readability = preflight_source_readability(body_text)
        pre_gate = {
            "final_status": "" if readability.usable else "needs_review",
            "failure_reason": "" if readability.usable else "insufficient_source_content",
            "page_type_hint": "",
            "reason": readability.reason,
        }
        source_metadata = {
            "source_url": source_url,
            "source_title": str(page.get("source_title") or ""),
            "source_family": strict_row.get("source_family") or page.get("source_family") or page.get("discovered_source_family") or "",
            "raw_source_family": page.get("discovered_source_family") or page.get("source_family") or "",
            "source_lane": strict_row.get("source_lane") or page.get("source_lane") or "",
            "query_id": strict_row.get("query_id") or page.get("query_id") or "",
            "duplicate_for_candidate": occurrence > 1,
            "duplicate_ordinal_for_candidate_url": occurrence,
            "accepted_from_snippet": bool(page.get("accepted_from_snippet")),
            "pre_gate_page_type_hint": pre_gate["page_type_hint"],
        }
        candidate_metadata = direct_candidate_metadata(
            context,
            page=page,
            strict_row=strict_row,
        )

        if not judge_config.use_llm_evidence_judge:
            primary = keyword_baseline_primary(page, source_url)
            skeptic = pre_gate_skeptic({"reason": "LLM judge disabled; keyword baseline only.", "failure_reason": "none"})
            final = deterministic_arbiter(primary, skeptic)
        else:
            call_key = direct_call_identity(candidate_id, normalized_url, occurrence)
            if saved_direct_calls is not None:
                if call_key not in saved_direct_calls:
                    raise LLMEvidenceValidationUnavailable(
                        f"saved_direct_call_missing:{call_key}"
                    )
                direct_call = saved_direct_calls[call_key]
            else:
                assert direct_client is not None
                direct_call = direct_client.judge(
                    candidate=candidate_metadata,
                    source=source_metadata,
                    body_text=body_text,
                )
            raw_outputs["primary"].append(raw_direct_call_record(direct_call, candidate_id, source_url))
            if direct_call.status not in {"ok", "not_requested_unusable_source"}:
                schema_failures.append(direct_schema_failure_row(direct_call, candidate_id, source_url))
            primary = direct_call.parsed or {}
            final = derive_local_page_result(
                semantic=direct_call.parsed,
                candidate=candidate_metadata,
                body_text=body_text,
                call_status=direct_call.status,
            )
            skeptic = {}

        if judge_config.use_llm_evidence_judge:
            judgment = direct_page_judgment_row(
                page=page,
                strict_row=strict_row,
                prior_llm_row=prior_llm_row,
                semantic=primary,
                local_result=final,
                source_metadata=source_metadata,
                candidate_metadata=candidate_metadata,
            )
        else:
            judgment = page_judgment_row(
                page=page,
                strict_row=strict_row,
                prior_llm_row=prior_llm_row,
                primary=primary,
                final=final,
                source_metadata=source_metadata,
            )
        judgment_rows.append(judgment)
        normalized_rows.append(
            {
                "candidate_id": candidate_id,
                "source_url": source_url,
                "candidate_metadata": candidate_metadata,
                "source_metadata": source_metadata,
                "pre_gate": pre_gate,
                "raw_semantic_judgment": primary,
                "guarded_local_result": final,
                "raw_keyword_status": judgment["raw_keyword_status"],
                "prior_strict_qa_status_for_evaluation_only": judgment["prior_strict_qa_status"],
                "prior_llm_replay_status_for_evaluation_only": judgment["prior_llm_replay_status"],
            }
        )
        disagreement_rows.extend(build_disagreement_rows(judgment, final))

    return IntegratedValidationResult(
        judgment_rows=judgment_rows,
        span_rows=span_rows,
        disagreement_rows=disagreement_rows,
        schema_failure_rows=schema_failures,
        normalized_rows=normalized_rows,
        raw_model_outputs=raw_outputs,
        config=judge_config,
    )


def direct_call_identity(candidate_id: str, normalized_url: str, occurrence: int) -> str:
    return f"{candidate_id}\x1f{normalized_url}\x1f{occurrence}"


def build_direct_judge_inputs(
    *,
    page: dict[str, Any],
    strict_row: dict[str, str],
    case_cards: dict[str, dict[str, str]],
    occurrence: int = 1,
    smoke_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the exact candidate/source/body tuple used by the direct judge."""

    candidate_id = page_candidate_id(page)
    source_url = str(page.get("source_url") or "")
    context = candidate_context_from_sources(
        candidate_id, page, strict_row, case_cards
    )
    readability = preflight_source_readability(load_body_text(page, smoke_dir=smoke_dir))
    source_metadata = {
        "source_url": source_url,
        "source_title": str(page.get("source_title") or ""),
        "source_family": strict_row.get("source_family") or page.get("source_family") or page.get("discovered_source_family") or "",
        "raw_source_family": page.get("discovered_source_family") or page.get("source_family") or "",
        "source_lane": strict_row.get("source_lane") or page.get("source_lane") or "",
        "query_id": strict_row.get("query_id") or page.get("query_id") or "",
        "duplicate_for_candidate": occurrence > 1,
        "duplicate_ordinal_for_candidate_url": occurrence,
        "accepted_from_snippet": bool(page.get("accepted_from_snippet")),
        "pre_gate_page_type_hint": "",
    }
    body = load_body_text(page, smoke_dir=smoke_dir)
    return {
        "call_key": direct_call_identity(
            candidate_id, normalize_url(source_url), occurrence
        ),
        "candidate_metadata": direct_candidate_metadata(
            context,
            page=page,
            strict_row=strict_row,
        ),
        "source_metadata": source_metadata,
        "body_text": body,
        "readability": readability.reason,
    }


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def index_rows_by_candidate_url_occurrence(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("candidate_id") or ""), normalize_url(str(row.get("source_url") or "")))
        counts[key] += 1
        indexed[(key[0], key[1], counts[key])] = row
    return indexed


def page_candidate_id(page: dict[str, Any]) -> str:
    return str(page.get("phase2_3_sample_id") or page.get("raw_candidate_id") or page.get("prior_phase2_sample_id") or "")


def candidate_context_from_sources(
    candidate_id: str,
    page: dict[str, Any],
    strict_row: dict[str, str],
    case_cards: dict[str, dict[str, str]],
) -> CandidateContext:
    card = case_cards.get(candidate_id, {})
    county = strict_row.get("county") or card.get("county") or str(page.get("county") or "")
    fips = card.get("county_fips") or str(page.get("FIPS") or page.get("county_fips") or "")
    wet_window = strict_row.get("event_window") or card.get("wet_window") or ""
    return CandidateContext(
        candidate_id=candidate_id,
        county=county,
        county_fips=fips,
        state=str(page.get("state") or "California"),
        drought_window=card.get("drought_window", ""),
        wet_window=wet_window,
        candidate_stratum=card.get("candidate_stratum") or str(page.get("stratum") or ""),
        # Static county-to-place hints are prohibited in forward production.
        # County/state/FIPS and event-window grounding remain unchanged.
        locality_hints=[],
    )


def target_hazard_axis_for_page(
    page: Mapping[str, Any],
    strict_row: Mapping[str, Any] | None = None,
) -> str:
    strict_row = strict_row or {}
    target_gap = str(
        page.get("target_gap")
        or page.get("missing_component")
        or strict_row.get("target_gap")
        or strict_row.get("missing_component")
        or ""
    ).strip().lower()
    expansion_trigger = str(
        page.get("expansion_trigger")
        or strict_row.get("expansion_trigger")
        or ""
    ).strip().lower()
    source_lane = str(
        page.get("source_lane")
        or strict_row.get("source_lane")
        or ""
    ).strip().lower()
    drought_targeted = (
        "drought" in target_gap
        or "drought_impact_missing" in expansion_trigger
        or (
            target_gap in {"impact", "impact_enrichment"}
            and source_lane == "agriculture_drought_impact"
        )
    )
    return "drought" if drought_targeted else "wet"


def direct_candidate_metadata(
    context: CandidateContext,
    *,
    page: Mapping[str, Any] | None = None,
    strict_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    wet_start, wet_end = _direct_window_bounds(context.wet_window)
    drought_start, drought_end = _direct_window_bounds(context.drought_window)
    target_axis = target_hazard_axis_for_page(page or {}, strict_row)
    target_start, target_end = (
        (drought_start, drought_end)
        if target_axis == "drought"
        else (wet_start, wet_end)
    )
    return {
        "candidate_id": context.candidate_id,
        "state": context.state,
        "county": context.county,
        "county_fips": context.county_fips,
        "wet_window_start": wet_start,
        "wet_window_end": wet_end,
        "drought_window": context.drought_window,
        "target_hazard_axis": target_axis,
        "target_window_start": target_start,
        "target_window_end": target_end,
        "target_hazards": (
            ["drought", "dry conditions", "water shortage", "water restrictions"]
            if target_axis == "drought"
            else ["rainfall", "flood"]
        ),
        "locality_hints": list(context.locality_hints),
    }


def _direct_window_bounds(value: str) -> tuple[str, str]:
    parts = re.split(r"\s+(?:to|through)\s+", str(value or "").strip(), maxsplit=1)
    if len(parts) == 1:
        parts.append(parts[0])
    return _normalized_window_boundary(parts[0], end=False), _normalized_window_boundary(
        parts[1],
        end=True,
    )


def _normalized_window_boundary(value: str, *, end: bool) -> str:
    value = str(value or "").strip()[:10]
    month_match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if not month_match:
        return value
    year, month = (int(part) for part in month_match.groups())
    day = calendar.monthrange(year, month)[1] if end else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def load_body_text(page: dict[str, Any], *, smoke_dir: Path | None = None) -> str:
    body = str(page.get("body_text_or_archived_body_text") or "")
    if body:
        return body
    rel_path = page.get("body_text_path")
    if rel_path:
        candidates = [config.PROJECT_ROOT / str(rel_path)]
        if smoke_dir:
            candidates.append(smoke_dir / str(rel_path))
        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
    return ""


def pre_gate_page(
    *,
    page: dict[str, Any],
    context: CandidateContext,
    source_url: str,
    source_family: str,
    duplicate_for_candidate: bool,
    spans: list[EvidenceSpan],
) -> dict[str, str]:
    lower_url = source_url.lower()
    joined_generic = ";".join(signal for span in spans for signal in span.generic_page_signals).lower()
    selected_has_local_impact = any(
        span.location_terms_found and span.time_terms_found and span.impact_terms_found
        for span in spans
        if span.selected_for_llm
    )
    if duplicate_for_candidate:
        return {
            "final_status": "rejected",
            "failure_reason": "duplicate_url",
            "page_type_hint": "duplicate",
            "reason": "Duplicate normalized URL for the same candidate; not eligible as another accepted evidence row.",
        }
    if page.get("accepted_from_snippet"):
        return {
            "final_status": "rejected",
            "failure_reason": "weak_impact_grounding",
            "page_type_hint": "other",
            "reason": "Snippet-only evidence is not eligible for final acceptance.",
        }
    if context.state.lower() == "california" and _wrong_state_url_or_title(page, lower_url):
        return {
            "final_status": "rejected",
            "failure_reason": "weak_location_grounding",
            "page_type_hint": "unrelated",
            "reason": "Wrong-state page for a California candidate.",
        }
    if _is_broad_resource(lower_url, source_family, joined_generic) and not selected_has_local_impact:
        return {
            "final_status": "context_only",
            "failure_reason": "broad_resource_page",
            "page_type_hint": "broad_resource_page",
            "reason": "Broad federal/statewide/resource page without a compact local event-impact span.",
        }
    return {"final_status": "", "failure_reason": "", "page_type_hint": "generic_planning_page" if "emergency plan" in joined_generic else "", "reason": ""}


def _wrong_state_url_or_title(page: dict[str, Any], lower_url: str) -> bool:
    title = str(page.get("source_title") or "").lower()
    wrong_state_terms = (" texas", " tx ", ".tx.us", "txdot", "texas emergency management")
    text = f" {lower_url} {title} "
    return any(term in text for term in wrong_state_terms)


def _is_broad_resource(lower_url: str, source_family: str, generic_signals: str) -> bool:
    return (
        "house.gov" in lower_url
        or "congresswoman" in generic_signals
        or "congressman" in generic_signals
        or ("resource" in generic_signals and "official" in source_family.lower())
    )


def pre_gate_primary(page: dict[str, Any], source_url: str, gate: dict[str, str]) -> dict[str, Any]:
    status = gate["final_status"]
    return {
        "candidate_id": page_candidate_id(page),
        "source_url": source_url,
        "source_title": str(page.get("source_title") or ""),
        "page_type": gate["page_type_hint"] or "other",
        "status": status,
        "component_supported": "none",
        "hazard_supported": False,
        "impact_supported": False,
        "impact_types": [],
        "location_match": "no_match" if status == "rejected" else "regional_only",
        "time_match": "no_match",
        "source_role": "context" if status == "context_only" else "reject",
        "explicit_transition_support": False,
        "quoted_supporting_spans": [],
        "reason": gate["reason"],
        "failure_reason_if_rejected": gate["failure_reason"] or "none",
    }


def pre_gate_skeptic(gate: dict[str, str]) -> dict[str, Any]:
    final_status = gate.get("final_status") or "needs_review"
    if final_status == "rejected":
        decision = "reject"
    elif final_status == "context_only":
        decision = "downgrade_to_context_only"
    else:
        decision = "downgrade_to_needs_review"
    return {
        "skeptic_decision": decision,
        "skeptic_reason": gate.get("reason") or "Pre-gate fail-safe.",
        "identified_failure_modes": [gate.get("failure_reason") or "pre_gate"],
        "minimum_safe_status": final_status,
    }


def pre_gate_final(gate: dict[str, str]) -> dict[str, Any]:
    status = gate["final_status"]
    return {
        "final_status": status,
        "final_component_supported": "none",
        "final_impact_supported": False,
        "final_impact_types": [],
        "final_source_role": "context" if status == "context_only" else "reject",
        "final_reason": gate["reason"],
        "final_quoted_spans": [],
        "disagreement_resolution": "deterministic_pre_gate",
        "failure_reason_if_rejected": gate.get("failure_reason") or "none",
    }


def keyword_baseline_primary(page: dict[str, Any], source_url: str) -> dict[str, Any]:
    accepted = bool(page.get("accepted"))
    return {
        "candidate_id": page_candidate_id(page),
        "source_url": source_url,
        "source_title": str(page.get("source_title") or ""),
        "page_type": "other",
        "status": "accepted" if accepted else "rejected",
        "component_supported": "wet" if accepted else "none",
        "hazard_supported": accepted,
        "impact_supported": accepted,
        "impact_types": ["other"] if accepted else [],
        "location_match": "same_county" if accepted else "no_match",
        "time_match": "year_only" if accepted else "no_match",
        "source_role": "supporting_impact" if accepted else "reject",
        "explicit_transition_support": False,
        "quoted_supporting_spans": [],
        "reason": "Keyword validator baseline only.",
        "failure_reason_if_rejected": "none",
    }


def fail_safe_primary(candidate_id: str, source_url: str, source_title: str, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_url": source_url,
        "source_title": source_title,
        "page_type": "other",
        "status": "needs_review",
        "component_supported": "none",
        "hazard_supported": False,
        "impact_supported": False,
        "impact_types": [],
        "location_match": "no_match",
        "time_match": "no_match",
        "source_role": "reject",
        "explicit_transition_support": False,
        "quoted_supporting_spans": [],
        "reason": f"LLM primary judge failed; fail-safe. {reason}",
        "failure_reason_if_rejected": "none",
    }


def final_from_primary(primary: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_status": primary["status"],
        "final_component_supported": primary["component_supported"],
        "final_impact_supported": primary["impact_supported"],
        "final_impact_types": primary["impact_types"],
        "final_source_role": primary["source_role"],
        "final_reason": f"Skeptic confirmed primary judgment. {primary.get('reason', '')}",
        "final_quoted_spans": primary.get("quoted_supporting_spans", []),
        "disagreement_resolution": "skeptic_confirmed",
    }


def page_judgment_row(
    *,
    page: dict[str, Any],
    strict_row: dict[str, str],
    prior_llm_row: dict[str, str],
    primary: dict[str, Any],
    final: dict[str, Any],
    source_metadata: dict[str, Any],
) -> dict[str, str]:
    final_status = str(final.get("final_status") or "")
    return {
        "candidate_id": page_candidate_id(page),
        "source_url": str(page.get("source_url") or ""),
        "source_title": str(page.get("source_title") or ""),
        "source_family": str(source_metadata.get("source_family") or ""),
        "raw_keyword_status": "accepted" if page.get("accepted") else "rejected",
        "prior_strict_qa_status": canonical_status(strict_row.get("evidence_status") or ""),
        "prior_llm_replay_status": canonical_status(prior_llm_row.get("final_agent_status") or ""),
        "integrated_llm_status": canonical_status(final_status),
        "component_supported": str(final.get("final_component_supported") or primary.get("component_supported") or ""),
        "impact_supported": str(bool(final.get("final_impact_supported"))).lower(),
        "impact_types": ";".join(final.get("final_impact_types") or []),
        "location_match": str(primary.get("location_match") or ""),
        "time_match": str(primary.get("time_match") or ""),
        "page_type": str(primary.get("page_type") or ""),
        "failure_reason_if_rejected": str(final.get("failure_reason_if_rejected") or primary.get("failure_reason_if_rejected") or "none"),
        "quoted_supporting_spans": " || ".join(final.get("final_quoted_spans") or []),
        "final_reason": str(final.get("final_reason") or ""),
    }


def direct_page_judgment_row(
    *,
    page: dict[str, Any],
    strict_row: dict[str, str],
    prior_llm_row: dict[str, str],
    semantic: dict[str, Any],
    local_result: dict[str, Any],
    source_metadata: dict[str, Any],
    candidate_metadata: Mapping[str, Any],
) -> dict[str, str]:
    page_result = str(local_result.get("page_result") or "unresolved")
    status_map = {
        "supports": "accepted",
        "does_not_support": "rejected",
        "unresolved": "needs_review",
        "insufficient_source_content": "needs_review",
    }
    impact_support = str(local_result.get("realized_impact_support") or "unresolved")
    target_axis = str(candidate_metadata.get("target_hazard_axis") or "wet")
    component = (
        target_axis
        if local_result.get("candidate_hazard_support") == "yes"
        else "none"
    )
    quotes = [str(value) for value in semantic.get("supporting_quotes") or []]
    failure_reason = "none"
    if page_result == "insufficient_source_content":
        failure_reason = "insufficient_source_content"
    elif page_result == "unresolved":
        failure_reason = "unresolved_semantic_evidence"
    elif page_result == "does_not_support":
        failure_reason = _direct_rejection_reason(local_result)
    return {
        "candidate_id": page_candidate_id(page),
        "source_url": str(page.get("source_url") or ""),
        "source_title": str(page.get("source_title") or ""),
        "source_family": str(source_metadata.get("source_family") or ""),
        "raw_keyword_status": "accepted" if page.get("accepted") else "rejected",
        "prior_strict_qa_status": canonical_status(strict_row.get("evidence_status") or ""),
        "prior_llm_replay_status": canonical_status(prior_llm_row.get("final_agent_status") or ""),
        "integrated_llm_status": status_map[page_result],
        "component_supported": component,
        "impact_supported": "true" if impact_support == "yes" else "false",
        "impact_types": ";".join(str(value.get("claim") or "") for value in semantic.get("realized_impacts") or []),
        "location_match": str(semantic.get("event_location_relation") or "unknown"),
        "time_match": str(semantic.get("event_date_relation") or "unknown"),
        "page_type": str(semantic.get("event_identity") or "unresolved_event_identity"),
        "failure_reason_if_rejected": failure_reason,
        "quoted_supporting_spans": " || ".join(quotes),
        "final_reason": "; ".join(
            f"{action.get('guard')}:{action.get('action')}:{action.get('reason')}"
            for action in local_result.get("guard_actions") or []
        ),
    }


def _direct_rejection_reason(local_result: dict[str, Any]) -> str:
    for action in local_result.get("guard_actions") or []:
        if action.get("action") == "rejected":
            guard = str(action.get("guard") or "")
            return {
                "event_date": "wrong_time_window",
                "event_location": "weak_location_grounding",
                "hazard_compatibility": "incompatible_hazard",
                "event_actuality": "prospective_or_administrative_only",
                "event_identity": "zero_concrete_relevant_event",
            }.get(guard, "does_not_support")
    return "does_not_support"


def raw_direct_call_record(call: Any, candidate_id: str, source_url: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_url": source_url,
        "role": "full_body_direct_judge",
        "model": call.model,
        "status": call.status,
        "attempts": call.attempts,
        "retry_reason": call.retry_reason,
        "request_sha256": call.request_sha256,
        "usage": call.usage,
        "error": call.error,
        "parsed": call.parsed,
        "raw_output_text": call.raw_output_text,
        "raw_responses": call.raw_responses,
        "response_ids": [
            response.get("id", "") for response in call.raw_responses if isinstance(response, dict)
        ],
    }


def direct_schema_failure_row(call: Any, candidate_id: str, source_url: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "source_url": source_url,
        "role": "full_body_direct_judge",
        "model": call.model,
        "status": call.status,
        "attempts": str(call.attempts),
        "error": call.error,
        "raw_output_path": "raw_primary_model_outputs.jsonl",
    }


def build_disagreement_rows(judgment: dict[str, str], final: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    strict = canonical_status(judgment["prior_strict_qa_status"])
    prior_llm = canonical_status(judgment["prior_llm_replay_status"])
    integrated = canonical_status(judgment["integrated_llm_status"])
    if strict and strict != integrated:
        rows.append(disagreement_row(judgment, "integrated_vs_prior_strict_qa", f"strict={strict}; integrated={integrated}"))
    if prior_llm and prior_llm != integrated:
        rows.append(disagreement_row(judgment, "integrated_vs_prior_llm_replay", f"prior_llm={prior_llm}; integrated={integrated}"))
    if judgment["raw_keyword_status"] == "accepted" and integrated != "accepted":
        rows.append(disagreement_row(judgment, "raw_keyword_accept_downgraded", str(final.get("final_reason") or "")))
    return rows


def disagreement_row(judgment: dict[str, str], kind: str, explanation: str) -> dict[str, str]:
    return {
        "candidate_id": judgment["candidate_id"],
        "source_url": judgment["source_url"],
        "prior_strict_qa_status": judgment["prior_strict_qa_status"],
        "prior_llm_replay_status": judgment["prior_llm_replay_status"],
        "integrated_llm_status": judgment["integrated_llm_status"],
        "disagreement_type": kind,
        "explanation": explanation,
    }


def raw_call_record(call: Any, candidate_id: str, source_url: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_url": source_url,
        "role": call.role,
        "model": call.model,
        "status": call.status,
        "attempts": call.attempts,
        "usage": call.usage,
        "error": call.error,
        "parsed": call.parsed,
        "raw_output_text": call.raw_output_text,
        "response_id": (call.raw_response or {}).get("id") if isinstance(call.raw_response, dict) else "",
    }


def schema_failure_row(call: Any, candidate_id: str, source_url: str, output_path: str) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "source_url": source_url,
        "role": call.role,
        "model": call.model,
        "status": call.status,
        "attempts": str(call.attempts),
        "error": call.error,
        "raw_output_path": output_path,
    }


def canonical_status(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
