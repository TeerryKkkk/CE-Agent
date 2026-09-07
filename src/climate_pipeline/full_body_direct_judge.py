from __future__ import annotations

import json
import re
import socket
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from typing import Any, Callable

import tiktoken

from climate_pipeline.bounded_http import (
    no_redirect_opener,
    read_bounded,
    read_http_error,
    validate_headers,
)


EXACT_MODEL = "gpt-5.5-2026-04-23"
REASONING_EFFORT = "high"
MIN_MAX_OUTPUT_TOKENS = 5_000
DEFAULT_MAX_OUTPUT_TOKENS = 12_000
RESPONSES_URL = "https://api.openai.com/v1/responses"

SOURCE_READABILITY = {"readable", "unusable", "uncertain"}
EVENT_IDENTITIES = {
    "zero_concrete_relevant_event",
    "one_uniquely_relevant_event",
    "unresolved_event_identity",
}
EVENT_ACTUALITIES = {
    "observed",
    "retrospective_observed",
    "warning_or_forecast",
    "order_or_administrative_action",
    "planning_or_generic_context",
    "mixed_or_uncertain",
}
DATE_ROLES = {
    "event_start",
    "event_end",
    "event_day",
    "incident_period",
    "publication",
    "update",
    "warning",
    "forecast",
    "order",
    "declaration",
    "response",
    "recovery",
    "historical_context",
    "unknown",
}
NATIVE_DATE_ROLES = {
    "event_start",
    "event_end",
    "event_day",
    "incident_period",
    "publication",
    "update",
    "observed_status_as_of",
    "warning_period",
    "declaration_period",
    "administrative_date",
    "historical_context",
    "unknown",
}
RELATIVE_ANCHOR_ROLES = {"publication", "update", "observed_status_as_of", "document_date"}
RELATIONS = {"aligned", "outside_candidate", "not_aligned", "unknown"}
TARGET_HAZARD_STATUSES = {
    "observed",
    "anticipated_or_warning",
    "incompatible_hazard_only",
    "absent_or_context",
    "uncertain",
}
TRISTATE = {"yes", "no", "uncertain"}
MODEL_SUPPORT = {"supports", "does_not_support", "needs_review"}
LOCAL_RESULTS = {"supports", "does_not_support", "unresolved", "insufficient_source_content"}
ACTIVE_GUARD_POLICY_VERSION = "candidate_event_guard_v3_production"
SHADOW_GUARD_POLICY_VERSION = "candidate_event_guard_v4_validated"
GUARD_POLICY_VERSIONS = {ACTIVE_GUARD_POLICY_VERSION, SHADOW_GUARD_POLICY_VERSION}


def _claim_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claim", "quote"],
        "properties": {
            "claim": {"type": "string"},
            "quote": {"type": "string"},
        },
    }


def _bounded_claim_schema() -> dict[str, Any]:
    schema = _claim_schema()
    schema["properties"]["claim"]["maxLength"] = 240
    schema["properties"]["quote"]["maxLength"] = 700
    return schema


LEGACY_DIRECT_JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_readability",
        "readability_reason",
        "event_identity",
        "event_actuality",
        "date_mentions",
        "event_date_relation",
        "event_locations",
        "location_quotes",
        "event_location_relation",
        "observed_hazards",
        "prospective_hazards",
        "hazard_quotes",
        "target_hazard_status",
        "realized_impacts",
        "potential_impacts_or_risks",
        "asset_or_economic_values",
        "aid_or_administrative_actions",
        "target_hazard_observed_impact",
        "explicit_target_hazard_to_impact_attribution",
        "attribution_quotes",
        "explicit_drought_to_wet_transition",
        "transition_quotes",
        "candidate_support_recommendation",
        "supporting_quotes",
        "uncertainty",
    ],
    "properties": {
        "source_readability": {"type": "string", "enum": sorted(SOURCE_READABILITY)},
        "readability_reason": {"type": "string"},
        "event_identity": {"type": "string", "enum": sorted(EVENT_IDENTITIES)},
        "event_actuality": {"type": "string", "enum": sorted(EVENT_ACTUALITIES)},
        "date_mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "semantic_role", "normalized_start", "normalized_end", "quote"],
                "properties": {
                    "text": {"type": "string"},
                    "semantic_role": {"type": "string", "enum": sorted(DATE_ROLES)},
                    "normalized_start": {"type": "string"},
                    "normalized_end": {"type": "string"},
                    "quote": {"type": "string"},
                },
            },
        },
        "event_date_relation": {"type": "string", "enum": ["aligned", "outside_candidate", "unknown"]},
        "event_locations": {"type": "array", "items": {"type": "string"}},
        "location_quotes": {"type": "array", "items": {"type": "string"}},
        "event_location_relation": {"type": "string", "enum": ["aligned", "not_aligned", "unknown"]},
        "observed_hazards": {"type": "array", "items": {"type": "string"}},
        "prospective_hazards": {"type": "array", "items": {"type": "string"}},
        "hazard_quotes": {"type": "array", "items": {"type": "string"}},
        "target_hazard_status": {"type": "string", "enum": sorted(TARGET_HAZARD_STATUSES)},
        "realized_impacts": {"type": "array", "items": _claim_schema()},
        "potential_impacts_or_risks": {"type": "array", "items": _claim_schema()},
        "asset_or_economic_values": {"type": "array", "items": _claim_schema()},
        "aid_or_administrative_actions": {"type": "array", "items": _claim_schema()},
        "target_hazard_observed_impact": {"type": "string", "enum": sorted(TRISTATE)},
        "explicit_target_hazard_to_impact_attribution": {"type": "string", "enum": sorted(TRISTATE)},
        "attribution_quotes": {"type": "array", "items": {"type": "string"}},
        "explicit_drought_to_wet_transition": {"type": "string", "enum": sorted(TRISTATE)},
        "transition_quotes": {"type": "array", "items": {"type": "string"}},
        "candidate_support_recommendation": {"type": "string", "enum": sorted(MODEL_SUPPORT)},
        "supporting_quotes": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "string"},
    },
}


def _date_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "semantic_role",
            "precision",
            "normalized_start",
            "normalized_end",
            "continuous_event_period",
            "relative_anchor",
            "quote",
        ],
        "properties": {
            "semantic_role": {"type": "string", "enum": sorted(NATIVE_DATE_ROLES)},
            "precision": {
                "type": "string",
                "enum": ["exact_day", "exact_interval", "month", "year", "relative", "unknown"],
            },
            "normalized_start": {"type": "string", "maxLength": 10},
            "normalized_end": {"type": "string", "maxLength": 10},
            "continuous_event_period": {"type": "boolean"},
            "relative_anchor": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["anchor_date", "anchor_role", "anchor_quote"],
                        "properties": {
                            "anchor_date": {"type": "string", "maxLength": 10},
                            "anchor_role": {"type": "string", "enum": sorted(RELATIVE_ANCHOR_ROLES)},
                            "anchor_quote": {"type": "string", "maxLength": 500},
                        },
                    },
                    {"type": "null"},
                ]
            },
            "quote": {"type": "string", "maxLength": 700},
        },
    }


def _observed_status_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["normalized_date", "date_quote", "status_quote"],
        "properties": {
            "normalized_date": {"type": "string", "maxLength": 10},
            "date_quote": {"type": "string", "maxLength": 500},
            "status_quote": {"type": "string", "maxLength": 700},
        },
    }


def _location_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["location", "kind", "quote"],
        "properties": {
            "location": {"type": "string", "maxLength": 160},
            "kind": {
                "type": "string",
                "enum": [
                    "event_location",
                    "document_scope",
                    "agency_jurisdiction",
                    "eligibility_area",
                    "unrelated_event_location",
                    "unknown",
                ],
            },
            "quote": {"type": "string", "maxLength": 700},
        },
    }


def _hazard_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hazard", "actuality", "quote"],
        "properties": {
            "hazard": {"type": "string", "maxLength": 160},
            "actuality": {"type": "string", "enum": ["observed", "prospective", "context_or_unknown"]},
            "quote": {"type": "string", "maxLength": 700},
        },
    }


CANDIDATE_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "event_summary",
        "binding_status",
        "actuality",
        "date_relation",
        "date_evidence",
        "observed_status_as_of",
        "location_relation",
        "location_evidence",
        "hazard_status",
        "hazard_evidence",
        "realized_impacts",
        "potential_impacts_or_risks",
        "asset_or_economic_values",
        "aid_or_administrative_actions",
        "observed_impact",
        "attribution",
        "attribution_quotes",
        "transition",
        "transition_quotes",
        "supporting_quotes",
    ],
    "properties": {
        "event_summary": {"type": "string", "maxLength": 360},
        "binding_status": {"type": "string", "enum": ["explicit_same_event", "unresolved"]},
        "actuality": {"type": "string", "enum": sorted(EVENT_ACTUALITIES)},
        "date_relation": {"type": "string", "enum": ["aligned", "outside_candidate", "unknown"]},
        "date_evidence": {"type": "array", "maxItems": 5, "items": _date_evidence_schema()},
        "observed_status_as_of": {"anyOf": [_observed_status_schema(), {"type": "null"}]},
        "location_relation": {"type": "string", "enum": ["aligned", "not_aligned", "unknown"]},
        "location_evidence": {"type": "array", "maxItems": 4, "items": _location_evidence_schema()},
        "hazard_status": {"type": "string", "enum": sorted(TARGET_HAZARD_STATUSES)},
        "hazard_evidence": {"type": "array", "maxItems": 4, "items": _hazard_evidence_schema()},
        "realized_impacts": {"type": "array", "maxItems": 4, "items": _bounded_claim_schema()},
        "potential_impacts_or_risks": {"type": "array", "maxItems": 2, "items": _bounded_claim_schema()},
        "asset_or_economic_values": {"type": "array", "maxItems": 2, "items": _bounded_claim_schema()},
        "aid_or_administrative_actions": {"type": "array", "maxItems": 2, "items": _bounded_claim_schema()},
        "observed_impact": {"type": "string", "enum": sorted(TRISTATE)},
        "attribution": {"type": "string", "enum": sorted(TRISTATE)},
        "attribution_quotes": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 700}},
        "transition": {"type": "string", "enum": sorted(TRISTATE)},
        "transition_quotes": {"type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 700}},
        "supporting_quotes": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 700}},
    },
}


DIRECT_JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_readability",
        "readability_reason",
        "event_identity",
        "candidate_event",
        "candidate_support_recommendation",
        "uncertainty",
    ],
    "properties": {
        "source_readability": {"type": "string", "enum": sorted(SOURCE_READABILITY)},
        "readability_reason": {"type": "string", "maxLength": 280},
        "event_identity": {"type": "string", "enum": sorted(EVENT_IDENTITIES)},
        "candidate_event": {"anyOf": [CANDIDATE_EVENT_SCHEMA, {"type": "null"}]},
        "candidate_support_recommendation": {"type": "string", "enum": sorted(MODEL_SUPPORT)},
        "uncertainty": {"type": "string", "maxLength": 360},
    },
}


DIRECT_JUDGE_INSTRUCTIONS = """You are CE-Agent's single direct webpage semantic judge. Judge only CANDIDATE and the complete SOURCE_TEXT. Do not browse, use outside knowledge, use prior labels, or infer why the source was selected.

Treat CANDIDATE.target_hazard_axis, CANDIDATE.target_window_start/end, and CANDIDATE.target_hazards as the only hazard-corroboration target for this request. A drought-targeted request must be judged against the drought target window and observed drought evidence; a wet-targeted request must be judged against the wet target window and observed rain/flood evidence. The query target scopes the question but is never evidence by itself.

First decide whether the source is readable and whether it contains zero concrete candidate-relevant events, exactly one uniquely candidate-relevant event, or unresolved event identity. A page may mention many events, but candidate_event must be either null or one single event selected as candidate-relevant. If more than one event remains plausible, or decisive fields cannot all be assigned to the same event, return unresolved_event_identity with candidate_event=null. Never combine fields from different events.

All decisive evidence belongs inside candidate_event and therefore must describe that same event. Keep event dates separate from publication, update, observed-status-as-of, warning-period, declaration-period, administrative, and historical-context dates. Record normalized dates only when mechanically supported by the exact quote. Month- or year-level text is coarse; it does not establish a narrow candidate window unless the quote explicitly states one continuous event period covering the window. Set continuous_event_period=true only for such explicit periods. For relative dates, use precision=relative and provide the exact relative quote plus one exact, grounded publication/update/status/document anchor; otherwise leave normalization empty. observed_status_as_of is only for a report that explicitly describes a currently observed condition as of an exact date, never for a bare publication or update timestamp.

Type each location quote. Only event_location means the event occurred there. Keep document_scope, agency_jurisdiction, eligibility_area, and unrelated_event_location separate. Distinguish observed or retrospective events from warnings, forecasts, orders, planning, and generic context. Bind hazard evidence, realized impacts, attribution, and transition to candidate_event only. Separate realized impacts from potential risk, exposure, asset/crop/economic value, aid, eligibility, budgets, declarations, and administrative action. An evacuation warning or order is not proof that evacuation occurred. Values are not losses unless the source explicitly calls them damage or loss.

Explicit hazard-to-impact attribution requires source language connecting a realized impact to the same event's target hazard; co-occurrence is not attribution. Explicit drought-to-wet transition requires source language describing that transition for the same event; chronology is insufficient.

Every quote must be an exact substring of SOURCE_TEXT. Keep at most five date items, four location items, four hazard items, four realized impacts, three attribution quotes, two transition quotes, and four supporting quotes. Prefer the smallest decisive evidence set. Keep event_summary, reasons, and uncertainty short. Use empty arrays rather than invented evidence. Set binding_status=explicit_same_event only when every populated decisive field is bound to the one candidate_event; otherwise use unresolved. The candidate-support recommendation is audit-only, but needs_review or does_not_support must never be contradicted by a newly constructed local positive. Return only the required JSON."""


class DirectJudgeSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class DirectCallResult:
    model: str
    status: str
    parsed: dict[str, Any] | None
    raw_output_text: str
    raw_responses: list[dict[str, Any]]
    attempts: int
    usage: dict[str, Any]
    error: str = ""
    retry_reason: str = ""
    request_sha256: str = ""


@dataclass(frozen=True)
class ReadabilityResult:
    usable: bool
    reason: str


def preflight_source_readability(body_text: str) -> ReadabilityResult:
    text = str(body_text or "").strip()
    if not text:
        return ReadabilityResult(False, "empty_body")
    lower = text.lower()
    captcha_markers = (
        "performing security verification",
        "this page maybe requiring captcha",
        "verify you are not a bot",
        "captcha-only",
    )
    if any(marker in lower for marker in captcha_markers):
        return ReadabilityResult(False, "captcha_or_security_challenge")
    if "markdown content:" in lower:
        tail = lower.split("markdown content:", 1)[1].strip()
        if not tail:
            return ReadabilityResult(False, "empty_extracted_content")
    replacement_count = text.count("\ufffd")
    if replacement_count >= 20 and replacement_count / max(1, len(text)) >= 0.02:
        return ReadabilityResult(False, "binary_or_corrupted_text")
    letters = sum(character.isalpha() for character in text)
    if len(text) < 120 or letters < 40:
        return ReadabilityResult(False, "insufficient_extracted_text")
    return ReadabilityResult(True, "readable")


def validate_legacy_direct_judgment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DirectJudgeSchemaError("direct_judgment_not_object")
    missing = [key for key in LEGACY_DIRECT_JUDGMENT_SCHEMA["required"] if key not in payload]
    extras = [key for key in payload if key not in LEGACY_DIRECT_JUDGMENT_SCHEMA["properties"]]
    if missing:
        raise DirectJudgeSchemaError(f"missing_required_keys:{','.join(missing)}")
    if extras:
        raise DirectJudgeSchemaError(f"unexpected_keys:{','.join(extras)}")
    _require_enum(payload, "source_readability", SOURCE_READABILITY)
    _require_enum(payload, "event_identity", EVENT_IDENTITIES)
    _require_enum(payload, "event_actuality", EVENT_ACTUALITIES)
    _require_enum(payload, "event_date_relation", {"aligned", "outside_candidate", "unknown"})
    _require_enum(payload, "event_location_relation", {"aligned", "not_aligned", "unknown"})
    _require_enum(payload, "target_hazard_status", TARGET_HAZARD_STATUSES)
    _require_enum(payload, "target_hazard_observed_impact", TRISTATE)
    _require_enum(payload, "explicit_target_hazard_to_impact_attribution", TRISTATE)
    _require_enum(payload, "explicit_drought_to_wet_transition", TRISTATE)
    _require_enum(payload, "candidate_support_recommendation", MODEL_SUPPORT)
    for key in (
        "event_locations",
        "location_quotes",
        "observed_hazards",
        "prospective_hazards",
        "hazard_quotes",
        "attribution_quotes",
        "transition_quotes",
        "supporting_quotes",
    ):
        _require_string_list(payload, key)
    if not isinstance(payload["date_mentions"], list):
        raise DirectJudgeSchemaError("invalid_date_mentions")
    for index, mention in enumerate(payload["date_mentions"]):
        if not isinstance(mention, dict):
            raise DirectJudgeSchemaError(f"invalid_date_mention:{index}")
        expected = {"text", "semantic_role", "normalized_start", "normalized_end", "quote"}
        if set(mention) != expected:
            raise DirectJudgeSchemaError(f"invalid_date_mention_keys:{index}")
        if mention["semantic_role"] not in DATE_ROLES:
            raise DirectJudgeSchemaError(f"invalid_date_role:{index}")
        if any(not isinstance(mention[key], str) for key in expected):
            raise DirectJudgeSchemaError(f"invalid_date_mention_value:{index}")
    for key in (
        "realized_impacts",
        "potential_impacts_or_risks",
        "asset_or_economic_values",
        "aid_or_administrative_actions",
    ):
        claims = payload[key]
        if not isinstance(claims, list):
            raise DirectJudgeSchemaError(f"invalid_{key}")
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict) or set(claim) != {"claim", "quote"}:
                raise DirectJudgeSchemaError(f"invalid_{key}_claim:{index}")
            if not isinstance(claim["claim"], str) or not isinstance(claim["quote"], str):
                raise DirectJudgeSchemaError(f"invalid_{key}_value:{index}")
    return json.loads(json.dumps(payload, ensure_ascii=False))


def validate_direct_judgment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DirectJudgeSchemaError("direct_judgment_not_object")
    missing = [key for key in DIRECT_JUDGMENT_SCHEMA["required"] if key not in payload]
    extras = [key for key in payload if key not in DIRECT_JUDGMENT_SCHEMA["properties"]]
    if missing:
        raise DirectJudgeSchemaError(f"missing_required_keys:{','.join(missing)}")
    if extras:
        raise DirectJudgeSchemaError(f"unexpected_keys:{','.join(extras)}")
    _require_enum(payload, "source_readability", SOURCE_READABILITY)
    _require_enum(payload, "event_identity", EVENT_IDENTITIES)
    _require_enum(payload, "candidate_support_recommendation", MODEL_SUPPORT)
    if not isinstance(payload.get("readability_reason"), str) or not isinstance(payload.get("uncertainty"), str):
        raise DirectJudgeSchemaError("invalid_top_level_string")
    event = payload.get("candidate_event")
    if event is None:
        if payload["event_identity"] == "one_uniquely_relevant_event":
            raise DirectJudgeSchemaError("unique_event_requires_candidate_event")
        return json.loads(json.dumps(payload, ensure_ascii=False))
    if payload["event_identity"] != "one_uniquely_relevant_event":
        raise DirectJudgeSchemaError("candidate_event_requires_unique_event_identity")
    if not isinstance(event, dict):
        raise DirectJudgeSchemaError("candidate_event_not_object_or_null")
    expected = set(CANDIDATE_EVENT_SCHEMA["required"])
    if set(event) != expected:
        raise DirectJudgeSchemaError("invalid_candidate_event_keys")
    for key in ("event_summary",):
        if not isinstance(event[key], str):
            raise DirectJudgeSchemaError(f"invalid_candidate_event_{key}")
    _require_enum(event, "binding_status", {"explicit_same_event", "unresolved"})
    _require_enum(event, "actuality", EVENT_ACTUALITIES)
    _require_enum(event, "date_relation", {"aligned", "outside_candidate", "unknown"})
    _require_enum(event, "location_relation", {"aligned", "not_aligned", "unknown"})
    _require_enum(event, "hazard_status", TARGET_HAZARD_STATUSES)
    _require_enum(event, "observed_impact", TRISTATE)
    _require_enum(event, "attribution", TRISTATE)
    _require_enum(event, "transition", TRISTATE)
    if not isinstance(event["date_evidence"], list):
        raise DirectJudgeSchemaError("invalid_candidate_event_date_evidence")
    if len(event["date_evidence"]) > 5:
        raise DirectJudgeSchemaError("too_many_candidate_event_date_evidence_items")
    date_keys = set(_date_evidence_schema()["required"])
    for index, mention in enumerate(event["date_evidence"]):
        if not isinstance(mention, dict) or set(mention) != date_keys:
            raise DirectJudgeSchemaError(f"invalid_candidate_event_date_evidence_keys:{index}")
        if mention["semantic_role"] not in NATIVE_DATE_ROLES:
            raise DirectJudgeSchemaError(f"invalid_candidate_event_date_role:{index}")
        if mention["precision"] not in {"exact_day", "exact_interval", "month", "year", "relative", "unknown"}:
            raise DirectJudgeSchemaError(f"invalid_candidate_event_date_precision:{index}")
        if not isinstance(mention["continuous_event_period"], bool):
            raise DirectJudgeSchemaError(f"invalid_candidate_event_continuous_period:{index}")
        if any(
            not isinstance(mention[key], str)
            for key in date_keys - {"continuous_event_period", "relative_anchor"}
        ):
            raise DirectJudgeSchemaError(f"invalid_candidate_event_date_value:{index}")
        anchor = mention["relative_anchor"]
        if anchor is not None:
            if not isinstance(anchor, dict) or set(anchor) != {"anchor_date", "anchor_role", "anchor_quote"}:
                raise DirectJudgeSchemaError(f"invalid_candidate_event_relative_anchor:{index}")
            if anchor["anchor_role"] not in RELATIVE_ANCHOR_ROLES:
                raise DirectJudgeSchemaError(f"invalid_candidate_event_relative_anchor_role:{index}")
            if any(not isinstance(anchor[key], str) for key in anchor):
                raise DirectJudgeSchemaError(f"invalid_candidate_event_relative_anchor_value:{index}")
        if mention["precision"] == "relative" and anchor is None:
            raise DirectJudgeSchemaError(f"relative_date_requires_anchor:{index}")
        if mention["precision"] != "relative" and anchor is not None:
            raise DirectJudgeSchemaError(f"nonrelative_date_forbids_anchor:{index}")
    status = event["observed_status_as_of"]
    if status is not None:
        if not isinstance(status, dict) or set(status) != {"normalized_date", "date_quote", "status_quote"}:
            raise DirectJudgeSchemaError("invalid_candidate_event_observed_status_as_of")
        if any(not isinstance(status[key], str) for key in status):
            raise DirectJudgeSchemaError("invalid_candidate_event_observed_status_value")
    if not isinstance(event["location_evidence"], list):
        raise DirectJudgeSchemaError("invalid_candidate_event_location_evidence")
    if len(event["location_evidence"]) > 4:
        raise DirectJudgeSchemaError("too_many_candidate_event_location_evidence_items")
    location_kinds = {
        "event_location",
        "document_scope",
        "agency_jurisdiction",
        "eligibility_area",
        "unrelated_event_location",
        "unknown",
    }
    for index, evidence in enumerate(event["location_evidence"]):
        if not isinstance(evidence, dict) or set(evidence) != {"location", "kind", "quote"}:
            raise DirectJudgeSchemaError(f"invalid_candidate_event_location_evidence:{index}")
        if evidence["kind"] not in location_kinds or any(not isinstance(evidence[key], str) for key in evidence):
            raise DirectJudgeSchemaError(f"invalid_candidate_event_location_value:{index}")
    if not isinstance(event["hazard_evidence"], list):
        raise DirectJudgeSchemaError("invalid_candidate_event_hazard_evidence")
    if len(event["hazard_evidence"]) > 4:
        raise DirectJudgeSchemaError("too_many_candidate_event_hazard_evidence_items")
    for index, evidence in enumerate(event["hazard_evidence"]):
        if not isinstance(evidence, dict) or set(evidence) != {"hazard", "actuality", "quote"}:
            raise DirectJudgeSchemaError(f"invalid_candidate_event_hazard_evidence:{index}")
        if evidence["actuality"] not in {"observed", "prospective", "context_or_unknown"}:
            raise DirectJudgeSchemaError(f"invalid_candidate_event_hazard_actuality:{index}")
        if any(not isinstance(evidence[key], str) for key in evidence):
            raise DirectJudgeSchemaError(f"invalid_candidate_event_hazard_value:{index}")
    for key in (
        "realized_impacts",
        "potential_impacts_or_risks",
        "asset_or_economic_values",
        "aid_or_administrative_actions",
    ):
        claims = event[key]
        if not isinstance(claims, list):
            raise DirectJudgeSchemaError(f"invalid_candidate_event_{key}")
        limit = 4 if key == "realized_impacts" else 2
        if len(claims) > limit:
            raise DirectJudgeSchemaError(f"too_many_candidate_event_{key}_items")
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict) or set(claim) != {"claim", "quote"}:
                raise DirectJudgeSchemaError(f"invalid_candidate_event_{key}_claim:{index}")
            if not isinstance(claim["claim"], str) or not isinstance(claim["quote"], str):
                raise DirectJudgeSchemaError(f"invalid_candidate_event_{key}_value:{index}")
    for key, limit in (("attribution_quotes", 3), ("transition_quotes", 2), ("supporting_quotes", 4)):
        _require_string_list(event, key)
        if len(event[key]) > limit:
            raise DirectJudgeSchemaError(f"too_many_candidate_event_{key}_items")
    return json.loads(json.dumps(payload, ensure_ascii=False))


class DirectPageJudgeClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = EXACT_MODEL,
        reasoning_effort: str = REASONING_EFFORT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        responses_url: str = RESPONSES_URL,
        timeout_seconds: int = 360,
        opener: Callable[..., Any] | None = None,
        instructions: str = DIRECT_JUDGE_INSTRUCTIONS,
        output_schema: dict[str, Any] = DIRECT_JUDGMENT_SCHEMA,
        schema_name: str = "ce_agent_full_body_direct_judgment_v4_targeted_hazard_axis",
        validator: Callable[[Any], dict[str, Any]] = validate_direct_judgment,
        max_input_tokens: int = 131_072,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 262_144,
        max_error_response_bytes: int = 65_536,
        max_redirects: int = 0,
        max_redirect_response_bytes: int = 16_384,
        max_header_bytes: int = 65_536,
        max_redirect_location_bytes: int = 4_096,
    ) -> None:
        if not api_key:
            raise ValueError("missing_openai_api_key")
        if model != EXACT_MODEL:
            raise ValueError(f"direct_judge_requires_exact_model:{EXACT_MODEL}")
        if reasoning_effort != "high":
            raise ValueError("direct_judge_requires_high_reasoning")
        if max_output_tokens < MIN_MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_output_tokens_below_minimum:{MIN_MAX_OUTPUT_TOKENS}")
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.responses_url = responses_url
        self.timeout_seconds = timeout_seconds
        if int(max_redirects) != 0:
            raise ValueError("direct_judge_redirects_must_be_disabled")
        self.opener = opener or no_redirect_opener()
        self.instructions = instructions
        self.output_schema = output_schema
        self.schema_name = schema_name
        self.validator = validator
        for name, value in (
            ("max_input_tokens", max_input_tokens),
            ("max_request_bytes", max_request_bytes),
            ("max_response_bytes", max_response_bytes),
            ("max_error_response_bytes", max_error_response_bytes),
            ("max_redirect_response_bytes", max_redirect_response_bytes),
            ("max_header_bytes", max_header_bytes),
            ("max_redirect_location_bytes", max_redirect_location_bytes),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name}_must_be_positive")
        self.max_input_tokens = int(max_input_tokens)
        self.max_request_bytes = int(max_request_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.max_error_response_bytes = int(max_error_response_bytes)
        self.max_redirects = 0
        self.max_redirect_response_bytes = int(max_redirect_response_bytes)
        self.max_header_bytes = int(max_header_bytes)
        self.max_redirect_location_bytes = int(max_redirect_location_bytes)
        self._tokenizer = tiktoken.get_encoding("o200k_base")

    def judge(self, *, candidate: dict[str, Any], source: dict[str, Any], body_text: str) -> DirectCallResult:
        readability = preflight_source_readability(body_text)
        if not readability.usable:
            return DirectCallResult(
                model=self.model,
                status="not_requested_unusable_source",
                parsed=None,
                raw_output_text="",
                raw_responses=[],
                attempts=0,
                usage={},
                error=readability.reason,
            )
        payload = self._payload(candidate=candidate, source=source, body_text=body_text)
        wire = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request_hash = sha256(wire).hexdigest()
        raw_responses: list[dict[str, Any]] = []
        usage_totals: dict[str, int] = {}
        last_error = ""
        retry_reason = ""
        raw_text = ""
        for attempt in (1, 2):
            try:
                response = self._request(wire)
                raw_responses.append(response)
                _add_usage(usage_totals, response.get("usage") or {})
                if _response_is_explicitly_incomplete(response):
                    last_error = _incomplete_reason(response)
                    if attempt == 1:
                        retry_reason = last_error
                        continue
                    return DirectCallResult(
                        model=self.model,
                        status="incomplete",
                        parsed=None,
                        raw_output_text=_extract_output_text(response),
                        raw_responses=raw_responses,
                        attempts=attempt,
                        usage=usage_totals,
                        error=last_error,
                        retry_reason=retry_reason,
                        request_sha256=request_hash,
                    )
                raw_text = _extract_output_text(response)
                try:
                    parsed = self.validator(json.loads(raw_text))
                except (json.JSONDecodeError, DirectJudgeSchemaError) as exc:
                    return DirectCallResult(
                        model=self.model,
                        status="invalid_output",
                        parsed=None,
                        raw_output_text=raw_text,
                        raw_responses=raw_responses,
                        attempts=attempt,
                        usage=usage_totals,
                        error=_safe_error(exc, self.api_key),
                        retry_reason=retry_reason,
                        request_sha256=request_hash,
                    )
                return DirectCallResult(
                    model=self.model,
                    status="ok",
                    parsed=parsed,
                    raw_output_text=raw_text,
                    raw_responses=raw_responses,
                    attempts=attempt,
                    usage=usage_totals,
                    retry_reason=retry_reason,
                    request_sha256=request_hash,
                )
            except urllib.error.HTTPError as exc:
                _, detail = read_http_error(
                    exc,
                    max_error_bytes=self.max_error_response_bytes,
                    max_redirect_bytes=self.max_redirect_response_bytes,
                    max_header_bytes=self.max_header_bytes,
                    max_location_bytes=self.max_redirect_location_bytes,
                )
                last_error = f"http_{exc.code}:{_redact(detail, self.api_key)}"
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_error = f"transport:{_safe_error(exc, self.api_key)}"
                retryable = True
            except Exception as exc:  # provider/transport envelope failure, never semantic repair
                last_error = f"provider:{_safe_error(exc, self.api_key)}"
                retryable = True
            if attempt == 1 and retryable:
                retry_reason = last_error
                continue
            return DirectCallResult(
                model=self.model,
                status="request_failed",
                parsed=None,
                raw_output_text=raw_text,
                raw_responses=raw_responses,
                attempts=attempt,
                usage=usage_totals,
                error=last_error,
                retry_reason=retry_reason,
                request_sha256=request_hash,
            )
        raise AssertionError("unreachable")

    def _payload(self, *, candidate: dict[str, Any], source: dict[str, Any], body_text: str) -> dict[str, Any]:
        user_payload = {
            "CANDIDATE": candidate,
            "SOURCE": source,
            "SOURCE_TEXT": body_text,
        }
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": self.instructions}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(user_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        }
                    ],
                },
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "strict": True,
                    "schema": self.output_schema,
                }
            },
        }

    def _request(self, wire: bytes) -> dict[str, Any]:
        if len(wire) > self.max_request_bytes:
            raise ValueError(
                f"openai_request_body_exceeds_limit:{self.max_request_bytes}"
            )
        input_tokens = len(
            self._tokenizer.encode(wire.decode("utf-8", errors="strict"))
        )
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                f"openai_input_tokens_exceed_limit:{self.max_input_tokens}"
            )
        request = urllib.request.Request(
            self.responses_url,
            data=wire,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with self.opener(request, timeout=self.timeout_seconds) as response:
            validate_headers(
                getattr(response, "headers", {}),
                max_header_bytes=self.max_header_bytes,
                max_location_bytes=self.max_redirect_location_bytes,
            )
            raw_bytes = read_bounded(
                response,
                max_bytes=self.max_response_bytes,
                label="direct_judge_response_body",
            ).data
        parsed = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        if not isinstance(parsed, dict):
            raise ValueError("provider_response_not_object")
        return parsed


def parse_saved_direct_responses(
    *,
    model: str,
    raw_responses: list[dict[str, Any]],
    request_sha256: str,
    attempts: int,
    provider_status: str,
    provider_error: str = "",
    retry_reason: str = "",
    validator: Callable[[Any], dict[str, Any]] = validate_direct_judgment,
) -> DirectCallResult:
    """Reparse a durable provider boundary without issuing a model call."""

    usage: dict[str, int] = {}
    for response in raw_responses:
        _add_usage(usage, response.get("usage") or {})
    if provider_status == "not_requested_unusable_source":
        return DirectCallResult(
            model=model, status=provider_status, parsed=None,
            raw_output_text="", raw_responses=[], attempts=0, usage={},
            error=provider_error, request_sha256=request_sha256,
        )
    if not raw_responses:
        return DirectCallResult(
            model=model, status="request_failed", parsed=None,
            raw_output_text="", raw_responses=[], attempts=attempts,
            usage=usage, error=provider_error or "saved_provider_response_missing",
            retry_reason=retry_reason, request_sha256=request_sha256,
        )
    response = raw_responses[-1]
    if _response_is_explicitly_incomplete(response):
        return DirectCallResult(
            model=model, status="incomplete", parsed=None,
            raw_output_text=_extract_output_text(response),
            raw_responses=raw_responses, attempts=attempts, usage=usage,
            error=_incomplete_reason(response), retry_reason=retry_reason,
            request_sha256=request_sha256,
        )
    raw_text = _extract_output_text(response)
    try:
        parsed = validator(json.loads(raw_text))
    except (json.JSONDecodeError, DirectJudgeSchemaError) as exc:
        return DirectCallResult(
            model=model, status="invalid_output", parsed=None,
            raw_output_text=raw_text, raw_responses=raw_responses,
            attempts=attempts, usage=usage, error=str(exc)[:800],
            retry_reason=retry_reason, request_sha256=request_sha256,
        )
    return DirectCallResult(
        model=model, status="ok", parsed=parsed, raw_output_text=raw_text,
        raw_responses=raw_responses, attempts=attempts, usage=usage,
        retry_reason=retry_reason, request_sha256=request_sha256,
    )


def derive_local_page_result(
    *,
    semantic: dict[str, Any] | None,
    candidate: dict[str, Any],
    body_text: str,
    call_status: str = "ok",
    guard_policy_version: str = ACTIVE_GUARD_POLICY_VERSION,
) -> dict[str, Any]:
    if guard_policy_version not in GUARD_POLICY_VERSIONS:
        raise ValueError(f"unsupported_guard_policy_version:{guard_policy_version}")
    shadow_v4 = guard_policy_version == SHADOW_GUARD_POLICY_VERSION
    date_relation_deriver = (
        derive_candidate_event_date_relation_shadow_v4
        if shadow_v4
        else derive_candidate_event_date_relation
    )

    def negative_reason_for(selected_event: dict[str, Any]) -> str:
        return _candidate_event_negative_reason(
            selected_event,
            candidate,
            date_relation_deriver=date_relation_deriver,
        )

    actions: list[dict[str, str]] = []
    if shadow_v4:
        _action(actions, "guard_policy_version", "applied", SHADOW_GUARD_POLICY_VERSION)
    readability = preflight_source_readability(body_text)
    if not readability.usable:
        _action(actions, "source_readability", "blocked", readability.reason)
        # An unusable source contributes no affirmative webpage evidence and
        # must not contaminate every case axis with semantic uncertainty.
        return _local_result("insufficient_source_content", "no", "no", "no", "no", actions)
    _action(actions, "source_readability", "passed", "body_passed_general_readability_preflight")
    if semantic is None or call_status != "ok":
        _action(actions, "output_completeness", "blocked", f"terminal_call_status={call_status}")
        return _local_result("unresolved", *_unresolved_without_semantic_axes(body_text), actions)
    legacy_adapter_used = "candidate_event" not in semantic
    try:
        if legacy_adapter_used:
            semantic = _adapt_legacy_judgment(validate_legacy_direct_judgment(semantic), candidate)
        else:
            semantic = validate_direct_judgment(semantic)
    except DirectJudgeSchemaError as exc:
        _action(actions, "output_completeness", "blocked", str(exc))
        return _local_result("unresolved", *_unresolved_without_semantic_axes(body_text), actions)
    _action(actions, "output_completeness", "passed", "strict_schema_complete")
    if legacy_adapter_used:
        _action(actions, "legacy_candidate_event_adapter", "applied", "frozen_page_level_output_adapted_fail_closed")

    semantic, removed_quote_count = _filter_ungrounded_evidence(semantic, body_text)
    if removed_quote_count:
        _action(
            actions,
            "quotation_grounding",
            "filtered",
            f"ungrounded_evidence_items_removed={removed_quote_count}",
        )
    else:
        _action(actions, "quotation_grounding", "passed", "all_nonempty_evidence_quotes_grounded")

    if legacy_adapter_used and isinstance(semantic.get("candidate_event"), dict):
        witnesses = _legacy_same_event_support_witnesses(semantic["candidate_event"], candidate)
        semantic["_legacy_binding_witnesses"] = witnesses
        event = semantic["candidate_event"]
        eligible_positive = (
            semantic["event_identity"] == "one_uniquely_relevant_event"
            and semantic["candidate_support_recommendation"] == "supports"
            and event["actuality"] in {"observed", "retrospective_observed"}
            and event["date_relation"] == "aligned"
            and event["location_relation"] == "aligned"
            and event["hazard_status"] == "observed"
            and bool(witnesses)
        )
        event["binding_status"] = "explicit_same_event" if eligible_positive else "unresolved"
        _action(
            actions,
            "legacy_same_event_package",
            "passed" if eligible_positive else "blocked",
            (
                f"mechanical_same_event_witnesses={len(witnesses)}"
                if eligible_positive
                else "separate_legacy_arrays_do_not_mechanically_prove_one_event"
            ),
        )

    if semantic["source_readability"] == "unusable":
        _action(
            actions,
            "model_readability",
            "corrected_or_preserved",
            "model_marked_unusable_but_objective_body_preflight_passed;continue_with_complete_semantic_fields",
        )
    elif semantic["source_readability"] == "uncertain":
        _action(
            actions,
            "model_readability",
            "corrected_or_preserved",
            "model_readability_uncertain_but_objective_body_preflight_passed;continue_with_complete_semantic_fields",
        )
    else:
        _action(actions, "model_readability", "passed", "model_marked_readable")
    if (
        semantic["source_readability"] in {"unusable", "uncertain"}
        and semantic["candidate_support_recommendation"] == "supports"
    ):
        _action(
            actions,
            "field_consistency",
            "blocked",
            "positive_semantics_conflict_with_model_readability",
        )
        return _local_result("unresolved", *_unresolved_without_semantic_axes(body_text), actions)

    recommendation = semantic["candidate_support_recommendation"]
    identity = semantic["event_identity"]
    if identity == "unresolved_event_identity":
        if shadow_v4 and recommendation == "does_not_support":
            _action(
                actions,
                "shadow_full_body_negative_closure",
                "rejected",
                "completed_candidate_relative_judge_found_no_unique_supporting_event",
            )
            return _local_result("does_not_support", "no", "no", "no", "no", actions)
        _action(actions, "event_identity", "blocked", "unique_candidate_relevant_event_not_established")
        return _local_result("unresolved", *_unresolved_without_semantic_axes(body_text), actions)
    if identity == "zero_concrete_relevant_event":
        _action(actions, "event_identity", "rejected", "zero_concrete_candidate_relevant_event")
        return _local_result(
            "does_not_support",
            "no",
            "no",
            "no",
            "no",
            actions,
            source_axes={
                "observed_target_hazard": "no",
                "realized_target_hazard_impact": "no",
                "explicit_target_hazard_to_impact_attribution": "no",
                "explicit_drought_to_wet_transition": "no",
            },
        )
    event = semantic.get("candidate_event")
    if not isinstance(event, dict):
        _action(actions, "event_identity", "blocked", "candidate_event_missing_for_unique_identity")
        return _local_result("unresolved", *_unresolved_without_semantic_axes(body_text), actions)
    _action(actions, "event_identity", "passed", "one_uniquely_relevant_event")
    source_axes = _derive_source_event_axes(semantic, candidate)
    legacy_witnesses = semantic.get("_legacy_binding_witnesses") or []
    unresolved_axes = _unresolved_candidate_axes(
        event,
        candidate,
        legacy_adapter_used=legacy_adapter_used,
        legacy_witnesses=legacy_witnesses,
    )

    if legacy_adapter_used and recommendation != "supports":
        negative_reason = negative_reason_for(event)
        legacy_outcome = "does_not_support" if negative_reason else "unresolved"
        action = "rejected" if negative_reason else "blocked"
        _action(
            actions,
            "legacy_model_recommendation_ceiling",
            action,
            negative_reason or "legacy_negative_not_mechanically_bound_to_selected_event",
        )
        if legacy_outcome == "does_not_support":
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    shadow_observed_hazard_promotion = bool(
        shadow_v4
        and recommendation == "needs_review"
        and _shadow_v4_observed_hazard_only_promotion_is_safe(
            event,
            candidate,
            date_relation_deriver=date_relation_deriver,
        )
    )
    shadow_bound_model_negative = bool(
        shadow_v4
        and recommendation == "does_not_support"
        and event.get("binding_status") == "explicit_same_event"
        and event.get("date_relation") == "outside_candidate"
        and _shadow_v4_model_outside_closure_is_safe(event, candidate)
    )
    if shadow_bound_model_negative:
        _action(
            actions,
            "shadow_full_body_negative_closure",
            "rejected",
            "completed_candidate_relative_judge_bound_selected_event_outside_candidate",
        )
        return _local_result(
            "does_not_support",
            "no",
            "no",
            "no",
            "no",
            actions,
            source_axes=source_axes,
        )
    shadow_all_axes_negative = bool(
        shadow_v4
        and recommendation != "supports"
        and event.get("binding_status") == "explicit_same_event"
        and unresolved_axes == ("no", "no", "no", "no")
    )
    if shadow_all_axes_negative:
        _action(
            actions,
            "shadow_bounded_axis_negative_closure",
            "rejected",
            "selected_event_has_no_remaining_candidate_axis_support_path",
        )
        return _local_result(
            "does_not_support",
            "no",
            "no",
            "no",
            "no",
            actions,
            source_axes=source_axes,
        )
    if recommendation != "supports" and not shadow_observed_hazard_promotion:
        negative_reason = negative_reason_for(event)
        if negative_reason:
            _action(actions, "model_recommendation_ceiling", "rejected", negative_reason)
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        _action(actions, "model_recommendation_ceiling", "blocked", f"model_recommendation={recommendation}")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    if shadow_observed_hazard_promotion:
        _action(
            actions,
            "shadow_observed_hazard_promotion",
            "passed",
            "needs_review_contains_separable_observed_target_hazard_with_complete_candidate_binding",
        )

    # A mechanically verified necessary failure can close a page even when the
    # model's audit-only recommendation says support. This runs before the
    # positive binding gate because an outside event cannot have an aligned
    # legacy witness package.
    negative_reason = negative_reason_for(event)
    if negative_reason:
        _action(actions, "candidate_event_negative_closure", "rejected", negative_reason)
        return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)

    if event["binding_status"] != "explicit_same_event":
        _action(actions, "candidate_event_binding", "blocked", "decisive_fields_not_proven_to_belong_to_one_event")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    _action(actions, "candidate_event_binding", "passed", "all_decisive_fields_bound_to_candidate_event")

    actuality = event["actuality"]
    if actuality in {"warning_or_forecast", "order_or_administrative_action", "planning_or_generic_context"}:
        negative_reason = negative_reason_for(event)
        if negative_reason:
            _action(actions, "event_actuality", "rejected", negative_reason)
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        _action(actions, "event_actuality", "blocked", f"non_observed_actuality_not_mechanically_proven={actuality}")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    if actuality == "mixed_or_uncertain":
        negative_reason = negative_reason_for(event)
        if negative_reason and event["observed_impact"] == "no":
            _action(
                actions,
                "field_consistency",
                "rejected",
                f"mixed_page_with_mechanically_verified_failure:{negative_reason}",
            )
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        if shadow_observed_hazard_promotion:
            _action(
                actions,
                "event_actuality",
                "passed",
                "observed_target_hazard_separable_from_prospective_impact_language",
            )
        else:
            _action(actions, "event_actuality", "blocked", "observed_event_not_separable_from_prospective_or_context_content")
            return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    if actuality != "mixed_or_uncertain":
        _action(actions, "event_actuality", "passed", actuality)

    model_date_relation = event["date_relation"]
    if model_date_relation == "outside_candidate":
        if date_relation_deriver(event, candidate) == "outside_candidate":
            _action(actions, "event_date", "rejected", "selected_event_exact_date_is_outside_candidate")
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        _action(actions, "event_date", "blocked", "outside_relation_not_mechanically_proven_by_bound_event_date")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    if model_date_relation == "unknown":
        _action(actions, "event_date", "blocked", "model_selected_event_timing_unknown")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    event_relation = (
        _legacy_bound_date_relation(event, candidate, legacy_witnesses)
        if legacy_adapter_used
        else date_relation_deriver(event, candidate)
    )
    if event_relation == "outside_candidate":
        _action(actions, "event_date", "rejected", "event_dates_outside_candidate_window")
        return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
    if event_relation == "unknown":
        _action(actions, "event_date", "blocked", "event_timing_not_established_by_event_role_dates")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    _action(actions, "event_date", "passed", "event_date_overlaps_candidate_window")

    model_location_relation = event["location_relation"]
    if model_location_relation == "not_aligned":
        if derive_candidate_event_location_relation(event, candidate) == "not_aligned":
            _action(actions, "event_location", "rejected", "selected_event_location_is_outside_candidate")
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        _action(actions, "event_location", "blocked", "not_aligned_relation_not_proven_by_event_location")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    if model_location_relation == "unknown":
        _action(actions, "event_location", "blocked", "model_selected_event_location_unknown")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    location_relation = (
        _legacy_bound_location_relation(event, candidate, legacy_witnesses)
        if legacy_adapter_used
        else derive_candidate_event_location_relation(event, candidate)
    )
    if location_relation == "not_aligned":
        _action(actions, "event_location", "rejected", "event_location_not_candidate_aligned")
        return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
    if location_relation == "unknown":
        _action(actions, "event_location", "blocked", "event_location_not_locally_verifiable")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    _action(actions, "event_location", "passed", "candidate_county_or_locality_grounded")

    model_hazard_status = event["hazard_status"]
    if model_hazard_status in {"anticipated_or_warning", "incompatible_hazard_only", "absent_or_context"}:
        negative_reason = negative_reason_for(event)
        if negative_reason:
            _action(actions, "hazard_compatibility", "rejected", negative_reason)
            return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
        _action(actions, "hazard_compatibility", "blocked", f"negative_hazard_status_not_mechanically_proven={model_hazard_status}")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    if model_hazard_status == "uncertain":
        _action(actions, "hazard_compatibility", "blocked", "model_selected_event_hazard_uncertain")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    hazard_relation = (
        _legacy_bound_hazard_relation(event, candidate, legacy_witnesses)
        if legacy_adapter_used
        else derive_candidate_event_hazard_relation(event, candidate)
    )
    if hazard_relation == "incompatible":
        _action(actions, "hazard_compatibility", "rejected", "observed_hazards_incompatible_with_candidate_targets")
        return _local_result("does_not_support", "no", "no", "no", "no", actions, source_axes=source_axes)
    if hazard_relation == "unknown":
        _action(actions, "hazard_compatibility", "blocked", "observed_compatible_hazard_not_mechanically_established")
        return _local_result("unresolved", *unresolved_axes, actions, source_axes=source_axes)
    _action(actions, "hazard_compatibility", "passed", "observed_compatible_target_hazard")

    realized = event["realized_impacts"]
    if legacy_adapter_used:
        realized = [
            claim
            for claim in realized
            if _legacy_axis_quote_bound_to_candidate_event(
                str(claim.get("quote") or ""), legacy_witnesses, event, candidate
            )
        ]
    impact_support = "yes" if realized and event["observed_impact"] == "yes" else "no"
    if not realized and (
        event["potential_impacts_or_risks"]
        or event["asset_or_economic_values"]
        or event["aid_or_administrative_actions"]
    ):
        _action(actions, "realized_impact", "corrected_or_preserved", "risk_value_aid_or_action_not_realized_impact")
    elif impact_support == "yes":
        _action(actions, "realized_impact", "passed", "realized_target_hazard_impact_grounded")
    elif event["observed_impact"] == "uncertain":
        impact_support = "unresolved"
        _action(actions, "realized_impact", "blocked", "realized_impact_uncertain")
    else:
        _action(actions, "realized_impact", "passed", "no_realized_target_hazard_impact")

    attribution = "no"
    attribution_quotes = event["attribution_quotes"]
    if legacy_adapter_used:
        attribution_quotes = [
            quote
            for quote in attribution_quotes
            if _legacy_axis_quote_bound_to_candidate_event(quote, legacy_witnesses, event, candidate)
            and _quote_has_target_hazard_and_impact(quote, candidate)
        ]
    if impact_support == "yes" and event["attribution"] == "yes" and attribution_quotes:
        attribution = "yes"
        _action(actions, "explicit_attribution", "passed", "explicit_grounded_hazard_to_impact_language")
    elif event["attribution"] == "uncertain" and impact_support != "no" and attribution_quotes:
        attribution = "unresolved"
        _action(actions, "explicit_attribution", "blocked", "attribution_uncertain")
    else:
        _action(actions, "explicit_attribution", "corrected_or_preserved", "cooccurrence_or_missing_explicit_link_not_attribution")

    transition = "no"
    transition_quotes = event["transition_quotes"]
    if legacy_adapter_used:
        transition_quotes = [
            quote
            for quote in transition_quotes
            if _legacy_axis_quote_bound_to_candidate_event(quote, legacy_witnesses, event, candidate)
            and "drought" in _hazard_families(quote)
            and bool(_target_hazard_families(candidate) & _hazard_families(quote))
        ]
    if event["transition"] == "yes" and transition_quotes:
        transition = "yes"
        _action(actions, "explicit_transition", "passed", "explicit_grounded_drought_to_wet_transition")
    elif event["transition"] == "uncertain" and transition_quotes:
        transition = "unresolved"
        _action(actions, "explicit_transition", "blocked", "transition_uncertain")
    else:
        _action(actions, "explicit_transition", "corrected_or_preserved", "chronology_or_absent_language_not_transition")

    _action(actions, "field_consistency", "passed", "support_validated_without_constructing_new_positive_facts")
    return _local_result("supports", "yes", impact_support, attribution, transition, actions, source_axes=source_axes)


def derive_candidate_event_date_relation(event: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Validate timing without selecting a different page-level date."""
    window_start, window_end = _candidate_window(candidate)
    if not window_start or not window_end:
        return "unknown"
    event_roles = {"event_start", "event_end", "event_day", "incident_period"}
    intervals: list[tuple[date, date]] = []
    coarse_overlap = False
    for mention in event.get("date_evidence") or []:
        if mention.get("semantic_role") not in event_roles or not str(mention.get("quote") or "").strip():
            continue
        precision = mention.get("precision")
        verified = _verified_date_evidence_interval(mention)
        if verified is None:
            coarse_start = _parse_iso_date(str(mention.get("normalized_start") or ""))
            coarse_end = _parse_iso_date(str(mention.get("normalized_end") or "")) or coarse_start
            if precision in {"month", "year"} and coarse_start and coarse_end and coarse_start <= window_end and coarse_end >= window_start:
                coarse_overlap = True
            continue
        intervals.append(verified)
    if not intervals:
        return "unknown"
    if any(start <= window_end and end >= window_start for start, end in intervals):
        return "aligned"
    if coarse_overlap:
        return "unknown"
    return "outside_candidate"


def _shadow_v4_month_day_in_quote(quote: str, value: date) -> bool:
    text = _normalize_quote(quote).lower()
    month_names = [name for name, number in _MONTHS.items() if number == value.month]
    names = "|".join(sorted({re.escape(name) for name in month_names}, key=len, reverse=True))
    return bool(
        re.search(
            rf"\b(?:{names})\.?\s+{value.day}(?:st|nd|rd|th)?\b",
            text,
        )
    )


def _shadow_v4_anchor_dates(event: dict[str, Any]) -> set[date]:
    anchors: set[date] = set()
    for mention in event.get("date_evidence") or []:
        if mention.get("semantic_role") not in RELATIVE_ANCHOR_ROLES:
            continue
        verified = _verified_date_evidence_interval(mention)
        if verified and verified[0] == verified[1]:
            anchors.add(verified[0])
    return anchors


def _shadow_v4_model_outside_closure_is_safe(
    event: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    """Allow a model-negative distant event anchor without accepting overlap ambiguity."""

    window_start, window_end = _candidate_window(candidate)
    if not window_start or not window_end:
        return False
    anchors = _shadow_v4_anchor_dates(event)
    if not anchors:
        return False
    # A publication/update anchor is only a safe negative discriminator when
    # it is well separated from the candidate window.  It never supplies
    # positive event timing and cannot override a coarse interval that could
    # overlap the candidate.
    if any(window_start - timedelta(days=31) <= value <= window_end + timedelta(days=31) for value in anchors):
        return False
    for mention in event.get("date_evidence") or []:
        if mention.get("semantic_role") in RELATIVE_ANCHOR_ROLES:
            continue
        if mention.get("precision") not in {"month", "year"}:
            continue
        start = _parse_iso_date(str(mention.get("normalized_start") or ""))
        end = _parse_iso_date(str(mention.get("normalized_end") or "")) or start
        if start and end and start <= window_end and end >= window_start:
            return False
    return True


def _shadow_v4_quote_names_interval(
    quote: str,
    start: date,
    end: date,
    anchors: set[date],
) -> bool:
    explicit = _explicit_dates_in_quote(quote)
    if start in explicit and end in explicit:
        return True
    if start.year != end.year or start.month != end.month:
        return False
    text = _normalize_quote(quote).lower()
    month_names = [name for name, number in _MONTHS.items() if number == start.month]
    names = "|".join(sorted({re.escape(name) for name in month_names}, key=len, reverse=True))
    matched = re.search(
        rf"\b(?:{names})\.?\s+{start.day}(?:st|nd|rd|th)?\s*"
        rf"(?:-|\u2013|\u2014|to|through|and)\s*{end.day}(?:st|nd|rd|th)?\b",
        text,
    )
    if not matched:
        return False
    return bool(re.search(rf"\b{start.year}\b", text) or any(value.year == start.year for value in anchors))


def _shadow_v4_relative_interval(
    mention: dict[str, Any],
) -> tuple[date, date] | None:
    anchor = mention.get("relative_anchor")
    if not isinstance(anchor, dict):
        return None
    anchor_date = _parse_iso_date(str(anchor.get("anchor_date") or ""))
    anchor_quote = str(anchor.get("anchor_quote") or "")
    if (
        not anchor_date
        or anchor.get("anchor_role") not in RELATIVE_ANCHOR_ROLES
        or anchor_date not in _explicit_dates_in_quote(anchor_quote)
    ):
        return None
    quote = str(mention.get("quote") or "")
    text = _normalize_quote(quote).lower()
    resolved_day = _relative_date_from_quote(quote, anchor_date)
    weekdays = re.findall(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
    )
    resolved: list[date] = []
    for weekday in weekdays:
        target = _WEEKDAYS[weekday]
        delta = (anchor_date.weekday() - target) % 7
        resolved.append(anchor_date - timedelta(days=delta))
    if resolved_day:
        interval = (resolved_day, resolved_day)
    elif resolved:
        interval = (min(resolved), max(resolved))
    elif re.search(r"\b(?:last|another|the)\s+weekend\b", text):
        days_since_sunday = (anchor_date.weekday() - _WEEKDAYS["sunday"]) % 7
        sunday = anchor_date - timedelta(days=days_since_sunday or 7)
        interval = (sunday - timedelta(days=1), sunday)
    elif re.search(r"\b(now|currently|as of)\b", text):
        interval = (anchor_date, anchor_date)
    else:
        return None
    start = _parse_iso_date(str(mention.get("normalized_start") or ""))
    end = _parse_iso_date(str(mention.get("normalized_end") or "")) or start
    if start is None and end is not None and interval[0] == interval[1] == end:
        return interval
    return interval if start == interval[0] and end == interval[1] else None


def _shadow_v4_verified_date_evidence_interval(
    mention: dict[str, Any],
    *,
    anchors: set[date],
) -> tuple[date, date] | None:
    strict = _verified_date_evidence_interval(mention)
    if strict:
        return strict
    quote = str(mention.get("quote") or "")
    precision = mention.get("precision")
    if precision == "relative":
        return _shadow_v4_relative_interval(mention)
    start = _parse_iso_date(str(mention.get("normalized_start") or ""))
    end = _parse_iso_date(str(mention.get("normalized_end") or "")) or start
    if not quote or not start or not end:
        return None
    if precision == "exact_day" and start == end:
        if _shadow_v4_month_day_in_quote(quote, start) and start in anchors:
            return (start, end)
        if mention.get("semantic_role") == "observed_status_as_of" and start in _explicit_dates_in_quote(quote):
            return (start, end)
    if precision == "exact_interval" and _shadow_v4_quote_names_interval(quote, start, end, anchors):
        return (min(start, end), max(start, end))
    return None


def _shadow_v4_date_quote_has_observed_target_hazard(
    mention: dict[str, Any],
    event: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    date_quote = _normalize_quote(str(mention.get("quote") or "")).lower()
    if not date_quote:
        return False
    targets = _target_hazard_families(candidate)
    for evidence in event.get("hazard_evidence") or []:
        if evidence.get("actuality") != "observed":
            continue
        hazard_quote = _normalize_quote(str(evidence.get("quote") or "")).lower()
        if not hazard_quote or not (hazard_quote in date_quote or date_quote in hazard_quote):
            continue
        grounded = _hazard_families(evidence.get("hazard")) & _hazard_families(hazard_quote)
        if grounded & targets:
            return True
    return False


def derive_candidate_event_date_relation_shadow_v4(
    event: dict[str, Any], candidate: dict[str, Any]
) -> str:
    """Shadow timing verifier with narrowly anchored contemporary-date forms.

    The production v3 verifier remains the default. This version accepts only
    additional forms that retain an explicit same-page anchor and never uses a
    publication date by itself as event evidence.
    """

    window_start, window_end = _candidate_window(candidate)
    if not window_start or not window_end:
        return "unknown"
    anchors = _shadow_v4_anchor_dates(event)
    event_roles = {
        "event_start",
        "event_end",
        "event_day",
        "incident_period",
        "observed_status_as_of",
    }
    intervals: list[tuple[date, date]] = []
    coarse_overlap = False
    mentions = event.get("date_evidence") or []
    for mention in mentions:
        role = mention.get("semantic_role")
        if role == "warning_period" and _shadow_v4_date_quote_has_observed_target_hazard(
            mention, event, candidate
        ):
            role = "event_day"
        if role not in event_roles or not str(mention.get("quote") or "").strip():
            continue
        precision = mention.get("precision")
        verified = _shadow_v4_verified_date_evidence_interval(mention, anchors=anchors)
        if verified is None and role == "event_start":
            start = _parse_iso_date(str(mention.get("normalized_start") or ""))
            language = " ".join(
                [str(mention.get("quote") or ""), str(event.get("event_summary") or "")]
            ).lower()
            nearby_anchors = [
                value
                for value in anchors
                if start and 0 <= (value - start).days <= 31
            ]
            if (
                start
                and nearby_anchors
                and _shadow_v4_month_day_in_quote(str(mention.get("quote") or ""), start)
                and re.search(r"\b(ongoing|continues?|continuing|remain(?:s|ed)?)\b", language)
            ):
                verified = (start, start)
        if verified is None:
            coarse_start = _parse_iso_date(str(mention.get("normalized_start") or ""))
            coarse_end = _parse_iso_date(str(mention.get("normalized_end") or "")) or coarse_start
            if (
                precision in {"month", "year"}
                and coarse_start
                and coarse_end
                and coarse_start <= window_end
                and coarse_end >= window_start
            ):
                coarse_overlap = True
            continue
        start, end = verified
        if mention.get("semantic_role") == "event_start":
            language = " ".join(
                [str(mention.get("quote") or ""), str(event.get("event_summary") or "")]
            ).lower()
            later_anchors = [value for value in anchors if value >= start]
            if later_anchors and re.search(r"\b(ongoing|continues?|continuing|remain(?:s|ed)?)\b", language):
                end = max(later_anchors)
        intervals.append((min(start, end), max(start, end)))
    if not intervals:
        return "unknown"
    if any(start <= window_end and end >= window_start for start, end in intervals):
        return "aligned"
    if coarse_overlap:
        return "unknown"
    return "outside_candidate"


def _shadow_v4_observed_hazard_only_promotion_is_safe(
    event: dict[str, Any],
    candidate: dict[str, Any],
    *,
    date_relation_deriver: Callable[[dict[str, Any], dict[str, Any]], str],
) -> bool:
    if (
        event.get("binding_status") != "explicit_same_event"
        or event.get("actuality") != "mixed_or_uncertain"
        or event.get("date_relation") != "aligned"
        or event.get("location_relation") != "aligned"
        or event.get("hazard_status") != "observed"
        or event.get("observed_impact") != "no"
        or event.get("attribution") != "no"
        or date_relation_deriver(event, candidate) != "aligned"
        or derive_candidate_event_location_relation(event, candidate) != "aligned"
        or derive_candidate_event_hazard_relation(event, candidate) != "compatible"
    ):
        return False
    return any(
        _shadow_v4_date_quote_has_observed_target_hazard(mention, event, candidate)
        for mention in event.get("date_evidence") or []
    )


def derive_local_page_result_shadow_v4(
    *,
    semantic: dict[str, Any] | None,
    candidate: dict[str, Any],
    body_text: str,
    call_status: str = "ok",
) -> dict[str, Any]:
    return derive_local_page_result(
        semantic=semantic,
        candidate=candidate,
        body_text=body_text,
        call_status=call_status,
        guard_policy_version=SHADOW_GUARD_POLICY_VERSION,
    )


def _candidate_event_negative_reason(
    event: dict[str, Any],
    candidate: dict[str, Any],
    *,
    date_relation_deriver: Callable[[dict[str, Any], dict[str, Any]], str] = derive_candidate_event_date_relation,
) -> str:
    # These native-only closures are allowed to correct a model relation in
    # the negative direction, but only after the model has explicitly bound
    # every populated decisive field to one candidate_event. Legacy arrays do
    # not satisfy this condition.
    if event.get("binding_status") == "explicit_same_event":
        if date_relation_deriver(event, candidate) == "outside_candidate":
            return "bound_candidate_event_dates_mechanically_outside_candidate"
        if _derive_observed_status_relation_for_negative_closure(event, candidate) == "outside_candidate":
            return "bound_observed_status_date_mechanically_outside_candidate"
        if _explicit_non_candidate_geography_is_complete(event, candidate):
            return "bound_candidate_event_geography_explicitly_outside_candidate"
    if event["actuality"] in {"warning_or_forecast", "order_or_administrative_action", "planning_or_generic_context"}:
        if _nonobserved_actuality_is_mechanically_supported(event, candidate):
            return f"model_negative_with_non_observed_actuality={event['actuality']}"
        return ""
    if event["date_relation"] == "outside_candidate":
        if date_relation_deriver(event, candidate) == "outside_candidate":
            return "model_negative_with_selected_event_outside_candidate"
        return ""
    if event["location_relation"] == "not_aligned":
        if derive_candidate_event_location_relation(event, candidate) == "not_aligned":
            return "model_negative_with_selected_event_location_not_aligned"
        return ""
    if event["hazard_status"] == "incompatible_hazard_only":
        if derive_candidate_event_hazard_relation(event, candidate) == "incompatible":
            return "model_negative_with_hazard_status=incompatible_hazard_only"
        return ""
    if event["hazard_status"] == "anticipated_or_warning":
        prospective = any(
            value.get("actuality") == "prospective" and str(value.get("quote") or "").strip()
            for value in event.get("hazard_evidence") or []
        )
        observed = derive_candidate_event_hazard_relation(event, candidate)
        if prospective and observed != "compatible":
            return "model_negative_with_hazard_status=anticipated_or_warning"
        return ""
    if event["hazard_status"] == "absent_or_context":
        has_grounded_context = bool(
            event.get("supporting_quotes")
            or event.get("potential_impacts_or_risks")
            or event.get("aid_or_administrative_actions")
            or event.get("hazard_evidence")
        )
        if has_grounded_context and derive_candidate_event_hazard_relation(event, candidate) != "compatible":
            return "model_negative_with_hazard_status=absent_or_context"
    return ""


def derive_candidate_event_location_relation(event: dict[str, Any], candidate: dict[str, Any]) -> str:
    terms = _candidate_location_terms(candidate)
    event_location_seen = False
    for evidence in event.get("location_evidence") or []:
        if evidence.get("kind") != "event_location" or not str(evidence.get("quote") or "").strip():
            continue
        event_location_seen = True
        haystack = _normalize_location(str(evidence.get("quote") or ""))
        if any(term and term in haystack for term in terms):
            return "aligned"
    return "not_aligned" if event_location_seen else "unknown"


def derive_candidate_event_hazard_relation(event: dict[str, Any], candidate: dict[str, Any]) -> str:
    observed_families: set[str] = set()
    for value in event.get("hazard_evidence") or []:
        if value.get("actuality") != "observed" or not str(value.get("quote") or "").strip():
            continue
        observed_families.update(
            _hazard_families(value.get("hazard")) & _hazard_families(value.get("quote"))
        )
    target_families: set[str] = set()
    for target in candidate.get("target_hazards") or []:
        target_families.update(_hazard_families(target))
    if not observed_families:
        return "unknown"
    return "compatible" if observed_families & target_families else "incompatible"


def _evidence_text_bound_to_quote(text: str, quote: str) -> bool:
    normalized_text = _normalize_quote(text).lower()
    normalized_quote = _normalize_quote(quote).lower()
    if normalized_text and normalized_text in normalized_quote:
        return True
    # Permit only a narrow OCR reversal: spaces inserted within words or
    # ordinal suffixes.  The alphanumeric sequence must otherwise be exact.
    compact_text = re.sub(r"[^a-z0-9]+", "", normalized_text)
    compact_quote = re.sub(r"[^a-z0-9]+", "", normalized_quote)
    return bool(len(compact_text) >= 8 and compact_text in compact_quote)


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _explicit_dates_in_quote(quote: str) -> set[date]:
    text = _normalize_quote(quote).lower()
    found: set[date] = set()
    for year, month, day in re.findall(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", text):
        parsed = _safe_date(int(year), int(month), int(day))
        if parsed:
            found.add(parsed)
    for month, day, year in re.findall(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+((?:19|20)\d{2})\b",
        text,
    ):
        parsed = _safe_date(int(year), _MONTHS[month.rstrip(".")], int(day))
        if parsed:
            found.add(parsed)
    for day, month, year in re.findall(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?[,]?\s+((?:19|20)\d{2})\b",
        text,
    ):
        parsed = _safe_date(int(year), _MONTHS[month.rstrip(".")], int(day))
        if parsed:
            found.add(parsed)
    for month, day, year in re.findall(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b", text):
        parsed = _safe_date(int(year), int(month), int(day))
        if parsed:
            found.add(parsed)
    for year in re.findall(r"\bnew year['’]?s eve\s+((?:19|20)\d{2})\b", text):
        found.add(date(int(year), 12, 31))
    for year in re.findall(r"\bnew year['’]?s day\s+((?:19|20)\d{2})\b", text):
        found.add(date(int(year), 1, 1))
    return found


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _relative_date_from_quote(quote: str, anchor_date: date) -> date | None:
    text = _normalize_quote(quote).lower()
    if "tomorrow" in text or "next week" in text:
        return None
    if re.search(r"\b(today|this morning|this afternoon|this evening|tonight)\b", text):
        return anchor_date
    if "yesterday" in text:
        return anchor_date - timedelta(days=1)
    days_ago = re.search(r"\b(\d{1,2})\s+days?\s+ago\b", text)
    if days_ago:
        return anchor_date - timedelta(days=int(days_ago.group(1)))
    return None


def _relative_weekday_interval_from_quote(quote: str, anchor_date: date) -> tuple[date, date] | None:
    """Resolve only weekday wording with one unambiguous anchored interpretation.

    Bare weekday names can refer to more than one prior week and therefore do
    not resolve by themselves. Explicit ``last``, ``previous`` or ``this past``
    modifiers select the immediately preceding occurrence. A coordinated
    interval is accepted only when every weekday carries an explicit modifier.
    """
    text = _normalize_quote(quote).lower()
    matches = list(
        re.finditer(
            r"\b(?:(last|previous|this\s+past)|the\s+(previous))\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            text,
        )
    )
    all_weekdays = re.findall(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
    )
    if not all_weekdays or len(matches) != len(all_weekdays):
        return None
    resolved: list[date] = []
    for match in matches:
        target = _WEEKDAYS[match.group(3)]
        delta = (anchor_date.weekday() - target) % 7
        if delta == 0:
            delta = 7
        resolved.append(anchor_date - timedelta(days=delta))
    interval = (min(resolved), max(resolved))
    # Coordinated weekdays must describe one contiguous interval. This blocks
    # accidental combination of separate weekly references in one quote.
    if len(resolved) > 1 and (interval[1] - interval[0]).days != len(set(resolved)) - 1:
        return None
    return interval


def _verified_date_evidence_interval(mention: dict[str, Any]) -> tuple[date, date] | None:
    quote = str(mention.get("quote") or "")
    if not quote:
        return None
    precision = mention.get("precision")
    start = _parse_iso_date(str(mention.get("normalized_start") or ""))
    end = _parse_iso_date(str(mention.get("normalized_end") or "")) or start
    # Frozen v1/v2 adapters retain the old text field. Their already-audited
    # contract is a narrowly reversible text-to-quote binding followed by the
    # saved normalization. Native v3 never has this field and therefore must
    # satisfy the stricter explicit/relative checks below.
    if "text" in mention:
        if not start or not end:
            return None
        if not _evidence_text_bound_to_quote(str(mention.get("text") or ""), quote):
            return None
        if precision == "exact_day":
            return (start, end)
        if precision == "exact_interval" and mention.get("continuous_event_period") is True:
            return (min(start, end), max(start, end))
        return None
    if precision == "relative":
        anchor = mention.get("relative_anchor")
        if not isinstance(anchor, dict):
            return None
        anchor_date = _parse_iso_date(str(anchor.get("anchor_date") or ""))
        anchor_quote = str(anchor.get("anchor_quote") or "")
        if (
            not anchor_date
            or anchor.get("anchor_role") not in RELATIVE_ANCHOR_ROLES
            or anchor_date not in _explicit_dates_in_quote(anchor_quote)
        ):
            return None
        resolved_day = _relative_date_from_quote(quote, anchor_date)
        resolved_interval = (
            (resolved_day, resolved_day)
            if resolved_day
            else _relative_weekday_interval_from_quote(quote, anchor_date)
        )
        if resolved_interval is None:
            return None
        if start or end:
            return resolved_interval if start == resolved_interval[0] and end == resolved_interval[1] else None
        return resolved_interval
    if not start or not end:
        return None
    explicit = _explicit_dates_in_quote(quote)
    if precision == "exact_day":
        return (start, end) if start == end and start in explicit else None
    if precision == "exact_interval" and mention.get("continuous_event_period") is True:
        return (min(start, end), max(start, end)) if start in explicit and end in explicit else None
    return None


def _complete_observed_status_span_date(
    quote: str, normalized_value: str, candidate: dict[str, Any]
) -> date | None:
    normalized = _parse_iso_date(normalized_value)
    if not normalized or normalized not in _explicit_dates_in_quote(quote):
        return None
    normalized_span = _normalize_location(quote)
    location_bound = any(term and term in normalized_span for term in _candidate_location_terms(candidate))
    hazard_bound = bool(_hazard_families(quote) & _target_hazard_families(candidate))
    observed_language = bool(
        re.search(
            r"\b(as of|current status|currently|remain|remains|continued|continues|"
            r"closed|flooded|damaged|responded|has responded|due to)\b",
            quote.lower(),
        )
    )
    return normalized if location_bound and hazard_bound and observed_language else None


def _verified_observed_status_dates_for_negative_closure(
    event: dict[str, Any], candidate: dict[str, Any]
) -> list[date]:
    """Return status dates proved inside one span or one internally linked object.

    Separate date/location/hazard arrays are never joined here. A date-evidence
    quote may stand alone only when that same quote contains every required
    element. The dedicated status object may be used only when its two quotes
    overlap by literal containment; mere co-membership in candidate_event is
    not enough.
    """
    dates: list[date] = []
    for mention in event.get("date_evidence") or []:
        if mention.get("semantic_role") != "observed_status_as_of":
            continue
        start = str(mention.get("normalized_start") or "")
        end = str(mention.get("normalized_end") or "") or start
        if not start or start != end:
            continue
        verified = _complete_observed_status_span_date(
            str(mention.get("quote") or ""), start, candidate
        )
        if verified:
            dates.append(verified)
    value = event.get("observed_status_as_of")
    if isinstance(value, dict):
        date_quote = str(value.get("date_quote") or "")
        status_quote = str(value.get("status_quote") or "")
        if _quotes_overlap(date_quote, status_quote):
            complete_span = date_quote if len(date_quote) >= len(status_quote) else status_quote
            verified = _complete_observed_status_span_date(
                complete_span, str(value.get("normalized_date") or ""), candidate
            )
            if verified:
                dates.append(verified)
    return sorted(set(dates))


def _derive_observed_status_relation_for_negative_closure(
    event: dict[str, Any], candidate: dict[str, Any]
) -> str:
    dates = _verified_observed_status_dates_for_negative_closure(event, candidate)
    window_start, window_end = _candidate_window(candidate)
    if not dates or not window_start or not window_end:
        return "unknown"
    if any(window_start <= value <= window_end for value in dates):
        return "aligned"
    return "outside_candidate"


_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
}


def _explicit_non_candidate_geography_is_complete(
    event: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    evidence = [
        value
        for value in event.get("location_evidence") or []
        if value.get("kind") == "event_location" and str(value.get("quote") or "").strip()
    ]
    if not evidence:
        return False
    candidate_terms = _candidate_location_terms(candidate)
    candidate_county = _normalize_location(candidate.get("county") or "")
    candidate_state = _normalize_location(candidate.get("state") or "")
    nonexhaustive = (
        "including", "such as", "for example", "among", "parts of", "around", "near ",
        "across", "elsewhere", "one of", "through portions", "some areas",
    )
    occurrence = re.compile(
        r"\b(occurred|happened|struck|hit|made landfall|was centered|was located|"
        r"affected|impacted|flooded|experienced)\b"
    )
    for value in evidence:
        quote = _normalize_location(str(value.get("quote") or ""))
        location = _normalize_location(str(value.get("location") or ""))
        if any(term and term in quote for term in candidate_terms):
            return False
        if any(marker in quote for marker in nonexhaustive) or not occurrence.search(quote):
            return False
        # The structured location must itself be literally present in the
        # event-occurrence quote; model normalization alone is not evidence.
        if not location or location not in quote:
            return False
        county_names = {
            match.group(1).strip()
            for match in re.finditer(r"\b([a-z][a-z0-9 .'-]{0,45}) county\b", location)
        }
        state_names = {name for name in _US_STATE_NAMES if re.search(rf"\b{re.escape(name)}\b", location)}
        explicit_other_county = bool(county_names) and candidate_county not in {
            f"{name} county" for name in county_names
        }
        explicit_other_state = bool(state_names) and candidate_state not in state_names
        if not (explicit_other_county or explicit_other_state):
            return False
    return True


def _target_hazard_families(candidate: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    for target in candidate.get("target_hazards") or []:
        families.update(_hazard_families(target))
    return families


def _single_date_evidence_relation(mention: dict[str, Any], candidate: dict[str, Any]) -> str:
    if mention.get("semantic_role") not in {"event_start", "event_end", "event_day", "incident_period"}:
        return "unknown"
    if not str(mention.get("quote") or "").strip():
        return "unknown"
    interval = _verified_date_evidence_interval(mention)
    if interval is None:
        return "unknown"
    start, end = interval
    window_start, window_end = _candidate_window(candidate)
    if not start or not end or not window_start or not window_end:
        return "unknown"
    return "aligned" if min(start, end) <= window_end and max(start, end) >= window_start else "outside_candidate"


def _legacy_same_event_support_witnesses(event: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Return a quote component that mechanically binds every decisive positive fact.

    Legacy v1 fields were arrays rather than an event object.  We therefore do
    not treat co-presence on a page as binding.  Quotes may join a component
    only by literal containment, or by an explicit adjacent anaphoric phrase
    with shared event anchors.  This is deliberately narrower than general
    coreference resolution and never searches for a replacement semantic fact.
    """
    if event.get("actuality") not in {"observed", "retrospective_observed"}:
        return []
    if event.get("date_relation") != "aligned" or event.get("location_relation") != "aligned":
        return []
    if event.get("hazard_status") != "observed":
        return []
    aligned_dates = [value for value in event.get("date_evidence") or [] if _single_date_evidence_relation(value, candidate) == "aligned"]
    if not aligned_dates:
        return []
    target_families = _target_hazard_families(candidate)
    supporting = [str(value) for value in event.get("supporting_quotes") or [] if str(value).strip()]
    date_quotes = [str(value.get("quote") or "") for value in aligned_dates]
    location_quotes = [
        str(value.get("quote") or "")
        for value in event.get("location_evidence") or []
        if str(value.get("quote") or "").strip()
        and (
            value.get("kind") == "event_location"
            or (
                event.get("location_relation") == "aligned"
                and value.get("kind") == "unrelated_event_location"
            )
        )
    ]
    hazard_quotes = [
        str(value.get("quote") or "")
        for value in event.get("hazard_evidence") or []
        if value.get("actuality") == "observed"
        and str(value.get("quote") or "").strip()
        and (_hazard_families(value.get("hazard")) & _hazard_families(value.get("quote")) & target_families)
    ]
    quotes = list(dict.fromkeys([*supporting, *date_quotes, *location_quotes, *hazard_quotes]))
    if not location_quotes or not hazard_quotes:
        return []

    labels: list[set[str]] = []
    for quote in quotes:
        normalized = _normalize_quote(quote).lower()
        quote_labels: set[str] = set()
        if any(_quotes_overlap(quote, value) for value in date_quotes):
            quote_labels.add("date")
        if any(_quotes_overlap(quote, value) for value in location_quotes):
            quote_labels.add("location")
        if any(_quotes_overlap(quote, value) for value in hazard_quotes):
            quote_labels.add("hazard")
        if any(term and term in _normalize_location(quote) for term in _candidate_location_terms(candidate)):
            quote_labels.add("location")
        if _hazard_families(quote) & target_families:
            quote_labels.add("hazard")
        if any(_normalize_quote(value).lower() in normalized for value in date_quotes):
            quote_labels.add("date")
        labels.append(quote_labels)

    adjacency = [set() for _ in quotes]
    for left in range(len(quotes)):
        for right in range(left + 1, len(quotes)):
            if _quotes_overlap(quotes[left], quotes[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
    support_index = {quote: index for index, quote in enumerate(quotes)}
    for index in range(1, len(supporting)):
        previous = supporting[index - 1]
        current = supporting[index]
        if _explicit_adjacent_event_coreference(previous, current):
            left, right = support_index[previous], support_index[current]
            adjacency[left].add(right)
            adjacency[right].add(left)

    seen: set[int] = set()
    for start in range(len(quotes)):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        component_labels: set[str] = set()
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            component.append(index)
            component_labels.update(labels[index])
            stack.extend(adjacency[index] - seen)
        if {"date", "location", "hazard"} <= component_labels:
            return [quotes[index] for index in component]
    return []


def _quotes_overlap(left: str, right: str) -> bool:
    normalized_left = _normalize_quote(left).lower()
    normalized_right = _normalize_quote(right).lower()
    if not normalized_left or not normalized_right:
        return False
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    return len(shorter) >= 8 and shorter in longer


def _explicit_adjacent_event_coreference(previous: str, current: str) -> bool:
    current_text = _normalize_quote(current).lower()
    markers = (
        "this series of storms",
        "these storms",
        "the storm series",
        "the flooding that ensued",
        "in total",
        "between those days",
    )
    if not any(marker in current_text for marker in markers):
        return False
    anchor_terms = {
        "storm",
        "series",
        "flood",
        "rain",
        "river",
        "creek",
        "levee",
        "sheriff",
        "office",
        "call",
    }

    def anchors(value: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", _normalize_quote(value).lower())
        stems = {
            token[:-1]
            if token.endswith("s") and len(token) > 3 and token != "series" and not token.endswith("ss")
            else token
            for token in tokens
        }
        return stems & anchor_terms

    return len(anchors(previous) & anchors(current)) >= 2


def _quote_in_witness_component(quote: str, witnesses: list[str]) -> bool:
    return any(_quotes_overlap(quote, witness) for witness in witnesses)


def _legacy_bound_date_relation(event: dict[str, Any], candidate: dict[str, Any], witnesses: list[str]) -> str:
    relations = [
        _single_date_evidence_relation(value, candidate)
        for value in event.get("date_evidence") or []
        if _quote_in_witness_component(str(value.get("quote") or ""), witnesses)
    ]
    if "aligned" in relations:
        return "aligned"
    return "outside_candidate" if relations and all(value == "outside_candidate" for value in relations) else "unknown"


def _legacy_bound_location_relation(
    event: dict[str, Any], candidate: dict[str, Any], witnesses: list[str]
) -> str:
    bound = any(
        _quote_in_witness_component(str(value.get("quote") or ""), witnesses)
        for value in event.get("location_evidence") or []
    )
    if not bound:
        bound = any(
            term and term in _normalize_location(witness)
            for term in _candidate_location_terms(candidate)
            for witness in witnesses
        )
    return "aligned" if event.get("location_relation") == "aligned" and bound else "unknown"


def _legacy_bound_hazard_relation(
    event: dict[str, Any], candidate: dict[str, Any], witnesses: list[str]
) -> str:
    observed: set[str] = set()
    for value in event.get("hazard_evidence") or []:
        quote = str(value.get("quote") or "")
        if value.get("actuality") != "observed" or not _quote_in_witness_component(quote, witnesses):
            continue
        observed.update(_hazard_families(value.get("hazard")) & _hazard_families(quote))
    if not observed:
        if event.get("hazard_status") == "observed" and any(
            _hazard_families(witness) & _target_hazard_families(candidate) for witness in witnesses
        ):
            return "compatible"
        return "unknown"
    return "compatible" if observed & _target_hazard_families(candidate) else "incompatible"


def _legacy_axis_quote_bound_to_candidate_event(
    quote: str,
    witnesses: list[str],
    event: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    normalized = _normalize_quote(quote).lower()
    if not normalized:
        return False
    for witness in witnesses:
        normalized_witness = _normalize_quote(witness).lower()
        if normalized in normalized_witness or normalized_witness in normalized:
            return True
    location_bound = any(
        term and term in _normalize_location(quote) for term in _candidate_location_terms(candidate)
    )
    hazard_bound = bool(_hazard_families(quote) & _target_hazard_families(candidate))
    date_bound = any(
        _single_date_evidence_relation(value, candidate) == "aligned"
        and _normalize_quote(str(value.get("quote") or "")).lower() in normalized
        for value in event.get("date_evidence") or []
    )
    return location_bound and hazard_bound and date_bound


def _quote_has_target_hazard_and_impact(quote: str, candidate: dict[str, Any]) -> bool:
    if not (_hazard_families(quote) & _target_hazard_families(candidate)):
        return False
    text = _normalize_quote(quote).lower()
    impact_markers = (
        "damage",
        "damaged",
        "destroyed",
        "closed",
        "closure",
        "outage",
        "rescued",
        "rescue",
        "inundated",
        "fatal",
        "died",
        "death",
        "washed out",
        "trapped",
        "loss",
        "flooded",
        "flooding",
        "home",
        "road",
        "street",
        "neighborhood",
        "community",
        "business",
        "vehicle",
        "school",
        "power outage",
    )
    return any(marker in text for marker in impact_markers)


def _nonobserved_actuality_is_mechanically_supported(event: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if derive_candidate_event_hazard_relation(event, candidate) == "compatible":
        return False
    actuality = event.get("actuality")
    if actuality == "warning_or_forecast":
        return any(
            value.get("actuality") == "prospective" and str(value.get("quote") or "").strip()
            for value in event.get("hazard_evidence") or []
        ) or bool(event.get("potential_impacts_or_risks"))
    if actuality == "order_or_administrative_action":
        return bool(event.get("aid_or_administrative_actions") or event.get("potential_impacts_or_risks"))
    if actuality == "planning_or_generic_context":
        return bool(
            event.get("aid_or_administrative_actions")
            or event.get("potential_impacts_or_risks")
            or event.get("supporting_quotes")
        )
    return False


def _adapt_legacy_judgment(legacy: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed adapter for frozen v1 outputs; it never overturns a model negative."""
    date_evidence = []
    for mention in legacy["date_mentions"]:
        precision, continuous = _infer_legacy_date_precision(mention)
        date_evidence.append(
            {
                **mention,
                "precision": precision,
                "continuous_event_period": continuous,
            }
        )

    location_evidence = []
    for quote in legacy["location_quotes"]:
        location_evidence.append(
            {
                "location": quote,
                "kind": _classify_legacy_location_quote(
                    quote,
                    candidate,
                    relation=legacy["event_location_relation"],
                ),
                "quote": quote,
            }
        )

    hazard_evidence: list[dict[str, str]] = []
    for hazard in legacy["observed_hazards"]:
        quote = next(
            (
                value
                for value in legacy["hazard_quotes"]
                if _hazard_families(hazard) & _hazard_families(value)
            ),
            "",
        )
        if quote:
            hazard_evidence.append({"hazard": hazard, "actuality": "observed", "quote": quote})
    for hazard in legacy["prospective_hazards"]:
        quote = next(
            (
                value
                for value in legacy["hazard_quotes"]
                if _hazard_families(hazard) & _hazard_families(value)
            ),
            "",
        )
        if quote:
            hazard_evidence.append({"hazard": hazard, "actuality": "prospective", "quote": quote})

    event = {
        "event_summary": "legacy_frozen_single_candidate_event",
        "binding_status": "unresolved",
        "actuality": legacy["event_actuality"],
        "date_relation": legacy["event_date_relation"],
        "date_evidence": date_evidence,
        "location_relation": legacy["event_location_relation"],
        "location_evidence": location_evidence,
        "hazard_status": legacy["target_hazard_status"],
        "hazard_evidence": hazard_evidence,
        "realized_impacts": legacy["realized_impacts"],
        "potential_impacts_or_risks": legacy["potential_impacts_or_risks"],
        "asset_or_economic_values": legacy["asset_or_economic_values"],
        "aid_or_administrative_actions": legacy["aid_or_administrative_actions"],
        "observed_impact": legacy["target_hazard_observed_impact"],
        "attribution": legacy["explicit_target_hazard_to_impact_attribution"],
        "attribution_quotes": legacy["attribution_quotes"],
        "transition": legacy["explicit_drought_to_wet_transition"],
        "transition_quotes": legacy["transition_quotes"],
        "supporting_quotes": legacy["supporting_quotes"],
    }
    # The frozen schema did not encode bindings. Grounded mechanical binding is
    # evaluated only after ungrounded evidence has been filtered from the adapter.
    return {
        "source_readability": legacy["source_readability"],
        "readability_reason": legacy["readability_reason"],
        "event_identity": legacy["event_identity"],
        "candidate_event": event if legacy["event_identity"] == "one_uniquely_relevant_event" else None,
        "candidate_support_recommendation": legacy["candidate_support_recommendation"],
        "uncertainty": legacy["uncertainty"],
        "_legacy_original": legacy,
    }


def _infer_legacy_date_precision(mention: dict[str, Any]) -> tuple[str, bool]:
    text = f"{mention.get('text', '')} {mention.get('quote', '')}".lower()
    start = _parse_iso_date(str(mention.get("normalized_start") or ""))
    end = _parse_iso_date(str(mention.get("normalized_end") or ""))
    if not start or not end:
        return "relative_or_unknown", False
    has_day = bool(
        re.search(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", text)
        or re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
            text,
        )
        or re.search(r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", text)
        or re.search(r"\b\d{1,2}\s*(?:st|nd|rd|th)\b", text)
        or re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
        or "new year" in text
    )
    if start == end:
        if has_day:
            return "exact_day", False
        return ("year" if start.month == 1 and start.day == 1 else "month"), False
    month_or_year_only = not has_day and bool(re.search(r"\b(?:19|20)\d{2}\b", text))
    if month_or_year_only:
        return ("year" if (end - start).days > 31 else "month"), False
    continuous = bool(
        re.search(r"\b(?:from|between|beginning|starting)\b.{0,80}\b(?:to|through|until|and|continued)\b", text)
        or re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\b.{0,40}(?:-|–|—|through|to)\s*.{0,20}\b\d{1,2}(?:st|nd|rd|th)?\b", text)
    )
    return "exact_interval", continuous


def _classify_legacy_location_quote(
    quote: str,
    candidate: dict[str, Any],
    *,
    relation: str,
) -> str:
    normalized = _normalize_location(quote)
    administrative_markers = (
        "eligible",
        "eligibility",
        "assistance",
        "apply",
        "resource",
        "website",
        "jurisdiction",
        "service area",
        "office of",
        "department of",
    )
    if any(marker in normalized for marker in administrative_markers):
        return "eligibility_area" if any(marker in normalized for marker in ("eligible", "eligibility", "assistance", "apply")) else "document_scope"
    if relation == "not_aligned":
        return "event_location"
    if not any(term and term in normalized for term in _candidate_location_terms(candidate)):
        return "unrelated_event_location"
    return "event_location"


def _candidate_window(candidate: dict[str, Any]) -> tuple[date | None, date | None]:
    target_axis = str(candidate.get("target_hazard_axis") or "").strip().lower()
    if target_axis in {"drought", "wet"}:
        start = _parse_iso_date(str(candidate.get("target_window_start") or ""))
        end = _parse_iso_date(str(candidate.get("target_window_end") or ""))
        fallback_window = str(
            candidate.get("drought_window")
            if target_axis == "drought"
            else candidate.get("wet_window") or candidate.get("event_window") or ""
        )
    else:
        start = _parse_iso_date(str(candidate.get("wet_window_start") or ""))
        end = _parse_iso_date(str(candidate.get("wet_window_end") or ""))
        fallback_window = str(
            candidate.get("wet_window") or candidate.get("event_window") or ""
        )
    if not start or not end:
        parsed_start, parsed_end = _parse_window_text(fallback_window)
        start = start or parsed_start
        end = end or parsed_end
    return start, end


def _candidate_location_terms(candidate: dict[str, Any]) -> list[str]:
    values = [candidate.get("county") or "", *(candidate.get("locality_hints") or [])]
    terms = [_normalize_location(value) for value in values if _normalize_location(value)]
    county = _normalize_location(candidate.get("county") or "")
    if county.endswith(" county"):
        terms.append(county[: -len(" county")].strip())
    return list(dict.fromkeys(value for value in terms if value))


def derive_event_date_relation(date_mentions: list[dict[str, Any]], candidate: dict[str, Any]) -> str:
    window_start, window_end = _candidate_window(candidate)
    if not window_start or not window_end:
        start, end = _parse_window_text(str(candidate.get("wet_window") or candidate.get("event_window") or ""))
        window_start = window_start or start
        window_end = window_end or end
    if not window_start or not window_end:
        return "unknown"
    event_roles = {"event_start", "event_end", "event_day", "incident_period"}
    intervals: list[tuple[date, date]] = []
    for mention in date_mentions:
        if mention.get("semantic_role") not in event_roles:
            continue
        start = _parse_iso_date(str(mention.get("normalized_start") or ""))
        end = _parse_iso_date(str(mention.get("normalized_end") or "")) or start
        if start and end:
            intervals.append((min(start, end), max(start, end)))
    if not intervals:
        return "unknown"
    if any(start <= window_end and end >= window_start for start, end in intervals):
        return "aligned"
    return "outside_candidate"


def derive_event_location_relation(semantic: dict[str, Any], candidate: dict[str, Any]) -> str:
    raw_relation = semantic.get("event_location_relation")
    if raw_relation == "not_aligned":
        return "not_aligned"
    if raw_relation != "aligned":
        return "unknown"
    evidence = " ".join(str(value) for value in semantic.get("location_quotes") or [])
    normalized_evidence = _normalize_location(evidence)
    candidate_terms = [str(candidate.get("county") or "")]
    candidate_terms.extend(str(value) for value in candidate.get("locality_hints") or [])
    for term in candidate_terms:
        normalized_term = _normalize_location(term)
        if normalized_term and normalized_term in normalized_evidence:
            return "aligned"
    return "unknown"


def derive_hazard_compatibility(observed_hazards: list[str], target_hazards: list[str]) -> str:
    observed = {_hazard_family(value) for value in observed_hazards}
    targets = {_hazard_family(value) for value in target_hazards}
    observed.discard("")
    targets.discard("")
    if not observed:
        return "unknown"
    if observed & targets:
        return "compatible"
    return "incompatible"


def _hazard_family(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if any(term in text for term in ("flash flood", "flood", "inundat", "overflow", "high water")):
        return "flood"
    if any(term in text for term in ("rain", "precipitation", "atmospheric river", "cloudburst")):
        return "rainfall"
    if any(term in text for term in ("debris flow", "mudslide", "mud flow", "landslide")):
        return "debris_flow"
    if any(term in text for term in ("wind", "tornado", "downburst")):
        return "wind"
    if any(
        term in text
        for term in (
            "drought",
            "dryness",
            "dry condition",
            "water shortage",
            "water restriction",
        )
    ):
        return "drought"
    return text


def _hazard_families(value: Any) -> set[str]:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    families: set[str] = set()
    if any(term in text for term in ("flash flood", "flood", "inundat", "overflow", "high water")):
        families.add("flood")
    if any(term in text for term in ("rain", "precipitation", "atmospheric river", "cloudburst")):
        families.add("rainfall")
    if any(term in text for term in ("debris flow", "mudslide", "mud flow", "landslide")):
        families.add("debris_flow")
    if any(term in text for term in ("wind", "tornado", "downburst")):
        families.add("wind")
    if any(
        term in text
        for term in (
            "drought",
            "dryness",
            "dry condition",
            "water shortage",
            "water restriction",
        )
    ):
        families.add("drought")
    if not families and text:
        families.add(text)
    return families


def _filter_ungrounded_evidence(semantic: dict[str, Any], body_text: str) -> tuple[dict[str, Any], int]:
    filtered = json.loads(json.dumps(semantic, ensure_ascii=False))
    removed = 0
    event = filtered.get("candidate_event")
    if not isinstance(event, dict):
        return filtered, removed
    for key in ("attribution_quotes", "transition_quotes", "supporting_quotes"):
        values = event.get(key) or []
        kept = [value for value in values if not str(value).strip() or quote_is_grounded(str(value), body_text)]
        removed += len(values) - len(kept)
        event[key] = kept
    mentions = event.get("date_evidence") or []
    kept_mentions = [
        value
        for value in mentions
        if (
            (not str(value.get("quote") or "").strip() or quote_is_grounded(str(value.get("quote") or ""), body_text))
            and (
                not isinstance(value.get("relative_anchor"), dict)
                or quote_is_grounded(str(value["relative_anchor"].get("anchor_quote") or ""), body_text)
            )
        )
    ]
    removed += len(mentions) - len(kept_mentions)
    event["date_evidence"] = kept_mentions
    status = event.get("observed_status_as_of")
    if isinstance(status, dict) and not (
        quote_is_grounded(str(status.get("date_quote") or ""), body_text)
        and quote_is_grounded(str(status.get("status_quote") or ""), body_text)
    ):
        event["observed_status_as_of"] = None
        removed += 1
    for key in ("location_evidence", "hazard_evidence"):
        values = event.get(key) or []
        kept = [
            value
            for value in values
            if not str(value.get("quote") or "").strip() or quote_is_grounded(str(value.get("quote") or ""), body_text)
        ]
        removed += len(values) - len(kept)
        event[key] = kept
    for key in (
        "realized_impacts",
        "potential_impacts_or_risks",
        "asset_or_economic_values",
        "aid_or_administrative_actions",
    ):
        values = event.get(key) or []
        kept = [
            value
            for value in values
            if not str(value.get("quote") or "").strip() or quote_is_grounded(str(value.get("quote") or ""), body_text)
        ]
        removed += len(values) - len(kept)
        event[key] = kept
    return filtered, removed


def quote_is_grounded(quote: str, body_text: str) -> bool:
    normalized_quote = _normalize_quote(quote)
    return bool(normalized_quote and normalized_quote in _normalize_quote(body_text))


def _normalize_quote(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00a0": " "}))
    return re.sub(r"\s+", " ", text).strip()


def _normalize_location(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _parse_window_text(value: str) -> tuple[date | None, date | None]:
    parts = re.split(r"\s+(?:to|through)\s+", value.strip(), maxsplit=1)
    if len(parts) == 1:
        parts.append(parts[0])
    return _parse_iso_date(parts[0]), _parse_iso_date(parts[1])


def _parse_iso_date(value: str) -> date | None:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _local_result(
    result: str,
    hazard_support: str,
    impact_support: str,
    attribution_support: str,
    transition_support: str,
    actions: list[dict[str, str]],
    *,
    source_axes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if result not in LOCAL_RESULTS:
        raise ValueError(f"invalid_local_result:{result}")
    resolved_source_axes = source_axes or {
        "observed_target_hazard": "unresolved",
        "realized_target_hazard_impact": "unresolved",
        "explicit_target_hazard_to_impact_attribution": "unresolved",
        "explicit_drought_to_wet_transition": "unresolved",
    }
    if result == "unresolved" and source_axes is not None:
        if hazard_support == "unresolved" and resolved_source_axes["observed_target_hazard"] == "no":
            hazard_support = "no"
        if impact_support == "unresolved" and resolved_source_axes["realized_target_hazard_impact"] == "no":
            impact_support = "no"
        if attribution_support == "unresolved" and resolved_source_axes["explicit_target_hazard_to_impact_attribution"] == "no":
            attribution_support = "no"
        if transition_support == "unresolved" and resolved_source_axes["explicit_drought_to_wet_transition"] == "no":
            transition_support = "no"
    return {
        "page_result": result,
        "candidate_hazard_support": hazard_support,
        "realized_impact_support": impact_support,
        "explicit_attribution_support": attribution_support,
        "explicit_drought_to_wet_transition_support": transition_support,
        "source_event_axes": resolved_source_axes,
        "guard_actions": actions,
    }


def _unresolved_candidate_axes(
    event: dict[str, Any],
    candidate: dict[str, Any],
    *,
    legacy_adapter_used: bool,
    legacy_witnesses: list[str],
) -> tuple[str, str, str, str]:
    """Limit uncertainty to candidate axes the saved event could still change."""
    observed_actuality = event.get("actuality") in {
        "observed",
        "retrospective_observed",
        "mixed_or_uncertain",
    }
    hazard_relation = derive_candidate_event_hazard_relation(event, candidate)
    lexical_target_hazard = any(
        _hazard_families(str(value.get("quote") or "")) & _target_hazard_families(candidate)
        for value in event.get("hazard_evidence") or []
        if value.get("actuality") == "observed"
    )
    hazard_possible = (
        observed_actuality
        and event.get("hazard_status") in {"observed", "uncertain"}
        and (hazard_relation == "compatible" or lexical_target_hazard)
    )
    hazard = "unresolved" if hazard_possible else "no"

    impact_possible = bool(
        hazard_possible
        and event.get("observed_impact") in {"yes", "uncertain"}
        and event.get("realized_impacts")
    )
    impact = "unresolved" if impact_possible else "no"

    attribution_possible = bool(
        impact_possible
        and event.get("attribution") in {"yes", "uncertain"}
        and any(
            _quote_has_target_hazard_and_impact(str(quote), candidate)
            for quote in event.get("attribution_quotes") or []
        )
    )
    attribution = "unresolved" if attribution_possible else "no"

    transition_quotes = [str(value) for value in event.get("transition_quotes") or []]
    transition_bound = bool(
        event.get("binding_status") == "explicit_same_event"
        and transition_quotes
        and (
            not legacy_adapter_used
            or any(
                _legacy_axis_quote_bound_to_candidate_event(
                    quote, legacy_witnesses, event, candidate
                )
                for quote in transition_quotes
            )
        )
    )
    transition_possible = bool(
        hazard_possible
        and event.get("transition") in {"yes", "uncertain"}
        and transition_bound
        and any(
            "drought" in _hazard_families(quote)
            and bool(_target_hazard_families(candidate) & _hazard_families(quote))
            for quote in transition_quotes
        )
    )
    transition = "unresolved" if transition_possible else "no"
    return hazard, impact, attribution, transition


def _unresolved_without_semantic_axes(body_text: str) -> tuple[str, str, str, str]:
    """Close only an axis whose required explicit vocabulary is absent.

    Explicit drought-to-wet linkage necessarily names drought/dryness.  Its
    lexical absence is therefore a safe negative for that axis even when a
    model output is incomplete; no other semantic axis is inferred here.
    """
    transition = "unresolved" if "drought" in _hazard_families(body_text) else "no"
    return "unresolved", "unresolved", "unresolved", transition


def _derive_source_event_axes(semantic: dict[str, Any], candidate: dict[str, Any]) -> dict[str, str]:
    event = semantic.get("candidate_event")
    if not isinstance(event, dict):
        return {
            "observed_target_hazard": "unresolved",
            "realized_target_hazard_impact": "unresolved",
            "explicit_target_hazard_to_impact_attribution": "unresolved",
            "explicit_drought_to_wet_transition": "unresolved",
        }
    actuality = event["actuality"]
    observed_actuality = actuality in {"observed", "retrospective_observed"}
    hazard_relation = derive_candidate_event_hazard_relation(event, candidate)
    hazard = "yes" if observed_actuality and hazard_relation == "compatible" else "no"
    impact = (
        "yes"
        if hazard == "yes" and event["realized_impacts"] and event["observed_impact"] == "yes"
        else ("unresolved" if event["observed_impact"] == "uncertain" else "no")
    )
    attribution = (
        "yes"
        if impact == "yes"
        and event["attribution"] == "yes"
        and event["attribution_quotes"]
        else (
            "unresolved"
            if impact != "no" and event["attribution"] == "uncertain"
            else "no"
        )
    )
    transition = (
        "yes"
        if event["transition"] == "yes" and event["transition_quotes"]
        else ("unresolved" if event["transition"] == "uncertain" else "no")
    )
    return {
        "observed_target_hazard": hazard,
        "realized_target_hazard_impact": impact,
        "explicit_target_hazard_to_impact_attribution": attribution,
        "explicit_drought_to_wet_transition": transition,
    }


def _action(actions: list[dict[str, str]], guard: str, action: str, reason: str) -> None:
    actions.append({"guard": guard, "action": action, "reason": reason})


def _require_enum(payload: dict[str, Any], key: str, values: set[str]) -> None:
    if payload.get(key) not in values:
        raise DirectJudgeSchemaError(f"invalid_{key}:{payload.get(key)}")


def _require_string_list(payload: dict[str, Any], key: str) -> None:
    values = payload.get(key)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise DirectJudgeSchemaError(f"invalid_{key}_string_list")


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text") or ""))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _response_is_explicitly_incomplete(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "incomplete" or bool(payload.get("incomplete_details"))


def _incomplete_reason(payload: dict[str, Any]) -> str:
    details = payload.get("incomplete_details")
    if isinstance(details, dict):
        return f"explicit_incomplete:{details.get('reason') or 'unknown'}"
    return "explicit_incomplete:unknown"


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _safe_error(exc: Exception, api_key: str) -> str:
    return _redact(str(exc), api_key)[:800]


def _redact(text: str, api_key: str) -> str:
    safe = text.replace(api_key, "[REDACTED_OPENAI_KEY]") if api_key else text
    return re.sub(r"sk-[A-Za-z0-9_\-]+", "[REDACTED_OPENAI_KEY]", safe)


def estimated_cost_usd(
    usage: dict[str, Any],
    *,
    input_per_million: float = 5.0,
    output_per_million: float = 30.0,
) -> float:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return round(input_tokens * input_per_million / 1_000_000 + output_tokens * output_per_million / 1_000_000, 6)
