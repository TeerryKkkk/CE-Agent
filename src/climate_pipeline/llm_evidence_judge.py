from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import tiktoken

from climate_pipeline.bounded_http import (
    no_redirect_opener,
    read_bounded,
    read_http_error,
    validate_headers,
)

from . import config


PRIMARY_STATUSES = {"accepted", "needs_review", "context_only", "rejected"}
PAGE_TYPES = {
    "event_impact_page",
    "official_response_page",
    "news_impact_page",
    "broad_resource_page",
    "generic_planning_page",
    "duplicate",
    "unrelated",
    "other",
}
COMPONENTS = {"drought", "wet", "both", "none"}
IMPACT_TYPES = {
    "transportation",
    "evacuation_or_rescue",
    "property_damage",
    "human_impact",
    "agriculture",
    "public_services",
    "emergency_declaration",
    "other",
}
LOCATION_MATCHES = {"same_county", "mapped_city_or_place", "regional_only", "no_match"}
TIME_MATCHES = {
    "exact_window",
    "same_month",
    "same_storm_sequence",
    "broad_incident_period",
    "year_only",
    "no_match",
}
SOURCE_ROLES = {"primary_impact", "supporting_impact", "context", "reject"}
FAILURE_REASONS = {
    "wrong_time_window",
    "generic_planning_page",
    "broad_resource_page",
    "duplicate_url",
    "weak_location_grounding",
    "weak_time_grounding",
    "weak_impact_grounding",
    "unrelated",
    "none",
}
SKEPTIC_DECISIONS = {"confirm", "downgrade_to_needs_review", "downgrade_to_context_only", "reject"}
FINAL_STATUSES = {"accepted", "needs_review", "context_only", "rejected"}

IMPACT_TYPE_ALIASES = {
    "damage": "property_damage",
    "evacuation": "evacuation_or_rescue",
    "evacuations": "evacuation_or_rescue",
    "power outage": "public_services",
    "power outages": "public_services",
    "power_outage": "public_services",
    "rescue": "evacuation_or_rescue",
    "rescues": "evacuation_or_rescue",
    "road closure": "transportation",
    "road closures": "transportation",
    "shelter": "public_services",
    "shelters": "public_services",
}
FAILURE_REASON_ALIASES = {
    "regional_only": "weak_location_grounding",
    "year_only": "weak_time_grounding",
    "year_only_or_weak_time": "weak_time_grounding",
}
PAGE_TYPE_ALIASES = {
    "regional_only": "broad_resource_page",
}


PRIMARY_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "source_url": {"type": "string"},
        "source_title": {"type": "string"},
        "page_type": {"type": "string", "enum": sorted(PAGE_TYPES)},
        "status": {"type": "string", "enum": sorted(PRIMARY_STATUSES)},
        "component_supported": {"type": "string", "enum": sorted(COMPONENTS)},
        "hazard_supported": {"type": "boolean"},
        "impact_supported": {"type": "boolean"},
        "impact_types": {"type": "array", "items": {"type": "string", "enum": sorted(IMPACT_TYPES)}},
        "location_match": {"type": "string", "enum": sorted(LOCATION_MATCHES)},
        "time_match": {"type": "string", "enum": sorted(TIME_MATCHES)},
        "source_role": {"type": "string", "enum": sorted(SOURCE_ROLES)},
        "explicit_transition_support": {"type": "boolean"},
        "quoted_supporting_spans": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "failure_reason_if_rejected": {"type": "string", "enum": sorted(FAILURE_REASONS)},
    },
    "required": [
        "candidate_id",
        "source_url",
        "source_title",
        "page_type",
        "status",
        "component_supported",
        "hazard_supported",
        "impact_supported",
        "impact_types",
        "location_match",
        "time_match",
        "source_role",
        "explicit_transition_support",
        "quoted_supporting_spans",
        "reason",
        "failure_reason_if_rejected",
    ],
    "additionalProperties": False,
}

SKEPTIC_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "skeptic_decision": {"type": "string", "enum": sorted(SKEPTIC_DECISIONS)},
        "skeptic_reason": {"type": "string"},
        "identified_failure_modes": {"type": "array", "items": {"type": "string"}},
        "minimum_safe_status": {"type": "string", "enum": sorted(FINAL_STATUSES)},
    },
    "required": ["skeptic_decision", "skeptic_reason", "identified_failure_modes", "minimum_safe_status"],
    "additionalProperties": False,
}

ARBITER_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_status": {"type": "string", "enum": sorted(FINAL_STATUSES)},
        "final_component_supported": {"type": "string", "enum": sorted(COMPONENTS)},
        "final_impact_supported": {"type": "boolean"},
        "final_impact_types": {"type": "array", "items": {"type": "string", "enum": sorted(IMPACT_TYPES)}},
        "final_source_role": {"type": "string", "enum": sorted(SOURCE_ROLES)},
        "final_reason": {"type": "string"},
        "final_quoted_spans": {"type": "array", "items": {"type": "string"}},
        "disagreement_resolution": {"type": "string"},
    },
    "required": [
        "final_status",
        "final_component_supported",
        "final_impact_supported",
        "final_impact_types",
        "final_source_role",
        "final_reason",
        "final_quoted_spans",
        "disagreement_resolution",
    ],
    "additionalProperties": False,
}


class JudgeSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ModelCallResult:
    role: str
    model: str
    status: str
    parsed: dict[str, Any] | None
    raw_output_text: str
    raw_response: dict[str, Any] | None
    attempts: int
    usage: dict[str, Any]
    error: str = ""


def read_openai_api_key(path: Path | None = None) -> str:
    if config.OPENAI_API_KEY:
        return config.OPENAI_API_KEY
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY", "")
    candidate_paths = []
    if path is not None:
        candidate_paths.append(Path(path))
    candidate_paths.extend(
        [
            config.PROJECT_ROOT / "openai_apikey.txt",
            config.PROJECT_ROOT / "apikeys" / "openai_apikey.txt",
        ]
    )
    for key_path in candidate_paths:
        if not key_path.exists():
            continue
        with key_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    return value
    return ""


class OpenAIEvidenceJudgeClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        opener=None,
        responses_url: str | None = None,
        timeout_seconds: int | None = None,
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
        if not api_key:
            raise ValueError("missing_openai_api_key")
        self.api_key = api_key
        self.model = model or config.OPENAI_MODEL
        if int(max_redirects) != 0:
            raise ValueError("openai_redirects_must_be_disabled")
        self.opener = opener or no_redirect_opener()
        self.responses_url = responses_url or config.OPENAI_RESPONSES_URL
        self.timeout_seconds = timeout_seconds or config.OPENAI_TIMEOUT_SECONDS
        for name, value in (
            ("max_input_tokens", max_input_tokens),
            ("max_output_tokens", max_output_tokens),
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
        self.max_output_tokens = int(max_output_tokens)
        self.max_request_bytes = int(max_request_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.max_error_response_bytes = int(max_error_response_bytes)
        self.max_redirects = 0
        self.max_redirect_response_bytes = int(max_redirect_response_bytes)
        self.max_header_bytes = int(max_header_bytes)
        self.max_redirect_location_bytes = int(max_redirect_location_bytes)
        self._tokenizer = tiktoken.get_encoding("o200k_base")

    def primary_judge(
        self,
        *,
        candidate_metadata: dict[str, Any],
        source_metadata: dict[str, Any],
        spans: list[dict[str, Any]],
    ) -> ModelCallResult:
        instructions = (
            "You are the primary evidence judge for CE-Agent. Judge only the supplied candidate metadata, "
            "source metadata, and extracted page spans. Do not browse. Do not use prior knowledge. "
            "Return strict JSON only. Accepted impact evidence must quote a span that body-grounds location, "
            "time, hazard, and impact in the same span or nearby supplied sentence window. Year-only time "
            "matches are not enough for wet/flood impact. Regional-only or broad resource pages are context "
            "unless they quote target county/city plus concrete impact. Generic emergency planning pages are "
            "rejected unless they quote event-date impacts. Explicit drought-to-wet transition support is "
            "false unless the source directly says the transition itself caused or shaped impact."
        )
        user_payload = {
            "candidate_metadata": candidate_metadata,
            "source_metadata": source_metadata,
            "extracted_spans": spans,
            "output_schema_name": "primary_evidence_judgment",
        }
        return self._call_json(
            role="primary_judge",
            instructions=instructions,
            user_payload=user_payload,
            schema=PRIMARY_JUDGMENT_SCHEMA,
            schema_name="primary_evidence_judgment",
            validator=validate_primary_judgment,
        )

    def skeptic_judge(
        self,
        *,
        candidate_metadata: dict[str, Any],
        source_metadata: dict[str, Any],
        spans: list[dict[str, Any]],
        primary_judgment: dict[str, Any],
    ) -> ModelCallResult:
        instructions = (
            "You are the skeptic/auditor evidence judge for CE-Agent. Check whether the primary judgment "
            "over-accepted the page. Use only the supplied spans and metadata. Verify quoted spans actually "
            "name the target county/city/place, match the candidate event window or same storm sequence, "
            "and state a concrete impact. Downgrade generic planning pages, broad resource pages, duplicate "
            "URLs, wrong-window pages, regional-only pages, and weak title-only support."
        )
        user_payload = {
            "candidate_metadata": candidate_metadata,
            "source_metadata": source_metadata,
            "extracted_spans": spans,
            "primary_judgment": primary_judgment,
            "output_schema_name": "skeptic_evidence_judgment",
        }
        return self._call_json(
            role="skeptic_judge",
            instructions=instructions,
            user_payload=user_payload,
            schema=SKEPTIC_JUDGMENT_SCHEMA,
            schema_name="skeptic_evidence_judgment",
            validator=validate_skeptic_judgment,
        )

    def arbiter_judge(
        self,
        *,
        candidate_metadata: dict[str, Any],
        source_metadata: dict[str, Any],
        spans: list[dict[str, Any]],
        primary_judgment: dict[str, Any],
        skeptic_judgment: dict[str, Any],
    ) -> ModelCallResult:
        instructions = (
            "You are the conservative arbiter for CE-Agent evidence validation. Resolve disagreement between "
            "the primary judge and skeptic. Use only the supplied metadata, spans, and judgments. If the page "
            "does not quote target location, aligned event time, hazard, and concrete impact, do not accept. "
            "Duplicate URLs should not create additional accepted evidence. Broad resource pages should be "
            "context-only unless the supplied span gives local event-window impact."
        )
        user_payload = {
            "candidate_metadata": candidate_metadata,
            "source_metadata": source_metadata,
            "extracted_spans": spans,
            "primary_judgment": primary_judgment,
            "skeptic_judgment": skeptic_judgment,
            "output_schema_name": "arbiter_evidence_judgment",
        }
        return self._call_json(
            role="arbiter_judge",
            instructions=instructions,
            user_payload=user_payload,
            schema=ARBITER_JUDGMENT_SCHEMA,
            schema_name="arbiter_evidence_judgment",
            validator=validate_arbiter_judgment,
        )

    def _call_json(
        self,
        *,
        role: str,
        instructions: str,
        user_payload: dict[str, Any],
        schema: dict[str, Any],
        schema_name: str,
        validator,
        max_attempts: int = 2,
    ) -> ModelCallResult:
        last_error = ""
        raw_text = ""
        raw_response: dict[str, Any] | None = None
        wire_attempts = 0
        while wire_attempts < max_attempts:
            body: dict[str, Any] = {
                "model": self.model,
                "instructions": instructions,
                "input": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": False,
                    }
                },
                "temperature": 0,
                "max_output_tokens": self.max_output_tokens,
            }
            try:
                wire_attempts += 1
                raw_response = self._request(body)
                raw_text = self._extract_output_text(raw_response)
                parsed = json.loads(raw_text)
                normalized = validator(parsed)
                return ModelCallResult(
                    role=role,
                    model=self.model,
                    status="ok",
                    parsed=normalized,
                    raw_output_text=raw_text,
                    raw_response=raw_response,
                    attempts=wire_attempts,
                    usage=_usage(raw_response),
                )
            except urllib.error.HTTPError as exc:
                _, detail = read_http_error(
                    exc,
                    max_error_bytes=self.max_error_response_bytes,
                    max_redirect_bytes=self.max_redirect_response_bytes,
                    max_header_bytes=self.max_header_bytes,
                    max_location_bytes=self.max_redirect_location_bytes,
                )
                if (
                    exc.code == 400
                    and "temperature" in detail.lower()
                    and wire_attempts < max_attempts
                ):
                    try:
                        body.pop("temperature", None)
                        wire_attempts += 1
                        raw_response = self._request(body)
                        raw_text = self._extract_output_text(raw_response)
                        parsed = json.loads(raw_text)
                        normalized = validator(parsed)
                        return ModelCallResult(
                            role=role,
                            model=self.model,
                            status="ok",
                            parsed=normalized,
                            raw_output_text=raw_text,
                            raw_response=raw_response,
                            attempts=wire_attempts,
                            usage=_usage(raw_response),
                        )
                    except Exception as retry_exc:
                        last_error = _safe_error(retry_exc, self.api_key)
                else:
                    last_error = f"http_{exc.code}: {_redact(detail, self.api_key)}"
            except (json.JSONDecodeError, JudgeSchemaError) as exc:
                last_error = _safe_error(exc, self.api_key)
            except (TimeoutError, socket.timeout) as exc:
                last_error = f"timeout: {_safe_error(exc, self.api_key)}"
            except Exception as exc:
                last_error = _safe_error(exc, self.api_key)
        return ModelCallResult(
            role=role,
            model=self.model,
            status="failed",
            parsed=None,
            raw_output_text=raw_text,
            raw_response=raw_response,
            attempts=wire_attempts,
            usage=_usage(raw_response or {}),
            error=last_error,
        )

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        wire = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(wire) > self.max_request_bytes:
            raise ValueError(
                f"openai_request_body_exceeds_limit:{self.max_request_bytes}"
            )
        token_count = len(
            self._tokenizer.encode(wire.decode("utf-8", errors="strict"))
        )
        if token_count > self.max_input_tokens:
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
                label="openai_response_body",
            ).data
        parsed = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        if not isinstance(parsed, dict):
            raise ValueError("openai_response_not_object")
        return parsed

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        chunks: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    chunks.append(str(content.get("text") or ""))
        return "\n".join(chunk for chunk in chunks if chunk).strip()


def validate_primary_judgment(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JudgeSchemaError("primary_judgment_not_object")
    normalized = dict(payload)
    _normalize_primary_aliases(normalized)
    _require_keys(normalized, PRIMARY_JUDGMENT_SCHEMA["required"])
    _require_enum(normalized, "page_type", PAGE_TYPES)
    _require_enum(normalized, "status", PRIMARY_STATUSES)
    _require_enum(normalized, "component_supported", COMPONENTS)
    _require_bool(normalized, "hazard_supported")
    _require_bool(normalized, "impact_supported")
    _require_list_enum(normalized, "impact_types", IMPACT_TYPES)
    _require_enum(normalized, "location_match", LOCATION_MATCHES)
    _require_enum(normalized, "time_match", TIME_MATCHES)
    _require_enum(normalized, "source_role", SOURCE_ROLES)
    _require_bool(normalized, "explicit_transition_support")
    _require_string_list(normalized, "quoted_supporting_spans")
    _require_enum(normalized, "failure_reason_if_rejected", FAILURE_REASONS)
    if normalized["status"] == "accepted" and not normalized["quoted_supporting_spans"]:
        raise JudgeSchemaError("accepted_without_quoted_span")
    if normalized["status"] == "accepted" and normalized["time_match"] in {"year_only", "no_match"}:
        raise JudgeSchemaError("accepted_with_weak_time_match")
    if normalized["status"] == "accepted" and not normalized["impact_supported"]:
        raise JudgeSchemaError("accepted_without_impact_supported")
    return normalized


def validate_skeptic_judgment(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JudgeSchemaError("skeptic_judgment_not_object")
    _require_keys(payload, SKEPTIC_JUDGMENT_SCHEMA["required"])
    normalized = dict(payload)
    _require_enum(normalized, "skeptic_decision", SKEPTIC_DECISIONS)
    _require_string_list(normalized, "identified_failure_modes")
    _require_enum(normalized, "minimum_safe_status", FINAL_STATUSES)
    return normalized


def validate_arbiter_judgment(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise JudgeSchemaError("arbiter_judgment_not_object")
    normalized = dict(payload)
    _normalize_arbiter_aliases(normalized)
    _require_keys(normalized, ARBITER_JUDGMENT_SCHEMA["required"])
    _require_enum(normalized, "final_status", FINAL_STATUSES)
    _require_enum(normalized, "final_component_supported", COMPONENTS)
    _require_bool(normalized, "final_impact_supported")
    _require_list_enum(normalized, "final_impact_types", IMPACT_TYPES)
    _require_enum(normalized, "final_source_role", SOURCE_ROLES)
    _require_string_list(normalized, "final_quoted_spans")
    if normalized["final_status"] == "accepted" and not normalized["final_quoted_spans"]:
        raise JudgeSchemaError("accepted_final_without_quoted_span")
    return normalized


def _normalization_notes(payload: dict[str, Any]) -> list[str]:
    notes = payload.get("schema_normalization_notes")
    if isinstance(notes, list):
        return notes
    notes = []
    payload["schema_normalization_notes"] = notes
    return notes


def _canonical_alias(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


def _normalize_enum_alias(payload: dict[str, Any], key: str, aliases: dict[str, str]) -> None:
    value = payload.get(key)
    if not isinstance(value, str):
        return
    alias_key = _canonical_alias(value)
    replacement = aliases.get(value.strip().lower()) or aliases.get(alias_key)
    if replacement and replacement != value:
        payload[f"original_{key}"] = value
        payload[key] = replacement
        _normalization_notes(payload).append(f"{key}:{value}->{replacement}")


def _normalize_impact_type_list(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, list):
        return
    normalized: list[str] = []
    changed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            normalized.append(item)
            continue
        alias_key = _canonical_alias(item)
        replacement = IMPACT_TYPE_ALIASES.get(item.strip().lower()) or IMPACT_TYPE_ALIASES.get(alias_key) or item
        if replacement != item:
            changed.append(f"{item}->{replacement}")
        if replacement not in normalized:
            normalized.append(replacement)
    if changed:
        payload[f"original_{key}"] = list(value)
        payload[key] = normalized
        _normalization_notes(payload).append(f"{key}:{','.join(changed)}")


def _conservative_failure_reason(payload: dict[str, Any]) -> str:
    page_type = str(payload.get("page_type") or payload.get("final_page_type") or "").strip()
    location_match = str(payload.get("location_match") or "").strip()
    time_match = str(payload.get("time_match") or "").strip()
    if page_type == "generic_planning_page":
        return "generic_planning_page"
    if page_type == "broad_resource_page":
        return "broad_resource_page"
    if location_match in {"regional_only", "no_match"}:
        return "weak_location_grounding"
    if time_match in {"year_only", "no_match"}:
        return "weak_time_grounding"
    if not payload.get("impact_supported") and not payload.get("final_impact_supported"):
        return "weak_impact_grounding"
    return "none"


def _normalize_primary_aliases(payload: dict[str, Any]) -> None:
    _normalize_enum_alias(payload, "page_type", PAGE_TYPE_ALIASES)
    _normalize_enum_alias(payload, "failure_reason_if_rejected", FAILURE_REASON_ALIASES)
    _normalize_impact_type_list(payload, "impact_types")
    if "failure_reason_if_rejected" not in payload:
        status = str(payload.get("status") or "").strip()
        if status in {"accepted", "needs_review", "context_only", "rejected"}:
            reason = "none" if status == "accepted" else _conservative_failure_reason(payload)
            payload["failure_reason_if_rejected"] = reason
            _normalization_notes(payload).append(f"failure_reason_if_rejected:<missing>->{reason}")


def _normalize_arbiter_aliases(payload: dict[str, Any]) -> None:
    _normalize_impact_type_list(payload, "final_impact_types")


def deterministic_arbiter(primary: dict[str, Any], skeptic: dict[str, Any]) -> dict[str, Any]:
    decision = skeptic.get("skeptic_decision")
    minimum = skeptic.get("minimum_safe_status", "needs_review")
    primary_status = primary.get("status", "needs_review")
    quoted = list(primary.get("quoted_supporting_spans") or [])
    if decision == "confirm":
        final_status = primary_status
    elif decision == "reject":
        final_status = "rejected"
    elif decision == "downgrade_to_context_only":
        final_status = "context_only"
    elif decision == "downgrade_to_needs_review":
        final_status = "needs_review"
    else:
        final_status = minimum if minimum in FINAL_STATUSES else "needs_review"
    if final_status == "accepted" and not quoted:
        final_status = "needs_review"
    return {
        "final_status": final_status,
        "final_component_supported": primary.get("component_supported", "none") if final_status == "accepted" else "none",
        "final_impact_supported": bool(primary.get("impact_supported")) and final_status == "accepted",
        "final_impact_types": primary.get("impact_types", []) if final_status == "accepted" else [],
        "final_source_role": primary.get("source_role", "context") if final_status == "accepted" else ("context" if final_status == "context_only" else "reject"),
        "final_reason": f"Deterministic conservative arbiter applied skeptic decision: {decision}. {skeptic.get('skeptic_reason', '')}",
        "final_quoted_spans": quoted if final_status == "accepted" else [],
        "disagreement_resolution": "deterministic_conservative_rule",
    }


def apply_safety_gates(
    *,
    final: dict[str, Any],
    primary: dict[str, Any],
    skeptic: dict[str, Any],
    source_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any] | None = None,
    body_text: str | None = None,
    body_spans: list[str] | None = None,
) -> dict[str, Any]:
    guarded = dict(final)
    reasons: list[str] = []
    duplicate_for_candidate = bool(source_metadata.get("duplicate_for_candidate"))
    url = str(source_metadata.get("source_url") or "").lower()
    source_family = str(source_metadata.get("source_family") or source_metadata.get("raw_source_family") or "").lower()
    page_type = str(primary.get("page_type") or "")
    time_match = str(primary.get("time_match") or "")
    location_match = str(primary.get("location_match") or "")
    status = guarded.get("final_status")
    if duplicate_for_candidate:
        guarded = _force_final(
            guarded,
            "rejected",
            "duplicate_url",
            "Duplicate URL for the same candidate; it cannot add another accepted evidence row.",
        )
        reasons.append("duplicate_url")
    elif "house.gov" in url and status == "accepted":
        guarded = _force_final(
            guarded,
            "context_only",
            "broad_resource_page",
            "Federal congressional resource page is context-only unless local event-window impact is explicit.",
        )
        reasons.append("broad_resource_page")
    elif page_type == "generic_planning_page" and status in {"accepted", "needs_review", "context_only"}:
        guarded = _force_final(
            guarded,
            "rejected",
            "generic_planning_page",
            "Generic emergency/planning page cannot be accepted without event-date impact spans.",
        )
        reasons.append("generic_planning_page")
    elif status == "accepted" and time_match in {"year_only", "no_match"}:
        guarded = _force_final(
            guarded,
            "rejected",
            "weak_time_grounding",
            "Accepted wet-impact evidence cannot rely on year-only or missing time alignment.",
        )
        reasons.append("weak_time_grounding")
    elif status == "accepted" and not guarded.get("final_quoted_spans"):
        guarded = _force_final(
            guarded,
            "needs_review",
            "weak_impact_grounding",
            "Accepted evidence requires at least one exact quoted supporting span.",
        )
        reasons.append("missing_quoted_span")
    elif status == "accepted" and body_text is not None and not quoted_spans_are_body_grounded(
        guarded.get("final_quoted_spans") or [],
        body_text=body_text,
        body_spans=body_spans or [],
    ):
        guarded = _force_final(
            guarded,
            "needs_review",
            "weak_impact_grounding",
            "Accepted web evidence requires at least one quoted supporting span found in the fetched body text or selected body spans after whitespace normalization.",
        )
        reasons.append("quote_not_body_grounded")
    elif status == "accepted" and location_match == "regional_only":
        guarded = _force_final(
            guarded,
            "context_only",
            "weak_location_grounding",
            "Regional-only location support is context-only unless the target county/city is quoted with impact.",
        )
        reasons.append("regional_only_location")
    elif (
        status == "accepted"
        and time_match == "same_storm_sequence"
        and "news" in source_family
        and _same_storm_news_timing_is_weak(guarded, candidate_metadata or {})
    ):
        guarded = _force_final(
            guarded,
            "needs_review",
            "weak_time_grounding",
            "News evidence marked only as same-storm sequence was downgraded because quoted dates are outside the tight candidate wet-window allowance.",
        )
        reasons.append("weak_same_storm_news_timing")
    if reasons:
        previous = guarded.get("disagreement_resolution", "")
        guarded["disagreement_resolution"] = "; ".join([part for part in [previous, f"safety_gates={','.join(reasons)}"] if part])
    return guarded


def quoted_spans_are_body_grounded(
    quoted_spans: list[Any],
    *,
    body_text: str,
    body_spans: list[str] | None = None,
) -> bool:
    body_norm = _normalize_quote_grounding_text(body_text)
    span_norms = [_normalize_quote_grounding_text(span) for span in (body_spans or [])]
    for quote in quoted_spans:
        quote_norm = _normalize_quote_grounding_text(str(quote or ""))
        if not quote_norm:
            continue
        if body_norm and quote_norm in body_norm:
            return True
        if any(quote_norm in span_norm for span_norm in span_norms if span_norm):
            return True
    return False


def _normalize_quote_grounding_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def judgments_disagree(primary: dict[str, Any], skeptic: dict[str, Any]) -> bool:
    if skeptic.get("skeptic_decision") != "confirm":
        return True
    minimum = skeptic.get("minimum_safe_status")
    return bool(minimum and minimum != primary.get("status"))


def _force_final(final: dict[str, Any], status: str, failure_reason: str, reason: str) -> dict[str, Any]:
    updated = dict(final)
    updated["final_status"] = status
    updated["final_component_supported"] = "none" if status != "accepted" else updated.get("final_component_supported", "none")
    updated["final_impact_supported"] = bool(updated.get("final_impact_supported")) and status == "accepted"
    if status != "accepted":
        updated["final_impact_types"] = []
        updated["final_quoted_spans"] = []
    updated["final_source_role"] = "context" if status == "context_only" else ("reject" if status == "rejected" else updated.get("final_source_role", "supporting_impact"))
    updated["final_reason"] = f"{reason} Previous arbiter reason: {updated.get('final_reason', '')}".strip()
    updated["failure_reason_if_rejected"] = failure_reason
    return updated


def _same_storm_news_timing_is_weak(final: dict[str, Any], candidate_metadata: dict[str, Any]) -> bool:
    wet_window = str(
        candidate_metadata.get("wet_window")
        or candidate_metadata.get("rain_flood_window")
        or candidate_metadata.get("event_window")
        or ""
    )
    start, end = _parse_wet_window(wet_window)
    if not start or not end:
        return False
    quoted = " ".join(str(value) for value in final.get("final_quoted_spans") or [])
    quote_dates = _dates_from_text(quoted, default_year=end.year)
    if not quote_dates:
        return False
    allowance_end = end + timedelta(days=1)
    return not any(start <= value <= allowance_end for value in quote_dates)


def _parse_wet_window(value: str) -> tuple[date | None, date | None]:
    parts = [part.strip() for part in re.split(r"\s+to\s+", value or "", maxsplit=1)]
    if len(parts) == 1:
        parts.append(parts[0])
    return _parse_iso_date(parts[0]), _parse_iso_date(parts[1])


def _parse_iso_date(value: str) -> date | None:
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _dates_from_text(text: str, *, default_year: int) -> list[date]:
    month_lookup = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    values: list[date] = []
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            values.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        except ValueError:
            continue
    for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text):
        year = int(match.group(3))
        if year < 100:
            year += 2000
        try:
            values.append(date(year, int(match.group(1)), int(match.group(2))))
        except ValueError:
            continue
    month_pattern = "|".join(sorted(month_lookup, key=len, reverse=True))
    for match in re.finditer(rf"\b({month_pattern})\.?\s+(\d{{1,2}})(?:,\s*(\d{{4}}))?\b", text, flags=re.IGNORECASE):
        month = month_lookup[match.group(1).lower().rstrip(".")]
        year = int(match.group(3) or default_year)
        try:
            values.append(date(year, month, int(match.group(2))))
        except ValueError:
            continue
    unique: list[date] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _require_keys(payload: dict[str, Any], required: list[str]) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise JudgeSchemaError(f"missing_required_keys:{','.join(missing)}")


def _require_enum(payload: dict[str, Any], key: str, allowed: set[str]) -> None:
    value = payload.get(key)
    if value not in allowed:
        raise JudgeSchemaError(f"invalid_{key}:{value}")


def _require_bool(payload: dict[str, Any], key: str) -> None:
    if not isinstance(payload.get(key), bool):
        raise JudgeSchemaError(f"invalid_{key}_bool")


def _require_string_list(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise JudgeSchemaError(f"invalid_{key}_string_list")


def _require_list_enum(payload: dict[str, Any], key: str, allowed: set[str]) -> None:
    value = payload.get(key)
    if not isinstance(value, list):
        raise JudgeSchemaError(f"invalid_{key}_list")
    invalid = [item for item in value if item not in allowed]
    if invalid:
        raise JudgeSchemaError(f"invalid_{key}:{invalid}")


def _safe_error(exc: Exception, api_key: str) -> str:
    return _redact(str(exc), api_key)[:800]


def _redact(text: str, api_key: str) -> str:
    safe = text.replace(api_key, "[REDACTED_OPENAI_KEY]") if api_key else text
    safe = re.sub(r"sk-[A-Za-z0-9_\-]+", "[REDACTED_OPENAI_KEY]", safe)
    return safe


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return usage if isinstance(usage, dict) else {}
