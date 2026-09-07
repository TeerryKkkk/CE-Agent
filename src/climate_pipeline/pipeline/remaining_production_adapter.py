"""Explicit remaining-368 adapters into the frozen repaired-v2 schemas.

This module only maps completed production-stage artifacts into canonical
records.  It does not aggregate axes or maintain a second truth table.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping

from .frozen_webpage_adapter import FROZEN_WEBPAGE_LANE
from .official_adapters import USDM_LANE
from .schemas import (
    ADAPTER_VERSION_ENVELOPE,
    AxisCoverageState,
    AxisName,
    AxisValue,
    EvidenceDisposition,
    FrozenWebpageExecutionDetails,
    JudgmentCoverageState,
    LaneExecutionRecord,
    LaneExecutionStatus,
    LaneReadiness,
    ReasonCode,
    RetrievalCoverageState,
    RuleId,
    SourceRole,
    SpatialAlignmentGrade,
    StructuredEvidenceRecord,
    TemporalAlignmentGrade,
    VersionLineage,
    WebpageCoverage,
)


USDM_LIVE_ADAPTER_VERSION = f"{ADAPTER_VERSION_ENVELOPE}.usdm_county_week_live_checkpoint.1"
WEBPAGE_PRODUCTION_ADAPTER_VERSION = f"{ADAPTER_VERSION_ENVELOPE}.frozen_webpage_production.1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _axis_map(**values: AxisValue) -> dict[AxisName, AxisValue]:
    result = {axis: AxisValue.NO for axis in AxisName}
    for name, value in values.items():
        result[AxisName(name)] = value
    return result


def adapt_usdm_checkpoint(
    *,
    case: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> tuple[LaneExecutionRecord, tuple[StructuredEvidenceRecord, ...]]:
    """Map one completed USDM county/week checkpoint using repaired-v2 rules."""

    candidate_id = str(case["candidate_id"])
    package_id = f"remaining368-usdm-checkpoint-{candidate_id}"
    package_sha = _stable_hash(source_row)
    status = str(source_row.get("final_status") or "")
    d1_text = str(source_row.get("usdm_d1_area_percent") or "").strip()
    source_id = str(source_row.get("structured_record_id") or "")
    source_url = str(source_row.get("source_url") or "")
    if status == "lane_failed" or not d1_text or not source_id:
        execution = LaneExecutionRecord(
            candidate_id=candidate_id,
            lane=USDM_LANE,
            enabled=True,
            required=True,
            source_mode=str(source_row.get("usdm_source_mode") or "live_or_checkpoint"),
            package_id=package_id,
            package_sha256=package_sha,
            readiness=LaneReadiness.UNAVAILABLE,
            execution_status=LaneExecutionStatus.RETRYABLE_FAILURE,
            coverage=AxisCoverageState.FAILED,
            records_examined=0,
            matches_found=0,
            accepted_count=0,
            contextual_count=0,
            unresolved_count=0,
            lineage=VersionLineage(adapter_version=USDM_LIVE_ADAPTER_VERSION),
        )
        return execution, ()

    try:
        d1 = float(d1_text)
    except ValueError:
        d1 = math.nan
    if not math.isfinite(d1) or not 0.0 <= d1 <= 100.0:
        execution = LaneExecutionRecord(
            candidate_id=candidate_id,
            lane=USDM_LANE,
            enabled=True,
            required=True,
            source_mode=str(source_row.get("usdm_source_mode") or "live_or_checkpoint"),
            package_id=package_id,
            package_sha256=package_sha,
            readiness=LaneReadiness.UNAVAILABLE,
            execution_status=LaneExecutionStatus.TERMINAL_FAILURE,
            coverage=AxisCoverageState.FAILED,
            records_examined=1,
            matches_found=0,
            accepted_count=0,
            contextual_count=0,
            unresolved_count=0,
            lineage=VersionLineage(adapter_version=USDM_LIVE_ADAPTER_VERSION),
        )
        return execution, ()
    supports = d1 > 0.0
    disposition = EvidenceDisposition.ACCEPTED if supports else EvidenceDisposition.REJECTED_NON_SUPPORT
    assessments = _axis_map(
        drought_corroboration=AxisValue.YES if supports else AxisValue.NO,
    )
    evidence_payload = {
        "candidate_id": candidate_id,
        "source_record_id": source_id,
        "source_checkpoint": dict(source_row),
        "axis_assessments": {axis.value: value.value for axis, value in assessments.items()},
    }
    evidence = StructuredEvidenceRecord(
        candidate_id=candidate_id,
        lane=USDM_LANE,
        package_id=package_id,
        source_role=SourceRole.USDM_COUNTY_WEEK_DROUGHT_CONTEXT,
        disposition=disposition,
        source_record_id=source_id,
        source_event_id="",
        source_file="structured_usdm_checkpoint.json",
        source_row="1",
        source_url=source_url,
        source_sha256=package_sha,
        source_version=USDM_LIVE_ADAPTER_VERSION,
        source_date=str(source_row.get("source_date_if_available") or ""),
        raw_value_provenance=evidence_payload,
        spatial_grade=SpatialAlignmentGrade.COUNTY_CONTEXT,
        temporal_grade=TemporalAlignmentGrade.PREVIOUS_OR_SAME_TUESDAY,
        axis_assessments=assessments,
        rule_ids=(
            RuleId.USDM_REFERENCE_WEEK,
            RuleId.USDM_D1_POSITIVE if supports else RuleId.USDM_D1_ZERO,
            RuleId.USDM_AXIS_SCOPE,
            RuleId.PROHIBIT_USDM_NOAA_LINKAGE,
        ),
        reason_codes=(
            ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD
            if supports
            else ReasonCode.NO_SUPPORT_IN_SUCCESSFULLY_EVALUATED_SCOPE,
        ),
        record_hash=_stable_hash(evidence_payload),
        lineage=VersionLineage(adapter_version=USDM_LIVE_ADAPTER_VERSION),
    )
    execution = LaneExecutionRecord(
        candidate_id=candidate_id,
        lane=USDM_LANE,
        enabled=True,
        required=True,
        source_mode=str(source_row.get("usdm_source_mode") or "live_or_checkpoint"),
        package_id=package_id,
        package_sha256=package_sha,
        readiness=LaneReadiness.READY,
        execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
        coverage=AxisCoverageState.COMPLETE,
        records_examined=1,
        matches_found=1,
        accepted_count=int(supports),
        contextual_count=0,
        unresolved_count=0,
        completed_at="checkpoint_complete",
        lineage=VersionLineage(adapter_version=USDM_LIVE_ADAPTER_VERSION),
    )
    return execution, (evidence,)


def _axis_value(raw: Any) -> AxisValue:
    try:
        return AxisValue(str(raw or "unresolved"))
    except ValueError:
        return AxisValue.UNRESOLVED


def _spatial_grade(raw: Any) -> SpatialAlignmentGrade:
    value = str(raw or "").strip().casefold()
    if value in {"aligned", "same_county", "county"}:
        return SpatialAlignmentGrade.COUNTY
    if value in {"mismatch", "wrong_location", "wrong_county"}:
        return SpatialAlignmentGrade.MISMATCH
    return SpatialAlignmentGrade.UNRESOLVED


def _temporal_grade(raw: Any) -> TemporalAlignmentGrade:
    value = str(raw or "").strip().casefold()
    if value in {"aligned", "exact_window", "candidate_window_exact_overlap"}:
        return TemporalAlignmentGrade.CANDIDATE_WINDOW_EXACT_OVERLAP
    if value in {"mismatch", "wrong_time", "wrong_date"}:
        return TemporalAlignmentGrade.MISMATCH
    return TemporalAlignmentGrade.UNRESOLVED


@dataclass(frozen=True)
class WebpageAdapterResult:
    execution: LaneExecutionRecord
    evidence: tuple[StructuredEvidenceRecord, ...]


def adapt_webpage_checkpoints(
    *,
    candidate_id: str,
    pages: Iterable[Mapping[str, Any]],
    webpage_rows: Iterable[Mapping[str, Any]],
    request_payloads: Iterable[Mapping[str, Any]],
    package_id: str,
) -> WebpageAdapterResult:
    """Map frozen bodies plus completed direct-judge outputs into repaired-v2."""

    page_rows = tuple(dict(row) for row in pages)
    judged_rows = tuple(dict(row) for row in webpage_rows)
    requests = tuple(dict(row) for row in request_payloads)
    page_by_url = {str(row.get("source_url") or ""): row for row in page_rows}
    package_payload = {"pages": page_rows, "requests": requests}
    package_sha = _stable_hash(package_payload)
    evidence: list[StructuredEvidenceRecord] = []
    for index, row in enumerate(judged_rows, start=1):
        url = str(row.get("source_url") or "")
        page = page_by_url.get(url, {})
        body_hash = str(page.get("frozen_body_sha256") or row.get("body_sha256") or "")
        if not SHA256_RE.fullmatch(body_hash):
            body_hash = _stable_hash(str(page.get("body_text_or_archived_body_text") or ""))
        page_result = str(row.get("page_result") or "unresolved")
        explicit_drought_axis = row.get("drought_hazard_support")
        explicit_wet_axis = row.get("wet_hazard_support")
        assessments = _axis_map(
            drought_corroboration=(
                _axis_value(explicit_drought_axis)
                if explicit_drought_axis not in {None, ""}
                else AxisValue.UNRESOLVED
            ),
            wet_hazard_corroboration=(
                _axis_value(explicit_wet_axis)
                if explicit_wet_axis not in {None, ""}
                else _axis_value(row.get("candidate_hazard_support"))
            ),
            realized_impact=_axis_value(row.get("realized_impact_support")),
            hazard_to_impact_attribution=_axis_value(row.get("explicit_attribution_support")),
            drought_to_wet_linkage=_axis_value(row.get("explicit_drought_to_wet_transition_support")),
        )
        if page_result == "supports" and AxisValue.YES in assessments.values():
            disposition = EvidenceDisposition.ACCEPTED
            reason = ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD
        elif page_result == "does_not_support":
            disposition = EvidenceDisposition.REJECTED_NON_SUPPORT
            reason = ReasonCode.NO_SUPPORT_IN_SUCCESSFULLY_EVALUATED_SCOPE
        else:
            disposition = EvidenceDisposition.UNRESOLVED
            reason = (
                ReasonCode.SOURCE_CONTENT_INSUFFICIENT
                if page_result == "insufficient_source_content"
                else ReasonCode.CANDIDATE_RELEVANT_UNRESOLVED
            )
        record_id = str(page.get("page_id") or row.get("page_id") or _stable_hash({"candidate_id": candidate_id, "url": url, "body": body_hash})[:24])
        raw_provenance = {
            "adapter_action": "completed_direct_judge_checkpoint_mapping",
            "page_result": page_result,
            "webpage_row": row,
            "body_sha256": body_hash,
        }
        evidence.append(
            StructuredEvidenceRecord(
                candidate_id=candidate_id,
                lane=FROZEN_WEBPAGE_LANE,
                package_id=package_id,
                source_role=SourceRole.FROZEN_WEBPAGE_RECORD,
                disposition=disposition,
                source_record_id=record_id,
                source_event_id="",
                source_file="fetched_pages.jsonl",
                source_row=str(index),
                source_url=url,
                source_sha256=body_hash,
                source_version=WEBPAGE_PRODUCTION_ADAPTER_VERSION,
                source_date="",
                raw_value_provenance=raw_provenance,
                spatial_grade=_spatial_grade(row.get("location_match")),
                temporal_grade=_temporal_grade(row.get("time_match")),
                axis_assessments=assessments,
                rule_ids=(RuleId.FROZEN_WEB_PRESERVE,),
                reason_codes=(reason,),
                record_hash=_stable_hash({"record_id": record_id, "raw": raw_provenance, "assessments": {axis.value: value.value for axis, value in assessments.items()}}),
                lineage=VersionLineage(adapter_version=WEBPAGE_PRODUCTION_ADAPTER_VERSION),
            )
        )

    logical = len(page_rows)
    readable = sum(bool(str(row.get("body_text_or_archived_body_text") or "").strip()) for row in page_rows)
    completed = len(requests)
    guarded = len(judged_rows)
    reported_insufficient = sum(str(row.get("page_result") or "") == "insufficient_source_content" for row in judged_rows)
    insufficient = max(reported_insufficient, logical - readable)
    unresolved = sum(str(row.get("page_result") or "") == "unresolved" for row in judged_rows)
    all_complete = logical > 0 and completed == logical and guarded == logical
    if logical == 0:
        web_coverage = WebpageCoverage.ZERO_PAGES_RETRIEVED
        retrieval_coverage = RetrievalCoverageState.ZERO_PAGES_RETRIEVED
        judgment_coverage = JudgmentCoverageState.COMPLETE
        axis_coverage = AxisCoverageState.PARTIAL
        execution_status = LaneExecutionStatus.UNAVAILABLE
        derivation = RuleId.WEB_AXIS_ZERO_PAGES
    elif not all_complete:
        web_coverage = WebpageCoverage.JUDGMENT_INCOMPLETE_OR_FAILED
        retrieval_coverage = RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
        judgment_coverage = JudgmentCoverageState.INCOMPLETE_OR_FAILED
        axis_coverage = AxisCoverageState.FAILED
        execution_status = LaneExecutionStatus.TERMINAL_FAILURE
        derivation = RuleId.WEB_AXIS_UNRESOLVED_DEPENDENCY
    elif insufficient or unresolved:
        web_coverage = WebpageCoverage.SOURCE_CONTENT_INSUFFICIENT
        retrieval_coverage = RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
        judgment_coverage = JudgmentCoverageState.SOURCE_CONTENT_INSUFFICIENT
        axis_coverage = AxisCoverageState.PARTIAL
        execution_status = LaneExecutionStatus.COMPLETED_WITH_MATCHES
        derivation = RuleId.WEB_AXIS_UNRESOLVED_EVIDENCE
    else:
        web_coverage = WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE
        retrieval_coverage = RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
        judgment_coverage = JudgmentCoverageState.COMPLETE
        axis_coverage = AxisCoverageState.COMPLETE
        execution_status = LaneExecutionStatus.COMPLETED_WITH_MATCHES
        derivation = RuleId.WEB_AXIS_NO

    hashes = {
        "manifest": _stable_hash([str(row.get("source_url") or "") for row in page_rows]),
        "page": _stable_hash([str(row.get("page_result") or "") for row in judged_rows]),
        "body": _stable_hash([str(row.get("frozen_body_sha256") or "") for row in page_rows]),
        "judgment": _stable_hash(requests),
        "guard": _stable_hash(judged_rows),
    }
    details = FrozenWebpageExecutionDetails(
        retrieval_performed=True,
        logical_page_count=logical,
        successfully_fetched_or_readable_count=readable,
        source_content_insufficient_count=insufficient,
        completed_judgment_count=completed,
        candidate_relevant_unresolved_count=unresolved,
        guarded_page_count=guarded,
        zero_pages_retrieved=logical == 0,
        retrieval_coverage_state=retrieval_coverage,
        judgment_coverage_state=judgment_coverage,
        webpage_coverage=web_coverage,
        frozen_manifest_hash=hashes["manifest"],
        page_result_bundle_hash=hashes["page"],
        body_bundle_hash=hashes["body"],
        judgment_bundle_hash=hashes["judgment"],
        guard_bundle_hash=hashes["guard"],
        derivation_rule_id=derivation,
        adapter_version=WEBPAGE_PRODUCTION_ADAPTER_VERSION,
    )
    accepted_count = sum(row.disposition is EvidenceDisposition.ACCEPTED for row in evidence)
    unresolved_count = sum(row.disposition is EvidenceDisposition.UNRESOLVED for row in evidence)
    execution = LaneExecutionRecord(
        candidate_id=candidate_id,
        lane=FROZEN_WEBPAGE_LANE,
        enabled=True,
        required=True,
        source_mode="bounded_retrieval_frozen_body_direct_judge",
        package_id=package_id,
        package_sha256=package_sha,
        readiness=LaneReadiness.READY if execution_status.is_successful_completion else LaneReadiness.UNAVAILABLE,
        execution_status=execution_status,
        coverage=axis_coverage,
        records_examined=logical,
        matches_found=logical,
        accepted_count=accepted_count,
        contextual_count=0,
        unresolved_count=unresolved_count,
        completed_at="checkpoint_complete" if execution_status.is_successful_completion else "",
        webpage=details,
        lineage=VersionLineage(adapter_version=WEBPAGE_PRODUCTION_ADAPTER_VERSION),
    )
    return WebpageAdapterResult(execution=execution, evidence=tuple(evidence))


__all__ = [
    "USDM_LIVE_ADAPTER_VERSION",
    "WEBPAGE_PRODUCTION_ADAPTER_VERSION",
    "WebpageAdapterResult",
    "adapt_usdm_checkpoint",
    "adapt_webpage_checkpoints",
]
