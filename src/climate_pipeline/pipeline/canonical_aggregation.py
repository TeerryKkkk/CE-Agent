"""The single production truth aggregator for repaired-v2.

Every runner mode calls :func:`aggregate_repaired_v2_case`.  Other helpers in
this module derive views or summarize already-computed truth; they do not
provide an alternate axis calculator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .frozen_webpage_adapter import (
    FROZEN_WEBPAGE_LANE,
    ACTIVE_WEB_AXIS_SCOPE_POLICY_VERSION,
    SHADOW_WEB_AXIS_SCOPE_POLICY_VERSION,
    evaluate_bounded_web_axis,
)
from .official_adapters import NOAA_LANE, USDM_LANE
from .schemas import (
    AGGREGATOR_VERSION_ENVELOPE,
    AxisCoverageState,
    AxisName,
    AxisValue,
    CanonicalSchemaError,
    CaseAxisRecord,
    CaseOutputRecord,
    ComponentCorroborationView,
    DerivedViewsRecord,
    EvidenceDisposition,
    ImpactAttributionView,
    LaneExecutionRecord,
    LaneExecutionStatus,
    LegacyCompatibilityView,
    LinkageView,
    NavigationSummary,
    PhysicalStatusRecord,
    ReasonCode,
    RuleId,
    SpatialAlignmentGrade,
    StructuredEvidenceRecord,
    TemporalAlignmentGrade,
    VersionLineage,
    validate_unique_lane_execution,
)


CANONICAL_AGGREGATOR_VERSION = f"{AGGREGATOR_VERSION_ENVELOPE}.canonical.1"
SHADOW_CANONICAL_AGGREGATOR_VERSION = (
    f"{AGGREGATOR_VERSION_ENVELOPE}.canonical.2.candidate-scope"
)
ACTIVE_CANDIDATE_SCOPE_POLICY_VERSION = "candidate_scope_v1_production"
SHADOW_CANDIDATE_SCOPE_POLICY_VERSION = "candidate_scope_v2_validated"
CANDIDATE_SCOPE_POLICY_VERSIONS = {
    ACTIVE_CANDIDATE_SCOPE_POLICY_VERSION,
    SHADOW_CANDIDATE_SCOPE_POLICY_VERSION,
}

DEFAULT_REQUIRED_LANES = (USDM_LANE, NOAA_LANE, FROZEN_WEBPAGE_LANE)
LANE_AXIS_RELEVANCE: Mapping[str, frozenset[AxisName]] = {
    USDM_LANE: frozenset({AxisName.DROUGHT_CORROBORATION}),
    NOAA_LANE: frozenset(
        {
            AxisName.WET_HAZARD_CORROBORATION,
            AxisName.REALIZED_IMPACT,
            AxisName.HAZARD_TO_IMPACT_ATTRIBUTION,
        }
    ),
    FROZEN_WEBPAGE_LANE: frozenset(
        {
            AxisName.DROUGHT_CORROBORATION,
            AxisName.WET_HAZARD_CORROBORATION,
            AxisName.REALIZED_IMPACT,
            AxisName.HAZARD_TO_IMPACT_ATTRIBUTION,
            AxisName.DROUGHT_TO_WET_LINKAGE,
        }
    ),
    "openfema": frozenset(),
}


class CanonicalAggregationError(CanonicalSchemaError):
    """Raised when upstream state cannot safely produce repaired-v2 truth."""


@dataclass(frozen=True)
class CanonicalAggregationResult:
    case_output: CaseOutputRecord
    source_mix: Mapping[str, Any]


_SPATIAL_STRENGTH = {
    SpatialAlignmentGrade.LOCAL: 7,
    SpatialAlignmentGrade.COUNTY: 6,
    SpatialAlignmentGrade.ZONE: 5,
    SpatialAlignmentGrade.COUNTY_CONTEXT: 4,
    SpatialAlignmentGrade.CONTEXT: 3,
    SpatialAlignmentGrade.UNRESOLVED: 2,
    SpatialAlignmentGrade.MISMATCH: 1,
}
_TEMPORAL_STRENGTH = {
    TemporalAlignmentGrade.CANDIDATE_WINDOW_EXACT_OVERLAP: 6,
    TemporalAlignmentGrade.EXACT_TIME_SAME_EVENT_EPISODE: 5,
    TemporalAlignmentGrade.PREVIOUS_OR_SAME_TUESDAY: 4,
    TemporalAlignmentGrade.NEAR_WINDOW_BROADER_STORM_SEQUENCE: 3,
    TemporalAlignmentGrade.UNRESOLVED: 2,
    TemporalAlignmentGrade.MISMATCH: 1,
}


def _ordered_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _strongest(values: tuple[Any, ...], order: Mapping[Any, int]) -> Any | None:
    return max(values, key=order.__getitem__) if values else None


def _execution_reason(record: LaneExecutionRecord) -> ReasonCode:
    if record.execution_status is LaneExecutionStatus.UNAVAILABLE:
        return ReasonCode.LANE_UNAVAILABLE
    if record.execution_status in {
        LaneExecutionStatus.RETRYABLE_FAILURE,
        LaneExecutionStatus.TERMINAL_FAILURE,
    }:
        return ReasonCode.SOURCE_FAILURE
    if record.coverage is AxisCoverageState.NOT_RUN:
        return ReasonCode.LANE_NOT_RUN
    return ReasonCode.COVERAGE_WARNING


def _web_target_hazard_axis(record: StructuredEvidenceRecord) -> str:
    provenance = record.raw_value_provenance
    for key in ("webpage_row", "original_page_result"):
        row = provenance.get(key)
        if not isinstance(row, Mapping):
            continue
        target_axis = str(row.get("target_hazard_axis") or "").strip().lower()
        if target_axis in {"drought", "wet"}:
            return target_axis
    return ""


def _web_record_applies_to_axis(
    record: StructuredEvidenceRecord,
    axis: AxisName,
) -> bool:
    if axis not in {
        AxisName.DROUGHT_CORROBORATION,
        AxisName.WET_HAZARD_CORROBORATION,
    }:
        return True
    target_axis = _web_target_hazard_axis(record)
    if target_axis:
        return target_axis == (
            "drought"
            if axis is AxisName.DROUGHT_CORROBORATION
            else "wet"
        )
    # Historical frozen webpage rows predate explicit hazard-axis targeting
    # and were judged only against the wet-event window.
    return axis is AxisName.WET_HAZARD_CORROBORATION


def derive_repaired_v2_views(
    *,
    candidate_id: str,
    axes: Mapping[AxisName, CaseAxisRecord],
    legacy_compatibility: LegacyCompatibilityView,
    aggregator_version: str = CANONICAL_AGGREGATOR_VERSION,
) -> DerivedViewsRecord:
    drought = axes[AxisName.DROUGHT_CORROBORATION].axis_value
    wet = axes[AxisName.WET_HAZARD_CORROBORATION].axis_value
    impact = axes[AxisName.REALIZED_IMPACT].axis_value
    attribution = axes[AxisName.HAZARD_TO_IMPACT_ATTRIBUTION].axis_value
    linkage = axes[AxisName.DROUGHT_TO_WET_LINKAGE].axis_value

    if AxisValue.UNRESOLVED in {drought, wet}:
        component = ComponentCorroborationView.COMPONENT_STATUS_UNRESOLVED
    elif drought is AxisValue.YES and wet is AxisValue.YES:
        component = ComponentCorroborationView.BOTH_COMPONENTS_CORROBORATED
    elif drought is AxisValue.YES:
        component = ComponentCorroborationView.DROUGHT_ONLY_CORROBORATED
    elif wet is AxisValue.YES:
        component = ComponentCorroborationView.WET_ONLY_CORROBORATED
    else:
        component = ComponentCorroborationView.NO_COMPONENT_CORROBORATION_IN_EVALUATED_SCOPE

    if impact is AxisValue.UNRESOLVED:
        impact_view = ImpactAttributionView.IMPACT_STATUS_UNRESOLVED
    elif impact is AxisValue.NO and attribution is AxisValue.NO:
        impact_view = ImpactAttributionView.NO_DOCUMENTED_IMPACT_IN_EVALUATED_SCOPE
    elif impact is AxisValue.YES and attribution is AxisValue.YES:
        impact_view = ImpactAttributionView.EXPLICITLY_ATTRIBUTED_IMPACT
    elif impact is AxisValue.YES and attribution is AxisValue.NO:
        impact_view = ImpactAttributionView.DOCUMENTED_IMPACT_WITHOUT_ATTRIBUTION
    elif impact is AxisValue.YES and attribution is AxisValue.UNRESOLVED:
        impact_view = ImpactAttributionView.DOCUMENTED_IMPACT_ATTRIBUTION_UNRESOLVED
    else:
        impact_view = ImpactAttributionView.IMPACT_STATUS_UNRESOLVED

    if linkage is AxisValue.YES:
        linkage_view = LinkageView.EXPLICIT_LINKAGE_SUPPORTED
    elif linkage is AxisValue.NO:
        linkage_view = LinkageView.NO_EXPLICIT_LINKAGE_IN_EVALUATED_SCOPE
    else:
        linkage_view = LinkageView.LINKAGE_UNRESOLVED

    if AxisValue.UNRESOLVED in {
        drought,
        wet,
        impact,
        attribution,
        linkage,
    }:
        navigation = NavigationSummary.REVIEW_REQUIRED
    elif drought is AxisValue.YES and wet is AxisValue.YES and impact is AxisValue.YES:
        navigation = NavigationSummary.BOTH_COMPONENTS_WITH_DOCUMENTED_IMPACT
    elif drought is AxisValue.YES and wet is AxisValue.YES:
        navigation = NavigationSummary.BOTH_COMPONENTS_CORROBORATED
    elif (drought is AxisValue.YES) ^ (wet is AxisValue.YES):
        navigation = NavigationSummary.ONE_COMPONENT_CORROBORATED
    else:
        navigation = NavigationSummary.CANDIDATE_ONLY

    return DerivedViewsRecord(
        candidate_id=candidate_id,
        component_corroboration=component,
        impact_attribution=impact_view,
        linkage=linkage_view,
        navigation_summary=navigation,
        legacy_compatibility=legacy_compatibility,
        lineage=VersionLineage(aggregator_version=aggregator_version),
    )


def _official_unresolved_is_candidate_relevant(
    record: StructuredEvidenceRecord,
    axis: AxisName,
    *,
    shadow_candidate_scope: bool,
) -> bool:
    if not (
        record.disposition is EvidenceDisposition.UNRESOLVED
        or record.axis_assessments.get(axis) is AxisValue.UNRESOLVED
    ):
        return False
    if not shadow_candidate_scope or record.lane != NOAA_LANE:
        return True
    # NOAA negative scope is candidate-bound, not a scan of every California
    # row in the +/-72 h inventory.  County/local records and zones mapped to
    # the target county can block a bounded No; spatially unbound background,
    # other counties, and unrelated statewide rows cannot.
    return bool(
        record.axis_assessments.get(axis) is AxisValue.UNRESOLVED
        and record.spatial_grade
        in {
            SpatialAlignmentGrade.LOCAL,
            SpatialAlignmentGrade.COUNTY,
            SpatialAlignmentGrade.ZONE,
        }
    )


def _validate_inputs(
    *,
    physical_status: PhysicalStatusRecord,
    execution_records: tuple[LaneExecutionRecord, ...],
    evidence_records: tuple[StructuredEvidenceRecord, ...],
    required_lanes: tuple[str, ...],
    expected_package_hashes: Mapping[str, str] | None,
) -> Mapping[str, LaneExecutionRecord]:
    try:
        validate_unique_lane_execution(execution_records)
    except CanonicalSchemaError as exc:
        raise CanonicalAggregationError(str(exc)) from exc
    candidate_id = physical_status.candidate_id
    if any(record.candidate_id != candidate_id for record in execution_records):
        raise CanonicalAggregationError("lane_execution_candidate_mismatch")
    if any(record.candidate_id != candidate_id for record in evidence_records):
        raise CanonicalAggregationError("evidence_candidate_mismatch")
    by_lane = {record.lane: record for record in execution_records}
    missing = sorted(set(required_lanes) - set(by_lane))
    if missing:
        raise CanonicalAggregationError(f"missing_required_execution_records:{','.join(missing)}")
    for record in evidence_records:
        execution = by_lane.get(record.lane)
        if execution is None:
            raise CanonicalAggregationError(f"evidence_without_lane_execution:{record.lane}")
        if record.package_id != execution.package_id:
            raise CanonicalAggregationError(f"evidence_package_mismatch:{record.lane}")
    if expected_package_hashes:
        for lane, expected in expected_package_hashes.items():
            if lane not in by_lane or by_lane[lane].package_sha256 != expected:
                raise CanonicalAggregationError(f"stale_package_hash:{lane}")
    return by_lane


def _aggregate_axis(
    *,
    candidate_id: str,
    axis: AxisName,
    by_lane: Mapping[str, LaneExecutionRecord],
    evidence: tuple[StructuredEvidenceRecord, ...],
    scope_policy_version: str,
    aggregator_version: str,
) -> CaseAxisRecord:
    shadow_candidate_scope = (
        scope_policy_version == SHADOW_CANDIDATE_SCOPE_POLICY_VERSION
    )
    applicable_webpages = tuple(
        record
        for record in evidence
        if record.lane == FROZEN_WEBPAGE_LANE
        and _web_record_applies_to_axis(record, axis)
    )
    relevant_lanes = {
        lane for lane, axes in LANE_AXIS_RELEVANCE.items() if axis in axes and lane in by_lane
    }
    if (
        axis is AxisName.DROUGHT_CORROBORATION
        and not applicable_webpages
    ):
        # Legacy frozen webpage packages did not assess the drought axis.
        # Keep that lane out of drought aggregation unless a record explicitly
        # declares drought as its target.
        relevant_lanes.discard(FROZEN_WEBPAGE_LANE)
    official = tuple(
        record
        for record in evidence
        if record.lane in relevant_lanes and record.lane != FROZEN_WEBPAGE_LANE
    )
    webpages = tuple(
        record for record in applicable_webpages if record.lane in relevant_lanes
    )
    official_support = _ordered_unique(
        record.source_record_id
        for record in official
        if record.disposition is EvidenceDisposition.ACCEPTED
        and record.axis_assessments.get(axis) is AxisValue.YES
    )
    official_context = _ordered_unique(
        record.source_record_id
        for record in official
        if record.disposition is EvidenceDisposition.CONTEXTUAL
    )
    official_unresolved = tuple(
        record
        for record in official
        if _official_unresolved_is_candidate_relevant(
            record,
            axis,
            shadow_candidate_scope=shadow_candidate_scope,
        )
    )

    web_axis: CaseAxisRecord | None = None
    if FROZEN_WEBPAGE_LANE in relevant_lanes:
        web_axis = evaluate_bounded_web_axis(
            axis_name=axis,
            evidence_records=webpages,
            execution_record=by_lane[FROZEN_WEBPAGE_LANE],
            scope_policy_version=(
                SHADOW_WEB_AXIS_SCOPE_POLICY_VERSION
                if shadow_candidate_scope
                else ACTIVE_WEB_AXIS_SCOPE_POLICY_VERSION
            ),
        )
    webpage_support = web_axis.webpage_supporting_record_ids if web_axis else ()
    webpage_context = web_axis.webpage_contextual_record_ids if web_axis else ()
    support_exists = bool(official_support or webpage_support)

    relevant_execution = tuple(by_lane[lane] for lane in sorted(relevant_lanes))
    def effective_complete(record: LaneExecutionRecord) -> bool:
        if not record.required:
            return True
        if not record.successful_completion:
            return False
        if record.coverage is AxisCoverageState.COMPLETE:
            return True
        if not shadow_candidate_scope:
            return False
        if record.lane == NOAA_LANE:
            return not official_unresolved
        if record.lane == FROZEN_WEBPAGE_LANE:
            return bool(
                web_axis and web_axis.coverage_state is AxisCoverageState.COMPLETE
            )
        return False

    blocking_execution = tuple(
        record for record in relevant_execution if not effective_complete(record)
    )
    unresolved_reasons = list(
        reason
        for record in official_unresolved
        for reason in (record.reason_codes or (ReasonCode.CANDIDATE_RELEVANT_UNRESOLVED,))
    )
    if web_axis and web_axis.axis_value is AxisValue.UNRESOLVED:
        unresolved_reasons.extend(web_axis.unresolved_reason_codes)
    unresolved_reasons.extend(_execution_reason(record) for record in blocking_execution)

    all_complete = all(effective_complete(record) for record in relevant_execution) and (
        web_axis is None or web_axis.coverage_state is AxisCoverageState.COMPLETE
    )

    review_reasons: tuple[ReasonCode, ...] = ()
    negative_scope = ""
    if support_exists:
        value = AxisValue.YES
        coverage = AxisCoverageState.COMPLETE if all_complete else AxisCoverageState.PARTIAL
        decision = RuleId.AXIS_VALUE
        unresolved = ()
        if not all_complete:
            review_reasons = (ReasonCode.COVERAGE_WARNING,)
    elif unresolved_reasons:
        value = AxisValue.UNRESOLVED
        failure_states = {record.coverage for record in blocking_execution}
        if AxisCoverageState.FAILED in failure_states:
            coverage = AxisCoverageState.FAILED
        elif AxisCoverageState.UNAVAILABLE in failure_states:
            coverage = AxisCoverageState.UNAVAILABLE
        elif AxisCoverageState.NOT_RUN in failure_states:
            coverage = AxisCoverageState.NOT_RUN
        else:
            coverage = AxisCoverageState.PARTIAL
        decision = RuleId.AXIS_UNRESOLVED
        unresolved = _ordered_unique(unresolved_reasons)
    elif all_complete:
        value = AxisValue.NO
        coverage = AxisCoverageState.COMPLETE
        decision = RuleId.AXIS_NO
        unresolved = ()
        negative_scope = "bounded_evaluated_scope_only"
    else:
        value = AxisValue.UNRESOLVED
        coverage = AxisCoverageState.PARTIAL
        decision = RuleId.AXIS_UNRESOLVED
        unresolved = (ReasonCode.EXECUTION_NOT_PROVEN,)

    considered = official + webpages
    spatial = _ordered_unique(record.spatial_grade for record in considered)
    temporal = _ordered_unique(record.temporal_grade for record in considered)
    return CaseAxisRecord(
        candidate_id=candidate_id,
        axis_name=axis,
        axis_value=value,
        coverage_state=coverage,
        official_supporting_record_ids=official_support,
        official_contextual_record_ids=official_context,
        webpage_supporting_record_ids=webpage_support,
        webpage_contextual_record_ids=webpage_context,
        strongest_spatial_grade=_strongest(spatial, _SPATIAL_STRENGTH),
        all_spatial_grades=spatial,
        strongest_temporal_grade=_strongest(temporal, _TEMPORAL_STRENGTH),
        all_temporal_grades=temporal,
        decision_rule_id=decision,
        unresolved_reason_codes=unresolved,
        review_reason_codes=review_reasons,
        source_package_ids=_ordered_unique(record.package_id for record in relevant_execution),
        negative_scope=negative_scope,
        lineage=VersionLineage(aggregator_version=aggregator_version),
    )


def _source_mix(
    *,
    candidate_id: str,
    axes: Mapping[AxisName, CaseAxisRecord],
    execution: tuple[LaneExecutionRecord, ...],
    evidence: tuple[StructuredEvidenceRecord, ...],
) -> Mapping[str, Any]:
    contributions = {
        (candidate_id, record.lane, axis.value)
        for record in evidence
        for axis, value in record.axis_assessments.items()
        if record.disposition is EvidenceDisposition.ACCEPTED
        and value is AxisValue.YES
        and (
            record.lane != FROZEN_WEBPAGE_LANE
            or _web_record_applies_to_axis(record, axis)
        )
    }
    axis_source_state: dict[str, str] = {}
    for axis, record in axes.items():
        official = bool(record.official_supporting_record_ids)
        webpage = bool(record.webpage_supporting_record_ids)
        if record.axis_value is AxisValue.UNRESOLVED:
            state = "unresolved"
        elif official and webpage:
            state = "both"
        elif official:
            state = "official_only"
        elif webpage:
            state = "webpage_only"
        else:
            state = "neither"
        axis_source_state[axis.value] = state
    return {
        "metric_version": "ce_agent_repaired_v2_source_mix_v1",
        "main_metric": "candidate_x_source_family_x_axis_accepted_contribution",
        "candidate_source_axis_contributions": [
            {"candidate_id": item[0], "source_family": item[1], "axis": item[2]}
            for item in sorted(contributions)
        ],
        "successful_or_eligible_lane_coverage": [
            {
                "candidate_id": record.candidate_id,
                "lane": record.lane,
                "enabled": record.enabled,
                "required": record.required,
                "execution_status": record.execution_status.value,
                "coverage": record.coverage.value,
            }
            for record in sorted(execution, key=lambda item: item.lane)
        ],
        "axis_source_state": axis_source_state,
        "unique_source_record_ids": sorted({record.source_record_id for record in evidence}),
        "unique_event_ids": sorted({record.source_event_id for record in evidence if record.source_event_id}),
        "lane_state_counts": {
            state.value: sum(record.execution_status is state for record in execution)
            for state in LaneExecutionStatus
        },
        "historical_compatibility_row_occurrence": {
            "explicitly_non_authoritative": True,
            "raw_evidence_row_count": len(evidence),
        },
    }


def aggregate_repaired_v2_case(
    *,
    physical_status: PhysicalStatusRecord,
    execution_records: Iterable[LaneExecutionRecord],
    evidence_records: Iterable[StructuredEvidenceRecord],
    legacy_compatibility: LegacyCompatibilityView,
    required_lanes: tuple[str, ...] = DEFAULT_REQUIRED_LANES,
    expected_package_hashes: Mapping[str, str] | None = None,
    scope_policy_version: str = ACTIVE_CANDIDATE_SCOPE_POLICY_VERSION,
) -> CanonicalAggregationResult:
    """Calculate all five repaired-v2 axes through the sole production API."""

    if scope_policy_version not in CANDIDATE_SCOPE_POLICY_VERSIONS:
        raise CanonicalAggregationError(
            f"unsupported_candidate_scope_policy:{scope_policy_version}"
        )
    aggregator_version = (
        SHADOW_CANONICAL_AGGREGATOR_VERSION
        if scope_policy_version == SHADOW_CANDIDATE_SCOPE_POLICY_VERSION
        else CANONICAL_AGGREGATOR_VERSION
    )
    execution = tuple(execution_records)
    evidence = tuple(evidence_records)
    by_lane = _validate_inputs(
        physical_status=physical_status,
        execution_records=execution,
        evidence_records=evidence,
        required_lanes=required_lanes,
        expected_package_hashes=expected_package_hashes,
    )
    axes = {
        axis: _aggregate_axis(
            candidate_id=physical_status.candidate_id,
            axis=axis,
            by_lane=by_lane,
            evidence=evidence,
            scope_policy_version=scope_policy_version,
            aggregator_version=aggregator_version,
        )
        for axis in AxisName
    }
    views = derive_repaired_v2_views(
        candidate_id=physical_status.candidate_id,
        axes=axes,
        legacy_compatibility=legacy_compatibility,
        aggregator_version=aggregator_version,
    )
    output = CaseOutputRecord(
        candidate_id=physical_status.candidate_id,
        physical_status=physical_status,
        axes=axes,
        derived_views=views,
        lineage=VersionLineage(aggregator_version=aggregator_version),
    )
    return CanonicalAggregationResult(
        case_output=output,
        source_mix=_source_mix(
            candidate_id=physical_status.candidate_id,
            axes=axes,
            execution=execution,
            evidence=evidence,
        ),
    )


def combine_repaired_v2_source_mix(
    results: Iterable[CanonicalAggregationResult],
) -> Mapping[str, Any]:
    """Combine per-case source mix without multiplying shared source/event IDs.

    This is a summary over canonical case results, not a second truth
    aggregator. Candidate-specific contributions remain distinct while global
    source-record and event identities are set-deduplicated.
    """

    result_rows = tuple(results)
    contributions = {
        (
            str(row["candidate_id"]),
            str(row["source_family"]),
            str(row["axis"]),
        )
        for result in result_rows
        for row in result.source_mix["candidate_source_axis_contributions"]
    }
    coverage = {
        (
            str(row["candidate_id"]),
            str(row["lane"]),
            bool(row["enabled"]),
            bool(row["required"]),
            str(row["execution_status"]),
            str(row["coverage"]),
        )
        for result in result_rows
        for row in result.source_mix["successful_or_eligible_lane_coverage"]
    }
    source_ids = {
        str(value)
        for result in result_rows
        for value in result.source_mix["unique_source_record_ids"]
    }
    event_ids = {
        str(value)
        for result in result_rows
        for value in result.source_mix["unique_event_ids"]
    }
    return {
        "metric_version": "ce_agent_repaired_v2_source_mix_batch_v1",
        "main_metric": "candidate_x_source_family_x_axis_accepted_contribution",
        "candidate_source_axis_contributions": [
            {"candidate_id": item[0], "source_family": item[1], "axis": item[2]}
            for item in sorted(contributions)
        ],
        "successful_or_eligible_lane_coverage": [
            {
                "candidate_id": item[0],
                "lane": item[1],
                "enabled": item[2],
                "required": item[3],
                "execution_status": item[4],
                "coverage": item[5],
            }
            for item in sorted(coverage)
        ],
        "unique_source_record_ids": sorted(source_ids),
        "unique_event_ids": sorted(event_ids),
        "candidate_count": len(
            {item[0] for item in contributions} | {item[0] for item in coverage}
        ),
        "historical_compatibility_row_occurrence": {
            "explicitly_non_authoritative": True,
            "raw_evidence_row_count": sum(
                int(result.source_mix["historical_compatibility_row_occurrence"]["raw_evidence_row_count"])
                for result in result_rows
            ),
        },
    }


__all__ = [
    "CANONICAL_AGGREGATOR_VERSION",
    "CanonicalAggregationError",
    "CanonicalAggregationResult",
    "DEFAULT_REQUIRED_LANES",
    "LANE_AXIS_RELEVANCE",
    "aggregate_repaired_v2_case",
    "combine_repaired_v2_source_mix",
    "derive_repaired_v2_views",
]
