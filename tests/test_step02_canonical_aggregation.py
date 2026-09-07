from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import pytest

from src.climate_pipeline import case_aggregation as legacy_aggregation
from src.climate_pipeline.pipeline.canonical_aggregation import (
    CANONICAL_AGGREGATOR_VERSION,
    SHADOW_CANDIDATE_SCOPE_POLICY_VERSION,
    CanonicalAggregationError,
    aggregate_repaired_v2_case,
    combine_repaired_v2_source_mix,
    derive_repaired_v2_views,
)
from src.climate_pipeline.pipeline.frozen_webpage_adapter import FROZEN_WEBPAGE_LANE
from src.climate_pipeline.pipeline.official_adapters import NOAA_LANE, USDM_LANE
from src.climate_pipeline.pipeline.schemas import (
    AxisCoverageState,
    AxisName,
    AxisValue,
    CaseAxisRecord,
    ComponentCorroborationView,
    EvidenceDisposition,
    FrozenWebpageExecutionDetails,
    ImpactAttributionView,
    JudgmentCoverageState,
    LaneExecutionRecord,
    LaneExecutionStatus,
    LaneReadiness,
    LegacyCompatibilityView,
    LinkageView,
    NavigationSummary,
    PhysicalCandidateStatus,
    PhysicalStatusRecord,
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


CANDIDATE = "fixture-candidate"
HASH = "a" * 64


def _physical() -> PhysicalStatusRecord:
    return PhysicalStatusRecord(
        candidate_id=CANDIDATE,
        status=PhysicalCandidateStatus.AUTHORITATIVE_PHYSICAL_CANDIDATE,
        authoritative_inventory_id="authoritative-408",
        physical_input_hashes=(HASH,),
    )


def _legacy(tier: str = "historical-tier") -> LegacyCompatibilityView:
    return LegacyCompatibilityView(legacy_tier=tier, source_artifact_id="frozen-history")


def _official_execution(
    lane: str,
    *,
    status: LaneExecutionStatus = LaneExecutionStatus.COMPLETED_ZERO_MATCH,
    coverage: AxisCoverageState = AxisCoverageState.COMPLETE,
    required: bool = True,
    package_hash: str = HASH,
) -> LaneExecutionRecord:
    matches = 0 if status is LaneExecutionStatus.COMPLETED_ZERO_MATCH else 1
    ready = LaneReadiness.READY if status.is_successful_completion else LaneReadiness.UNAVAILABLE
    return LaneExecutionRecord(
        candidate_id=CANDIDATE,
        lane=lane,
        enabled=True,
        required=required,
        source_mode="local_read_only",
        package_id=f"{lane}-package",
        package_sha256=package_hash,
        readiness=ready,
        execution_status=status,
        coverage=coverage,
        records_examined=matches,
        matches_found=matches,
        accepted_count=0,
        contextual_count=1 if status is LaneExecutionStatus.COMPLETED_CONTEXT_ONLY else 0,
        unresolved_count=0,
        completed_at="done" if status.is_successful_completion else "",
    )


def _web_execution(
    *,
    coverage: WebpageCoverage = WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE,
    status: LaneExecutionStatus = LaneExecutionStatus.COMPLETED_WITH_MATCHES,
    accepted: int = 0,
) -> LaneExecutionRecord:
    zero = coverage is WebpageCoverage.ZERO_PAGES_RETRIEVED
    logical = 0 if zero else 1
    complete = coverage is WebpageCoverage.BOUNDED_COMPLETE_FOR_FROZEN_SCOPE
    failed = coverage is WebpageCoverage.JUDGMENT_INCOMPLETE_OR_FAILED
    details = FrozenWebpageExecutionDetails(
        retrieval_performed=True,
        logical_page_count=logical,
        successfully_fetched_or_readable_count=logical,
        source_content_insufficient_count=(1 if coverage is WebpageCoverage.SOURCE_CONTENT_INSUFFICIENT else 0),
        completed_judgment_count=(0 if failed else logical),
        candidate_relevant_unresolved_count=0,
        guarded_page_count=(0 if failed else logical),
        zero_pages_retrieved=zero,
        retrieval_coverage_state=(
            RetrievalCoverageState.ZERO_PAGES_RETRIEVED
            if zero else RetrievalCoverageState.COMPLETE_FOR_FROZEN_SCOPE
        ),
        judgment_coverage_state=(
            JudgmentCoverageState.INCOMPLETE_OR_FAILED
            if failed else JudgmentCoverageState.COMPLETE
        ),
        webpage_coverage=coverage,
        frozen_manifest_hash=HASH,
        page_result_bundle_hash=HASH,
        body_bundle_hash=HASH,
        judgment_bundle_hash=HASH,
        guard_bundle_hash=HASH,
        derivation_rule_id=(
            RuleId.WEB_AXIS_ZERO_PAGES if zero
            else RuleId.WEB_AXIS_UNRESOLVED_DEPENDENCY if failed
            else RuleId.FROZEN_WEB_PRESERVE
        ),
    )
    if zero:
        status = LaneExecutionStatus.UNAVAILABLE
    axis_coverage = (
        AxisCoverageState.COMPLETE if complete
        else AxisCoverageState.FAILED if failed
        else AxisCoverageState.PARTIAL
    )
    return LaneExecutionRecord(
        candidate_id=CANDIDATE,
        lane=FROZEN_WEBPAGE_LANE,
        enabled=True,
        required=True,
        source_mode="frozen_local_read_only",
        package_id="frozen-web-package",
        package_sha256=HASH,
        readiness=LaneReadiness.READY if status.is_successful_completion else LaneReadiness.UNAVAILABLE,
        execution_status=status,
        coverage=axis_coverage,
        records_examined=logical,
        matches_found=logical,
        accepted_count=accepted,
        contextual_count=0,
        unresolved_count=0,
        completed_at="done" if status.is_successful_completion else "",
        webpage=details,
    )


def _evidence(
    *,
    lane: str,
    record_id: str,
    assessments: Mapping[AxisName, AxisValue],
    disposition: EvidenceDisposition = EvidenceDisposition.ACCEPTED,
    event_id: str = "",
) -> StructuredEvidenceRecord:
    return StructuredEvidenceRecord(
        candidate_id=CANDIDATE,
        lane=lane,
        package_id=("frozen-web-package" if lane == FROZEN_WEBPAGE_LANE else f"{lane}-package"),
        source_role=(
            SourceRole.FROZEN_WEBPAGE_RECORD if lane == FROZEN_WEBPAGE_LANE
            else SourceRole.USDM_COUNTY_WEEK_DROUGHT_CONTEXT if lane == USDM_LANE
            else SourceRole.DIRECT_PRECIPITATION_EVENT if lane == NOAA_LANE
            else SourceRole.ADMINISTRATIVE_RESPONSE
        ),
        disposition=disposition,
        source_record_id=record_id,
        source_event_id=event_id,
        source_file="fixture.json",
        source_row="1",
        source_url="",
        source_sha256=HASH,
        source_version="fixture-v1",
        source_date="2023-01-01",
        raw_value_provenance={"fixture": True},
        spatial_grade=SpatialAlignmentGrade.COUNTY,
        temporal_grade=TemporalAlignmentGrade.CANDIDATE_WINDOW_EXACT_OVERLAP,
        axis_assessments=assessments,
        rule_ids=(RuleId.AXIS_VALUE,),
        reason_codes=(ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD,),
        record_hash=(record_id.encode().hex() + HASH)[:64].ljust(64, "a"),
    )


def _web_no_record() -> StructuredEvidenceRecord:
    return _evidence(
        lane=FROZEN_WEBPAGE_LANE,
        record_id="web-no",
        disposition=EvidenceDisposition.REJECTED_NON_SUPPORT,
        assessments={axis: AxisValue.NO for axis in AxisName},
    )


def _base_executions(web: LaneExecutionRecord | None = None) -> tuple[LaneExecutionRecord, ...]:
    return (
        _official_execution(USDM_LANE),
        _official_execution(NOAA_LANE),
        web or _web_execution(),
    )


def _aggregate(
    evidence: tuple[StructuredEvidenceRecord, ...] = (),
    executions: tuple[LaneExecutionRecord, ...] | None = None,
    legacy: LegacyCompatibilityView | None = None,
):
    if executions is None:
        executions = _base_executions()
    if not any(record.lane == FROZEN_WEBPAGE_LANE for record in evidence):
        evidence = evidence + (_web_no_record(),)
    return aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=evidence,
        legacy_compatibility=legacy or _legacy(),
    )


def test_exactly_one_canonical_production_api_is_exported_to_legacy_module() -> None:
    assert legacy_aggregation.aggregate_repaired_v2_case is aggregate_repaired_v2_case


def test_physical_candidate_does_not_imply_any_evidence_axis() -> None:
    result = _aggregate()
    assert all(record.axis_value is AxisValue.NO for record in result.case_output.axes.values())


def test_usdm_drought_and_noaa_wet_do_not_imply_linkage_or_impact() -> None:
    drought = _evidence(
        lane=USDM_LANE,
        record_id="usdm",
        assessments={AxisName.DROUGHT_CORROBORATION: AxisValue.YES},
    )
    wet = _evidence(
        lane=NOAA_LANE,
        record_id="noaa",
        assessments={
            AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES,
            AxisName.REALIZED_IMPACT: AxisValue.NO,
            AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: AxisValue.NO,
            AxisName.DROUGHT_TO_WET_LINKAGE: AxisValue.NO,
        },
        event_id="event-1",
    )
    executions = (
        replace(_official_execution(USDM_LANE), execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
                records_examined=1, matches_found=1, accepted_count=1),
        replace(_official_execution(NOAA_LANE), execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
                records_examined=1, matches_found=1, accepted_count=1),
        _web_execution(),
    )
    axes = _aggregate((drought, wet), executions).case_output.axes
    assert axes[AxisName.DROUGHT_CORROBORATION].axis_value is AxisValue.YES
    assert axes[AxisName.WET_HAZARD_CORROBORATION].axis_value is AxisValue.YES
    assert axes[AxisName.REALIZED_IMPACT].axis_value is AxisValue.NO
    assert axes[AxisName.HAZARD_TO_IMPACT_ATTRIBUTION].axis_value is AxisValue.NO
    assert axes[AxisName.DROUGHT_TO_WET_LINKAGE].axis_value is AxisValue.NO


def test_support_yes_survives_required_partial_coverage_with_warning() -> None:
    web = _web_execution(coverage=WebpageCoverage.SOURCE_CONTENT_INSUFFICIENT, accepted=1)
    evidence = _evidence(
        lane=FROZEN_WEBPAGE_LANE,
        record_id="web-yes",
        assessments={AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES},
    )
    axes = _aggregate((evidence,), _base_executions(web)).case_output.axes
    wet = axes[AxisName.WET_HAZARD_CORROBORATION]
    assert wet.axis_value is AxisValue.YES
    assert wet.coverage_state is AxisCoverageState.PARTIAL
    assert wet.review_reason_codes == (ReasonCode.COVERAGE_WARNING,)


def test_hazard_targeting_keeps_drought_and_wet_web_records_axis_specific() -> None:
    drought_page = _evidence(
        lane=FROZEN_WEBPAGE_LANE,
        record_id="web-drought-yes",
        assessments={
            AxisName.DROUGHT_CORROBORATION: AxisValue.YES,
            AxisName.WET_HAZARD_CORROBORATION: AxisValue.UNRESOLVED,
        },
    )
    drought_page = replace(
        drought_page,
        raw_value_provenance={
            "webpage_row": {"target_hazard_axis": "drought"}
        },
    )
    axes = _aggregate(
        (drought_page,),
        _base_executions(_web_execution(accepted=1)),
    ).case_output.axes

    drought = axes[AxisName.DROUGHT_CORROBORATION]
    wet = axes[AxisName.WET_HAZARD_CORROBORATION]
    assert drought.axis_value is AxisValue.YES
    assert drought.webpage_supporting_record_ids == ("web-drought-yes",)
    assert wet.axis_value is AxisValue.NO
    assert wet.webpage_supporting_record_ids == ()


@pytest.mark.parametrize(
    ("status", "coverage"),
    [
        (LaneExecutionStatus.UNAVAILABLE, AxisCoverageState.UNAVAILABLE),
        (LaneExecutionStatus.RETRYABLE_FAILURE, AxisCoverageState.FAILED),
        (LaneExecutionStatus.TERMINAL_FAILURE, AxisCoverageState.FAILED),
    ],
)
def test_no_support_plus_required_lane_failure_is_unresolved(
    status: LaneExecutionStatus, coverage: AxisCoverageState
) -> None:
    noaa = _official_execution(NOAA_LANE, status=status, coverage=coverage)
    result = _aggregate(executions=(_official_execution(USDM_LANE), noaa, _web_execution()))
    for axis in (
        AxisName.WET_HAZARD_CORROBORATION,
        AxisName.REALIZED_IMPACT,
        AxisName.HAZARD_TO_IMPACT_ATTRIBUTION,
    ):
        assert result.case_output.axes[axis].axis_value is AxisValue.UNRESOLVED


def test_shadow_scope_excludes_spatially_unbound_noaa_background_but_not_county_records() -> None:
    noaa = replace(
        _official_execution(NOAA_LANE),
        execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
        coverage=AxisCoverageState.PARTIAL,
        records_examined=1,
        matches_found=1,
        unresolved_count=1,
    )
    unresolved = _evidence(
        lane=NOAA_LANE,
        record_id="unbound-statewide-background",
        disposition=EvidenceDisposition.UNRESOLVED,
        assessments={
            AxisName.WET_HAZARD_CORROBORATION: AxisValue.UNRESOLVED,
            AxisName.REALIZED_IMPACT: AxisValue.UNRESOLVED,
            AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: AxisValue.UNRESOLVED,
        },
    )
    unresolved = replace(unresolved, spatial_grade=SpatialAlignmentGrade.UNRESOLVED)
    executions = (_official_execution(USDM_LANE), noaa, _web_execution())
    production = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=(unresolved, _web_no_record()),
        legacy_compatibility=_legacy(),
    )
    shadow = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=(unresolved, _web_no_record()),
        legacy_compatibility=_legacy(),
        scope_policy_version=SHADOW_CANDIDATE_SCOPE_POLICY_VERSION,
    )
    assert production.case_output.axes[AxisName.WET_HAZARD_CORROBORATION].axis_value is AxisValue.UNRESOLVED
    assert shadow.case_output.axes[AxisName.WET_HAZARD_CORROBORATION].axis_value is AxisValue.NO

    county_bound = replace(unresolved, spatial_grade=SpatialAlignmentGrade.COUNTY)
    shadow_with_relevant_unresolved = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=(county_bound, _web_no_record()),
        legacy_compatibility=_legacy(),
        scope_policy_version=SHADOW_CANDIDATE_SCOPE_POLICY_VERSION,
    )
    assert (
        shadow_with_relevant_unresolved.case_output.axes[
            AxisName.WET_HAZARD_CORROBORATION
        ].axis_value
        is AxisValue.UNRESOLVED
    )


def test_shadow_scope_keeps_web_uncertainty_axis_specific() -> None:
    complete_web = _web_execution()
    assert complete_web.webpage is not None
    partial_details = replace(
        complete_web.webpage,
        candidate_relevant_unresolved_count=1,
        judgment_coverage_state=JudgmentCoverageState.SOURCE_CONTENT_INSUFFICIENT,
        webpage_coverage=WebpageCoverage.SOURCE_CONTENT_INSUFFICIENT,
        derivation_rule_id=RuleId.WEB_AXIS_UNRESOLVED_EVIDENCE,
    )
    partial_web = replace(
        complete_web,
        coverage=AxisCoverageState.PARTIAL,
        unresolved_count=1,
        webpage=partial_details,
    )
    axis_specific_no = _evidence(
        lane=FROZEN_WEBPAGE_LANE,
        record_id="page-unresolved-other-axis",
        disposition=EvidenceDisposition.UNRESOLVED,
        assessments={axis: AxisValue.NO for axis in AxisName},
    )
    executions = (_official_execution(USDM_LANE), _official_execution(NOAA_LANE), partial_web)
    production = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=(axis_specific_no,),
        legacy_compatibility=_legacy(),
    )
    shadow = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=(axis_specific_no,),
        legacy_compatibility=_legacy(),
        scope_policy_version=SHADOW_CANDIDATE_SCOPE_POLICY_VERSION,
    )
    assert production.case_output.axes[AxisName.WET_HAZARD_CORROBORATION].axis_value is AxisValue.UNRESOLVED
    assert shadow.case_output.axes[AxisName.WET_HAZARD_CORROBORATION].axis_value is AxisValue.NO

    wet_unresolved = replace(
        axis_specific_no,
        axis_assessments={
            **axis_specific_no.axis_assessments,
            AxisName.WET_HAZARD_CORROBORATION: AxisValue.UNRESOLVED,
        },
    )
    shadow_wet_unresolved = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=executions,
        evidence_records=(wet_unresolved,),
        legacy_compatibility=_legacy(),
        scope_policy_version=SHADOW_CANDIDATE_SCOPE_POLICY_VERSION,
    )
    assert (
        shadow_wet_unresolved.case_output.axes[
            AxisName.WET_HAZARD_CORROBORATION
        ].axis_value
        is AxisValue.UNRESOLVED
    )
    assert shadow_wet_unresolved.case_output.axes[AxisName.REALIZED_IMPACT].axis_value is AxisValue.NO


def test_zero_page_web_coverage_remains_unresolved_for_all_web_axes() -> None:
    result = aggregate_repaired_v2_case(
        physical_status=_physical(),
        execution_records=(_official_execution(USDM_LANE), _official_execution(NOAA_LANE), _web_execution(coverage=WebpageCoverage.ZERO_PAGES_RETRIEVED)),
        evidence_records=(),
        legacy_compatibility=_legacy(),
    )
    assert result.case_output.axes[AxisName.DROUGHT_CORROBORATION].axis_value is AxisValue.NO
    assert all(
        result.case_output.axes[axis].axis_value is AxisValue.UNRESOLVED
        for axis in AxisName
        if axis is not AxisName.DROUGHT_CORROBORATION
    )


def test_multiple_context_records_never_accumulate_into_support() -> None:
    contexts = tuple(
        _evidence(
            lane=NOAA_LANE,
            record_id=f"context-{index}",
            disposition=EvidenceDisposition.CONTEXTUAL,
            assessments={axis: AxisValue.NO for axis in AxisName},
        )
        for index in range(3)
    )
    noaa = replace(
        _official_execution(NOAA_LANE),
        execution_status=LaneExecutionStatus.COMPLETED_CONTEXT_ONLY,
        records_examined=3,
        matches_found=3,
        contextual_count=3,
    )
    result = _aggregate(contexts, (_official_execution(USDM_LANE), noaa, _web_execution()))
    assert result.case_output.axes[AxisName.WET_HAZARD_CORROBORATION].axis_value is AxisValue.NO
    assert len(result.case_output.axes[AxisName.WET_HAZARD_CORROBORATION].official_contextual_record_ids) == 3


def test_openfema_administrative_context_has_zero_axis_effect() -> None:
    baseline = _aggregate()
    openfema_execution = LaneExecutionRecord(
        candidate_id=CANDIDATE,
        lane="openfema",
        enabled=False,
        required=False,
        source_mode="disabled_administrative_only",
        package_id="none",
        package_sha256="",
        readiness=LaneReadiness.NOT_APPLICABLE,
        execution_status=LaneExecutionStatus.DISABLED,
        coverage=AxisCoverageState.NOT_RUN,
        records_examined=0,
        matches_found=0,
        accepted_count=0,
        contextual_count=0,
        unresolved_count=0,
    )
    admin = _evidence(
        lane="openfema",
        record_id="admin",
        disposition=EvidenceDisposition.CONTEXTUAL,
        assessments={axis: AxisValue.NO for axis in AxisName},
    )
    admin = replace(admin, package_id="none", record_hash="c" * 64)
    with_admin = _aggregate(
        (admin,),
        _base_executions() + (openfema_execution,),
    )
    assert {
        axis: record.axis_value for axis, record in baseline.case_output.axes.items()
    } == {
        axis: record.axis_value for axis, record in with_admin.case_output.axes.items()
    }


def test_official_and_webpage_support_ids_are_never_merged() -> None:
    official = _evidence(
        lane=NOAA_LANE,
        record_id="official",
        assessments={AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES},
    )
    web = _evidence(
        lane=FROZEN_WEBPAGE_LANE,
        record_id="web",
        assessments={AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES},
    )
    executions = (
        _official_execution(USDM_LANE),
        replace(_official_execution(NOAA_LANE), execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
                records_examined=1, matches_found=1, accepted_count=1),
        _web_execution(accepted=1),
    )
    axis = aggregate_repaired_v2_case(
        physical_status=_physical(), execution_records=executions,
        evidence_records=(official, web), legacy_compatibility=_legacy(),
    ).case_output.axes[AxisName.WET_HAZARD_CORROBORATION]
    assert axis.official_supporting_record_ids == ("official",)
    assert axis.webpage_supporting_record_ids == ("web",)


def _axis_record(axis: AxisName, value: AxisValue) -> CaseAxisRecord:
    return CaseAxisRecord(
        candidate_id=CANDIDATE,
        axis_name=axis,
        axis_value=value,
        coverage_state=(AxisCoverageState.COMPLETE if value is not AxisValue.UNRESOLVED else AxisCoverageState.PARTIAL),
        official_supporting_record_ids=((f"support-{axis.value}",) if value is AxisValue.YES else ()),
        official_contextual_record_ids=(),
        webpage_supporting_record_ids=(),
        webpage_contextual_record_ids=(),
        strongest_spatial_grade=None,
        all_spatial_grades=(),
        strongest_temporal_grade=None,
        all_temporal_grades=(),
        decision_rule_id=(RuleId.AXIS_VALUE if value is AxisValue.YES else RuleId.AXIS_NO if value is AxisValue.NO else RuleId.AXIS_UNRESOLVED),
        unresolved_reason_codes=((ReasonCode.CANDIDATE_RELEVANT_UNRESOLVED,) if value is AxisValue.UNRESOLVED else ()),
        review_reason_codes=(),
        source_package_ids=(),
        negative_scope=("bounded_evaluated_scope_only" if value is AxisValue.NO else ""),
    )


@pytest.mark.parametrize(
    ("drought", "wet", "expected"),
    [
        (AxisValue.YES, AxisValue.YES, ComponentCorroborationView.BOTH_COMPONENTS_CORROBORATED),
        (AxisValue.YES, AxisValue.NO, ComponentCorroborationView.DROUGHT_ONLY_CORROBORATED),
        (AxisValue.NO, AxisValue.YES, ComponentCorroborationView.WET_ONLY_CORROBORATED),
        (AxisValue.NO, AxisValue.NO, ComponentCorroborationView.NO_COMPONENT_CORROBORATION_IN_EVALUATED_SCOPE),
        (AxisValue.UNRESOLVED, AxisValue.NO, ComponentCorroborationView.COMPONENT_STATUS_UNRESOLVED),
    ],
)
def test_component_view_exhaustive_mapping(
    drought: AxisValue, wet: AxisValue, expected: ComponentCorroborationView
) -> None:
    values = {
        AxisName.DROUGHT_CORROBORATION: drought,
        AxisName.WET_HAZARD_CORROBORATION: wet,
        AxisName.REALIZED_IMPACT: AxisValue.NO,
        AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: AxisValue.NO,
        AxisName.DROUGHT_TO_WET_LINKAGE: AxisValue.NO,
    }
    views = derive_repaired_v2_views(
        candidate_id=CANDIDATE,
        axes={axis: _axis_record(axis, value) for axis, value in values.items()},
        legacy_compatibility=_legacy(),
    )
    assert views.component_corroboration is expected


@pytest.mark.parametrize(
    ("impact", "attribution", "expected"),
    [
        (AxisValue.NO, AxisValue.NO, ImpactAttributionView.NO_DOCUMENTED_IMPACT_IN_EVALUATED_SCOPE),
        (AxisValue.YES, AxisValue.NO, ImpactAttributionView.DOCUMENTED_IMPACT_WITHOUT_ATTRIBUTION),
        (AxisValue.YES, AxisValue.UNRESOLVED, ImpactAttributionView.DOCUMENTED_IMPACT_ATTRIBUTION_UNRESOLVED),
        (AxisValue.YES, AxisValue.YES, ImpactAttributionView.EXPLICITLY_ATTRIBUTED_IMPACT),
        (AxisValue.UNRESOLVED, AxisValue.NO, ImpactAttributionView.IMPACT_STATUS_UNRESOLVED),
    ],
)
def test_impact_attribution_view_exhaustive_mapping(
    impact: AxisValue, attribution: AxisValue, expected: ImpactAttributionView
) -> None:
    values = {
        AxisName.DROUGHT_CORROBORATION: AxisValue.NO,
        AxisName.WET_HAZARD_CORROBORATION: AxisValue.NO,
        AxisName.REALIZED_IMPACT: impact,
        AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: attribution,
        AxisName.DROUGHT_TO_WET_LINKAGE: AxisValue.NO,
    }
    views = derive_repaired_v2_views(
        candidate_id=CANDIDATE,
        axes={axis: _axis_record(axis, value) for axis, value in values.items()},
        legacy_compatibility=_legacy(),
    )
    assert views.impact_attribution is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (AxisValue.YES, LinkageView.EXPLICIT_LINKAGE_SUPPORTED),
        (AxisValue.NO, LinkageView.NO_EXPLICIT_LINKAGE_IN_EVALUATED_SCOPE),
        (AxisValue.UNRESOLVED, LinkageView.LINKAGE_UNRESOLVED),
    ],
)
def test_linkage_view_exhaustive_mapping(value: AxisValue, expected: LinkageView) -> None:
    values = {axis: AxisValue.NO for axis in AxisName}
    values[AxisName.DROUGHT_TO_WET_LINKAGE] = value
    views = derive_repaired_v2_views(
        candidate_id=CANDIDATE,
        axes={axis: _axis_record(axis, axis_value) for axis, axis_value in values.items()},
        legacy_compatibility=_legacy(),
    )
    assert views.linkage is expected


def test_navigation_summary_does_not_scalar_order_attribution_and_linkage() -> None:
    values = {
        AxisName.DROUGHT_CORROBORATION: AxisValue.YES,
        AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES,
        AxisName.REALIZED_IMPACT: AxisValue.YES,
        AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: AxisValue.YES,
        AxisName.DROUGHT_TO_WET_LINKAGE: AxisValue.NO,
    }
    first = derive_repaired_v2_views(
        candidate_id=CANDIDATE,
        axes={axis: _axis_record(axis, value) for axis, value in values.items()},
        legacy_compatibility=_legacy(),
    )
    values[AxisName.DROUGHT_TO_WET_LINKAGE] = AxisValue.YES
    second = derive_repaired_v2_views(
        candidate_id=CANDIDATE,
        axes={axis: _axis_record(axis, value) for axis, value in values.items()},
        legacy_compatibility=_legacy(),
    )
    assert first.navigation_summary is second.navigation_summary is NavigationSummary.BOTH_COMPONENTS_WITH_DOCUMENTED_IMPACT


def test_legacy_tier_is_copied_with_zero_truth_influence() -> None:
    first = _aggregate(legacy=_legacy("tier-a"))
    second = _aggregate(legacy=_legacy("tier-b"))
    assert {axis: record.axis_value for axis, record in first.case_output.axes.items()} == {
        axis: record.axis_value for axis, record in second.case_output.axes.items()
    }
    assert first.case_output.derived_views.legacy_compatibility.legacy_tier == "tier-a"
    assert second.case_output.derived_views.legacy_compatibility.legacy_tier == "tier-b"


def test_source_mix_uses_candidate_family_axis_dedup_and_event_id_dedup() -> None:
    first = _evidence(
        lane=NOAA_LANE,
        record_id="row-1",
        event_id="event-one",
        assessments={AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES},
    )
    second = replace(first, source_record_id="row-2", record_hash="b" * 64)
    noaa = replace(
        _official_execution(NOAA_LANE), execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
        records_examined=3, matches_found=3, accepted_count=3,
    )
    result = _aggregate(
        (first, first, second),
        (_official_execution(USDM_LANE), noaa, _web_execution()),
    )
    source_mix = result.source_mix
    assert len(source_mix["candidate_source_axis_contributions"]) == 1
    assert source_mix["unique_source_record_ids"] == ["row-1", "row-2", "web-no"]
    assert source_mix["unique_event_ids"] == ["event-one"]
    assert source_mix["historical_compatibility_row_occurrence"]["raw_evidence_row_count"] == 4


def test_source_mix_batch_preserves_candidate_contributions_but_deduplicates_reused_event() -> None:
    evidence = _evidence(
        lane=NOAA_LANE,
        record_id="shared-source-record",
        event_id="shared-event",
        assessments={AxisName.WET_HAZARD_CORROBORATION: AxisValue.YES},
    )
    noaa = replace(
        _official_execution(NOAA_LANE),
        execution_status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
        records_examined=1,
        matches_found=1,
        accepted_count=1,
    )
    first = _aggregate(
        (evidence,),
        (_official_execution(USDM_LANE), noaa, _web_execution()),
    )
    second_mix = dict(first.source_mix)
    second_mix["candidate_source_axis_contributions"] = [
        {**row, "candidate_id": "second-candidate"}
        for row in first.source_mix["candidate_source_axis_contributions"]
    ]
    second_mix["successful_or_eligible_lane_coverage"] = [
        {**row, "candidate_id": "second-candidate"}
        for row in first.source_mix["successful_or_eligible_lane_coverage"]
    ]
    combined = combine_repaired_v2_source_mix(
        (first, replace(first, source_mix=second_mix))
    )
    assert combined["candidate_count"] == 2
    assert len(combined["candidate_source_axis_contributions"]) == 2
    assert combined["unique_event_ids"] == ["shared-event"]
    assert combined["unique_source_record_ids"].count("shared-source-record") == 1


def test_duplicate_or_missing_execution_and_stale_package_fail_closed() -> None:
    execution = _base_executions()
    with pytest.raises(CanonicalAggregationError, match="duplicate_lane_execution"):
        aggregate_repaired_v2_case(
            physical_status=_physical(), execution_records=execution + (execution[0],),
            evidence_records=(_web_no_record(),), legacy_compatibility=_legacy(),
        )
    with pytest.raises(CanonicalAggregationError, match="missing_required_execution_records"):
        aggregate_repaired_v2_case(
            physical_status=_physical(), execution_records=execution[:2], evidence_records=(),
            legacy_compatibility=_legacy(),
        )
    with pytest.raises(CanonicalAggregationError, match="stale_package_hash"):
        aggregate_repaired_v2_case(
            physical_status=_physical(), execution_records=execution,
            evidence_records=(_web_no_record(),), legacy_compatibility=_legacy(),
            expected_package_hashes={USDM_LANE: "b" * 64},
        )


def test_case_output_uses_approved_aggregator_version_envelope() -> None:
    result = _aggregate()
    assert result.case_output.lineage.aggregator_version == CANONICAL_AGGREGATOR_VERSION
