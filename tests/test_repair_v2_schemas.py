from __future__ import annotations

from dataclasses import replace

import pytest

from src.climate_pipeline.pipeline.schemas import (
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
    LaneReadiness,
    LegacyCompatibilityView,
    LinkageView,
    NavigationSummary,
    PhysicalCandidateStatus,
    PhysicalStatusRecord,
    ReasonCode,
    RuleId,
    SourceRole,
    SpatialAlignmentGrade,
    StructuredEvidenceRecord,
    TemporalAlignmentGrade,
    VersionLineage,
    validate_lane_transition,
    validate_unique_lane_execution,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def lane_record(
    *,
    candidate_id: str = "candidate-1",
    status: LaneExecutionStatus = LaneExecutionStatus.COMPLETED_WITH_MATCHES,
) -> LaneExecutionRecord:
    return LaneExecutionRecord(
        candidate_id=candidate_id,
        lane="usdm",
        enabled=True,
        required=True,
        source_mode="frozen_local_read_only",
        package_id="fixture-package",
        package_sha256=SHA_A,
        readiness=LaneReadiness.READY,
        execution_status=status,
        coverage=AxisCoverageState.COMPLETE,
        records_examined=1,
        matches_found=1,
        accepted_count=1,
        contextual_count=0,
        unresolved_count=0,
        completed_at="2026-07-12T00:00:00Z",
    )


def evidence_record() -> StructuredEvidenceRecord:
    return StructuredEvidenceRecord(
        candidate_id="candidate-1",
        lane="usdm",
        package_id="fixture-package",
        source_role=SourceRole.USDM_COUNTY_WEEK_DROUGHT_CONTEXT,
        disposition=EvidenceDisposition.ACCEPTED,
        source_record_id="usdm-record-1",
        source_event_id="",
        source_file="fixture.csv",
        source_row="1",
        source_url="https://example.invalid/frozen-provenance-only",
        source_sha256=SHA_A,
        source_version="fixture-v1",
        source_date="2024-01-02",
        raw_value_provenance={"d1_plus": "10.0", "unit": "percent"},
        spatial_grade=SpatialAlignmentGrade.COUNTY_CONTEXT,
        temporal_grade=TemporalAlignmentGrade.PREVIOUS_OR_SAME_TUESDAY,
        axis_assessments={AxisName.DROUGHT_CORROBORATION: AxisValue.YES},
        rule_ids=(RuleId.USDM_D1_POSITIVE,),
        reason_codes=(ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD,),
        record_hash=SHA_B,
    )


def no_axis(candidate_id: str, axis_name: AxisName) -> CaseAxisRecord:
    return CaseAxisRecord(
        candidate_id=candidate_id,
        axis_name=axis_name,
        axis_value=AxisValue.NO,
        coverage_state=AxisCoverageState.COMPLETE,
        official_supporting_record_ids=(),
        official_contextual_record_ids=(),
        webpage_supporting_record_ids=(),
        webpage_contextual_record_ids=(),
        strongest_spatial_grade=None,
        all_spatial_grades=(),
        strongest_temporal_grade=None,
        all_temporal_grades=(),
        decision_rule_id=RuleId.AXIS_NO,
        unresolved_reason_codes=(),
        review_reason_codes=(),
        source_package_ids=("fixture-package",),
        negative_scope="bounded_evaluated_scope_only",
    )


def test_enum_exhaustiveness_for_core_product() -> None:
    assert {value.value for value in AxisValue} == {"yes", "no", "unresolved"}
    assert {value.value for value in AxisCoverageState} == {
        "complete",
        "partial",
        "not_run",
        "unavailable",
        "failed",
    }
    assert {value.value for value in LaneExecutionStatus} == {
        "completed_with_matches",
        "completed_zero_match",
        "completed_context_only",
        "disabled",
        "unavailable",
        "retryable_failure",
        "terminal_failure",
    }
    assert {value.value for value in EvidenceDisposition} == {
        "accepted",
        "rejected_non_support",
        "rejected_mismatch",
        "contextual",
        "unresolved",
    }


def test_lane_execution_round_trip_and_uniqueness() -> None:
    original = lane_record()
    assert LaneExecutionRecord.from_dict(original.to_dict()) == original
    validate_unique_lane_execution([original, lane_record(candidate_id="candidate-2")])
    with pytest.raises(CanonicalSchemaError, match="duplicate_lane_execution"):
        validate_unique_lane_execution([original, original])


def test_lane_execution_success_and_failure_states_are_not_interchangeable() -> None:
    assert lane_record().successful_completion is True
    with pytest.raises(CanonicalSchemaError, match="unsuccessful_execution_forbids_complete_coverage"):
        replace(lane_record(), execution_status=LaneExecutionStatus.TERMINAL_FAILURE)
    disabled = LaneExecutionRecord(
        candidate_id="candidate-1",
        lane="optional",
        enabled=False,
        required=False,
        source_mode="disabled",
        package_id="disabled-policy",
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
    assert disabled.successful_completion is False


def test_lane_transition_validation() -> None:
    previous = LaneExecutionRecord(
        candidate_id="candidate-1",
        lane="usdm",
        enabled=True,
        required=True,
        source_mode="frozen_local_read_only",
        package_id="fixture-package",
        package_sha256=SHA_A,
        readiness=LaneReadiness.READY,
        execution_status=LaneExecutionStatus.RETRYABLE_FAILURE,
        coverage=AxisCoverageState.FAILED,
        records_examined=0,
        matches_found=0,
        accepted_count=0,
        contextual_count=0,
        unresolved_count=0,
        attempt=1,
    )
    current = replace(lane_record(), attempt=2)
    validate_lane_transition(previous, current)
    with pytest.raises(CanonicalSchemaError, match="invalid_lane_transition"):
        validate_lane_transition(current, previous)


def test_structured_evidence_round_trip_preserves_semantics() -> None:
    original = evidence_record()
    assert StructuredEvidenceRecord.from_dict(original.to_dict()) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_status", "completed_zero_match"),
        ("zero_match", True),
        ("retryable_failure", True),
        ("coverage", "complete"),
    ],
)
def test_structured_evidence_rejects_execution_only_synthetic_states(field: str, value: object) -> None:
    raw = evidence_record().to_dict()
    raw[field] = value
    with pytest.raises(CanonicalSchemaError, match="execution_only_state"):
        StructuredEvidenceRecord.from_dict(raw)


def test_structured_evidence_rejects_derived_display_rows() -> None:
    raw = evidence_record().to_dict()
    raw["derived_review_row"] = True
    with pytest.raises(CanonicalSchemaError, match="derived_display_row_rejected"):
        StructuredEvidenceRecord.from_dict(raw)


@pytest.mark.parametrize(
    "value,coverage,supporting,reasons,scope,valid",
    [
        (AxisValue.YES, AxisCoverageState.COMPLETE, ("r1",), (), "", True),
        (AxisValue.YES, AxisCoverageState.PARTIAL, ("r1",), (), "", True),
        (AxisValue.YES, AxisCoverageState.UNAVAILABLE, ("r1",), (), "", False),
        (AxisValue.YES, AxisCoverageState.COMPLETE, (), (), "", False),
        (AxisValue.NO, AxisCoverageState.COMPLETE, (), (), "bounded_evaluated_scope_only", True),
        (AxisValue.NO, AxisCoverageState.PARTIAL, (), (), "bounded_evaluated_scope_only", False),
        (AxisValue.NO, AxisCoverageState.COMPLETE, ("r1",), (), "bounded_evaluated_scope_only", False),
        (AxisValue.UNRESOLVED, AxisCoverageState.COMPLETE, (), (ReasonCode.CANDIDATE_RELEVANT_UNRESOLVED,), "", True),
        (AxisValue.UNRESOLVED, AxisCoverageState.FAILED, (), (ReasonCode.SOURCE_FAILURE,), "", True),
        (AxisValue.UNRESOLVED, AxisCoverageState.PARTIAL, (), (), "", False),
    ],
)
def test_axis_value_coverage_product_validation(
    value: AxisValue,
    coverage: AxisCoverageState,
    supporting: tuple[str, ...],
    reasons: tuple[ReasonCode, ...],
    scope: str,
    valid: bool,
) -> None:
    kwargs = dict(
        candidate_id="candidate-1",
        axis_name=AxisName.REALIZED_IMPACT,
        axis_value=value,
        coverage_state=coverage,
        official_supporting_record_ids=supporting,
        official_contextual_record_ids=(),
        webpage_supporting_record_ids=(),
        webpage_contextual_record_ids=(),
        strongest_spatial_grade=None,
        all_spatial_grades=(),
        strongest_temporal_grade=None,
        all_temporal_grades=(),
        decision_rule_id=RuleId.AXIS_VALUE,
        unresolved_reason_codes=reasons,
        review_reason_codes=(),
        source_package_ids=("fixture",),
        negative_scope=scope,
    )
    if valid:
        record = CaseAxisRecord(**kwargs)
        assert CaseAxisRecord.from_dict(record.to_dict()) == record
    else:
        with pytest.raises(CanonicalSchemaError):
            CaseAxisRecord(**kwargs)


def test_complete_case_output_round_trip_and_legacy_non_influence() -> None:
    candidate_id = "candidate-1"
    physical = PhysicalStatusRecord(
        candidate_id=candidate_id,
        status=PhysicalCandidateStatus.AUTHORITATIVE_PHYSICAL_CANDIDATE,
        authoritative_inventory_id="california_408_authoritative_inventory",
        physical_input_hashes=(SHA_A, SHA_B),
    )
    legacy = LegacyCompatibilityView(
        legacy_tier="historical-only-value",
        source_artifact_id="historical-snapshot",
    )
    views = DerivedViewsRecord(
        candidate_id=candidate_id,
        component_corroboration=ComponentCorroborationView.NO_COMPONENT_CORROBORATION_IN_EVALUATED_SCOPE,
        impact_attribution=ImpactAttributionView.NO_DOCUMENTED_IMPACT_IN_EVALUATED_SCOPE,
        linkage=LinkageView.NO_EXPLICIT_LINKAGE_IN_EVALUATED_SCOPE,
        navigation_summary=NavigationSummary.CANDIDATE_ONLY,
        legacy_compatibility=legacy,
    )
    output = CaseOutputRecord(
        candidate_id=candidate_id,
        physical_status=physical,
        axes={axis: no_axis(candidate_id, axis) for axis in AxisName},
        derived_views=views,
    )
    assert CaseOutputRecord.from_dict(output.to_dict()) == output
    changed_legacy = replace(
        output,
        derived_views=replace(
            views,
            legacy_compatibility=replace(legacy, legacy_tier="different-history"),
        ),
    )
    assert changed_legacy.axes == output.axes
    assert changed_legacy.physical_status == output.physical_status


def test_legacy_compatibility_cannot_become_truth_input() -> None:
    with pytest.raises(CanonicalSchemaError, match="legacy_compatibility"):
        LegacyCompatibilityView(
            legacy_tier="bad",
            source_artifact_id="history",
            influences_repaired_v2_truth=True,
        )
