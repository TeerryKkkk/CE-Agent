from __future__ import annotations

from types import SimpleNamespace

from scripts.retrieve import (
    web_evidence_rows_from_validation as demo_web_evidence_rows,
)
from climate_pipeline.pipeline.canonical_aggregation import (
    aggregate_repaired_v2_case,
)
from climate_pipeline.pipeline.remaining_production_adapter import (
    adapt_webpage_checkpoints,
)
from climate_pipeline.pipeline.official_adapters import NOAA_LANE, USDM_LANE
from climate_pipeline.pipeline.schemas import (
    AxisCoverageState,
    AxisName,
    AxisValue,
    LaneExecutionRecord,
    LaneExecutionStatus,
    LaneReadiness,
    LegacyCompatibilityView,
    PhysicalCandidateStatus,
    PhysicalStatusRecord,
)
from climate_pipeline.production_manifest_protocol import (
    web_evidence_rows_from_validation as production_web_evidence_rows,
)


def _completed_zero_match_execution(
    candidate_id: str,
    lane: str,
) -> LaneExecutionRecord:
    return LaneExecutionRecord(
        candidate_id=candidate_id,
        lane=lane,
        enabled=True,
        required=True,
        source_mode="test_local_read_only",
        package_id=f"{lane}-test-package",
        package_sha256="a" * 64,
        readiness=LaneReadiness.READY,
        execution_status=LaneExecutionStatus.COMPLETED_ZERO_MATCH,
        coverage=AxisCoverageState.COMPLETE,
        records_examined=0,
        matches_found=0,
        accepted_count=0,
        contextual_count=0,
        unresolved_count=0,
        completed_at="done",
    )


def test_direct_judge_result_maps_into_mainline_evidence_axes() -> None:
    quote = "Example County reported heavy rain and flooding on January 10, 2023."
    result = SimpleNamespace(
        normalized_rows=[
            {
                "candidate_id": "candidate-example",
                "source_url": "https://example.test/report",
                "source_metadata": {"source_lane": "local_news", "source_family": "news"},
                "raw_semantic_judgment": {
                    "event_identity": "one_uniquely_relevant_event",
                    "candidate_event": {
                        "supporting_quotes": [quote],
                        "realized_impacts": [{"claim": "road flooding", "quote": quote}],
                        "location_relation": "aligned",
                        "date_relation": "aligned",
                    },
                },
                "guarded_local_result": {
                    "page_result": "supports",
                    "candidate_hazard_support": "yes",
                    "realized_impact_support": "yes",
                    "explicit_attribution_support": "yes",
                    "explicit_drought_to_wet_transition_support": "no",
                    "guard_actions": [],
                },
            }
        ]
    )
    pages = [
        {
            "candidate_id": "candidate-example",
            "source_url": "https://example.test/report",
            "source_title": "Example report",
            "body_text_or_archived_body_text": quote,
        }
    ]
    rows = demo_web_evidence_rows(
        result,
        pages,
        {"candidate-example": {"county": "Example County", "FIPS": "00001"}},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["final_status"] == "accepted"
    assert row["page_result"] == "supports"
    assert row["candidate_hazard_support"] == "yes"
    assert row["realized_impact_support"] == "yes"
    assert row["explicit_attribution_support"] == "yes"
    assert row["explicit_drought_to_wet_transition_support"] == "no"
    assert row["quote_body_grounded"] == "true"


def test_drought_targeted_direct_result_closes_drought_not_wet_axis() -> None:
    quote = (
        "Example County experienced severe drought conditions on June 15, 2022."
    )
    result = SimpleNamespace(
        normalized_rows=[
            {
                "candidate_id": "candidate-drought",
                "source_url": "https://example.test/drought-report",
                "candidate_metadata": {
                    "target_hazard_axis": "drought",
                    "target_window_start": "2022-05-01",
                    "target_window_end": "2022-12-31",
                },
                "source_metadata": {
                    "source_lane": "agriculture_drought_impact",
                    "source_family": "local_government",
                },
                "raw_semantic_judgment": {
                    "event_identity": "one_uniquely_relevant_event",
                    "candidate_event": {
                        "supporting_quotes": [quote],
                        "realized_impacts": [],
                        "location_relation": "aligned",
                        "date_relation": "aligned",
                    },
                },
                "guarded_local_result": {
                    "page_result": "supports",
                    "candidate_hazard_support": "yes",
                    "realized_impact_support": "no",
                    "explicit_attribution_support": "no",
                    "explicit_drought_to_wet_transition_support": "no",
                    "guard_actions": [],
                },
            }
        ]
    )
    pages = [
        {
            "candidate_id": "candidate-drought",
            "source_url": "https://example.test/drought-report",
            "source_title": "Drought report",
            "source_lane": "agriculture_drought_impact",
            "target_gap": "drought_hazard_public_corroboration",
            "body_text_or_archived_body_text": quote,
        }
    ]
    cases = {
        "candidate-drought": {
            "county": "Example County",
            "FIPS": "00001",
        }
    }

    for adapter in (demo_web_evidence_rows, production_web_evidence_rows):
        row = adapter(result, pages, cases)[0]
        assert row["component_supported"] == "drought"
        assert row["target_hazard_axis"] == "drought"
        assert row["drought_hazard_support"] == "yes"
        assert row["wet_hazard_support"] == "unresolved"

    production_row = production_web_evidence_rows(result, pages, cases)[0]
    canonical = adapt_webpage_checkpoints(
        candidate_id="candidate-drought",
        pages=pages,
        webpage_rows=[production_row],
        request_payloads=[{"call_key": "completed-drought-judge"}],
        package_id="test-drought-package",
    )
    record = canonical.evidence[0]
    assert (
        record.axis_assessments[AxisName.DROUGHT_CORROBORATION]
        is AxisValue.YES
    )
    assert (
        record.axis_assessments[AxisName.WET_HAZARD_CORROBORATION]
        is AxisValue.UNRESOLVED
    )

    aggregated = aggregate_repaired_v2_case(
        physical_status=PhysicalStatusRecord(
            candidate_id="candidate-drought",
            status=PhysicalCandidateStatus.AUTHORITATIVE_PHYSICAL_CANDIDATE,
            authoritative_inventory_id="test-inventory",
            physical_input_hashes=("b" * 64,),
        ),
        execution_records=(
            _completed_zero_match_execution("candidate-drought", USDM_LANE),
            _completed_zero_match_execution("candidate-drought", NOAA_LANE),
            canonical.execution,
        ),
        evidence_records=canonical.evidence,
        legacy_compatibility=LegacyCompatibilityView(
            legacy_tier="test-only",
            source_artifact_id="test-artifact",
        ),
    )
    drought_axis = aggregated.case_output.axes[
        AxisName.DROUGHT_CORROBORATION
    ]
    wet_axis = aggregated.case_output.axes[
        AxisName.WET_HAZARD_CORROBORATION
    ]
    assert drought_axis.axis_value is AxisValue.YES
    assert drought_axis.webpage_supporting_record_ids == (
        record.source_record_id,
    )
    assert wet_axis.axis_value is AxisValue.NO
    assert wet_axis.webpage_supporting_record_ids == ()
    assert aggregated.source_mix["candidate_source_axis_contributions"] == [
        {
            "candidate_id": "candidate-drought",
            "source_family": "frozen_webpage",
            "axis": AxisName.DROUGHT_CORROBORATION.value,
        }
    ]
