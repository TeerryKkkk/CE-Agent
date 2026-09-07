from __future__ import annotations

import inspect

from climate_pipeline import case_aggregation as aggregation


def structured(**overrides: str) -> dict[str, str]:
    row = {
        "deterministic_match_accepted": "true",
        "drought_hazard_support": "yes",
        "wet_hazard_support": "yes",
        "impact_support": "yes",
    }
    row.update(overrides)
    return row


def page(**overrides: str) -> dict[str, str]:
    row = {
        "page_result": "unresolved",
        "candidate_hazard_support": "no",
        "realized_impact_support": "no",
        "attribution_support": "no",
        "transition_support": "no",
    }
    row.update(overrides)
    return row


def test_unresolved_page_does_not_globally_contaminate_supported_axes() -> None:
    axes = aggregation.aggregate_case_axes([page()], [structured()])
    assert axes == {
        "drought_hazard_support": "yes",
        "wet_hazard_support": "yes",
        "impact_support": "yes",
        "explicit_hazard_to_impact_attribution_support": "no",
        "explicit_linkage_support": "no",
    }


def test_uncertainty_propagates_only_on_its_own_axis() -> None:
    axes = aggregation.aggregate_case_axes(
        [page(attribution_support="unresolved")],
        [structured()],
    )
    assert axes["wet_hazard_support"] == "yes"
    assert axes["impact_support"] == "yes"
    assert axes["explicit_hazard_to_impact_attribution_support"] == "unresolved"
    assert axes["explicit_linkage_support"] == "no"


def test_a_yes_on_one_web_axis_does_not_invent_another_axis() -> None:
    axes = aggregation.aggregate_case_axes(
        [page(candidate_hazard_support="yes")],
        [],
    )
    assert axes["wet_hazard_support"] == "yes"
    assert axes["impact_support"] == "no"
    assert axes["explicit_hazard_to_impact_attribution_support"] == "no"
    assert axes["explicit_linkage_support"] == "no"


def test_drought_targeted_web_evidence_closes_only_drought_axis() -> None:
    axes = aggregation.aggregate_case_axes(
        [
            page(
                drought_hazard_support="yes",
                wet_hazard_support="unresolved",
                candidate_hazard_support="yes",
            )
        ],
        [],
    )
    assert axes["drought_hazard_support"] == "yes"
    assert axes["wet_hazard_support"] == "unresolved"
    assert axes["impact_support"] == "no"


def test_rejected_structured_record_contributes_nothing() -> None:
    axes = aggregation.aggregate_case_axes([], [structured(deterministic_match_accepted="false")])
    assert all(value == "no" for value in axes.values())


def test_runtime_aggregation_has_no_audit_oracle_dependency() -> None:
    source = inspect.getsource(aggregation).lower()
    forbidden = (
        "page_level_independent_audit",
        "case_level_independent_audit",
        "independent_page_judgment",
        "expected_answer",
    )
    assert not any(value in source for value in forbidden)
