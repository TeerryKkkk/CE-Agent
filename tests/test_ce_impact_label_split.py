from __future__ import annotations

from climate_pipeline.ce_impact_labeling import (
    classify_evidence_impact,
    derive_case_level_split_labels,
    derive_case_use_label,
    derive_ce_status,
    derive_impact_status,
    derive_material_impact_pattern,
)


def _case(
    *,
    drought: bool = True,
    wet: bool = True,
    impact: bool = True,
    same_county: bool = True,
    same_window: bool = True,
    metadata: bool = True,
) -> dict[str, str]:
    return {
        "independent_case_id": "rawce_test",
        "county": "Test County",
        "event_window": "drought 2022-01..2022-08; wet 2022-11-09..2022-11-10"
        if metadata
        else "",
        "drought_window": "2022-01..2022-08" if metadata else "",
        "wet_event_window": "2022-11-09..2022-11-10" if metadata else "",
        "pipeline_found_drought_evidence": str(drought).lower(),
        "pipeline_found_wet_event_evidence": str(wet).lower(),
        "pipeline_found_impact_evidence": str(impact).lower(),
        "pipeline_same_county_match": str(same_county).lower(),
        "pipeline_same_window_match": str(same_window).lower(),
        "run_completed": "true",
        "reached_evidence_judgment": "true",
    }


def _impact_row(
    *,
    snippet: str,
    channel: str = "property / housing",
    supports_wet: bool = True,
    supports_impact: bool = True,
    same_county: bool = True,
    same_window: bool = True,
    decision: str = "accepted",
) -> dict[str, str]:
    return {
        "source_family": "noaa_ncei_storm_events_structured",
        "source_title": "NOAA/NCEI Storm Events Details CSV",
        "source_url": "structured://test",
        "retrieval_tier": "structured_official_first",
        "source_lane": "noaa_ncei_storm_events",
        "quoted_snippet_or_short_paraphrase": snippet,
        "supports_drought": "false",
        "supports_wet_event": str(supports_wet).lower(),
        "supports_impact": str(supports_impact).lower(),
        "impact_channel": channel,
        "same_county_or_local": str(same_county).lower(),
        "same_window_or_close": str(same_window).lower(),
        "pipeline_decision_label": decision,
    }


def _hazard_row(
    *,
    component: str,
    same_county: bool = True,
    same_window: bool = True,
    decision: str = "accepted",
) -> dict[str, str]:
    return {
        "source_family": "official_structured",
        "source_title": "Official hazard evidence",
        "source_url": "structured://hazard",
        "retrieval_tier": "structured_official_first",
        "source_lane": "official_hazard",
        "quoted_snippet_or_short_paraphrase": "Accepted local same-window hazard evidence.",
        "supports_drought": str(component in {"drought", "both"}).lower(),
        "supports_wet_event": str(component in {"wet", "both"}).lower(),
        "supports_impact": "false",
        "impact_channel": "",
        "same_county_or_local": str(same_county).lower(),
        "same_window_or_close": str(same_window).lower(),
        "pipeline_decision_label": decision,
    }


def test_ce_supported_with_material_impact_is_strong_case_study() -> None:
    evidence = [
        _hazard_row(component="drought"),
        _impact_row(
            snippet=(
                "NOAA event 1 (Flood) in TEST County; property=250.00K, crops=0.00K, "
                "deaths=0/0, injuries=0/0. Narrative: local homes and roads were damaged."
            ),
            channel="property / housing; roads / transport",
        )
    ]

    result = derive_case_level_split_labels(_case(), evidence)

    assert result.ce_status == "ce_supported"
    assert result.impact_status == "impact_material"
    assert result.material_impact_pattern == "wet_material_impact_only"
    assert result.as_dict()["material_impact_pattern"] == "wet_material_impact_only"
    assert result.case_use_label == "strong_ce_with_material_impact"


def test_material_impact_pattern_classifies_side_specific_material_gates() -> None:
    assert derive_material_impact_pattern(
        {
            "drought_material_impact_gate_passed": True,
            "wet_material_impact_gate_passed": True,
            "compound_material_impact_gate_passed": True,
        }
    ) == "compound_transition_material_impact"
    assert derive_material_impact_pattern(
        {
            "drought_material_impact_gate_passed": "true",
            "wet_material_impact_gate_passed": "true",
            "compound_material_impact_gate_passed": "false",
        }
    ) == "drought_and_wet_material_impact"
    assert derive_material_impact_pattern(
        {
            "drought_material_impact_gate_passed": False,
            "wet_material_impact_gate_passed": True,
            "compound_material_impact_gate_passed": False,
        }
    ) == "wet_material_impact_only"
    assert derive_material_impact_pattern(
        {
            "drought_material_impact_gate_passed": True,
            "wet_material_impact_gate_passed": False,
            "compound_material_impact_gate_passed": False,
        }
    ) == "drought_material_impact_only"
    assert derive_material_impact_pattern({}) == "no_material_impact"


def test_ce_supported_with_weak_road_only_low_damage_impact_is_weak_impact_case() -> None:
    evidence = [
        _hazard_row(component="drought"),
        _impact_row(
            snippet=(
                "NOAA event 2 (Flood) in TEST County; property=1.00K, crops=0.00K, "
                "deaths=0/0, injuries=0/0. Narrative: a roadway was flooded and briefly closed."
            ),
            channel="roads / transport",
        )
    ]

    result = derive_case_level_split_labels(_case(), evidence)

    assert classify_evidence_impact(evidence[1]).evidence_impact_status == "impact_weak"
    assert result.impact_status == "impact_weak"
    assert result.case_use_label == "ce_with_weak_impact"


def test_explicit_transition_material_impact_gets_compound_pattern() -> None:
    transition_impact = _impact_row(
        snippet=(
            "NOAA event 6 (Flood) in TEST County; property=250.00K, crops=0.00K, "
            "deaths=0/0, injuries=0/0. Narrative: direct local flood damage."
        ),
        channel="property / housing",
    )
    transition_impact["explicit_transition_support"] = "true"

    result = derive_case_level_split_labels(_case(), [_hazard_row(component="drought"), transition_impact])

    assert result.case_use_label == "strong_ce_with_material_impact"
    assert result.material_impact_pattern == "compound_transition_material_impact"
    assert result.compound_impact_status == "impact_material"


def test_wet_impact_with_weak_public_drought_is_separate_case_use_label() -> None:
    evidence = [
        _impact_row(
            snippet=(
                "NOAA event 3 (Flash Flood) in TEST County; property=0.00K, crops=0.00K, "
                "deaths=1/0, injuries=0/0. Narrative: one person died in flood waters."
            ),
            channel="human impact",
        )
    ]

    result = derive_case_level_split_labels(_case(drought=False), evidence)

    assert result.ce_status == "ce_public_drought_weak"
    assert result.impact_status == "impact_material"
    assert result.case_use_label == "wet_impact_drought_public_weak"


def test_ce_supported_but_no_public_impact_is_ce_without_public_impact() -> None:
    evidence = [_hazard_row(component="drought"), _hazard_row(component="wet")]

    result = derive_case_level_split_labels(_case(impact=False), evidence)

    assert result.ce_status == "ce_supported"
    assert result.impact_status == "impact_not_found"
    assert result.case_use_label == "ce_without_public_impact"


def test_impact_found_but_ce_unsupported_is_impact_only_not_ce() -> None:
    evidence = [
        _impact_row(
            snippet=(
                "NOAA event 4 (Flood) in TEST County; property=500.00K, crops=0.00K, "
                "deaths=0/0, injuries=0/0. Narrative: direct public infrastructure damage."
            ),
            supports_wet=False,
        )
    ]

    result = derive_case_level_split_labels(
        _case(drought=False, wet=False, same_county=False, same_window=False, metadata=False),
        evidence,
    )

    assert result.ce_status == "ce_unsupported"
    assert result.impact_status == "impact_material"
    assert result.case_use_label == "impact_only_not_ce"


def test_no_support_is_not_supported_or_review() -> None:
    result = derive_case_level_split_labels(
        _case(drought=False, wet=False, impact=False, same_county=False, same_window=False, metadata=False),
        [],
    )

    assert result.ce_status == "ce_unsupported"
    assert result.impact_status == "impact_not_found"
    assert result.case_use_label == "not_supported_or_review"


def test_impact_evidence_must_not_upgrade_ce_status() -> None:
    evidence = [
        _impact_row(
            snippet=(
                "NOAA event 5 (Flood) in TEST County; property=1.00M, crops=0.00K, "
                "deaths=0/0, injuries=0/0. Narrative: direct public infrastructure damage."
            ),
            supports_wet=False,
        )
    ]

    result = derive_case_level_split_labels(
        _case(drought=False, wet=False, same_county=False, same_window=False),
        evidence,
    )

    assert derive_ce_status(_case(drought=False, wet=False, same_county=False, same_window=False), evidence) == "ce_unsupported"
    assert derive_impact_status(_case(), evidence) == "impact_material"
    assert result.ce_status == "ce_unsupported"
    assert result.case_use_label == "impact_only_not_ce"


def test_case_use_combination_table_keeps_review_out_of_strong_labels() -> None:
    assert derive_case_use_label("needs_review", "impact_material") == "impact_only_not_ce"
    assert derive_case_use_label("ce_partial", "impact_not_found") == "ce_without_public_impact"
    assert derive_case_use_label("ce_unsupported", "impact_not_found") == "not_supported_or_review"
