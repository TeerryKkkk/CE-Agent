from __future__ import annotations

from climate_pipeline.ce_impact_labeling import (
    derive_accepted_gate_flags,
    derive_admin_response_status,
    derive_case_level_split_labels,
    derive_combined_impact_status,
    derive_direct_observed_impact_status,
)
from climate_pipeline.official_source_lanes import (
    OfficialSourceLaneControls,
    admin_response_flags_from_evidence_rows,
    openfema_evidence_row,
    run_openfema_admin_response_lane_for_case,
    support_flags_from_evidence_rows,
)


def _case(**overrides: str) -> dict[str, str]:
    case = {
        "raw_candidate_id": "rawce_openfema_test",
        "candidate_id": "rawce_openfema_test",
        "county": "Madera County",
        "state": "California",
        "state_abbrev": "CA",
        "FIPS": "06039",
        "county_fips": "06039",
        "rain_start": "2023-01-09",
        "rain_end": "2023-01-11",
        "drought_window": "2022-02 to 2022-10-31",
        "wet_event_window": "2023-01-09 to 2023-01-11",
        "event_window": "drought 2022-02 to 2022-10-31; wet 2023-01-09 to 2023-01-11",
        "run_completed": "true",
        "reached_evidence_judgment": "true",
    }
    case.update(overrides)
    return case


def _openfema_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "disasterNumber": 4683,
        "state": "CA",
        "fipsStateCode": "06",
        "fipsCountyCode": "039",
        "designatedArea": "Madera (County)",
        "declarationDate": "2023-01-14T00:00:00.000Z",
        "incidentBeginDate": "2022-12-27T00:00:00.000Z",
        "incidentEndDate": "2023-01-31T00:00:00.000Z",
        "incidentType": "Flood",
        "declarationTitle": "SEVERE WINTER STORMS, FLOODING, LANDSLIDES, AND MUDSLIDES",
        "declarationType": "DR",
        "paProgramDeclared": True,
        "iaProgramDeclared": False,
        "ihProgramDeclared": False,
        "hmProgramDeclared": True,
    }
    row.update(overrides)
    return row


def _direct_weak_row() -> dict[str, str]:
    return {
        "source_family": "noaa_ncei_storm_events_structured",
        "source_lane": "noaa_ncei_storm_events",
        "final_status": "accepted",
        "component_supported": "wet",
        "impact_supported": "true",
        "supports_impact": "true",
        "supports_wet_event": "true",
        "impact_types": "roads / transport",
        "impact_channel": "roads / transport",
        "location_match": "same_county",
        "time_match": "exact_window",
        "quoted_supporting_spans": (
            "NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. "
            "Narrative: a roadway was flooded and briefly closed."
        ),
    }


def test_openfema_county_level_matched_row_creates_admin_material_proxy() -> None:
    result = run_openfema_admin_response_lane_for_case(_case(), source_rows=[_openfema_row()])

    row = result.evidence_rows[0]

    assert row["final_status"] == "accepted"
    assert row["admin_response_status"] == "admin_material_proxy"
    assert row["admin_response_supported"] == "true"
    assert row["admin_proxy_material"] == "true"
    assert admin_response_flags_from_evidence_rows(result.evidence_rows)["admin_material_proxy"] is True
    assert derive_admin_response_status(result.evidence_rows) == "admin_material_proxy"


def test_openfema_wrong_county_does_not_create_accepted_admin_proxy() -> None:
    result = run_openfema_admin_response_lane_for_case(
        _case(),
        source_rows=[_openfema_row(fipsCountyCode="019", designatedArea="Fresno (County)")],
    )

    row = result.evidence_rows[0]

    assert row["final_status"] == "rejected"
    assert row["admin_response_status"] == "admin_not_found"
    assert admin_response_flags_from_evidence_rows(result.evidence_rows)["admin_material_proxy"] is False


def test_openfema_wrong_state_does_not_create_accepted_admin_proxy() -> None:
    row = openfema_evidence_row(
        _case(),
        _openfema_row(state="TX", fipsStateCode="48", designatedArea="Madera (County)"),
    )

    assert row["final_status"] == "rejected"
    assert row["admin_response_status"] == "admin_rejected"
    assert row["openfema_state_match"] == "false"


def test_openfema_unrelated_incident_type_does_not_create_accepted_admin_proxy() -> None:
    row = openfema_evidence_row(
        _case(),
        _openfema_row(incidentType="Fire", declarationTitle="WILDFIRE"),
    )

    assert row["final_status"] == "rejected"
    assert row["admin_response_status"] == "admin_rejected"
    assert row["openfema_incident_type_relevant"] == "false"


def test_openfema_statewide_or_countyless_declaration_does_not_accept() -> None:
    row = openfema_evidence_row(
        _case(),
        _openfema_row(designatedArea="Statewide", fipsCountyCode="000"),
    )

    assert row["final_status"] == "rejected"
    assert row["admin_response_status"] == "admin_rejected"
    assert row["openfema_county_match"] == "false"


def test_openfema_admin_proxy_does_not_close_drought_wet_or_transition_gates() -> None:
    row = run_openfema_admin_response_lane_for_case(_case(), source_rows=[_openfema_row()]).evidence_rows[0]
    flags = derive_accepted_gate_flags([row])

    assert flags["public_drought_found"] is False
    assert flags["wet_event_found"] is False
    assert flags["impact_found"] is False
    assert flags["drought_gate_passed"] is False
    assert flags["wet_event_gate_passed"] is False
    assert row["explicit_transition_support"] == "false"
    assert support_flags_from_evidence_rows([row]) == {"drought": False, "wet": False, "impact": False}


def test_direct_impact_and_admin_proxy_remain_separate() -> None:
    admin = run_openfema_admin_response_lane_for_case(_case(), source_rows=[_openfema_row()]).evidence_rows[0]
    rows = [_direct_weak_row(), admin]

    assert derive_direct_observed_impact_status(rows) == "direct_weak"
    assert derive_admin_response_status(rows) == "admin_material_proxy"
    assert derive_combined_impact_status("direct_weak", "admin_material_proxy") == "material_admin_proxy"


def test_admin_proxy_can_upgrade_combined_impact_without_changing_ce_status() -> None:
    admin = run_openfema_admin_response_lane_for_case(_case(), source_rows=[_openfema_row()]).evidence_rows[0]
    direct = _direct_weak_row()

    v1 = derive_case_level_split_labels(_case(), [direct])
    v2 = derive_case_level_split_labels(_case(), [direct, admin])
    combined = derive_combined_impact_status(
        derive_direct_observed_impact_status([direct, admin]),
        derive_admin_response_status([direct, admin]),
    )

    assert v1.ce_status == "ce_public_drought_weak"
    assert v2.ce_status == v1.ce_status
    assert v2.impact_status == v1.impact_status
    assert combined == "material_admin_proxy"


def test_openfema_outside_window_is_rejected() -> None:
    row = openfema_evidence_row(
        _case(),
        _openfema_row(
            incidentBeginDate="2024-01-01T00:00:00.000Z",
            incidentEndDate="2024-01-31T00:00:00.000Z",
        ),
        controls=OfficialSourceLaneControls(openfema_same_storm_sequence_days=7),
    )

    assert row["final_status"] == "rejected"
    assert row["admin_response_status"] == "admin_rejected"
    assert row["time_match"] == "outside_window"
