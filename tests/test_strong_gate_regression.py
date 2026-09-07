from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from src.climate_pipeline.ce_impact_labeling import (
    classify_evidence_impact,
    derive_accepted_gate_flags,
    derive_case_level_split_labels,
)
from src.climate_pipeline import official_source_lanes
from src.climate_pipeline.official_source_lanes import (
    OfficialSourceLaneControls,
    support_flags_from_evidence_rows,
    usdm_evidence_row,
)


def _evidence(
    *,
    component: str = "none",
    status: str = "accepted",
    supports_drought: bool = False,
    supports_wet: bool = False,
    supports_impact: bool = False,
    impact_channel: str = "",
    snippet: str = "Accepted local same-window evidence.",
    same_county: bool = True,
    same_window: bool = True,
    source_family: str = "official_test_source",
) -> dict[str, str]:
    return {
        "source_family": source_family,
        "source_title": "Test evidence",
        "source_url": "structured://test",
        "component_supported": component,
        "supports_drought": str(supports_drought).lower(),
        "supports_wet_event": str(supports_wet).lower(),
        "supports_impact": str(supports_impact).lower(),
        "impact_supported": str(supports_impact).lower(),
        "impact_channel": impact_channel,
        "impact_types": impact_channel,
        "quoted_snippet_or_short_paraphrase": snippet,
        "quoted_supporting_spans": snippet,
        "same_county_or_local": str(same_county).lower(),
        "same_window_or_close": str(same_window).lower(),
        "location_match": "same_county" if same_county else "no_match",
        "time_match": "exact_window" if same_window else "no_match",
        "pipeline_decision_label": status,
        "final_status": status,
    }


def _case() -> dict[str, str]:
    return {
        "independent_case_id": "rawce_regression",
        "county": "Test County",
        "event_window": "drought 2024-01..2024-08; wet 2024-10-01..2024-10-02",
        "drought_window": "2024-01..2024-08",
        "wet_event_window": "2024-10-01..2024-10-02",
        "run_completed": "true",
        "reached_evidence_judgment": "true",
    }


def _noaa_case() -> dict[str, str]:
    return {
        "raw_candidate_id": "rawce_noaa_test",
        "county": "Fresno County",
        "state": "California",
        "FIPS": "06019",
        "rain_start": "2023-01-09",
        "rain_end": "2023-01-11",
        "rainfall_year": "2023",
    }


def _write_noaa_cache(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "EVENT_ID",
        "BEGIN_DATE_TIME",
        "END_DATE_TIME",
        "STATE",
        "STATE_FIPS",
        "CZ_TYPE",
        "CZ_FIPS",
        "CZ_NAME",
        "EVENT_TYPE",
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "EPISODE_NARRATIVE",
        "EVENT_NARRATIVE",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_context_only_drought_row_does_not_set_public_drought_support() -> None:
    flags = derive_accepted_gate_flags(
        [_evidence(component="drought", status="context_only", supports_drought=True)]
    )

    assert flags["public_drought_found"] is False
    assert flags["drought_gate_passed"] is False


def test_needs_review_drought_row_does_not_set_public_drought_support() -> None:
    flags = derive_accepted_gate_flags(
        [_evidence(component="drought", status="needs_review", supports_drought=True)]
    )

    assert flags["public_drought_found"] is False
    assert flags["drought_gate_passed"] is False


def test_usdm_d1_zero_is_rejected_and_does_not_close_drought_gate(tmp_path: Path) -> None:
    case = {
        "raw_candidate_id": "rawce_usdm_zero",
        "county": "Alameda County",
        "FIPS": "06001",
        "drought_end_month_end": "2024-10-31",
    }
    cache_path = tmp_path / "usdm_06001_2024-10-29.csv"
    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ValidStart", "D0", "D1", "D2", "D3", "D4"])
        writer.writeheader()
        writer.writerow({"ValidStart": "2024-10-29", "D0": "0", "D1": "0", "D2": "0", "D3": "0", "D4": "0"})

    row = usdm_evidence_row(case, controls=OfficialSourceLaneControls(usdm_cache_dir=tmp_path))

    assert row["final_status"] == "rejected"
    assert row["component_supported"] == "none"
    assert row["supports_drought"] == "false"
    assert support_flags_from_evidence_rows([row])["drought"] is False
    assert derive_accepted_gate_flags([row])["public_drought_found"] is False
    assert derive_accepted_gate_flags([row])["public_drought_check_status"] == "success"


def test_usdm_cache_miss_live_fallback_accepts_and_writes_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = {
        "raw_candidate_id": "rawce_usdm_live",
        "county": "Fresno County",
        "FIPS": "06019",
        "drought_end_month_end": "2022-11-30",
    }

    def fake_fetch(
        fips: str,
        usdm_date,
        *,
        timeout_seconds: float,
    ) -> tuple[dict[str, str] | None, str, str]:
        assert fips == "06019"
        assert usdm_date.isoformat() == "2022-11-29"
        assert timeout_seconds > 0
        return (
            {
                "MapDate": "20221129",
                "FIPS": "06019",
                "County": "Fresno County",
                "State": "CA",
                "None": "0.00",
                "D0": "100.00",
                "D1": "100.00",
                "D2": "100.00",
                "D3": "98.60",
                "D4": "64.50",
                "ValidStart": "2022-11-29",
                "ValidEnd": "2022-12-05",
                "StatisticFormatID": "1",
            },
            "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi=06019",
            "",
        )

    monkeypatch.setattr(official_source_lanes, "_fetch_usdm_live_row", fake_fetch)

    row = official_source_lanes.usdm_evidence_row(
        case,
        controls=OfficialSourceLaneControls(usdm_cache_dir=tmp_path),
    )

    assert row["fetch_status"] == "live_api_fallback_success"
    assert row["final_status"] == "accepted"
    assert row["supports_drought"] == "true"
    assert row["component_supported"] == "drought"
    assert row["public_drought_check_status"] == "live_api_fallback_success"
    assert row["usdm_d1_area_percent"] == "100.00"
    assert (tmp_path / "usdm_06019_2022-11-29.csv").exists()


def test_usdm_cache_miss_without_live_fallback_is_lane_failed_not_rejected(tmp_path: Path) -> None:
    case = {
        "raw_candidate_id": "rawce_usdm_missing",
        "county": "Fresno County",
        "FIPS": "06019",
        "drought_end_month_end": "2022-11-30",
    }

    row = usdm_evidence_row(
        case,
        controls=OfficialSourceLaneControls(
            usdm_cache_dir=tmp_path,
            enable_usdm_live_api_fallback=False,
        ),
    )

    assert row["final_status"] == "lane_failed"
    assert row["supports_drought"] == ""
    assert row["public_drought_check_status"] == "lane_failed"
    flags = derive_accepted_gate_flags([row])
    assert flags["public_drought_found"] is False
    assert flags["public_drought_check_status"] == "lane_failed"


def test_noaa_cache_missing_lane_failed_not_no_match_rejected(tmp_path: Path) -> None:
    result = official_source_lanes.run_structured_official_lanes_for_case(
        _noaa_case(),
        controls=OfficialSourceLaneControls(
            enable_usdm_county_statistics=False,
            noaa_cache_dir=tmp_path / "missing_noaa_cache",
        ),
    )

    row = result.evidence_rows[0]
    assert row["source_lane"] == "noaa_ncei_storm_events"
    assert row["final_status"] == "lane_failed"
    assert row["fetch_status"] == "cache_dir_missing"
    assert row["source_check_status"] == "lane_failed"
    assert row["impact_supported"] == ""
    assert support_flags_from_evidence_rows([row]) == {"drought": False, "wet": False, "impact": False}


def test_noaa_successful_empty_county_window_is_substantive_rejected(tmp_path: Path) -> None:
    _write_noaa_cache(
        tmp_path / "StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz",
        [
            {
                "EVENT_ID": "1",
                "BEGIN_DATE_TIME": "09-Jan-23 00:00:00",
                "END_DATE_TIME": "09-Jan-23 01:00:00",
                "STATE": "CALIFORNIA",
                "STATE_FIPS": "06",
                "CZ_TYPE": "C",
                "CZ_FIPS": "001",
                "CZ_NAME": "ALAMEDA",
                "EVENT_TYPE": "Flood",
                "DAMAGE_PROPERTY": "0.00K",
                "DAMAGE_CROPS": "0.00K",
                "INJURIES_DIRECT": "0",
                "INJURIES_INDIRECT": "0",
                "DEATHS_DIRECT": "0",
                "DEATHS_INDIRECT": "0",
                "EPISODE_NARRATIVE": "",
                "EVENT_NARRATIVE": "Flooding occurred in another county.",
            }
        ],
    )

    result = official_source_lanes.run_structured_official_lanes_for_case(
        _noaa_case(),
        controls=OfficialSourceLaneControls(enable_usdm_county_statistics=False, noaa_cache_dir=tmp_path),
    )

    row = result.evidence_rows[0]
    assert row["final_status"] == "rejected"
    assert row["fetch_status"] == "local_structured_cache_no_match"
    assert row["source_check_status"] == "success"
    assert row["cache_status"] == "cache_hit"
    assert row["impact_supported"] == "false"


def test_fips_leading_zero_preserved_in_noaa_cache_keys(tmp_path: Path) -> None:
    _write_noaa_cache(
        tmp_path / "StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz",
        [
            {
                "EVENT_ID": "2",
                "BEGIN_DATE_TIME": "09-Jan-23 00:00:00",
                "END_DATE_TIME": "09-Jan-23 01:00:00",
                "STATE": "CALIFORNIA",
                "STATE_FIPS": "6",
                "CZ_TYPE": "C",
                "CZ_FIPS": "19",
                "CZ_NAME": "FRESNO",
                "EVENT_TYPE": "Flood",
                "DAMAGE_PROPERTY": "250.00K",
                "DAMAGE_CROPS": "0.00K",
                "INJURIES_DIRECT": "0",
                "INJURIES_INDIRECT": "0",
                "DEATHS_DIRECT": "0",
                "DEATHS_INDIRECT": "0",
                "EPISODE_NARRATIVE": "",
                "EVENT_NARRATIVE": "Flood damaged homes in Fresno County.",
            }
        ],
    )

    result = official_source_lanes.run_structured_official_lanes_for_case(
        {**_noaa_case(), "FIPS": "6019"},
        controls=OfficialSourceLaneControls(enable_usdm_county_statistics=False, noaa_cache_dir=tmp_path),
    )

    row = result.evidence_rows[0]
    assert row["final_status"] == "accepted"
    assert row["structured_record_id"] == "2"
    assert row["matched_record_count"] == "1"


def test_base_evidence_row_default_not_rejected() -> None:
    row = official_source_lanes._base_evidence_row({"raw_candidate_id": "rawce_base"})

    assert row["final_status"] == "uninitialized"
    assert row["final_status"] != "rejected"
    with pytest.raises(ValueError, match="evidence_rows_missing_explicit_final_status"):
        official_source_lanes.validate_evidence_row_statuses([row])


def test_failure_support_flags_unknown_not_false(tmp_path: Path) -> None:
    result = official_source_lanes.run_structured_official_lanes_for_case(
        _noaa_case(),
        controls=OfficialSourceLaneControls(
            enable_usdm_county_statistics=False,
            noaa_cache_dir=tmp_path / "missing_noaa_cache",
        ),
    )

    row = result.evidence_rows[0]
    assert row["final_status"] == "lane_failed"
    assert row["impact_supported"] == ""
    assert row["wet_impact_support"] == ""


def test_candidate_metadata_alone_cannot_produce_public_drought_support() -> None:
    result = derive_case_level_split_labels(_case(), [])

    assert result.ce_status == "ce_unsupported"
    assert result.case_use_label == "not_supported_or_review"


def test_context_only_wet_row_does_not_set_wet_event_support() -> None:
    flags = derive_accepted_gate_flags(
        [_evidence(component="wet", status="context_only", supports_wet=True)]
    )

    assert flags["wet_event_found"] is False
    assert flags["wet_event_gate_passed"] is False


def test_accepted_wet_row_requires_locality_and_time_alignment() -> None:
    unaligned = derive_accepted_gate_flags(
        [_evidence(component="wet", supports_wet=True, same_county=True, same_window=False)]
    )
    aligned = derive_accepted_gate_flags(
        [_evidence(component="wet", supports_wet=True, same_county=True, same_window=True)]
    )

    assert unaligned["wet_event_found"] is False
    assert aligned["wet_event_found"] is True


def test_accepted_road_only_zero_damage_noaa_impact_is_weak_not_material() -> None:
    row = _evidence(
        component="wet",
        supports_wet=True,
        supports_impact=True,
        impact_channel="roads / transport",
        source_family="noaa_ncei_storm_events_structured",
        snippet=(
            "NOAA event 10 (Flood) in TEST County; property=0.00K, crops=0.00K, "
            "deaths=0/0, injuries=0/0. Narrative: water covered a road and it was briefly closed."
        ),
    )

    assert classify_evidence_impact(row).evidence_impact_status == "impact_weak"
    assert derive_accepted_gate_flags([row])["material_impact_gate_passed"] is False


def test_low_damage_broad_noaa_narrative_is_weak_not_material() -> None:
    row = _evidence(
        component="wet",
        supports_wet=True,
        supports_impact=True,
        impact_channel="property / housing",
        source_family="noaa_ncei_storm_events_structured",
        snippet=(
            "NOAA event 11 (Storm) in TEST County; property=1.00K, crops=0.00K, "
            "deaths=0/0, injuries=0/0. Narrative: a broad regional storm brought heavy rain."
        ),
    )

    assert classify_evidence_impact(row).evidence_impact_status == "impact_weak"
    assert derive_accepted_gate_flags([row])["impact_found"] is True
    assert derive_accepted_gate_flags([row])["material_impact_gate_passed"] is False


@pytest.mark.parametrize(
    ("snippet", "channel"),
    [
        ("NOAA event; property=0.00K, crops=0.00K, deaths=1/0, injuries=0/0. Narrative: fatal flooding.", "human impact"),
        ("NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: swift-water rescue.", "human impact"),
        ("NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: evacuation order issued.", "human impact"),
        ("NOAA event; property=250.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: homes damaged.", "property / housing"),
        ("NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: bridge washed out.", "roads / transport"),
        ("NOAA event; property=0.00K, crops=200.00K, deaths=0/0, injuries=0/0. Narrative: crop loss reported.", "agriculture"),
        ("NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: power outage affected the town.", "energy / utility"),
        ("NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: school closure and public service disruption.", "public services / water"),
    ],
)
def test_material_impact_gate_accepts_concrete_local_consequences(snippet: str, channel: str) -> None:
    row = _evidence(
        component="wet",
        supports_wet=True,
        supports_impact=True,
        impact_channel=channel,
        snippet=snippet,
        source_family="noaa_ncei_storm_events_structured",
    )

    assert classify_evidence_impact(row).evidence_impact_status == "impact_material"
    assert derive_accepted_gate_flags([row])["material_impact_gate_passed"] is True


def test_strong_label_requires_accepted_drought_wet_and_material_impact() -> None:
    drought = _evidence(component="drought", supports_drought=True)
    wet_material = _evidence(
        component="wet",
        supports_wet=True,
        supports_impact=True,
        impact_channel="human impact",
        snippet="NOAA event; property=0.00K, crops=0.00K, deaths=1/0, injuries=0/0. Narrative: fatal flood.",
        source_family="noaa_ncei_storm_events_structured",
    )
    wet_weak = _evidence(
        component="wet",
        supports_wet=True,
        supports_impact=True,
        impact_channel="roads / transport",
        snippet="NOAA event; property=0.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: a road was briefly closed.",
        source_family="noaa_ncei_storm_events_structured",
    )

    assert derive_case_level_split_labels(_case(), [drought, wet_material]).case_use_label == "strong_ce_with_material_impact"
    assert derive_case_level_split_labels(_case(), [wet_material]).case_use_label == "wet_impact_drought_public_weak"
    assert derive_case_level_split_labels(_case(), [drought, wet_weak]).case_use_label == "ce_with_weak_impact"


def test_impact_evidence_does_not_upgrade_ce_status() -> None:
    impact_only = _evidence(
        component="none",
        supports_impact=True,
        impact_channel="property / housing",
        snippet="NOAA event; property=500.00K, crops=0.00K, deaths=0/0, injuries=0/0. Narrative: local damage.",
        source_family="noaa_ncei_storm_events_structured",
    )

    result = derive_case_level_split_labels(_case(), [impact_only])

    assert result.ce_status == "ce_unsupported"
    assert result.impact_status == "impact_material"
    assert result.case_use_label == "impact_only_not_ce"


def test_context_and_needs_review_rows_may_exist_but_cannot_close_gates() -> None:
    rows = [
        _evidence(component="drought", status="context_only", supports_drought=True),
        _evidence(component="wet", status="needs_review", supports_wet=True, supports_impact=True),
    ]

    flags = derive_accepted_gate_flags(rows)

    assert flags == {
        "public_drought_found": False,
        "wet_event_found": False,
        "impact_found": False,
        "same_county_match": False,
        "same_window_match": False,
        "drought_gate_passed": False,
        "wet_event_gate_passed": False,
        "impact_gate_passed": False,
        "material_impact_gate_passed": False,
        "drought_impact_found": False,
        "wet_impact_found": False,
        "compound_impact_found": False,
        "drought_material_impact_gate_passed": False,
        "wet_material_impact_gate_passed": False,
        "compound_material_impact_gate_passed": False,
        "ce_material_impact_gate_passed": False,
        "material_impact_pattern": "no_material_impact",
        "public_drought_check_status": "unknown",
    }


def test_drought_only_material_impact_does_not_close_ce_material_gate() -> None:
    drought_hazard = _evidence(component="drought", supports_drought=True)
    wet_hazard = _evidence(component="wet", supports_wet=True)
    drought_material = _evidence(
        component="drought",
        supports_drought=True,
        supports_impact=True,
        impact_channel="agriculture",
        snippet="NOAA event; property=0.00K, crops=200.00K, deaths=0/0, injuries=0/0. Narrative: crop loss reported.",
        source_family="noaa_ncei_storm_events_structured",
    )

    flags = derive_accepted_gate_flags([drought_hazard, wet_hazard, drought_material])
    result = derive_case_level_split_labels(_case(), [drought_hazard, wet_hazard, drought_material])

    assert flags["material_impact_gate_passed"] is True
    assert flags["drought_material_impact_gate_passed"] is True
    assert flags["wet_material_impact_gate_passed"] is False
    assert flags["ce_material_impact_gate_passed"] is False
    assert flags["material_impact_pattern"] == "drought_material_impact_only"
    assert result.ce_status == "ce_supported"
    assert result.impact_status == "impact_material"
    assert result.drought_impact_status == "impact_material"
    assert result.wet_impact_status == "impact_not_found"
    assert result.material_impact_pattern == "drought_material_impact_only"
    assert result.case_use_label == "ce_with_weak_impact"
