from __future__ import annotations

from .active_config import DATA_DIR, PROJECT_ROOT

CHINA_ARCHIVE_DATA_DIR = DATA_DIR / "archive" / "china_v0"
CHINA_ARCHIVE_DATASET_DIR = CHINA_ARCHIVE_DATA_DIR / "dataset"

PLUVIAL_CSV = CHINA_ARCHIVE_DATASET_DIR / "extreme_pluvial_1980_2024_china.csv"
DROUGHT_CSV = CHINA_ARCHIVE_DATASET_DIR / "spei3_drought_1980_2024.csv"

FIXTURE_PLUVIAL_CSV = PROJECT_ROOT / "tests" / "fixtures" / "tiny_pluvial.csv"
FIXTURE_DROUGHT_CSV = PROJECT_ROOT / "tests" / "fixtures" / "tiny_drought.csv"
FIXTURE_SEARCH_RESULTS = PROJECT_ROOT / "tests" / "fixtures" / "mock_search_results.jsonl"
FIXTURE_PAGE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "sample_pages"

PLUVIAL_HAZARD_TERMS = [
    "暴雨",
    "强降雨",
    "特大暴雨",
    "洪涝",
    "内涝",
    "山洪",
    "防汛",
    "超警",
    "编号洪水",
    "转移群众",
    "抢险救援",
    "防汛应急响应",
]
DROUGHT_HAZARD_TERMS = [
    "干旱",
    "旱情",
    "抗旱",
    "高温干旱",
    "缺水",
    "饮水困难",
    "受旱",
    "受旱面积",
    "农作物受灾",
    "送水",
    "灌溉",
    "抗旱应急响应",
]

AGENCY_TYPES = {
    "government_portal",
    "emergency_management",
    "water_resources",
    "meteorology",
    "agriculture_rural_affairs",
    "flood_drought_control",
    "disaster_relief_civil_affairs",
    "development_reform_or_energy",
    "transportation",
    "power_utility_public_agency",
    "other_public_agency",
    "unknown",
}

SOURCE_TIERS = {
    "national",
    "provincial",
    "municipal",
    "county",
    "township_or_village_mention_only",
    "basin_or_regional",
    "unknown",
}

ACTION_TYPES = {
    "discover_source",
    "search_evidence",
    "fetch_page",
    "classify_page",
    "extract_evidence",
    "drill_down_admin",
    "expand_agency",
    "stop_with_sufficient_evidence",
    "stop_no_public_official_evidence_found",
}

PAGE_TYPES = {
    "forecast_warning",
    "observed_hazard_report",
    "response_notice",
    "impact_report",
    "hydrological_report",
    "agricultural_impact_report",
    "recovery_notice",
    "termination_notice",
    "preparedness_plan",
    "responsibility_list",
    "policy_interpretation",
    "statistical_bulletin",
    "cross_region_repost",
    "historical_background",
    "irrelevant",
    "unknown",
}

__all__ = [
    "CHINA_ARCHIVE_DATA_DIR",
    "CHINA_ARCHIVE_DATASET_DIR",
    "PLUVIAL_CSV",
    "DROUGHT_CSV",
    "FIXTURE_PLUVIAL_CSV",
    "FIXTURE_DROUGHT_CSV",
    "FIXTURE_SEARCH_RESULTS",
    "FIXTURE_PAGE_DIR",
    "PLUVIAL_HAZARD_TERMS",
    "DROUGHT_HAZARD_TERMS",
    "AGENCY_TYPES",
    "SOURCE_TIERS",
    "ACTION_TYPES",
    "PAGE_TYPES",
]
