from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

from . import config

try:  # Canonical PYTHONPATH=src namespace.
    from io_utils import safe_write_json, safe_write_text
    from rainfall_p99_pipeline import (
        MappingResult,
        build_grid_to_county_lookup,
        coord_key,
        find_default_county_boundary_file,
    )
except ImportError:  # Compatibility for historical `src.climate_pipeline` imports.
    from ..io_utils import safe_write_json, safe_write_text
    from ..rainfall_p99_pipeline import (
        MappingResult,
        build_grid_to_county_lookup,
        coord_key,
        find_default_county_boundary_file,
    )

SCHEMA_VERSION = "raw_ce_pilot_v1"
DEFAULT_CONFIG_PATH = config.PROJECT_ROOT / "configs" / "raw_ce_pilot_texas_2021_2025.yaml"
DEFAULT_OUTPUT_DIR = config.OUTPUT_DIR / "raw_ce_pilot" / "texas_2021_2025"

SPATIAL_PRIORITY = {
    "same_grid": 3,
    "same_county_near_grid": 2,
    "same_county_only": 1,
    "unknown": 0,
}

COVERAGE_PRIORITY = {
    "county_broad": 3,
    "county_partial": 2,
    "county_sparse": 1,
    "unknown": 0,
}

SEVERITY_PRIORITY = {
    "none": 0,
    "abnormally dry": 0,
    "moderate": 1,
    "severe": 2,
    "extreme": 3,
    "exceptional": 4,
}

TIER_PRIORITY = {
    "exact_source_pair_match": 1,
    "exact_rain_drought_episode_match": 2,
    "spatiotemporal_window_match": 3,
    "component_only_match": 4,
    "no_match": 5,
}


@dataclass(frozen=True)
class PilotConfig:
    pilot_name: str = "raw_ce_pilot_texas_2021_2025"
    state: str = "Texas"
    state_abbrev: str = "TX"
    state_fips: str = "48"
    rainfall_start_date: date = date(2021, 1, 1)
    rainfall_end_date: date = date(2025, 12, 31)
    rainfall_threshold: str = "p99"
    policy_version: str = "raw_ce_pilot_texas_2021_2025_v1"
    temporal_policy_version: str = "temporal_post_drought_month_end_1_90d_v1"
    spatial_policy_version: str = "spatial_same_fips_grid_distance_75km_v1"
    candidate_id_policy_version: str = "raw_candidate_sha256_construction_fields_v1"
    deduplication_policy_version: str = "dedup_exact_then_same_rainfall_endmonth31_lagbin_v1"
    drought_grouping_policy_version: str = "drought_same_fips_overlap_or_gap_31d_v1"
    rainfall_grouping_policy_version: str = "rainfall_same_fips_event_or_overlap_gap_1d_v1"
    near_grid_threshold_km: float = 75.0
    primary_lag_min_days: int = 1
    primary_lag_max_days: int = 90
    spei3_input_path: Path = config.DATASET_DIR / "spei3_drought_us_2000_2025.csv"
    rainfall_input_path: Path = config.DATASET_DIR / "events_extreme_rain_pr3_p99_1985_2025.csv"
    dter_input_path: Path = config.DATASET_DIR / "dter_eca_events_us_2000_2025.csv"
    county_boundary_path: Path | None = config.DATA_DIR / "raw" / "tl_2024_us_county.zip"
    output_dir: Path = DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class DroughtSourceEvent:
    schema_version: str
    source_drought_key: str
    lat: float
    lon: float
    lat_key: str
    lon_key: str
    grid_coord_key: str
    drought_start_month: str
    drought_end_month: str
    drought_start_date: date
    drought_end_month_start: date
    drought_end_month_end: date
    drought_duration_months: int | None
    min_spei: float | None
    wmo_severity: str | None
    state: str | None
    county: str | None
    fips: str | None
    mapping_status: str
    source_event_id: str | None
    source_row_index: int


@dataclass(frozen=True)
class RainfallSourceEvent:
    schema_version: str
    source_rainfall_key: str
    rain_event_id: str
    rain_event_type: str
    lat: float
    lon: float
    lat_key: str
    lon_key: str
    grid_coord_key: str
    rain_start: date
    rain_end: date
    rain_duration_days: int | None
    rainfall_threshold: str
    state: str | None
    county: str | None
    fips: str | None
    mapping_status: str
    source_row_index: int


@dataclass(frozen=True)
class CountyDroughtEpisode:
    schema_version: str
    drought_episode_id: str
    state: str
    county: str
    fips: str
    start_month: str
    end_month: str
    drought_start_date: date
    drought_end_month_start: date
    drought_end_month_end: date
    duration_months: int | None
    source_drought_keys: list[str]
    source_grid_count: int
    source_grid_coords: list[str]
    min_spei_min: float | None
    wmo_severity_max: str | None
    drought_grouping_policy_version: str
    drought_coverage_class: str


@dataclass(frozen=True)
class CountyRainfallEpisode:
    schema_version: str
    rainfall_episode_id: str
    state: str
    county: str
    fips: str
    rain_start: date
    rain_end: date
    rain_duration_days: int | None
    rain_event_ids: list[str]
    rain_event_types: list[str]
    source_grid_count: int
    source_grid_coords: list[str]
    source_rainfall_keys: list[str]
    rainfall_threshold: str
    rainfall_grouping_policy_version: str


@dataclass(frozen=True)
class RawCECandidate:
    schema_version: str
    raw_candidate_id: str
    state: str
    county: str
    fips: str
    drought_episode_id: str
    rainfall_episode_id: str
    drought_start_month: str
    drought_end_month: str
    drought_end_month_end: date
    rain_start: date
    rain_end: date
    lag_days: int
    lag_bin: str
    temporal_match_status: str
    spatial_match_level: str
    min_grid_distance_km: float | None
    candidate_status: str
    drought_source_keys: list[str]
    rainfall_source_keys: list[str]
    rain_event_ids: list[str]
    rain_event_types: list[str]
    drought_coverage_class: str
    wmo_severity_max: str | None
    policy_version: str
    temporal_policy_version: str
    spatial_policy_version: str
    candidate_id_policy_version: str
    dedup_policy_version: str
    candidate_key_hash: str
    source_hash: str
    duplicate_of_candidate_id: str | None = None
    collapsed_candidate_ids: list[str] = field(default_factory=list)
    dedup_reason: str | None = None


@dataclass(frozen=True)
class DTERSourceRow:
    dter_event_id: str
    source_row_index: int
    source_drought_key: str
    rain_event_id: str
    rain_event_type: str | None
    lat: float
    lon: float
    rain_start: date
    rain_end: date
    drought_start_date: date
    drought_end_date: date
    drought_end_month_end: date
    state: str
    county: str
    fips: str
    mapping_status: str


@dataclass(frozen=True)
class DTERAlignmentRecord:
    schema_version: str
    alignment_record_id: str
    raw_candidate_id: str | None
    dter_event_id: str | None
    alignment_tier: str
    alignment_category: str
    state: str
    county: str | None
    fips: str | None
    raw_drought_episode_id: str | None
    raw_rainfall_episode_id: str | None
    dter_drought_key: str | None
    dter_rain_event_id: str | None
    raw_rain_event_ids: list[str]
    raw_drought_end_month_end: date | None
    dter_drought_end_month_end: date | None
    raw_rain_start: date | None
    dter_rain_start: date | None
    alignment_reason: str
    match_score: float | None
    review_priority: str


@dataclass
class ConstructionResult:
    drought_source_events: list[DroughtSourceEvent]
    rainfall_source_events: list[RainfallSourceEvent]
    drought_episodes: list[CountyDroughtEpisode]
    rainfall_episodes: list[CountyRainfallEpisode]
    raw_candidates: list[RawCECandidate]
    deduped_candidates: list[RawCECandidate]
    temporal_ambiguous_candidates: list[RawCECandidate]
    summary: dict[str, Any]
    paths: dict[str, Path]


@dataclass
class DTERAlignmentResult:
    dter_rows: list[DTERSourceRow]
    alignment_records: list[DTERAlignmentRecord]
    dter_only_records: list[DTERAlignmentRecord]
    summary: dict[str, Any]


def parse_day(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def decimal_key(value: Any, places: int | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return text
    if places is not None:
        decimal = decimal.quantize(Decimal("1").scaleb(-places), rounding=ROUND_HALF_UP)
    normalized = decimal.normalize()
    return format(normalized, "f")


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def month_label(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def inclusive_months(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month + 1


def stable_hash_payload(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def stable_raw_candidate_id(
    *,
    policy_version: str,
    fips: str,
    drought_episode_id: str,
    rainfall_episode_id: str,
    temporal_policy_version: str,
    spatial_policy_version: str,
) -> str:
    key = [
        policy_version,
        fips,
        drought_episode_id,
        rainfall_episode_id,
        temporal_policy_version,
        spatial_policy_version,
    ]
    return "rawce_" + stable_hash_payload(key, length=16)


def make_drought_source_key(row: dict[str, Any], *, severity_field: str = "wmo_severity") -> str:
    start = str(row.get("drought_start") or "").strip()
    end = str(row.get("drought_end") or "").strip()
    duration = decimal_key(row.get("drought_duration_months"))
    severity = str(row.get(severity_field) or "").strip().lower()
    payload = [
        decimal_key(row.get("lat"), places=6),
        decimal_key(row.get("lon"), places=6),
        start,
        end,
        duration,
        decimal_key(row.get("min_spei")),
        severity,
    ]
    return "spei3_" + stable_hash_payload(payload, length=20)


def make_rainfall_source_key(row: dict[str, Any]) -> str:
    payload = [
        str(row.get("event_id") or row.get("rain_event_id") or "").strip(),
        decimal_key(row.get("lat"), places=6),
        decimal_key(row.get("lon"), places=6),
        str(row.get("start_date") or row.get("rain_start") or "").strip(),
        str(row.get("end_date") or row.get("rain_end") or "").strip(),
        decimal_key(row.get("duration") or row.get("rain_duration_days")),
    ]
    return "rainp99_" + stable_hash_payload(payload, length=20)


def normalize_headers(row: dict[str, Any]) -> dict[str, str]:
    return {str(key or "").lstrip("\ufeff").strip(): "" if value is None else str(value).strip() for key, value in row.items()}


def iter_csv_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None:
            reader.fieldnames = [str(field or "").lstrip("\ufeff").strip() for field in reader.fieldnames]
        for index, row in enumerate(reader, start=1):
            yield index, normalize_headers(row)


def _parse_yaml_scalar(value: str) -> Any:
    text = value.strip().strip("'").strip('"')
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    parsed_date = parse_day(text)
    if parsed_date is not None:
        return parsed_date
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_pilot_config(path: Path = DEFAULT_CONFIG_PATH) -> PilotConfig:
    values: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _parse_yaml_scalar(value)

    def path_value(name: str, default: Path | None = None) -> Path | None:
        value = values.get(name)
        if value in {None, ""}:
            return default
        expanded = str(value)
        built_in_roots = {
            "${CE_AGENT_PROJECT_ROOT}": str(config.PROJECT_HOME),
            "${CE_AGENT_DATA_ROOT}": str(config.DATA_ROOT),
            "${CE_AGENT_RUN_ROOT}": str(config.RUN_ROOT),
            "%CE_AGENT_PROJECT_ROOT%": str(config.PROJECT_HOME),
            "%CE_AGENT_DATA_ROOT%": str(config.DATA_ROOT),
            "%CE_AGENT_RUN_ROOT%": str(config.RUN_ROOT),
        }
        for marker, replacement in built_in_roots.items():
            expanded = expanded.replace(marker, replacement)
        expanded = os.path.expandvars(expanded)
        if "$" in expanded or "%" in expanded:
            raise ValueError(f"Unresolved environment variable in {name}: {value}")
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = config.PROJECT_ROOT / candidate
        return candidate

    return PilotConfig(
        pilot_name=str(values.get("pilot_name", PilotConfig.pilot_name)),
        state=str(values.get("state", PilotConfig.state)),
        state_abbrev=str(values.get("state_abbrev", PilotConfig.state_abbrev)),
        state_fips=str(values.get("state_fips", PilotConfig.state_fips)).zfill(2),
        rainfall_start_date=values.get("rainfall_start_date", PilotConfig.rainfall_start_date),
        rainfall_end_date=values.get("rainfall_end_date", PilotConfig.rainfall_end_date),
        rainfall_threshold=str(values.get("rainfall_threshold", PilotConfig.rainfall_threshold)),
        policy_version=str(values.get("policy_version", PilotConfig.policy_version)),
        temporal_policy_version=str(values.get("temporal_policy_version", PilotConfig.temporal_policy_version)),
        spatial_policy_version=str(values.get("spatial_policy_version", PilotConfig.spatial_policy_version)),
        candidate_id_policy_version=str(values.get("candidate_id_policy_version", PilotConfig.candidate_id_policy_version)),
        deduplication_policy_version=str(values.get("deduplication_policy_version", PilotConfig.deduplication_policy_version)),
        drought_grouping_policy_version=str(values.get("drought_grouping_policy_version", PilotConfig.drought_grouping_policy_version)),
        rainfall_grouping_policy_version=str(values.get("rainfall_grouping_policy_version", PilotConfig.rainfall_grouping_policy_version)),
        near_grid_threshold_km=float(values.get("near_grid_threshold_km", PilotConfig.near_grid_threshold_km)),
        primary_lag_min_days=int(values.get("primary_lag_min_days", PilotConfig.primary_lag_min_days)),
        primary_lag_max_days=int(values.get("primary_lag_max_days", PilotConfig.primary_lag_max_days)),
        spei3_input_path=path_value("spei3_input_path", PilotConfig.spei3_input_path) or PilotConfig.spei3_input_path,
        rainfall_input_path=path_value("rainfall_input_path", PilotConfig.rainfall_input_path) or PilotConfig.rainfall_input_path,
        dter_input_path=path_value("dter_input_path", PilotConfig.dter_input_path) or PilotConfig.dter_input_path,
        county_boundary_path=path_value("county_boundary_path", PilotConfig.county_boundary_path),
        output_dir=path_value("output_dir", PilotConfig.output_dir) or PilotConfig.output_dir,
    )


def _row_coord(row: dict[str, Any]) -> tuple[float, float] | None:
    lat = parse_float(row.get("lat"))
    lon = parse_float(row.get("lon"))
    if lat is None or lon is None:
        return None
    return (lat, lon)


def _is_target_p99(row: dict[str, str], threshold: str) -> bool:
    haystack = " ".join(
        [
            str(row.get("event_type") or ""),
            str(row.get("event_id") or ""),
        ]
    ).lower()
    return threshold.lower() in haystack


def read_rainfall_rows(path: Path, pilot_config: PilotConfig) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    diagnostics: Counter[str] = Counter()
    for row_index, row in iter_csv_rows(path):
        diagnostics["rows_read"] += 1
        if not _is_target_p99(row, pilot_config.rainfall_threshold):
            diagnostics["skipped_not_p99"] += 1
            continue
        start = parse_day(row.get("start_date"))
        end = parse_day(row.get("end_date"))
        if start is None or end is None:
            diagnostics["skipped_invalid_date"] += 1
            continue
        if not pilot_config.rainfall_start_date <= start <= pilot_config.rainfall_end_date:
            diagnostics["skipped_outside_rainfall_date_scope"] += 1
            continue
        coord = _row_coord(row)
        if coord is None:
            diagnostics["skipped_invalid_coordinate"] += 1
            continue
        row["_source_row_index"] = str(row_index)
        rows.append(row)
        diagnostics["rows_in_scope"] += 1
    return rows, diagnostics


def read_drought_rows(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    diagnostics: Counter[str] = Counter()
    for row_index, row in iter_csv_rows(path):
        diagnostics["rows_read"] += 1
        start, end_start, end = parse_drought_months(row)
        if start is None or end_start is None or end is None:
            diagnostics["skipped_invalid_drought_months"] += 1
            continue
        coord = _row_coord(row)
        if coord is None:
            diagnostics["skipped_invalid_coordinate"] += 1
            continue
        row["_source_row_index"] = str(row_index)
        rows.append(row)
        diagnostics["rows_with_valid_window"] += 1
    return rows, diagnostics


def parse_drought_months(row: dict[str, Any]) -> tuple[date | None, date | None, date | None]:
    start_year = parse_int(row.get("drought_start_year"))
    start_month_num = parse_int(row.get("drought_start_month"))
    end_year = parse_int(row.get("drought_end_year"))
    end_month_num = parse_int(row.get("drought_end_month"))
    if start_year is None or start_month_num is None:
        parsed = parse_day(row.get("drought_start"))
        if parsed is not None:
            start_year = parsed.year
            start_month_num = parsed.month
    if end_year is None or end_month_num is None:
        parsed = parse_day(row.get("drought_end"))
        if parsed is not None:
            end_year = parsed.year
            end_month_num = parsed.month
    if None in {start_year, start_month_num, end_year, end_month_num}:
        return None, None, None
    try:
        start = date(int(start_year), int(start_month_num), 1)
        end_month_start = date(int(end_year), int(end_month_num), 1)
        end = month_end(int(end_year), int(end_month_num))
    except ValueError:
        return None, None, None
    return start, end_month_start, end


def build_mapping_lookup_for_rows(
    drought_rows: list[dict[str, str]],
    rainfall_rows: list[dict[str, str]],
    pilot_config: PilotConfig,
) -> tuple[dict[str, MappingResult], dict[str, Any]]:
    coords: set[tuple[float, float]] = set()
    for row in [*drought_rows, *rainfall_rows]:
        coord = _row_coord(row)
        if coord is not None:
            coords.add(coord)
    boundary_path = pilot_config.county_boundary_path or find_default_county_boundary_file()
    lookup, blocker, metadata = build_grid_to_county_lookup(coords, boundary_file=boundary_path)
    diagnostics = {
        "unique_coordinate_count": len(coords),
        "mapping_blocker": blocker,
        "county_boundary_path": str(boundary_path) if boundary_path else None,
        "boundary_metadata": boundary_metadata_for_pilot_summary(metadata),
    }
    return lookup, diagnostics


def boundary_metadata_for_pilot_summary(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    payload = metadata.as_dict()
    payload["local_boundary_file_used"] = True
    payload.pop("source_url", None)
    payload["discovered_by_search"] = False
    payload["source_selection_reason"] = "Local county boundary file configured for the no-external-run pilot."
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics.pop("known_us_interior_point_tests", None)
    payload["diagnostics"] = diagnostics
    return payload


def _mapping_for_row(row: dict[str, Any], lookup: dict[str, MappingResult]) -> MappingResult | None:
    coord = _row_coord(row)
    if coord is None:
        return None
    return lookup.get(coord_key(*coord))


def build_drought_source_events(
    rows: list[dict[str, str]],
    mapping_lookup: dict[str, MappingResult],
    pilot_config: PilotConfig,
) -> tuple[list[DroughtSourceEvent], Counter[str]]:
    events: list[DroughtSourceEvent] = []
    diagnostics: Counter[str] = Counter()
    for row in rows:
        mapping = _mapping_for_row(row, mapping_lookup)
        mapping_status = mapping.mapping_status if mapping else "mapping_missing"
        diagnostics[f"mapping_status:{mapping_status}"] += 1
        if mapping is None or mapping.mapping_status != "mapped_us_land_county":
            continue
        if mapping.state != pilot_config.state or mapping.state_fips != pilot_config.state_fips:
            diagnostics["mapped_outside_target_state"] += 1
            continue
        start, end_start, end = parse_drought_months(row)
        if start is None or end_start is None or end is None:
            diagnostics["skipped_invalid_drought_months_after_mapping"] += 1
            continue
        lat = float(row["lat"])
        lon = float(row["lon"])
        lat_key = decimal_key(lat, places=6)
        lon_key = decimal_key(lon, places=6)
        events.append(
            DroughtSourceEvent(
                schema_version=SCHEMA_VERSION,
                source_drought_key=make_drought_source_key(row, severity_field="wmo_severity"),
                lat=lat,
                lon=lon,
                lat_key=lat_key,
                lon_key=lon_key,
                grid_coord_key=f"{lat_key},{lon_key}",
                drought_start_month=month_label(start),
                drought_end_month=month_label(end),
                drought_start_date=start,
                drought_end_month_start=end_start,
                drought_end_month_end=end,
                drought_duration_months=parse_int(row.get("drought_duration_months")),
                min_spei=parse_float(row.get("min_spei")),
                wmo_severity=str(row.get("wmo_severity") or "").strip() or None,
                state=mapping.state,
                county=mapping.county,
                fips=mapping.county_fips,
                mapping_status=mapping.mapping_status,
                source_event_id=str(row.get("event_id") or "").strip() or None,
                source_row_index=int(row.get("_source_row_index") or 0),
            )
        )
        diagnostics["target_state_mapped_rows"] += 1
    events.sort(key=lambda item: (item.fips or "", item.drought_start_date, item.drought_end_month_end, item.source_drought_key))
    return events, diagnostics


def build_rainfall_source_events(
    rows: list[dict[str, str]],
    mapping_lookup: dict[str, MappingResult],
    pilot_config: PilotConfig,
) -> tuple[list[RainfallSourceEvent], Counter[str]]:
    events: list[RainfallSourceEvent] = []
    diagnostics: Counter[str] = Counter()
    for row in rows:
        mapping = _mapping_for_row(row, mapping_lookup)
        mapping_status = mapping.mapping_status if mapping else "mapping_missing"
        diagnostics[f"mapping_status:{mapping_status}"] += 1
        if mapping is None or mapping.mapping_status != "mapped_us_land_county":
            continue
        if mapping.state != pilot_config.state or mapping.state_fips != pilot_config.state_fips:
            diagnostics["mapped_outside_target_state"] += 1
            continue
        start = parse_day(row.get("start_date"))
        end = parse_day(row.get("end_date"))
        if start is None or end is None:
            diagnostics["skipped_invalid_rainfall_dates_after_mapping"] += 1
            continue
        lat = float(row["lat"])
        lon = float(row["lon"])
        lat_key = decimal_key(lat, places=6)
        lon_key = decimal_key(lon, places=6)
        events.append(
            RainfallSourceEvent(
                schema_version=SCHEMA_VERSION,
                source_rainfall_key=make_rainfall_source_key(row),
                rain_event_id=str(row.get("event_id") or "").strip(),
                rain_event_type=str(row.get("event_type") or "").strip(),
                lat=lat,
                lon=lon,
                lat_key=lat_key,
                lon_key=lon_key,
                grid_coord_key=f"{lat_key},{lon_key}",
                rain_start=start,
                rain_end=end,
                rain_duration_days=parse_int(row.get("duration")),
                rainfall_threshold=pilot_config.rainfall_threshold,
                state=mapping.state,
                county=mapping.county,
                fips=mapping.county_fips,
                mapping_status=mapping.mapping_status,
                source_row_index=int(row.get("_source_row_index") or 0),
            )
        )
        diagnostics["target_state_mapped_rows"] += 1
    events.sort(key=lambda item: (item.fips or "", item.rain_start, item.rain_end, item.source_rainfall_key))
    return events, diagnostics


def group_drought_episodes(
    events: list[DroughtSourceEvent],
    pilot_config: PilotConfig,
    *,
    close_gap_days: int = 31,
) -> list[CountyDroughtEpisode]:
    grouped: dict[str, list[DroughtSourceEvent]] = defaultdict(list)
    for event in events:
        if event.fips:
            grouped[event.fips].append(event)
    episodes: list[CountyDroughtEpisode] = []
    for fips, county_events in grouped.items():
        county_events.sort(key=lambda item: (item.drought_start_date, item.drought_end_month_end, item.source_drought_key))
        current: list[DroughtSourceEvent] = []
        current_end: date | None = None
        for event in county_events:
            if not current:
                current = [event]
                current_end = event.drought_end_month_end
                continue
            assert current_end is not None
            if event.drought_start_date <= current_end + timedelta(days=close_gap_days):
                current.append(event)
                current_end = max(current_end, event.drought_end_month_end)
            else:
                episodes.append(_drought_episode_from_group(current, pilot_config))
                current = [event]
                current_end = event.drought_end_month_end
        if current:
            episodes.append(_drought_episode_from_group(current, pilot_config))
    episodes.sort(key=lambda item: (item.fips, item.drought_start_date, item.drought_end_month_end, item.drought_episode_id))
    return episodes


def _coverage_class(source_grid_count: int) -> str:
    if source_grid_count <= 0:
        return "unknown"
    if source_grid_count == 1:
        return "county_sparse"
    if source_grid_count <= 4:
        return "county_partial"
    return "county_broad"


def _severity_rank(value: str | None) -> int:
    return SEVERITY_PRIORITY.get(str(value or "").strip().lower(), 0)


def _max_severity(values: Iterable[str | None]) -> str | None:
    cleaned = [str(value).strip() for value in values if value and str(value).strip()]
    if not cleaned:
        return None
    return max(cleaned, key=lambda value: (_severity_rank(value), value.lower()))


def _drought_episode_from_group(group: list[DroughtSourceEvent], pilot_config: PilotConfig) -> CountyDroughtEpisode:
    first = group[0]
    start = min(item.drought_start_date for item in group)
    end = max(item.drought_end_month_end for item in group)
    end_start = month_start(end)
    source_keys = sorted({item.source_drought_key for item in group})
    grid_coords = sorted({item.grid_coord_key for item in group})
    min_spei_values = [item.min_spei for item in group if item.min_spei is not None]
    episode_id = "drought_ep_" + stable_hash_payload(
        [
            pilot_config.drought_grouping_policy_version,
            first.fips,
            month_label(start),
            month_label(end),
            source_keys,
        ],
        length=16,
    )
    return CountyDroughtEpisode(
        schema_version=SCHEMA_VERSION,
        drought_episode_id=episode_id,
        state=first.state or pilot_config.state,
        county=first.county or "",
        fips=first.fips or "",
        start_month=month_label(start),
        end_month=month_label(end),
        drought_start_date=start,
        drought_end_month_start=end_start,
        drought_end_month_end=end,
        duration_months=inclusive_months(start, end),
        source_drought_keys=source_keys,
        source_grid_count=len(grid_coords),
        source_grid_coords=grid_coords,
        min_spei_min=min(min_spei_values) if min_spei_values else None,
        wmo_severity_max=_max_severity(item.wmo_severity for item in group),
        drought_grouping_policy_version=pilot_config.drought_grouping_policy_version,
        drought_coverage_class=_coverage_class(len(grid_coords)),
    )


def group_rainfall_episodes(
    events: list[RainfallSourceEvent],
    pilot_config: PilotConfig,
    *,
    close_gap_days: int = 1,
) -> list[CountyRainfallEpisode]:
    grouped: dict[str, list[RainfallSourceEvent]] = defaultdict(list)
    for event in events:
        if event.fips:
            grouped[event.fips].append(event)
    episodes: list[CountyRainfallEpisode] = []
    for fips, county_events in grouped.items():
        county_events.sort(key=lambda item: (item.rain_start, item.rain_end, item.rain_event_id, item.source_rainfall_key))
        current: list[RainfallSourceEvent] = []
        current_end: date | None = None
        current_event_ids: set[str] = set()
        for event in county_events:
            if not current:
                current = [event]
                current_end = event.rain_end
                current_event_ids = {event.rain_event_id}
                continue
            assert current_end is not None
            same_event = bool(event.rain_event_id and event.rain_event_id in current_event_ids)
            close_window = event.rain_start <= current_end + timedelta(days=close_gap_days)
            if same_event or close_window:
                current.append(event)
                current_end = max(current_end, event.rain_end)
                current_event_ids.add(event.rain_event_id)
            else:
                episodes.append(_rainfall_episode_from_group(current, pilot_config))
                current = [event]
                current_end = event.rain_end
                current_event_ids = {event.rain_event_id}
        if current:
            episodes.append(_rainfall_episode_from_group(current, pilot_config))
    episodes.sort(key=lambda item: (item.fips, item.rain_start, item.rain_end, item.rainfall_episode_id))
    return episodes


def _rainfall_episode_from_group(group: list[RainfallSourceEvent], pilot_config: PilotConfig) -> CountyRainfallEpisode:
    first = group[0]
    start = min(item.rain_start for item in group)
    end = max(item.rain_end for item in group)
    source_keys = sorted({item.source_rainfall_key for item in group})
    event_ids = sorted({item.rain_event_id for item in group if item.rain_event_id})
    event_types = sorted({item.rain_event_type for item in group if item.rain_event_type})
    grid_coords = sorted({item.grid_coord_key for item in group})
    episode_id = "rain_ep_" + stable_hash_payload(
        [
            pilot_config.rainfall_grouping_policy_version,
            first.fips,
            start.isoformat(),
            end.isoformat(),
            event_ids,
            source_keys,
        ],
        length=16,
    )
    return CountyRainfallEpisode(
        schema_version=SCHEMA_VERSION,
        rainfall_episode_id=episode_id,
        state=first.state or pilot_config.state,
        county=first.county or "",
        fips=first.fips or "",
        rain_start=start,
        rain_end=end,
        rain_duration_days=(end - start).days + 1,
        rain_event_ids=event_ids,
        rain_event_types=event_types,
        source_grid_count=len(grid_coords),
        source_grid_coords=grid_coords,
        source_rainfall_keys=source_keys,
        rainfall_threshold=pilot_config.rainfall_threshold,
        rainfall_grouping_policy_version=pilot_config.rainfall_grouping_policy_version,
    )


def classify_temporal_match(
    drought: CountyDroughtEpisode,
    rainfall: CountyRainfallEpisode,
    pilot_config: PilotConfig,
) -> tuple[str, int, str, str]:
    if drought.drought_end_month_start <= rainfall.rain_start <= drought.drought_end_month_end:
        lag_days = (rainfall.rain_start - drought.drought_end_month_end).days
        return "final_month_ambiguous", lag_days, "final_month_ambiguous", "temporal_ambiguous_candidate"
    lag_days = (rainfall.rain_start - drought.drought_end_month_end).days
    if lag_days < pilot_config.primary_lag_min_days:
        return "temporal_mismatch_before_or_during_drought", lag_days, "out_of_policy", "out_of_policy_candidate"
    if lag_days > pilot_config.primary_lag_max_days:
        return "temporal_mismatch_lag_too_long", lag_days, "out_of_policy", "out_of_policy_candidate"
    if lag_days <= 7:
        lag_bin = "1_7"
    elif lag_days <= 30:
        lag_bin = "8_30"
    elif lag_days <= 60:
        lag_bin = "31_60"
    else:
        lag_bin = "61_90"
    return "primary_post_drought", lag_days, lag_bin, "raw_climate_candidate"


def _coord_tuple(coord: str) -> tuple[float, float] | None:
    try:
        lat_text, lon_text = coord.split(",", 1)
        return float(lat_text), float(lon_text)
    except ValueError:
        return None


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = left
    lat2, lon2 = right
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def classify_spatial_match(
    drought: CountyDroughtEpisode,
    rainfall: CountyRainfallEpisode,
    *,
    near_grid_threshold_km: float = 75.0,
) -> tuple[str, float | None]:
    drought_coords = set(drought.source_grid_coords)
    rainfall_coords = set(rainfall.source_grid_coords)
    if drought_coords & rainfall_coords:
        return "same_grid", 0.0
    distances: list[float] = []
    for left_text in drought_coords:
        left = _coord_tuple(left_text)
        if left is None:
            continue
        for right_text in rainfall_coords:
            right = _coord_tuple(right_text)
            if right is None:
                continue
            distances.append(haversine_km(left, right))
    min_distance = min(distances) if distances else None
    if min_distance is not None and min_distance <= near_grid_threshold_km:
        return "same_county_near_grid", min_distance
    return "same_county_only", min_distance


def make_raw_candidate(
    drought: CountyDroughtEpisode,
    rainfall: CountyRainfallEpisode,
    pilot_config: PilotConfig,
    *,
    temporal_match_status: str,
    lag_days: int,
    lag_bin: str,
    candidate_status: str,
) -> RawCECandidate:
    spatial_level, min_distance = classify_spatial_match(
        drought,
        rainfall,
        near_grid_threshold_km=pilot_config.near_grid_threshold_km,
    )
    raw_candidate_id = stable_raw_candidate_id(
        policy_version=pilot_config.policy_version,
        fips=drought.fips,
        drought_episode_id=drought.drought_episode_id,
        rainfall_episode_id=rainfall.rainfall_episode_id,
        temporal_policy_version=pilot_config.temporal_policy_version,
        spatial_policy_version=pilot_config.spatial_policy_version,
    )
    candidate_key = [
        pilot_config.policy_version,
        drought.fips,
        drought.drought_episode_id,
        rainfall.rainfall_episode_id,
        pilot_config.temporal_policy_version,
        pilot_config.spatial_policy_version,
    ]
    source_hash = stable_hash_payload([drought.source_drought_keys, rainfall.source_rainfall_keys], length=16)
    return RawCECandidate(
        schema_version=SCHEMA_VERSION,
        raw_candidate_id=raw_candidate_id,
        state=drought.state,
        county=drought.county,
        fips=drought.fips,
        drought_episode_id=drought.drought_episode_id,
        rainfall_episode_id=rainfall.rainfall_episode_id,
        drought_start_month=drought.start_month,
        drought_end_month=drought.end_month,
        drought_end_month_end=drought.drought_end_month_end,
        rain_start=rainfall.rain_start,
        rain_end=rainfall.rain_end,
        lag_days=lag_days,
        lag_bin=lag_bin,
        temporal_match_status=temporal_match_status,
        spatial_match_level=spatial_level,
        min_grid_distance_km=round(min_distance, 3) if min_distance is not None else None,
        candidate_status=candidate_status,
        drought_source_keys=drought.source_drought_keys,
        rainfall_source_keys=rainfall.source_rainfall_keys,
        rain_event_ids=rainfall.rain_event_ids,
        rain_event_types=rainfall.rain_event_types,
        drought_coverage_class=drought.drought_coverage_class,
        wmo_severity_max=drought.wmo_severity_max,
        policy_version=pilot_config.policy_version,
        temporal_policy_version=pilot_config.temporal_policy_version,
        spatial_policy_version=pilot_config.spatial_policy_version,
        candidate_id_policy_version=pilot_config.candidate_id_policy_version,
        dedup_policy_version=pilot_config.deduplication_policy_version,
        candidate_key_hash=stable_hash_payload(candidate_key, length=16),
        source_hash=source_hash,
    )


def build_raw_candidates(
    drought_episodes: list[CountyDroughtEpisode],
    rainfall_episodes: list[CountyRainfallEpisode],
    pilot_config: PilotConfig,
) -> tuple[list[RawCECandidate], list[RawCECandidate], dict[str, Any]]:
    rainfall_by_fips: dict[str, list[CountyRainfallEpisode]] = defaultdict(list)
    for rainfall in rainfall_episodes:
        rainfall_by_fips[rainfall.fips].append(rainfall)

    raw_candidates: list[RawCECandidate] = []
    ambiguous_candidates: list[RawCECandidate] = []
    temporal_counts: Counter[str] = Counter()
    lag_bin_counts: Counter[str] = Counter()
    same_fips_pair_count = 0
    primary_pairs_by_drought: Counter[str] = Counter()
    primary_pairs_by_rainfall: Counter[str] = Counter()

    for drought in drought_episodes:
        for rainfall in rainfall_by_fips.get(drought.fips, []):
            same_fips_pair_count += 1
            status, lag_days, lag_bin, candidate_status = classify_temporal_match(drought, rainfall, pilot_config)
            temporal_counts[status] += 1
            lag_bin_counts[lag_bin] += 1
            if status == "primary_post_drought":
                candidate = make_raw_candidate(
                    drought,
                    rainfall,
                    pilot_config,
                    temporal_match_status=status,
                    lag_days=lag_days,
                    lag_bin=lag_bin,
                    candidate_status=candidate_status,
                )
                raw_candidates.append(candidate)
                primary_pairs_by_drought[drought.drought_episode_id] += 1
                primary_pairs_by_rainfall[rainfall.rainfall_episode_id] += 1
            elif status == "final_month_ambiguous":
                ambiguous_candidates.append(
                    make_raw_candidate(
                        drought,
                        rainfall,
                        pilot_config,
                        temporal_match_status=status,
                        lag_days=lag_days,
                        lag_bin=lag_bin,
                        candidate_status=candidate_status,
                    )
                )

    raw_candidates.sort(key=lambda item: (item.fips, item.rain_start, item.drought_end_month_end, item.raw_candidate_id))
    ambiguous_candidates.sort(key=lambda item: (item.fips, item.rain_start, item.drought_end_month_end, item.raw_candidate_id))
    diagnostics = {
        "same_fips_drought_rainfall_episode_pairs_evaluated": same_fips_pair_count,
        "temporal_match_status_counts": dict(sorted(temporal_counts.items())),
        "lag_bin_counts_all_same_fips_pairs": dict(sorted(lag_bin_counts.items())),
        "max_primary_pairs_per_drought_episode": max(primary_pairs_by_drought.values()) if primary_pairs_by_drought else 0,
        "max_primary_pairs_per_rainfall_episode": max(primary_pairs_by_rainfall.values()) if primary_pairs_by_rainfall else 0,
        "primary_pairs_by_drought_episode_top": primary_pairs_by_drought.most_common(10),
        "primary_pairs_by_rainfall_episode_top": primary_pairs_by_rainfall.most_common(10),
    }
    return raw_candidates, ambiguous_candidates, diagnostics


def _representative_sort_key(candidate: RawCECandidate) -> tuple[int, float, int, int, str]:
    distance = candidate.min_grid_distance_km if candidate.min_grid_distance_km is not None else float("inf")
    return (
        -SPATIAL_PRIORITY.get(candidate.spatial_match_level, 0),
        distance,
        -COVERAGE_PRIORITY.get(candidate.drought_coverage_class, 0),
        -_severity_rank(candidate.wmo_severity_max),
        candidate.source_hash,
    )


def choose_dedup_representative(candidates: list[RawCECandidate]) -> RawCECandidate:
    return sorted(candidates, key=_representative_sort_key)[0]


def deduplicate_candidates(candidates: list[RawCECandidate]) -> tuple[list[RawCECandidate], dict[str, Any]]:
    exact: dict[tuple[str, str, str, str, str, str], RawCECandidate] = {}
    exact_duplicate_counts: Counter[str] = Counter()
    for candidate in candidates:
        key = (
            candidate.policy_version,
            candidate.fips,
            candidate.drought_episode_id,
            candidate.rainfall_episode_id,
            candidate.temporal_policy_version,
            candidate.spatial_policy_version,
        )
        if key in exact:
            exact_duplicate_counts[exact[key].raw_candidate_id] += 1
            existing = exact[key]
            collapsed = sorted({*existing.collapsed_candidate_ids, candidate.raw_candidate_id})
            exact[key] = replace(
                existing,
                collapsed_candidate_ids=collapsed,
                dedup_reason="exact_candidate_key_duplicate",
            )
        else:
            exact[key] = candidate

    near_deduped: list[RawCECandidate] = []
    near_duplicate_groups = 0
    buckets: dict[tuple[str, str, str], list[RawCECandidate]] = defaultdict(list)
    for candidate in exact.values():
        buckets[(candidate.fips, candidate.rainfall_episode_id, candidate.lag_bin)].append(candidate)

    for _bucket, bucket_candidates in sorted(buckets.items()):
        bucket_candidates.sort(key=lambda item: (item.drought_end_month_end, item.raw_candidate_id))
        cluster: list[RawCECandidate] = []
        cluster_anchor: date | None = None
        for candidate in bucket_candidates:
            if not cluster:
                cluster = [candidate]
                cluster_anchor = candidate.drought_end_month_end
                continue
            assert cluster_anchor is not None
            if abs((candidate.drought_end_month_end - cluster_anchor).days) <= 31:
                cluster.append(candidate)
            else:
                near_deduped.append(_collapse_cluster(cluster))
                if len(cluster) > 1:
                    near_duplicate_groups += 1
                cluster = [candidate]
                cluster_anchor = candidate.drought_end_month_end
        if cluster:
            near_deduped.append(_collapse_cluster(cluster))
            if len(cluster) > 1:
                near_duplicate_groups += 1

    near_deduped.sort(key=lambda item: (item.fips, item.rain_start, item.drought_end_month_end, item.raw_candidate_id))
    duplicate_removed_count = len(candidates) - len(near_deduped)
    diagnostics = {
        "pre_dedup_candidate_count": len(candidates),
        "post_dedup_candidate_count": len(near_deduped),
        "exact_duplicate_representatives": dict(sorted(exact_duplicate_counts.items())),
        "near_duplicate_group_count": near_duplicate_groups,
        "duplicate_removed_count": duplicate_removed_count,
        "deduplication_ratio": round(len(candidates) / len(near_deduped), 6) if near_deduped else None,
    }
    return near_deduped, diagnostics


def _collapse_cluster(cluster: list[RawCECandidate]) -> RawCECandidate:
    if len(cluster) == 1:
        return cluster[0]
    representative = choose_dedup_representative(cluster)
    collapsed_ids = sorted(candidate.raw_candidate_id for candidate in cluster if candidate.raw_candidate_id != representative.raw_candidate_id)
    reason = "same_fips_same_rainfall_endmonth_within_31d_same_lag_bin"
    if representative.collapsed_candidate_ids:
        collapsed_ids = sorted({*collapsed_ids, *representative.collapsed_candidate_ids})
    return replace(
        representative,
        collapsed_candidate_ids=collapsed_ids,
        dedup_reason=reason,
    )


def construct_raw_candidates(pilot_config: PilotConfig) -> ConstructionResult:
    paths = output_paths(pilot_config.output_dir)
    rainfall_rows, rainfall_read_diagnostics = read_rainfall_rows(pilot_config.rainfall_input_path, pilot_config)
    drought_rows, drought_read_diagnostics = read_drought_rows(pilot_config.spei3_input_path)
    mapping_lookup, mapping_diagnostics = build_mapping_lookup_for_rows(drought_rows, rainfall_rows, pilot_config)
    drought_events, drought_mapping_diagnostics = build_drought_source_events(drought_rows, mapping_lookup, pilot_config)
    rainfall_events, rainfall_mapping_diagnostics = build_rainfall_source_events(rainfall_rows, mapping_lookup, pilot_config)
    drought_episodes = group_drought_episodes(drought_events, pilot_config)
    rainfall_episodes = group_rainfall_episodes(rainfall_events, pilot_config)
    raw_candidates, ambiguous_candidates, matching_diagnostics = build_raw_candidates(drought_episodes, rainfall_episodes, pilot_config)
    deduped_candidates, dedup_diagnostics = deduplicate_candidates(raw_candidates)

    spatial_counts_pre = Counter(candidate.spatial_match_level for candidate in raw_candidates)
    spatial_counts_post = Counter(candidate.spatial_match_level for candidate in deduped_candidates)
    lag_counts_pre = Counter(candidate.lag_bin for candidate in raw_candidates)
    lag_counts_post = Counter(candidate.lag_bin for candidate in deduped_candidates)
    coverage_counts = Counter(episode.drought_coverage_class for episode in drought_episodes)

    summary: dict[str, Any] = {
        "pilot_name": pilot_config.pilot_name,
        "state": pilot_config.state,
        "rainfall_start_date": pilot_config.rainfall_start_date.isoformat(),
        "rainfall_end_date": pilot_config.rainfall_end_date.isoformat(),
        "rainfall_threshold": pilot_config.rainfall_threshold,
        "policy_version": pilot_config.policy_version,
        "temporal_policy_version": pilot_config.temporal_policy_version,
        "spatial_policy_version": pilot_config.spatial_policy_version,
        "candidate_id_policy_version": pilot_config.candidate_id_policy_version,
        "deduplication_policy_version": pilot_config.deduplication_policy_version,
        "drought_grouping_policy_version": pilot_config.drought_grouping_policy_version,
        "rainfall_grouping_policy_version": pilot_config.rainfall_grouping_policy_version,
        "near_grid_threshold_km": pilot_config.near_grid_threshold_km,
        "primary_lag_window_days": [pilot_config.primary_lag_min_days, pilot_config.primary_lag_max_days],
        "drought_rows_read": dict(sorted(drought_read_diagnostics.items())),
        "rainfall_rows_read": dict(sorted(rainfall_read_diagnostics.items())),
        "mapping": mapping_diagnostics,
        "drought_mapping": dict(sorted(drought_mapping_diagnostics.items())),
        "rainfall_mapping": dict(sorted(rainfall_mapping_diagnostics.items())),
        "mapped_spei3_drought_source_rows": len(drought_events),
        "mapped_p99_rainfall_source_rows": len(rainfall_events),
        "county_drought_episode_count": len(drought_episodes),
        "county_rainfall_episode_count": len(rainfall_episodes),
        "candidate_pairs_before_deduplication": len(raw_candidates),
        "candidate_pairs_after_deduplication": len(deduped_candidates),
        "temporal_ambiguous_candidate_count": len(ambiguous_candidates),
        "counts_by_lag_bin_pre_dedup": dict(sorted(lag_counts_pre.items())),
        "counts_by_lag_bin_post_dedup": dict(sorted(lag_counts_post.items())),
        "counts_by_temporal_match_status": matching_diagnostics["temporal_match_status_counts"],
        "counts_by_spatial_match_level_pre_dedup": dict(sorted(spatial_counts_pre.items())),
        "counts_by_spatial_match_level_post_dedup": dict(sorted(spatial_counts_post.items())),
        "counts_by_drought_coverage_class": dict(sorted(coverage_counts.items())),
        "deduplication": dedup_diagnostics,
        "matching_diagnostics": matching_diagnostics,
        "construction_boundary": {
            "comparison_dataset_used_before_alignment": False,
            "external_evidence_stage_run": False,
            "component_validation_run": False,
            "pair_level_evidence_stage_run": False,
            "impact_extraction_run": False,
        },
    }
    return ConstructionResult(
        drought_source_events=drought_events,
        rainfall_source_events=rainfall_events,
        drought_episodes=drought_episodes,
        rainfall_episodes=rainfall_episodes,
        raw_candidates=raw_candidates,
        deduped_candidates=deduped_candidates,
        temporal_ambiguous_candidates=ambiguous_candidates,
        summary=summary,
        paths=paths,
    )


def read_dter_rows(path: Path, pilot_config: PilotConfig) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    diagnostics: Counter[str] = Counter()
    for row_index, row in iter_csv_rows(path):
        diagnostics["rows_read"] += 1
        rain_start = parse_day(row.get("rain_start"))
        rain_end = parse_day(row.get("rain_end"))
        if rain_start is None or rain_end is None:
            diagnostics["skipped_invalid_rain_dates"] += 1
            continue
        if not pilot_config.rainfall_start_date <= rain_start <= pilot_config.rainfall_end_date:
            diagnostics["skipped_outside_rainfall_date_scope"] += 1
            continue
        coord = _row_coord(row)
        if coord is None:
            diagnostics["skipped_invalid_coordinate"] += 1
            continue
        row["_source_row_index"] = str(row_index)
        rows.append(row)
        diagnostics["rows_in_rainfall_date_scope"] += 1
    return rows, diagnostics


def build_dter_mapping_lookup(
    dter_rows: list[dict[str, str]],
    pilot_config: PilotConfig,
) -> tuple[dict[str, MappingResult], dict[str, Any]]:
    coords = {_row_coord(row) for row in dter_rows if _row_coord(row) is not None}
    typed_coords = {coord for coord in coords if coord is not None}
    boundary_path = pilot_config.county_boundary_path or find_default_county_boundary_file()
    lookup, blocker, metadata = build_grid_to_county_lookup(typed_coords, boundary_file=boundary_path)
    return lookup, {
        "dter_unique_coordinate_count": len(typed_coords),
        "dter_mapping_blocker": blocker,
        "county_boundary_path": str(boundary_path) if boundary_path else None,
        "boundary_metadata": boundary_metadata_for_pilot_summary(metadata),
    }


def build_dter_source_rows(
    rows: list[dict[str, str]],
    mapping_lookup: dict[str, MappingResult],
    pilot_config: PilotConfig,
) -> tuple[list[DTERSourceRow], Counter[str]]:
    dter_rows: list[DTERSourceRow] = []
    diagnostics: Counter[str] = Counter()
    for row in rows:
        mapping = _mapping_for_row(row, mapping_lookup)
        mapping_status = mapping.mapping_status if mapping else "mapping_missing"
        diagnostics[f"mapping_status:{mapping_status}"] += 1
        if mapping is None or mapping.mapping_status != "mapped_us_land_county":
            continue
        if mapping.state != pilot_config.state or mapping.state_fips != pilot_config.state_fips:
            diagnostics["mapped_outside_target_state"] += 1
            continue
        rain_start = parse_day(row.get("rain_start"))
        rain_end = parse_day(row.get("rain_end"))
        drought_start = parse_day(row.get("drought_start"))
        drought_end = parse_day(row.get("drought_end"))
        drought_end_month_end = parse_day(row.get("drought_end_month_end"))
        if None in {rain_start, rain_end, drought_start, drought_end}:
            diagnostics["skipped_invalid_dates_after_mapping"] += 1
            continue
        assert rain_start is not None and rain_end is not None and drought_start is not None and drought_end is not None
        if drought_end_month_end is None:
            drought_end_month_end = month_end(drought_end.year, drought_end.month)
        dter_rows.append(
            DTERSourceRow(
                dter_event_id=str(row.get("dtp_event_id") or f"row_{row.get('_source_row_index')}").strip(),
                source_row_index=int(row.get("_source_row_index") or 0),
                source_drought_key=make_drought_source_key(row, severity_field="drought_severity"),
                rain_event_id=str(row.get("rain_event_id") or "").strip(),
                rain_event_type=str(row.get("rain_event_type") or "").strip() or None,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                rain_start=rain_start,
                rain_end=rain_end,
                drought_start_date=month_start(drought_start),
                drought_end_date=drought_end,
                drought_end_month_end=drought_end_month_end,
                state=mapping.state or pilot_config.state,
                county=mapping.county or "",
                fips=mapping.county_fips or "",
                mapping_status=mapping.mapping_status,
            )
        )
        diagnostics["target_state_mapped_rows"] += 1
    dter_rows.sort(key=lambda item: (item.fips, item.rain_start, item.drought_end_month_end, item.dter_event_id))
    return dter_rows, diagnostics


def windows_overlap(left_start: date, left_end: date, right_start: date, right_end: date) -> bool:
    return left_start <= right_end and right_start <= left_end


def classify_dter_alignment(raw: RawCECandidate, dter: DTERSourceRow) -> tuple[str | None, str | None, float | None]:
    if raw.fips != dter.fips:
        return None, None, None
    dter_drought_in_raw = dter.source_drought_key in set(raw.drought_source_keys)
    dter_rain_in_raw = dter.rain_event_id in set(raw.rain_event_ids)
    if dter_drought_in_raw and dter_rain_in_raw:
        return "exact_source_pair_match", "same FIPS, source drought key, and rain event id", 1.0
    if dter_rain_in_raw:
        raw_drought_start = datetime.strptime(raw.drought_start_month + "-01", "%Y-%m-%d").date()
        if raw_drought_start <= dter.drought_start_date and dter.drought_end_month_end <= raw.drought_end_month_end:
            return "exact_rain_drought_episode_match", "same FIPS and rain event id with DTER drought row inside raw drought episode", 0.9
    if windows_overlap(raw.rain_start, raw.rain_end, dter.rain_start, dter.rain_end):
        end_delta = abs((dter.drought_end_month_end - raw.drought_end_month_end).days)
        if end_delta <= 31:
            return "spatiotemporal_window_match", "same FIPS, rainfall windows overlap, and drought end month within 31 days", 0.7
    if dter_drought_in_raw or dter_rain_in_raw:
        return "component_only_match", "same FIPS and one exact component matched but the pair did not meet candidate-level tiers", 0.4
    return None, None, None


def make_alignment_record(
    *,
    raw: RawCECandidate | None,
    dter: DTERSourceRow | None,
    tier: str,
    category: str,
    reason: str,
    match_score: float | None,
) -> DTERAlignmentRecord:
    state = (raw.state if raw else None) or (dter.state if dter else "")
    county = (raw.county if raw else None) or (dter.county if dter else None)
    fips = (raw.fips if raw else None) or (dter.fips if dter else None)
    record_id = "dter_align_" + stable_hash_payload(
        [
            raw.raw_candidate_id if raw else None,
            dter.dter_event_id if dter else None,
            tier,
            category,
        ],
        length=16,
    )
    review_priority = "high" if tier in {"exact_source_pair_match", "exact_rain_drought_episode_match"} else "medium"
    if category in {"raw_only", "dter_only"}:
        review_priority = "low"
    return DTERAlignmentRecord(
        schema_version=SCHEMA_VERSION,
        alignment_record_id=record_id,
        raw_candidate_id=raw.raw_candidate_id if raw else None,
        dter_event_id=dter.dter_event_id if dter else None,
        alignment_tier=tier,
        alignment_category=category,
        state=state,
        county=county,
        fips=fips,
        raw_drought_episode_id=raw.drought_episode_id if raw else None,
        raw_rainfall_episode_id=raw.rainfall_episode_id if raw else None,
        dter_drought_key=dter.source_drought_key if dter else None,
        dter_rain_event_id=dter.rain_event_id if dter else None,
        raw_rain_event_ids=raw.rain_event_ids if raw else [],
        raw_drought_end_month_end=raw.drought_end_month_end if raw else None,
        dter_drought_end_month_end=dter.drought_end_month_end if dter else None,
        raw_rain_start=raw.rain_start if raw else None,
        dter_rain_start=dter.rain_start if dter else None,
        alignment_reason=reason,
        match_score=match_score,
        review_priority=review_priority,
    )


def align_dter_to_raw_candidates(
    raw_candidates: list[RawCECandidate],
    pilot_config: PilotConfig,
) -> DTERAlignmentResult:
    dter_raw_rows, dter_read_diagnostics = read_dter_rows(pilot_config.dter_input_path, pilot_config)
    dter_mapping_lookup, dter_mapping_diagnostics = build_dter_mapping_lookup(dter_raw_rows, pilot_config)
    dter_rows, dter_mapping_counts = build_dter_source_rows(dter_raw_rows, dter_mapping_lookup, pilot_config)

    raw_by_fips: dict[str, list[RawCECandidate]] = defaultdict(list)
    for raw in raw_candidates:
        raw_by_fips[raw.fips].append(raw)

    pair_records: list[DTERAlignmentRecord] = []
    component_records: list[DTERAlignmentRecord] = []
    raw_candidate_level_matches: set[str] = set()
    dter_candidate_level_matches: set[str] = set()
    raw_component_matches: set[str] = set()
    dter_component_matches: set[str] = set()

    for dter in dter_rows:
        for raw in raw_by_fips.get(dter.fips, []):
            tier, reason, score = classify_dter_alignment(raw, dter)
            if tier is None:
                continue
            if tier == "component_only_match":
                component_records.append(
                    make_alignment_record(
                        raw=raw,
                        dter=dter,
                        tier=tier,
                        category="component_only_overlap",
                        reason=reason or "",
                        match_score=score,
                    )
                )
                raw_component_matches.add(raw.raw_candidate_id)
                dter_component_matches.add(dter.dter_event_id)
                continue
            pair_records.append(
                make_alignment_record(
                    raw=raw,
                    dter=dter,
                    tier=tier,
                    category="dter_overlap",
                    reason=reason or "",
                    match_score=score,
                )
            )
            raw_candidate_level_matches.add(raw.raw_candidate_id)
            dter_candidate_level_matches.add(dter.dter_event_id)

    raw_only_records: list[DTERAlignmentRecord] = []
    for raw in raw_candidates:
        if raw.raw_candidate_id in raw_candidate_level_matches or raw.raw_candidate_id in raw_component_matches:
            continue
        raw_only_records.append(
            make_alignment_record(
                raw=raw,
                dter=None,
                tier="no_match",
                category="raw_only",
                reason="no DTER candidate-level or exact component overlap found after raw candidates were frozen",
                match_score=0.0,
            )
        )

    dter_only_records: list[DTERAlignmentRecord] = []
    for dter in dter_rows:
        if dter.dter_event_id in dter_candidate_level_matches or dter.dter_event_id in dter_component_matches:
            continue
        dter_only_records.append(
            make_alignment_record(
                raw=None,
                dter=dter,
                tier="no_match",
                category="dter_only",
                reason="no raw candidate-level or exact component overlap found",
                match_score=0.0,
            )
        )

    records = [*pair_records, *component_records, *raw_only_records, *dter_only_records]
    records.sort(
        key=lambda item: (
            TIER_PRIORITY.get(item.alignment_tier, 9),
            item.fips or "",
            item.raw_candidate_id or "",
            item.dter_event_id or "",
            item.alignment_record_id,
        )
    )
    tier_counts = Counter(record.alignment_tier for record in pair_records)
    category_counts = Counter(record.alignment_category for record in records)
    raw_to_dter: dict[str, set[str]] = defaultdict(set)
    dter_to_raw: dict[str, set[str]] = defaultdict(set)
    for record in pair_records:
        if record.raw_candidate_id and record.dter_event_id:
            raw_to_dter[record.raw_candidate_id].add(record.dter_event_id)
            dter_to_raw[record.dter_event_id].add(record.raw_candidate_id)
    summary = {
        "dter_rows_read": dict(sorted(dter_read_diagnostics.items())),
        "dter_mapping": dter_mapping_diagnostics,
        "dter_mapping_counts": dict(sorted(dter_mapping_counts.items())),
        "dter_rows_in_scope": len(dter_rows),
        "alignment_record_count": len(records),
        "dter_candidate_level_pair_count": len(pair_records),
        "dter_overlap_count_by_tier": dict(sorted(tier_counts.items())),
        "alignment_category_counts": dict(sorted(category_counts.items())),
        "raw_candidates_with_dter_overlap": len(raw_candidate_level_matches),
        "dter_rows_recovered_by_raw_candidate": len(dter_candidate_level_matches),
        "raw_only_count": len(raw_only_records),
        "dter_only_count": len(dter_only_records),
        "component_only_overlap_count": len(component_records),
        "one_to_many_match_count": sum(1 for values in raw_to_dter.values() if len(values) > 1),
        "many_to_one_match_count": sum(1 for values in dter_to_raw.values() if len(values) > 1),
    }
    return DTERAlignmentResult(
        dter_rows=dter_rows,
        alignment_records=records,
        dter_only_records=dter_only_records,
        summary=summary,
    )


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "county_drought_episodes": output_dir / "county_drought_episodes.csv",
        "county_rainfall_episodes": output_dir / "county_rainfall_episodes.csv",
        "raw_ce_candidates": output_dir / "raw_ce_candidates.csv",
        "raw_ce_candidates_deduped": output_dir / "raw_ce_candidates_deduped.csv",
        "dter_alignment": output_dir / "dter_alignment.csv",
        "dter_only": output_dir / "dter_only.csv",
        "review_sample_dter_overlap": output_dir / "review_sample_dter_overlap.csv",
        "review_sample_raw_only": output_dir / "review_sample_raw_only.csv",
        "review_sample_dter_only": output_dir / "review_sample_dter_only.csv",
        "candidate_construction_summary": output_dir / "candidate_construction_summary.json",
        "candidate_construction_report": output_dir / "candidate_construction_report.md",
        "temporal_ambiguous_candidates": output_dir / "temporal_ambiguous_candidates.csv",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
    return str(value)


def write_dataclass_csv(path: Path, records: Iterable[Any], fieldnames: list[str] | None = None) -> None:
    rows = list(records)
    if fieldnames is None:
        if rows:
            fieldnames = list(asdict(rows[0]).keys())
        else:
            fieldnames = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            data = asdict(record)
            writer.writerow({field: _csv_value(data.get(field)) for field in fieldnames})


def select_review_sample(records: Iterable[DTERAlignmentRecord], *, category: str, limit: int = 25) -> list[DTERAlignmentRecord]:
    filtered = [record for record in records if record.alignment_category == category]
    filtered.sort(
        key=lambda item: (
            TIER_PRIORITY.get(item.alignment_tier, 9),
            item.fips or "",
            item.raw_rain_start or date.min,
            item.dter_rain_start or date.min,
            item.raw_candidate_id or "",
            item.dter_event_id or "",
        )
    )
    return filtered[:limit]


def compute_red_flags(summary: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    dter_count = int(summary.get("dter_rows_in_scope") or 0)
    deduped = int(summary.get("candidate_pairs_after_deduplication") or 0)
    if dter_count > 0 and deduped > 20 * dter_count:
        flags.append("Deduped raw candidate count is more than 20x the in-scope DTER row count.")
    elif dter_count > 0 and deduped > 10 * dter_count:
        flags.append("Deduped raw candidate count is more than 10x the in-scope DTER row count.")

    spatial = summary.get("counts_by_spatial_match_level_post_dedup") or {}
    total_spatial = sum(int(value) for value in spatial.values())
    if total_spatial and int(spatial.get("same_county_only", 0)) > total_spatial / 2:
        flags.append("Most deduped raw candidates are same_county_only.")

    lag_counts = summary.get("counts_by_lag_bin_post_dedup") or {}
    total_lag = sum(int(value) for value in lag_counts.values())
    if total_lag and int(lag_counts.get("61_90", 0)) > total_lag / 2:
        flags.append("Most deduped raw candidates are in the 61_90 lag bin.")

    temporal = summary.get("counts_by_temporal_match_status") or {}
    temporal_total = sum(int(value) for value in temporal.values())
    if temporal_total and int(temporal.get("final_month_ambiguous", 0)) > temporal_total / 2:
        flags.append("Final-month ambiguous cases dominate same-FIPS temporal diagnostics.")

    matching = summary.get("matching_diagnostics") or {}
    if int(matching.get("max_primary_pairs_per_drought_episode") or 0) >= 100:
        flags.append("At least one drought episode pairs with 100 or more rainfall episodes.")
    if int(matching.get("max_primary_pairs_per_rainfall_episode") or 0) >= 10:
        flags.append("At least one rainfall episode pairs with many drought episodes.")

    overlap = int(summary.get("raw_candidates_with_dter_overlap") or 0)
    if dter_count and overlap == 0:
        flags.append("DTER overlap is zero at candidate-level tiers.")
    if dter_count and overlap >= int(0.95 * max(deduped, 1)) and deduped:
        flags.append("DTER overlap is near total; inspect for accidental pre-alignment key leakage.")

    drought_mapping = summary.get("drought_mapping") or {}
    rainfall_mapping = summary.get("rainfall_mapping") or {}
    if int(drought_mapping.get("target_state_mapped_rows") or 0) == 0:
        flags.append("No mapped Texas drought rows were available.")
    if int(rainfall_mapping.get("target_state_mapped_rows") or 0) == 0:
        flags.append("No mapped Texas p99 extreme-rainfall rows were available.")

    if not flags:
        flags.append("No automatic red flags triggered.")
    return flags


def render_report(summary: dict[str, Any], paths: dict[str, Path], red_flags: list[str]) -> str:
    lines = [
        "# Raw CE Pilot Candidate Construction Report",
        "",
        "## Scope",
        "",
        f"- State: {summary['state']}",
        f"- Rainfall start date range: {summary['rainfall_start_date']} to {summary['rainfall_end_date']}",
        f"- Rainfall threshold: {summary['rainfall_threshold']}",
        f"- Policy version: `{summary['policy_version']}`",
        f"- Temporal policy: `{summary['temporal_policy_version']}`",
        f"- Spatial policy: `{summary['spatial_policy_version']}`",
        f"- Candidate ID policy: `{summary['candidate_id_policy_version']}`",
        f"- De-duplication policy: `{summary['deduplication_policy_version']}`",
        "",
        "## Construction Counts",
        "",
        f"- Mapped SPEI3 drought source rows: {summary['mapped_spei3_drought_source_rows']}",
        f"- Mapped p99 extreme-rainfall source rows: {summary['mapped_p99_rainfall_source_rows']}",
        f"- CountyDroughtEpisodes: {summary['county_drought_episode_count']}",
        f"- CountyRainfallEpisodes: {summary['county_rainfall_episode_count']}",
        f"- Candidate pairs before de-duplication: {summary['candidate_pairs_before_deduplication']}",
        f"- Candidate pairs after de-duplication: {summary['candidate_pairs_after_deduplication']}",
        f"- De-duplication ratio: {summary['deduplication'].get('deduplication_ratio')}",
        f"- Temporal ambiguous candidates stored separately: {summary['temporal_ambiguous_candidate_count']}",
        "",
        "## Diagnostics",
        "",
        f"- Counts by lag_bin before de-duplication: `{summary['counts_by_lag_bin_pre_dedup']}`",
        f"- Counts by lag_bin after de-duplication: `{summary['counts_by_lag_bin_post_dedup']}`",
        f"- Counts by temporal_match_status: `{summary['counts_by_temporal_match_status']}`",
        f"- Counts by spatial_match_level before de-duplication: `{summary['counts_by_spatial_match_level_pre_dedup']}`",
        f"- Counts by spatial_match_level after de-duplication: `{summary['counts_by_spatial_match_level_post_dedup']}`",
        f"- Counts by drought_coverage_class: `{summary['counts_by_drought_coverage_class']}`",
        f"- Same-FIPS pairs evaluated for temporal diagnostics: {summary['matching_diagnostics']['same_fips_drought_rainfall_episode_pairs_evaluated']}",
        "",
        "## DTER Alignment",
        "",
        f"- DTER rows in scope: {summary.get('dter_rows_in_scope', 0)}",
        f"- DTER overlap count by tier: `{summary.get('dter_overlap_count_by_tier', {})}`",
        f"- Raw-only count: {summary.get('raw_only_count', 0)}",
        f"- DTER-only count: {summary.get('dter_only_count', 0)}",
        f"- Component-only overlap count: {summary.get('component_only_overlap_count', 0)}",
        f"- One-to-many match count: {summary.get('one_to_many_match_count', 0)}",
        f"- Many-to-one match count: {summary.get('many_to_one_match_count', 0)}",
        "",
        "## Review Samples",
        "",
        f"- DTER overlap sample: `{paths['review_sample_dter_overlap']}`",
        f"- Raw-only sample: `{paths['review_sample_raw_only']}`",
        f"- DTER-only sample: `{paths['review_sample_dter_only']}`",
        "",
        "## Guardrail Status",
        "",
        f"- Comparison dataset used before alignment: {summary['construction_boundary']['comparison_dataset_used_before_alignment']}",
        f"- External evidence stage run: {summary['construction_boundary']['external_evidence_stage_run']}",
        f"- Component validation run: {summary['construction_boundary']['component_validation_run']}",
        f"- Pair-level evidence stage run: {summary['construction_boundary']['pair_level_evidence_stage_run']}",
        f"- Impact extraction run: {summary['construction_boundary']['impact_extraction_run']}",
        "- P99 naming guard: rainfall components are written as p99 extreme-rainfall episodes.",
        "",
        "## Red Flags",
        "",
    ]
    lines.extend(f"- {flag}" for flag in red_flags)
    lines.extend(
        [
            "",
            "## Output Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in paths.values())
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    construction: ConstructionResult,
    alignment: DTERAlignmentResult,
    pilot_config: PilotConfig,
) -> dict[str, Any]:
    paths = construction.paths
    summary = {
        **construction.summary,
        **alignment.summary,
    }
    red_flags = compute_red_flags(summary)
    summary["red_flags"] = red_flags

    write_dataclass_csv(paths["county_drought_episodes"], construction.drought_episodes)
    write_dataclass_csv(paths["county_rainfall_episodes"], construction.rainfall_episodes)
    write_dataclass_csv(paths["raw_ce_candidates"], construction.raw_candidates)
    write_dataclass_csv(paths["raw_ce_candidates_deduped"], construction.deduped_candidates)
    write_dataclass_csv(paths["temporal_ambiguous_candidates"], construction.temporal_ambiguous_candidates)
    write_dataclass_csv(paths["dter_alignment"], alignment.alignment_records)
    write_dataclass_csv(paths["dter_only"], alignment.dter_only_records)
    write_dataclass_csv(
        paths["review_sample_dter_overlap"],
        select_review_sample(alignment.alignment_records, category="dter_overlap"),
    )
    write_dataclass_csv(
        paths["review_sample_raw_only"],
        select_review_sample(alignment.alignment_records, category="raw_only"),
    )
    write_dataclass_csv(
        paths["review_sample_dter_only"],
        select_review_sample(alignment.alignment_records, category="dter_only"),
    )
    summary["output_files"] = {name: str(path) for name, path in paths.items()}
    safe_write_json(paths["candidate_construction_summary"], summary)
    safe_write_text(paths["candidate_construction_report"], render_report(summary, paths, red_flags))
    return summary


def run_pilot(pilot_config: PilotConfig) -> dict[str, Any]:
    construction = construct_raw_candidates(pilot_config)
    alignment = align_dter_to_raw_candidates(construction.deduped_candidates, pilot_config)
    return write_outputs(construction, alignment, pilot_config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the no-evidence Texas 2021-2025 raw CE candidate pilot.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the raw CE pilot YAML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pilot_config = load_pilot_config(args.config)
    summary = run_pilot(pilot_config)
    print(json.dumps(
        {
            "output_dir": str(pilot_config.output_dir),
            "candidate_pairs_after_deduplication": summary["candidate_pairs_after_deduplication"],
            "dter_rows_in_scope": summary["dter_rows_in_scope"],
            "raw_only_count": summary["raw_only_count"],
            "dter_only_count": summary["dter_only_count"],
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
