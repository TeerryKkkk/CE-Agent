from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

try:  # Canonical PYTHONPATH=src imports.
    from climate_pipeline import config
    from hashing import make_id
    from io_utils import safe_write_text
except ImportError:  # Historical src.rainfall_p99_pipeline compatibility.
    from . import config
    from .hashing import make_id
    from .io_utils import safe_write_text


REQUIRED_RAINFALL_FIELDS = {
    "event_id",
    "event_type",
    "lat",
    "lon",
    "start_date",
    "end_date",
    "duration",
}

RAIN_P99_DATASET = "events_extreme_rain_pr3_p99_1985_2025"
RAIN_P99_INPUT = config.DATASET_DIR / f"{RAIN_P99_DATASET}.csv"

MAPPING_STATUSES = {
    "mapped_us_land_county",
    "outside_us",
    "offshore_or_water",
    "border_uncertain",
    "mapping_failed",
    "invalid_coordinates",
}

STATE_FIPS: dict[str, tuple[str, str]] = {
    "01": ("Alabama", "AL"),
    "02": ("Alaska", "AK"),
    "04": ("Arizona", "AZ"),
    "05": ("Arkansas", "AR"),
    "06": ("California", "CA"),
    "08": ("Colorado", "CO"),
    "09": ("Connecticut", "CT"),
    "10": ("Delaware", "DE"),
    "11": ("District of Columbia", "DC"),
    "12": ("Florida", "FL"),
    "13": ("Georgia", "GA"),
    "15": ("Hawaii", "HI"),
    "16": ("Idaho", "ID"),
    "17": ("Illinois", "IL"),
    "18": ("Indiana", "IN"),
    "19": ("Iowa", "IA"),
    "20": ("Kansas", "KS"),
    "21": ("Kentucky", "KY"),
    "22": ("Louisiana", "LA"),
    "23": ("Maine", "ME"),
    "24": ("Maryland", "MD"),
    "25": ("Massachusetts", "MA"),
    "26": ("Michigan", "MI"),
    "27": ("Minnesota", "MN"),
    "28": ("Mississippi", "MS"),
    "29": ("Missouri", "MO"),
    "30": ("Montana", "MT"),
    "31": ("Nebraska", "NE"),
    "32": ("Nevada", "NV"),
    "33": ("New Hampshire", "NH"),
    "34": ("New Jersey", "NJ"),
    "35": ("New Mexico", "NM"),
    "36": ("New York", "NY"),
    "37": ("North Carolina", "NC"),
    "38": ("North Dakota", "ND"),
    "39": ("Ohio", "OH"),
    "40": ("Oklahoma", "OK"),
    "41": ("Oregon", "OR"),
    "42": ("Pennsylvania", "PA"),
    "44": ("Rhode Island", "RI"),
    "45": ("South Carolina", "SC"),
    "46": ("South Dakota", "SD"),
    "47": ("Tennessee", "TN"),
    "48": ("Texas", "TX"),
    "49": ("Utah", "UT"),
    "50": ("Vermont", "VT"),
    "51": ("Virginia", "VA"),
    "53": ("Washington", "WA"),
    "54": ("West Virginia", "WV"),
    "55": ("Wisconsin", "WI"),
    "56": ("Wyoming", "WY"),
    "60": ("American Samoa", "AS"),
    "66": ("Guam", "GU"),
    "69": ("Northern Mariana Islands", "MP"),
    "72": ("Puerto Rico", "PR"),
    "78": ("U.S. Virgin Islands", "VI"),
}


@dataclass(frozen=True)
class ValidatedRainfallRow:
    row_index: int
    source_row_id: str
    event_type: str
    lat: float
    lon: float
    start_date: date
    end_date: date
    duration_days: int
    threshold_level: str
    precipitation_window_code: str | None
    raw_row: dict[str, str]


@dataclass
class PreflightResult:
    input_csv: Path
    source_dataset: str
    threshold_level: str
    rows_read: int = 0
    valid_rows: int = 0
    skipped_rows: int = 0
    skipped_reason_counts: Counter[str] = field(default_factory=Counter)
    event_type_counts: Counter[str] = field(default_factory=Counter)
    unique_grid_cells: set[tuple[float, float]] = field(default_factory=set)
    date_min: date | None = None
    date_max: date | None = None
    duration_min: int | None = None
    duration_max: int | None = None
    grid_events_path: Path | None = None
    skipped_rows_path: Path | None = None
    missing_required_fields: list[str] = field(default_factory=list)
    row_limit: int | None = None


@dataclass(frozen=True)
class CountyFeature:
    properties: dict[str, Any]
    geometry_type: str
    coordinates: Any
    bbox: tuple[float, float, float, float]


@dataclass
class BoundaryMetadata:
    path: Path
    file_format: str
    crs: str | None
    feature_count: int
    column_names: list[str]
    has_geoid: bool
    has_statefp_countyfp: bool
    has_county_name: bool
    has_geometry: bool
    geometry_valid_enough: bool
    boundary_source: str
    boundary_year: str | None
    boundary_geometry_type: str
    publisher: str | None = None
    source_url: str | None = None
    source_selection_reason: str | None = None
    discovered_by_search: bool = False
    bounds: tuple[float, float, float, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_format": self.file_format,
            "crs": self.crs,
            "feature_count": self.feature_count,
            "column_names": self.column_names,
            "has_geoid": self.has_geoid,
            "has_statefp_countyfp": self.has_statefp_countyfp,
            "has_county_name": self.has_county_name,
            "has_geometry": self.has_geometry,
            "geometry_valid_enough": self.geometry_valid_enough,
            "boundary_source": self.boundary_source,
            "boundary_year": self.boundary_year,
            "boundary_geometry_type": self.boundary_geometry_type,
            "publisher": self.publisher,
            "source_url": self.source_url,
            "source_selection_reason": self.source_selection_reason,
            "discovered_by_search": self.discovered_by_search,
            "bounds": list(self.bounds) if self.bounds else None,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class MappingResult:
    lat: float
    lon: float
    mapping_status: str
    state: str | None = None
    state_abbrev: str | None = None
    state_fips: str | None = None
    county: str | None = None
    county_fips: str | None = None
    mapping_method: str = "geojson_point_in_polygon"
    mapping_source: str | None = None
    skipped_reason: str | None = None
    boundary_source: str | None = None
    boundary_year: str | None = None
    boundary_geometry_type: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "mapping_status": self.mapping_status,
            "state": self.state,
            "state_name": self.state,
            "state_abbrev": self.state_abbrev,
            "state_fips": self.state_fips,
            "county": self.county,
            "county_name": self.county,
            "county_fips": self.county_fips,
            "mapping_method": self.mapping_method,
            "mapping_source": self.mapping_source,
            "skipped_reason": self.skipped_reason,
            "boundary_source": self.boundary_source,
            "boundary_year": self.boundary_year,
            "boundary_geometry_type": self.boundary_geometry_type,
        }


@dataclass
class PipelineRunResult:
    preflight: PreflightResult
    mapping_status_counts: Counter[str]
    candidate_count: int
    sample_count: int
    paths: dict[str, Path]
    blocker: str | None = None
    boundary_metadata: BoundaryMetadata | None = None


def parse_day(value: str | None) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def detect_threshold_level(row: dict[str, str], input_csv: Path | str, expected: str = "p99") -> tuple[str | None, list[str]]:
    haystack = " ".join(
        str(value or "")
        for value in (
            row.get("event_type"),
            row.get("event_id"),
            Path(input_csv).stem,
        )
    ).lower()
    matches = sorted(set(re.findall(r"(?<![A-Za-z0-9])p(?:95|99)(?![A-Za-z0-9])", haystack)))
    reasons: list[str] = []
    if not matches:
        reasons.append("threshold_not_identifiable")
        return None, reasons
    if len(matches) > 1:
        reasons.append("threshold_ambiguous")
        return None, reasons
    threshold = matches[0]
    if threshold != expected:
        reasons.append(f"unexpected_threshold_{threshold}")
        return threshold, reasons
    return threshold, reasons


def detect_precipitation_window_code(row: dict[str, str], input_csv: Path | str) -> str | None:
    haystack = " ".join(
        str(value or "")
        for value in (
            row.get("event_type"),
            row.get("event_id"),
            Path(input_csv).stem,
        )
    ).lower()
    match = re.search(r"(?<![A-Za-z0-9])pr\d+(?![A-Za-z0-9])", haystack)
    return match.group(0) if match else None


def validate_rainfall_row(
    row: dict[str, str],
    *,
    row_index: int,
    input_csv: Path | str,
    expected_threshold: str = "p99",
) -> tuple[ValidatedRainfallRow | None, list[str]]:
    reasons: list[str] = []
    missing = sorted(field for field in REQUIRED_RAINFALL_FIELDS if str(row.get(field) or "").strip() == "")
    reasons.extend(f"missing_{field}" for field in missing)

    threshold, threshold_reasons = detect_threshold_level(row, input_csv, expected_threshold)
    reasons.extend(threshold_reasons)

    source_row_id = str(row.get("event_id") or "").strip()
    event_type = str(row.get("event_type") or "").strip()

    lat: float | None
    lon: float | None
    try:
        lat = float(str(row.get("lat") or "").strip())
    except ValueError:
        lat = None
        reasons.append("lat_not_numeric")
    try:
        lon = float(str(row.get("lon") or "").strip())
    except ValueError:
        lon = None
        reasons.append("lon_not_numeric")

    start = parse_day(row.get("start_date"))
    end = parse_day(row.get("end_date"))
    if start is None:
        reasons.append("start_date_parse_failed")
    if end is None:
        reasons.append("end_date_parse_failed")
    if start is not None and end is not None and end < start:
        reasons.append("end_date_before_start_date")

    duration: int | None
    try:
        duration = int(str(row.get("duration") or "").strip())
    except ValueError:
        duration = None
        reasons.append("duration_not_integer")
    if duration is not None and duration <= 0:
        reasons.append("duration_not_positive")
    if start is not None and end is not None and duration is not None:
        computed = (end - start).days + 1
        if computed != duration:
            reasons.append(f"duration_mismatch_expected_{computed}")

    if reasons:
        return None, reasons
    assert lat is not None
    assert lon is not None
    assert start is not None
    assert end is not None
    assert duration is not None
    assert threshold is not None
    return (
        ValidatedRainfallRow(
            row_index=row_index,
            source_row_id=source_row_id,
            event_type=event_type,
            lat=lat,
            lon=lon,
            start_date=start,
            end_date=end,
            duration_days=duration,
            threshold_level=threshold,
            precipitation_window_code=detect_precipitation_window_code(row, input_csv),
            raw_row=dict(row),
        ),
        [],
    )


def normalize_rainfall_grid_event(row: ValidatedRainfallRow, source_dataset: str) -> dict[str, Any]:
    pipeline_event_id = make_id(
        "rain_grid_evt",
        source_dataset,
        row.source_row_id,
        row.lat,
        row.lon,
        row.start_date.isoformat(),
        row.end_date.isoformat(),
        row.threshold_level,
    )
    return {
        "pipeline_event_id": pipeline_event_id,
        "source_dataset": source_dataset,
        "source_row_id": row.source_row_id,
        "source_event_id": row.source_row_id,
        "source_row_index": row.row_index,
        "event_type": row.event_type,
        "threshold_level": row.threshold_level,
        "precipitation_window_code": row.precipitation_window_code,
        "precipitation_window_code_status": "unconfirmed_definition",
        "lat": row.lat,
        "lon": row.lon,
        "start_date": row.start_date.isoformat(),
        "end_date": row.end_date.isoformat(),
        "duration_days": row.duration_days,
        "hazard_family": "extreme_rainfall",
        "hazard_type": "extreme_rainfall",
        "input_status": "rainfall_grid_signal",
        "candidate_status": "climate_signal_candidate_only",
        "public_disaster_status": "not_claimed",
        "notes": [
            "Climate signal candidate only.",
            "p99 threshold exceedance is not public disaster evidence.",
            "County mapping, when present, is only an administrative anchor for later retrieval.",
        ],
    }


def _write_skipped_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_index", "event_id", "reasons"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stream_validate_and_normalize_p99(
    input_csv: Path = RAIN_P99_INPUT,
    *,
    output_dir: Path = config.OUTPUT_DIR,
    expected_threshold: str = "p99",
    row_limit: int | None = None,
) -> PreflightResult:
    input_csv = Path(input_csv)
    source_dataset = input_csv.stem
    grid_events_path = output_dir / "events" / "rainfall_grid_events_p99.jsonl"
    skipped_rows_path = output_dir / "diagnostics" / "rainfall_p99_skipped_rows.csv"
    result = PreflightResult(
        input_csv=input_csv,
        source_dataset=source_dataset,
        threshold_level=expected_threshold,
        grid_events_path=grid_events_path,
        skipped_rows_path=skipped_rows_path,
        row_limit=row_limit,
    )

    grid_events_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_rows_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        result.skipped_reason_counts["input_csv_missing"] += 1
        safe_write_text(grid_events_path, "")
        _write_skipped_rows_csv(
            skipped_rows_path,
            [{"row_index": None, "event_id": None, "reasons": f"input_csv_missing: {input_csv}"}],
        )
        return result

    with input_csv.open("r", encoding="utf-8-sig", newline="") as csv_handle, grid_events_path.open(
        "w", encoding="utf-8", newline=""
    ) as jsonl_handle, skipped_rows_path.open("w", encoding="utf-8", newline="") as skipped_handle:
        skipped_writer = csv.DictWriter(skipped_handle, fieldnames=["row_index", "event_id", "reasons"])
        skipped_writer.writeheader()
        reader = csv.DictReader(csv_handle)
        fieldnames = set(reader.fieldnames or [])
        missing_required = sorted(REQUIRED_RAINFALL_FIELDS - fieldnames)
        result.missing_required_fields = missing_required
        if missing_required:
            result.skipped_reason_counts["missing_required_columns"] += 1
            skipped_writer.writerow(
                {
                    "row_index": None,
                    "event_id": None,
                    "reasons": "missing_required_columns: " + ",".join(missing_required),
                }
            )
            return result

        for row_index, row in enumerate(reader, start=1):
            if row_limit is not None and result.rows_read >= row_limit:
                break
            result.rows_read += 1
            event_type = str(row.get("event_type") or "").strip()
            if event_type:
                result.event_type_counts[event_type] += 1
            valid_row, reasons = validate_rainfall_row(
                dict(row),
                row_index=row_index,
                input_csv=input_csv,
                expected_threshold=expected_threshold,
            )
            if valid_row is None:
                result.skipped_rows += 1
                for reason in reasons:
                    result.skipped_reason_counts[reason] += 1
                skipped_writer.writerow(
                    {
                        "row_index": row_index,
                        "event_id": str(row.get("event_id") or "").strip() or None,
                        "reasons": ";".join(reasons),
                    }
                )
                continue
            result.valid_rows += 1
            result.unique_grid_cells.add((valid_row.lat, valid_row.lon))
            result.date_min = valid_row.start_date if result.date_min is None or valid_row.start_date < result.date_min else result.date_min
            result.date_max = valid_row.end_date if result.date_max is None or valid_row.end_date > result.date_max else result.date_max
            result.duration_min = valid_row.duration_days if result.duration_min is None or valid_row.duration_days < result.duration_min else result.duration_min
            result.duration_max = valid_row.duration_days if result.duration_max is None or valid_row.duration_days > result.duration_max else result.duration_max
            normalized = normalize_rainfall_grid_event(valid_row, source_dataset)
            jsonl_handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")

    return result


def coord_key(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"


def valid_coordinate(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def feature_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    def collect_coords(value: Any) -> None:
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
            xs.append(float(value[0]))
            ys.append(float(value[1]))
            return
        if isinstance(value, list):
            for item in value:
                collect_coords(item)

    collect_coords(geometry.get("coordinates"))
    if not xs or not ys:
        return (math.inf, math.inf, -math.inf, -math.inf)
    return (min(xs), min(ys), max(xs), max(ys))


def _point_on_segment(lon: float, lat: float, a: list[float], b: list[float], eps: float) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cross = (lon - ax) * (by - ay) - (lat - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (lon - ax) * (bx - ax) + (lat - ay) * (by - ay)
    if dot < -eps:
        return False
    squared_len = (bx - ax) ** 2 + (by - ay) ** 2
    if squared_len <= eps:
        return False
    return dot <= squared_len + eps


def _point_in_ring(lon: float, lat: float, ring: list[list[float]], eps: float = 1e-10) -> tuple[bool, bool]:
    inside = False
    if len(ring) < 4:
        return False, False
    previous = ring[-1]
    for current in ring:
        if _point_on_segment(lon, lat, previous, current, eps):
            return True, True
        xi, yi = float(current[0]), float(current[1])
        xj, yj = float(previous[0]), float(previous[1])
        intersects = ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-300) + xi)
        if intersects:
            inside = not inside
        previous = current
    return inside, False


def _point_in_polygon(lon: float, lat: float, polygon: list[Any]) -> tuple[bool, bool]:
    if not polygon:
        return False, False
    exterior_inside, exterior_boundary = _point_in_ring(lon, lat, polygon[0])
    if exterior_boundary:
        return True, True
    if not exterior_inside:
        return False, False
    for hole in polygon[1:]:
        hole_inside, hole_boundary = _point_in_ring(lon, lat, hole)
        if hole_boundary:
            return True, True
        if hole_inside:
            return False, False
    return True, False


def point_in_feature(lon: float, lat: float, feature: CountyFeature) -> tuple[bool, bool]:
    min_lon, min_lat, max_lon, max_lat = feature.bbox
    if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
        return False, False
    if feature.geometry_type == "ShapefilePolygon":
        inside_count = 0
        on_boundary = False
        for ring in feature.coordinates:
            inside, boundary = _point_in_ring(lon, lat, ring)
            if boundary:
                on_boundary = True
            elif inside:
                inside_count += 1
        if on_boundary:
            return True, True
        return inside_count % 2 == 1, False
    if feature.geometry_type == "Polygon":
        return _point_in_polygon(lon, lat, feature.coordinates)
    if feature.geometry_type == "MultiPolygon":
        for polygon in feature.coordinates:
            inside, boundary = _point_in_polygon(lon, lat, polygon)
            if inside:
                return True, boundary
    return False, False


def _extract_county_properties(properties: dict[str, Any]) -> dict[str, str | None]:
    state_fips = _first_property(properties, "STATEFP", "STATE_FIPS", "state_fips")
    county_fips = _first_property(properties, "GEOID", "GEOIDFP", "FIPS", "county_fips")
    county_fp = _first_property(properties, "COUNTYFP", "COUNTY_FIPS")
    if not county_fips and state_fips and county_fp:
        county_fips = f"{state_fips}{county_fp}"
    state_name = _first_property(properties, "STATE_NAME", "STATE", "state", "STATE_NAM")
    state_abbrev = _first_property(properties, "STUSPS", "STATE_ABBR", "state_abbrev")
    if state_fips in STATE_FIPS:
        fips_name, fips_abbrev = STATE_FIPS[state_fips]
        state_name = state_name or fips_name
        state_abbrev = state_abbrev or fips_abbrev
    county = _first_property(properties, "NAMELSAD", "NAME", "COUNTY", "county")
    if county and "County" not in county and "Parish" not in county and "Borough" not in county and county not in {"District of Columbia"}:
        county = f"{county} County"
    return {
        "state": state_name,
        "state_abbrev": state_abbrev,
        "state_fips": state_fips,
        "county": county,
        "county_fips": county_fips,
    }


def _first_property(properties: dict[str, Any], *names: str) -> str | None:
    lowered = {key.lower(): value for key, value in properties.items()}
    for name in names:
        value = properties.get(name)
        if value is None:
            value = lowered.get(name.lower())
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None


def _coarse_us_area_status(lat: float, lon: float) -> str:
    conus = -125.5 <= lon <= -66 and 24 <= lat <= 49.5
    alaska = -180 <= lon <= -129 and 51 <= lat <= 72
    hawaii = -161 <= lon <= -154 and 18 <= lat <= 23
    puerto_rico = -68.5 <= lon <= -64.5 and 17 <= lat <= 19
    us_virgin_islands = -65.5 <= lon <= -64 and 17 <= lat <= 19
    if conus or alaska or hawaii or puerto_rico or us_virgin_islands:
        return "offshore_or_water"
    return "outside_us"


class BoundaryLoadError(ValueError):
    pass


def _infer_boundary_source(path: Path) -> dict[str, Any]:
    name = path.name.lower()
    if "tl_2024_us_county" in name:
        return {
            "boundary_source": "US Census TIGER/Line Shapefiles",
            "boundary_year": "2024",
            "publisher": "U.S. Census Bureau",
            "source_url": "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip",
            "source_selection_reason": "Primary requested source: national 2024 TIGER/Line county boundary with full county geometry.",
            "discovered_by_search": True,
        }
    if "cb_2024_us_county" in name:
        return {
            "boundary_source": "US Census Cartographic Boundary Files",
            "boundary_year": "2024",
            "publisher": "U.S. Census Bureau",
            "source_url": None,
            "source_selection_reason": "Secondary simplified Census cartographic county boundary.",
            "discovered_by_search": True,
        }
    year_match = re.search(r"(20\d{2})", name)
    return {
        "boundary_source": "US county boundary file",
        "boundary_year": year_match.group(1) if year_match else None,
        "publisher": None,
        "source_url": None,
        "source_selection_reason": "Local boundary file supplied by path.",
        "discovered_by_search": False,
    }


def _crs_is_lonlat_wgs84_compatible(prj: str | None) -> bool:
    if not prj:
        return True
    text = prj.upper()
    return "GEOGCS" in text and "UNIT[\"DEGREE\"" in text and ("NORTH_AMERICAN_1983" in text or "WGS_1984" in text)


def _geometry_bounds(features: list[CountyFeature]) -> tuple[float, float, float, float] | None:
    if not features:
        return None
    min_lon = min(feature.bbox[0] for feature in features)
    min_lat = min(feature.bbox[1] for feature in features)
    max_lon = max(feature.bbox[2] for feature in features)
    max_lat = max(feature.bbox[3] for feature in features)
    return (min_lon, min_lat, max_lon, max_lat)


def _parse_dbf_records(data: bytes, encoding: str = "latin-1") -> tuple[list[str], list[dict[str, str]]]:
    if len(data) < 32:
        raise BoundaryLoadError("invalid_dbf_header")
    record_count = struct.unpack("<I", data[4:8])[0]
    header_length = struct.unpack("<H", data[8:10])[0]
    record_length = struct.unpack("<H", data[10:12])[0]
    fields: list[tuple[str, int]] = []
    offset = 32
    while offset < len(data) and data[offset] != 0x0D:
        descriptor = data[offset : offset + 32]
        if len(descriptor) < 32:
            raise BoundaryLoadError("truncated_dbf_field_descriptor")
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        length = descriptor[16]
        if name:
            fields.append((name, length))
        offset += 32
    columns = [name for name, _length in fields]
    records: list[dict[str, str]] = []
    for idx in range(record_count):
        start = header_length + idx * record_length
        record = data[start : start + record_length]
        if len(record) < record_length:
            break
        if record[:1] == b"*":
            records.append({})
            continue
        cursor = 1
        row: dict[str, str] = {}
        for name, length in fields:
            raw = record[cursor : cursor + length]
            cursor += length
            row[name] = raw.decode(encoding, errors="replace").strip()
        records.append(row)
    return columns, records


def _parse_shp_polygons(data: bytes) -> tuple[int, list[dict[str, Any]], tuple[float, float, float, float] | None]:
    if len(data) < 100:
        raise BoundaryLoadError("invalid_shp_header")
    shape_type = struct.unpack("<i", data[32:36])[0]
    header_bounds = struct.unpack("<4d", data[36:68])
    shapes: list[dict[str, Any]] = []
    offset = 100
    while offset + 8 <= len(data):
        _record_number, content_length_words = struct.unpack(">2i", data[offset : offset + 8])
        offset += 8
        content_length = content_length_words * 2
        content = data[offset : offset + content_length]
        offset += content_length
        if len(content) < 4:
            continue
        record_shape_type = struct.unpack("<i", content[:4])[0]
        if record_shape_type == 0:
            shapes.append({"geometry_type": "Null", "rings": [], "bbox": None})
            continue
        if record_shape_type not in {5, 15, 25}:
            raise BoundaryLoadError(f"unsupported_shapefile_shape_type_{record_shape_type}; expected polygon geometry")
        if len(content) < 44:
            raise BoundaryLoadError("truncated_polygon_record")
        bbox = struct.unpack("<4d", content[4:36])
        num_parts, num_points = struct.unpack("<2i", content[36:44])
        parts_offset = 44
        points_offset = parts_offset + num_parts * 4
        if len(content) < points_offset + num_points * 16:
            raise BoundaryLoadError("truncated_polygon_points")
        parts = list(struct.unpack(f"<{num_parts}i", content[parts_offset:points_offset]))
        points: list[list[float]] = []
        for point_idx in range(num_points):
            x, y = struct.unpack("<2d", content[points_offset + point_idx * 16 : points_offset + (point_idx + 1) * 16])
            points.append([x, y])
        rings: list[list[list[float]]] = []
        for part_idx, start in enumerate(parts):
            end = parts[part_idx + 1] if part_idx + 1 < len(parts) else num_points
            ring = points[start:end]
            if len(ring) >= 4:
                rings.append(ring)
        shapes.append({"geometry_type": "ShapefilePolygon", "rings": rings, "bbox": bbox})
    return shape_type, shapes, header_bounds


def _read_shapefile_components(path: Path) -> tuple[bytes, bytes, str | None, str]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            shp_name = next((name for name in names if name.lower().endswith(".shp")), None)
            dbf_name = next((name for name in names if name.lower().endswith(".dbf")), None)
            prj_name = next((name for name in names if name.lower().endswith(".prj")), None)
            cpg_name = next((name for name in names if name.lower().endswith(".cpg")), None)
            if not shp_name or not dbf_name:
                raise BoundaryLoadError("zipped_shapefile_missing_shp_or_dbf")
            encoding = archive.read(cpg_name).decode("ascii", errors="ignore").strip() if cpg_name else "latin-1"
            prj = archive.read(prj_name).decode("utf-8", errors="replace").strip() if prj_name else None
            return archive.read(shp_name), archive.read(dbf_name), prj, encoding or "latin-1"
    if suffix == ".shp":
        dbf_path = path.with_suffix(".dbf")
        prj_path = path.with_suffix(".prj")
        cpg_path = path.with_suffix(".cpg")
        if not dbf_path.exists():
            raise BoundaryLoadError(f"shapefile_missing_dbf: {dbf_path}")
        encoding = cpg_path.read_text(encoding="ascii", errors="ignore").strip() if cpg_path.exists() else "latin-1"
        prj = prj_path.read_text(encoding="utf-8", errors="replace").strip() if prj_path.exists() else None
        return path.read_bytes(), dbf_path.read_bytes(), prj, encoding or "latin-1"
    raise BoundaryLoadError(f"unsupported_shapefile_container: {path.suffix}")


def _load_geojson_features(path: Path) -> tuple[list[CountyFeature], BoundaryMetadata]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_features = payload.get("features") if payload.get("type") == "FeatureCollection" else None
    if not isinstance(raw_features, list):
        raise BoundaryLoadError("invalid_geojson_feature_collection")
    features: list[CountyFeature] = []
    column_names: set[str] = set()
    geometry_types: Counter[str] = Counter()
    for feature in raw_features:
        properties = dict(feature.get("properties") or {})
        column_names.update(properties)
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            continue
        bbox = feature_bbox(geometry)
        if not all(math.isfinite(value) for value in bbox):
            continue
        geometry_types[geometry_type] += 1
        features.append(
            CountyFeature(
                properties=properties,
                geometry_type=geometry_type,
                coordinates=geometry.get("coordinates"),
                bbox=bbox,
            )
        )
    source = _infer_boundary_source(path)
    metadata = _metadata_from_features(
        path=path,
        file_format="GeoJSON",
        crs="EPSG:4326 assumed for GeoJSON input",
        features=features,
        column_names=sorted(column_names),
        boundary_geometry_type=",".join(sorted(geometry_types)) or "unknown",
        source=source,
    )
    return features, metadata


def _load_shapefile_features(path: Path) -> tuple[list[CountyFeature], BoundaryMetadata]:
    shp_bytes, dbf_bytes, prj, encoding = _read_shapefile_components(path)
    if prj and not _crs_is_lonlat_wgs84_compatible(prj):
        raise BoundaryLoadError(
            "unsupported_boundary_crs_requires_geospatial_dependencies: install geopandas plus pyogrio/fiona and pyproj, "
            "or provide a WGS84/NAD83 latitude-longitude county boundary file"
        )
    columns, records = _parse_dbf_records(dbf_bytes, encoding=encoding)
    shape_type, shapes, _header_bounds = _parse_shp_polygons(shp_bytes)
    features: list[CountyFeature] = []
    for idx, shape in enumerate(shapes):
        if shape.get("geometry_type") != "ShapefilePolygon" or not shape.get("rings") or not shape.get("bbox"):
            continue
        properties = records[idx] if idx < len(records) else {}
        features.append(
            CountyFeature(
                properties=properties,
                geometry_type="ShapefilePolygon",
                coordinates=shape["rings"],
                bbox=shape["bbox"],
            )
        )
    source = _infer_boundary_source(path)
    metadata = _metadata_from_features(
        path=path,
        file_format="zipped_shapefile" if path.suffix.lower() == ".zip" else "shapefile",
        crs=prj or "missing .prj; assumed WGS84/NAD83 lon/lat only if coordinates validate",
        features=features,
        column_names=columns,
        boundary_geometry_type=f"ESRI Shapefile shape type {shape_type}",
        source=source,
    )
    metadata.diagnostics["dbf_encoding"] = encoding
    metadata.diagnostics["dbf_record_count"] = len(records)
    metadata.diagnostics["shp_record_count"] = len(shapes)
    return features, metadata


def _metadata_from_features(
    *,
    path: Path,
    file_format: str,
    crs: str | None,
    features: list[CountyFeature],
    column_names: list[str],
    boundary_geometry_type: str,
    source: dict[str, Any],
) -> BoundaryMetadata:
    columns_upper = {column.upper() for column in column_names}
    return BoundaryMetadata(
        path=path,
        file_format=file_format,
        crs=crs,
        feature_count=len(features),
        column_names=column_names,
        has_geoid="GEOID" in columns_upper or "GEOIDFP" in columns_upper or "FIPS" in columns_upper,
        has_statefp_countyfp="STATEFP" in columns_upper and "COUNTYFP" in columns_upper,
        has_county_name=bool({"NAME", "NAMELSAD", "COUNTY"} & columns_upper),
        has_geometry=bool(features),
        geometry_valid_enough=bool(features) and all(feature.bbox for feature in features),
        boundary_geometry_type=boundary_geometry_type,
        bounds=_geometry_bounds(features),
        **source,
    )


class CountyBoundaryMapper:
    def __init__(self, boundary_file: Path | str):
        self.boundary_file = Path(boundary_file)
        self.features: list[CountyFeature] = []
        self.index: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.metadata: BoundaryMetadata
        self._load()

    def _load(self) -> None:
        suffix = self.boundary_file.suffix.lower()
        if suffix in {".json", ".geojson"}:
            self.features, self.metadata = _load_geojson_features(self.boundary_file)
        elif suffix in {".zip", ".shp"}:
            self.features, self.metadata = _load_shapefile_features(self.boundary_file)
        elif suffix == ".gpkg":
            if not importlib.util.find_spec("geopandas"):
                raise BoundaryLoadError(
                    "gpkg_support_requires_geospatial_dependencies: install geopandas with pyogrio or fiona, "
                    "or use the official Census zipped shapefile tl_2024_us_county.zip"
                )
            raise BoundaryLoadError("gpkg_support_not_enabled_in_standard_library_mapper")
        else:
            raise BoundaryLoadError(
                f"unsupported_boundary_format_{suffix}: supported formats are zipped shapefile, .shp, .geojson, and .gpkg"
            )
        for idx, feature in enumerate(self.features):
            min_lon, min_lat, max_lon, max_lat = feature.bbox
            for cell_lon in range(math.floor(min_lon), math.floor(max_lon) + 1):
                for cell_lat in range(math.floor(min_lat), math.floor(max_lat) + 1):
                    self.index[(cell_lon, cell_lat)].append(idx)
        if not self.features:
            raise BoundaryLoadError("no_polygon_features_in_boundary_file")

    def map_point(self, lat: float, lon: float) -> MappingResult:
        if not valid_coordinate(lat, lon):
            return MappingResult(
                lat=lat,
                lon=lon,
                mapping_status="invalid_coordinates",
                mapping_method="point_in_polygon",
                mapping_source=str(self.boundary_file),
                skipped_reason="coordinate_out_of_range",
                boundary_source=self.metadata.boundary_source,
                boundary_year=self.metadata.boundary_year,
                boundary_geometry_type=self.metadata.boundary_geometry_type,
            )
        candidate_indexes = self.index.get((math.floor(lon), math.floor(lat)), [])
        matches: list[CountyFeature] = []
        on_boundary = False
        for idx in candidate_indexes:
            feature = self.features[idx]
            inside, boundary = point_in_feature(lon, lat, feature)
            if inside:
                matches.append(feature)
                on_boundary = on_boundary or boundary
        if len(matches) == 1 and not on_boundary:
            props = _extract_county_properties(matches[0].properties)
            if not props["county_fips"]:
                return MappingResult(
                    lat=lat,
                    lon=lon,
                    mapping_status="mapping_failed",
                    mapping_method="point_in_polygon",
                    mapping_source=str(self.boundary_file),
                    skipped_reason="mapped_feature_missing_county_fips",
                    boundary_source=self.metadata.boundary_source,
                    boundary_year=self.metadata.boundary_year,
                    boundary_geometry_type=self.metadata.boundary_geometry_type,
                )
            return MappingResult(
                lat=lat,
                lon=lon,
                mapping_status="mapped_us_land_county",
                state=props["state"],
                state_abbrev=props["state_abbrev"],
                state_fips=props["state_fips"],
                county=props["county"],
                county_fips=props["county_fips"],
                mapping_method="point_in_polygon",
                mapping_source=str(self.boundary_file),
                boundary_source=self.metadata.boundary_source,
                boundary_year=self.metadata.boundary_year,
                boundary_geometry_type=self.metadata.boundary_geometry_type,
            )
        if matches:
            return MappingResult(
                lat=lat,
                lon=lon,
                mapping_status="border_uncertain",
                mapping_method="point_in_polygon",
                mapping_source=str(self.boundary_file),
                skipped_reason="point_on_or_inside_multiple_county_boundaries",
                boundary_source=self.metadata.boundary_source,
                boundary_year=self.metadata.boundary_year,
                boundary_geometry_type=self.metadata.boundary_geometry_type,
            )
        status = _coarse_us_area_status(lat, lon)
        return MappingResult(
            lat=lat,
            lon=lon,
            mapping_status=status,
            mapping_method="point_in_polygon",
            mapping_source=str(self.boundary_file),
            skipped_reason="no_county_polygon_contains_point",
            boundary_source=self.metadata.boundary_source,
            boundary_year=self.metadata.boundary_year,
            boundary_geometry_type=self.metadata.boundary_geometry_type,
        )


GeoJsonCountyBoundaryMapper = CountyBoundaryMapper


def find_default_county_boundary_file() -> Path | None:
    env_path = os.getenv("RAINFALL_COUNTY_BOUNDARY_FILE", "").strip()
    candidates = [
        Path(env_path) if env_path else None,
        config.DATA_DIR / "raw" / "tl_2024_us_county.zip",
        config.DATA_DIR / "geo" / "us_counties.geojson",
        config.DATA_DIR / "geo" / "counties.geojson",
        config.DATA_DIR / "raw" / "us_counties.geojson",
        config.DATASET_DIR / "us_counties.geojson",
        config.DATASET_DIR / "counties.geojson",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def build_missing_boundary_lookup(
    grid_cells: Iterable[tuple[float, float]],
    *,
    requested_boundary_file: Path | None,
) -> dict[str, MappingResult]:
    reason = (
        "county_boundary_file_missing; rerun with --county-boundaries PATH_TO_US_COUNTY_GEOJSON"
        if requested_boundary_file is None
        else f"county_boundary_file_missing: {requested_boundary_file}"
    )
    lookup: dict[str, MappingResult] = {}
    for lat, lon in sorted(grid_cells, reverse=True):
        status = "invalid_coordinates" if not valid_coordinate(lat, lon) else "mapping_failed"
        lookup[coord_key(lat, lon)] = MappingResult(
            lat=lat,
            lon=lon,
            mapping_status=status,
            mapping_method="not_run_missing_boundary_file",
            mapping_source=str(requested_boundary_file) if requested_boundary_file else None,
            skipped_reason="coordinate_out_of_range" if status == "invalid_coordinates" else reason,
        )
    return lookup


def build_grid_to_county_lookup(
    grid_cells: Iterable[tuple[float, float]],
    *,
    boundary_file: Path | None,
) -> tuple[dict[str, MappingResult], str | None, BoundaryMetadata | None]:
    grid_cell_list = list(grid_cells)
    if boundary_file is None or not Path(boundary_file).exists():
        return build_missing_boundary_lookup(grid_cell_list, requested_boundary_file=boundary_file), "county_boundary_file_missing", None
    try:
        mapper = CountyBoundaryMapper(boundary_file)
    except Exception as exc:
        lookup = {}
        for lat, lon in sorted(grid_cell_list, reverse=True):
            lookup[coord_key(lat, lon)] = MappingResult(
                lat=lat,
                lon=lon,
                mapping_status="mapping_failed" if valid_coordinate(lat, lon) else "invalid_coordinates",
                mapping_method="boundary_mapper_initialization_failed",
                mapping_source=str(boundary_file),
                skipped_reason=f"{type(exc).__name__}: {exc}",
            )
        return lookup, "county_boundary_file_unusable", None
    mapper.metadata.diagnostics["rainfall_point_bounds"] = list(bounds_from_points(grid_cell_list)) if grid_cell_list else None
    mapper.metadata.diagnostics["coordinate_order"] = "rainfall points constructed with x=longitude and y=latitude"
    mapper.metadata.diagnostics["known_us_interior_point_tests"] = {
        "salina_kansas": mapper.map_point(38.84, -97.61).as_row(),
        "fresno_california": mapper.map_point(36.74, -119.78).as_row(),
        "raleigh_north_carolina": mapper.map_point(35.78, -78.64).as_row(),
    }
    lookup = {coord_key(lat, lon): mapper.map_point(lat, lon) for lat, lon in sorted(grid_cell_list, reverse=True)}
    return lookup, None, mapper.metadata


def bounds_from_points(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    point_list = list(points)
    if not point_list:
        return (math.nan, math.nan, math.nan, math.nan)
    lats = [lat for lat, _lon in point_list]
    lons = [lon for _lat, lon in point_list]
    return (min(lons), min(lats), max(lons), max(lats))


def write_mapping_lookup_csv(path: Path, lookup: dict[str, MappingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lat",
        "lon",
        "mapping_status",
        "state",
        "state_name",
        "state_abbrev",
        "state_fips",
        "county",
        "county_name",
        "county_fips",
        "mapping_method",
        "mapping_source",
        "skipped_reason",
        "boundary_source",
        "boundary_year",
        "boundary_geometry_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mapping in lookup.values():
            writer.writerow(mapping.as_row())


def write_mapping_lookup_jsonl(path: Path, lookup: dict[str, MappingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for mapping in lookup.values():
            handle.write(json.dumps(mapping.as_row(), ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _event_date(value: str) -> date:
    parsed = parse_day(value)
    if parsed is None:
        raise ValueError(f"invalid normalized event date: {value}")
    return parsed


def aggregate_county_candidates(
    grid_events_path: Path,
    mapping_lookup: dict[str, MappingResult],
    *,
    close_window_days: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped_mapping_status: Counter[str] = Counter()
    read_count = 0
    eligible_count = 0
    for event in _read_jsonl(grid_events_path):
        read_count += 1
        mapping = mapping_lookup.get(coord_key(float(event["lat"]), float(event["lon"])))
        if mapping is None:
            skipped_mapping_status["mapping_failed"] += 1
            continue
        if mapping.mapping_status != "mapped_us_land_county":
            skipped_mapping_status[mapping.mapping_status] += 1
            continue
        eligible_count += 1
        item = {
            **event,
            "start_date_obj": _event_date(event["start_date"]),
            "end_date_obj": _event_date(event["end_date"]),
            "mapping": mapping,
        }
        groups[(mapping.county_fips or "", event["threshold_level"])].append(item)

    candidates: list[dict[str, Any]] = []
    for (_county_fips, threshold), events in groups.items():
        events.sort(key=lambda item: (item["start_date_obj"], item["end_date_obj"], item["pipeline_event_id"]))
        current: list[dict[str, Any]] = []
        current_end: date | None = None
        for event in events:
            start = event["start_date_obj"]
            end = event["end_date_obj"]
            if not current:
                current = [event]
                current_end = end
                continue
            assert current_end is not None
            if start <= current_end + timedelta(days=close_window_days):
                current.append(event)
                current_end = max(current_end, end)
            else:
                candidates.append(_candidate_from_events(current, threshold, close_window_days))
                current = [event]
                current_end = end
        if current:
            candidates.append(_candidate_from_events(current, threshold, close_window_days))

    candidates.sort(key=lambda item: (item["start_date"], item["state_abbrev"] or "", item["county_fips"], item["candidate_id"]))
    diagnostics = {
        "grid_events_read": read_count,
        "eligible_mapped_grid_events": eligible_count,
        "skipped_grid_events_by_mapping_status": dict(sorted(skipped_mapping_status.items())),
        "county_threshold_groups": len(groups),
        "candidate_count": len(candidates),
        "close_window_days": close_window_days,
        "aggregation_rule": "same county FIPS, same threshold, overlapping or gap <= close_window_days",
        "live_retrieval_run": False,
        "compound_pairing_run": False,
    }
    return candidates, diagnostics


def _candidate_from_events(events: list[dict[str, Any]], threshold: str, close_window_days: int) -> dict[str, Any]:
    first_mapping: MappingResult = events[0]["mapping"]
    start = min(item["start_date_obj"] for item in events)
    end = max(item["end_date_obj"] for item in events)
    grid_event_ids = sorted(item["pipeline_event_id"] for item in events)
    source_row_ids = sorted(item["source_row_id"] for item in events)
    lats = [float(item["lat"]) for item in events]
    lons = [float(item["lon"]) for item in events]
    unique_cells = sorted({coord_key(float(item["lat"]), float(item["lon"])) for item in events})
    candidate_id = make_id(
        "rain_county_candidate",
        first_mapping.county_fips,
        threshold,
        start.isoformat(),
        end.isoformat(),
        grid_event_ids,
    )
    return {
        "candidate_id": candidate_id,
        "candidate_status": "rainfall_county_candidate_climate_signal_only",
        "public_disaster_status": "not_claimed",
        "source_dataset": events[0]["source_dataset"],
        "hazard_family": "extreme_rainfall",
        "hazard_type": "extreme_rainfall",
        "state": first_mapping.state,
        "state_name": first_mapping.state,
        "state_abbrev": first_mapping.state_abbrev,
        "state_fips": first_mapping.state_fips,
        "county": first_mapping.county,
        "county_name": first_mapping.county,
        "county_fips": first_mapping.county_fips,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "duration_days": (end - start).days + 1,
        "threshold_level": threshold,
        "grid_event_count": len(events),
        "unique_grid_cell_count": len(unique_cells),
        "representative_lat": round(sum(lats) / len(lats), 6),
        "representative_lon": round(sum(lons) / len(lons), 6),
        "footprint_summary": {
            "lat_min": min(lats),
            "lat_max": max(lats),
            "lon_min": min(lons),
            "lon_max": max(lons),
            "unique_grid_cells": unique_cells,
        },
        "source_grid_event_ids": grid_event_ids,
        "source_row_ids": source_row_ids,
        "aggregation_diagnostics": {
            "close_window_days": close_window_days,
            "grouping_rule": "same_county_same_threshold_close_or_overlapping_windows",
            "merged_public_disaster_claim": False,
            "notes": [
                "County candidate is a validation unit only.",
                "It does not claim public disaster occurrence.",
            ],
        },
    }


def select_pilot_sample(
    candidates: list[dict[str, Any]],
    *,
    sample_size: int = 10,
    seed: int = 20260617,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    used_county_dates: set[tuple[str, str, str]] = set()
    selected_ids: set[str] = set()

    def add_candidate(candidate: dict[str, Any], reason: str, allow_duplicate_county_date: bool = False) -> bool:
        if len(selected) >= sample_size or candidate["candidate_id"] in selected_ids:
            return False
        county_date = (str(candidate.get("county_fips")), str(candidate.get("start_date")), str(candidate.get("end_date")))
        if county_date in used_county_dates and not allow_duplicate_county_date:
            return False
        selected_ids.add(candidate["candidate_id"])
        used_county_dates.add(county_date)
        selected.append({**candidate, "sample_selection_reason": reason, "sample_seed": seed})
        return True

    for bucket_name, key_func in (
        ("decade", lambda c: str(int(c["start_date"][:4]) // 10 * 10) + "s"),
        ("region", lambda c: region_for_state(str(c.get("state_abbrev") or ""))),
        ("duration_bucket", lambda c: duration_bucket(int(c.get("duration_days") or 0))),
        ("grid_count_bucket", lambda c: grid_count_bucket(int(c.get("unique_grid_cell_count") or 0))),
    ):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in shuffled:
            buckets[key_func(candidate)].append(candidate)
        for bucket in sorted(buckets):
            bucket_candidates = sorted(buckets[bucket], key=lambda c: stable_candidate_sort_key(c, seed))
            for candidate in bucket_candidates:
                if add_candidate(candidate, f"first_pass_{bucket_name}_{bucket}"):
                    break
            if len(selected) >= sample_size:
                break
        if len(selected) >= sample_size:
            break

    if len(selected) < sample_size:
        for candidate in sorted(shuffled, key=lambda c: stable_candidate_sort_key(c, seed)):
            if add_candidate(candidate, "deterministic_fill_avoid_duplicate_county_date"):
                continue
            if len(selected) >= sample_size:
                break
    if len(selected) < sample_size:
        for candidate in sorted(shuffled, key=lambda c: stable_candidate_sort_key(c, seed)):
            add_candidate(candidate, "deterministic_fill_duplicate_county_date_allowed", allow_duplicate_county_date=True)
            if len(selected) >= sample_size:
                break

    diagnostics = {
        "requested_sample_size": sample_size,
        "selected_sample_size": len(selected),
        "seed": seed,
        "selection_mode": "deterministic_seeded_stratified_passes",
        "strata": ["decade", "region", "duration_bucket", "grid_count_bucket"],
        "duplicate_same_county_date_avoidance": "avoid where possible, allow only if needed to fill sample",
        "live_retrieval_run": False,
        "compound_pairing_run": False,
    }
    return selected, diagnostics


def stable_candidate_sort_key(candidate: dict[str, Any], seed: int) -> str:
    return make_id("sample_order", seed, candidate["candidate_id"], candidate.get("county_fips"), candidate.get("start_date"))


def duration_bucket(duration_days: int) -> str:
    if duration_days <= 1:
        return "1_day"
    if duration_days <= 3:
        return "2_3_days"
    return "4_plus_days"


def grid_count_bucket(count: int) -> str:
    if count <= 1:
        return "1_grid_cell"
    if count <= 4:
        return "2_4_grid_cells"
    return "5_plus_grid_cells"


def region_for_state(state_abbrev: str) -> str:
    northeast = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
    midwest = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
    south = {"DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"}
    west = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}
    if state_abbrev in northeast:
        return "northeast"
    if state_abbrev in midwest:
        return "midwest"
    if state_abbrev in south:
        return "south"
    if state_abbrev in west:
        return "west"
    return "unknown"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_existing_grid_events(grid_events_path: Path, *, source_dataset: str = RAIN_P99_DATASET) -> PreflightResult:
    result = PreflightResult(
        input_csv=RAIN_P99_INPUT,
        source_dataset=source_dataset,
        threshold_level="p99",
        grid_events_path=grid_events_path,
        skipped_rows_path=config.OUTPUT_DIR / "diagnostics" / "rainfall_p99_skipped_rows.csv",
    )
    for event in _read_jsonl(grid_events_path):
        result.rows_read += 1
        result.valid_rows += 1
        if event.get("event_type"):
            result.event_type_counts[str(event["event_type"])] += 1
        lat = float(event["lat"])
        lon = float(event["lon"])
        result.unique_grid_cells.add((lat, lon))
        start = _event_date(str(event["start_date"]))
        end = _event_date(str(event["end_date"]))
        duration = int(event.get("duration_days") or (end - start).days + 1)
        result.date_min = start if result.date_min is None or start < result.date_min else result.date_min
        result.date_max = end if result.date_max is None or end > result.date_max else result.date_max
        result.duration_min = duration if result.duration_min is None or duration < result.duration_min else result.duration_min
        result.duration_max = duration if result.duration_max is None or duration > result.duration_max else result.duration_max
    return result


def write_preflight_report(path: Path, result: PreflightResult, output_paths: dict[str, Path]) -> None:
    text = "\n".join(
        [
            "# Rainfall P99 Preflight Report",
            "",
            f"- Input CSV: `{result.input_csv}`",
            f"- Source dataset: `{result.source_dataset}`",
            "- Read mode: streaming `csv.DictReader`; the full CSV is not loaded into memory.",
            f"- Row limit: `{result.row_limit}`" if result.row_limit is not None else "- Row limit: none",
            f"- Rows read: {result.rows_read}",
            f"- Valid rows normalized: {result.valid_rows}",
            f"- Skipped rows: {result.skipped_rows}",
            f"- Required columns missing: {', '.join(result.missing_required_fields) if result.missing_required_fields else 'none'}",
            f"- Unique valid grid cells: {len(result.unique_grid_cells)}",
            f"- Date range: {result.date_min.isoformat() if result.date_min else 'n/a'} to {result.date_max.isoformat() if result.date_max else 'n/a'}",
            f"- Duration range: {result.duration_min if result.duration_min is not None else 'n/a'} to {result.duration_max if result.duration_max is not None else 'n/a'} days",
            f"- Event type counts: `{dict(sorted(result.event_type_counts.items()))}`",
            f"- Skipped reason counts: `{dict(sorted(result.skipped_reason_counts.items()))}`",
            f"- Normalized grid events: `{output_paths['grid_events']}`",
            f"- Skipped-row diagnostics: `{output_paths['skipped_rows']}`",
            "",
            "Threshold and metadata notes:",
            "",
            "- Threshold detection requires `p99` in the file name, `event_type`, or source row ID.",
            "- `pr3` is preserved as the precipitation-window code when present and marked `unconfirmed_definition`.",
            "- Every normalized row is marked `climate_signal_candidate_only` and `public_disaster_status=not_claimed`.",
            "",
            "Execution boundary:",
            "",
            "- Live retrieval was not run.",
            "- Compound pairing was not run.",
            "- p95 was not processed.",
            "",
        ]
    )
    safe_write_text(path, text)


def write_mapping_report(
    path: Path,
    *,
    preflight: PreflightResult,
    lookup: dict[str, MappingResult],
    boundary_file: Path | None,
    boundary_metadata: BoundaryMetadata | None,
    blocker: str | None,
    output_paths: dict[str, Path],
) -> None:
    counts = Counter(item.mapping_status for item in lookup.values())
    boundary_lines = [
        f"- Boundary file: `{boundary_file}`" if boundary_file else "- Boundary file: not found",
        f"- Boundary file local/discovered: {'discovered by bounded official-source search' if boundary_metadata and boundary_metadata.discovered_by_search else 'local path or not loaded'}",
    ]
    if boundary_metadata:
        boundary_lines.extend(
            [
                f"- Boundary source: {boundary_metadata.boundary_source}",
                f"- Boundary publisher: {boundary_metadata.publisher or 'unknown'}",
                f"- Boundary source URL: {boundary_metadata.source_url or 'n/a'}",
                f"- Boundary selection reason: {boundary_metadata.source_selection_reason or 'n/a'}",
                f"- Boundary year: {boundary_metadata.boundary_year or 'unknown'}",
                f"- Boundary format: {boundary_metadata.file_format}",
                f"- Boundary CRS: `{boundary_metadata.crs or 'unknown'}`",
                f"- Boundary county features loaded: {boundary_metadata.feature_count}",
                f"- Boundary bounds (lon_min, lat_min, lon_max, lat_max): `{boundary_metadata.bounds}`",
                f"- Boundary column names: `{boundary_metadata.column_names}`",
                f"- GEOID exists: {boundary_metadata.has_geoid}",
                f"- STATEFP and COUNTYFP exist: {boundary_metadata.has_statefp_countyfp}",
                f"- County name exists: {boundary_metadata.has_county_name}",
                f"- Geometry exists: {boundary_metadata.has_geometry}",
                f"- Geometry valid enough for spatial join: {boundary_metadata.geometry_valid_enough}",
                f"- Boundary geometry type: {boundary_metadata.boundary_geometry_type}",
            ]
        )
    if blocker:
        action_lines = [
            "Action required if blocked:",
            "",
            "- Provide a local US county boundary file in zipped shapefile, .shp, .geojson, or .gpkg format.",
            "- For the preferred source, use `tl_2024_us_county.zip` from the official Census TIGER/Line COUNTY directory.",
            "- Rerun `python tools/run_rainfall_p99_pre_retrieval.py --reuse-grid-events --county-boundaries PATH_TO_BOUNDARY_FILE`.",
        ]
    else:
        action_lines = [
            "Mapping outcome:",
            "",
            "- Boundary loading succeeded and produced eligible `mapped_us_land_county` grid cells.",
            "- No reverse-geocoder or nearest-county fallback was used.",
        ]
    text = "\n".join(
        [
            "# Rainfall P99 Mapping Report",
            "",
            f"- Unique grid cells evaluated: {len(lookup)}",
            *boundary_lines,
            f"- Mapping blocker: `{blocker}`" if blocker else "- Mapping blocker: none",
            f"- Mapping status counts: `{dict(sorted(counts.items()))}`",
            f"- Clearly mapped US land counties: {counts.get('mapped_us_land_county', 0)}",
            f"- Outside US grid cells: {counts.get('outside_us', 0)}",
            f"- Offshore/water grid cells: {counts.get('offshore_or_water', 0)}",
            f"- Border-uncertain grid cells: {counts.get('border_uncertain', 0)}",
            f"- Mapping-failed grid cells: {counts.get('mapping_failed', 0)}",
            f"- Invalid-coordinate grid cells: {counts.get('invalid_coordinates', 0)}",
            f"- Lookup CSV: `{output_paths['mapping_lookup_csv']}`",
            f"- Lookup JSONL: `{output_paths['mapping_lookup_jsonl']}`",
            f"- Boundary validation JSON: `{output_paths['boundary_validation']}`",
            "",
            "Eligibility rule:",
            "",
            "- Only `mapped_us_land_county` grid cells are eligible for county-level p99 pilot candidates.",
            "- Mapped county is an administrative anchor only and is not disaster occurrence evidence.",
            "",
            "Mapping diagnostics:",
            "",
            f"- Rainfall point bounds: `{boundary_metadata.diagnostics.get('rainfall_point_bounds') if boundary_metadata else 'n/a'}`",
            f"- Coordinate order check: `{boundary_metadata.diagnostics.get('coordinate_order') if boundary_metadata else 'n/a'}`",
            f"- Known US interior point tests: `{boundary_metadata.diagnostics.get('known_us_interior_point_tests') if boundary_metadata else 'n/a'}`",
            "",
            "Execution boundary:",
            "",
            "- Live retrieval was not run.",
            "- Compound pairing was not run.",
            "- p95 was not processed.",
            "",
            *action_lines,
            "",
            f"Preflight input rows read: {preflight.rows_read}",
        ]
    )
    safe_write_text(path, text)


def write_aggregation_report(
    path: Path,
    *,
    diagnostics: dict[str, Any],
    candidate_path: Path,
    boundary_metadata: BoundaryMetadata | None,
) -> None:
    text = "\n".join(
        [
            "# Rainfall P99 Aggregation Report",
            "",
            f"- Boundary source: {boundary_metadata.boundary_source if boundary_metadata else 'not loaded'}",
            f"- Boundary year: {boundary_metadata.boundary_year if boundary_metadata else 'n/a'}",
            f"- Boundary format: {boundary_metadata.file_format if boundary_metadata else 'n/a'}",
            f"- Boundary CRS: `{boundary_metadata.crs if boundary_metadata else 'n/a'}`",
            f"- Boundary features loaded: {boundary_metadata.feature_count if boundary_metadata else 0}",
            f"- County-level candidate count: {diagnostics.get('candidate_count', 0)}",
            f"- Grid events read: {diagnostics.get('grid_events_read', 0)}",
            f"- Eligible mapped grid events: {diagnostics.get('eligible_mapped_grid_events', 0)}",
            f"- Skipped grid events by mapping status: `{diagnostics.get('skipped_grid_events_by_mapping_status', {})}`",
            f"- County/threshold groups: {diagnostics.get('county_threshold_groups', 0)}",
            f"- Close-window threshold: {diagnostics.get('close_window_days')} day(s)",
            f"- Aggregation rule: {diagnostics.get('aggregation_rule')}",
            f"- Candidate output: `{candidate_path}`",
            "",
            "Candidate semantics:",
            "",
            "- County candidates are rainfall validation units, not public disaster records.",
            "- p99 threshold exceedance is not treated as flood, impact, agriculture, response, or disaster evidence.",
            "",
            "Execution boundary:",
            "",
            "- Live retrieval was not run.",
            "- Compound pairing was not run.",
            "- p95 was not processed.",
            "",
        ]
    )
    safe_write_text(path, text)


def write_pilot_sample_report(
    path: Path,
    *,
    sample: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    sample_path: Path,
    blocker: str | None,
    boundary_metadata: BoundaryMetadata | None,
) -> None:
    summary_lines = [
        f"- {item['candidate_id']}: {item.get('county')}, {item.get('state_abbrev')} "
        f"{item['start_date']} to {item['end_date']}; grids={item.get('unique_grid_cell_count')}; "
        f"reason={item.get('sample_selection_reason')}"
        for item in sample
    ]
    if not summary_lines:
        summary_lines = ["- No candidates selected."]
    text = "\n".join(
        [
            "# Rainfall P99 Pilot Sample Report",
            "",
            f"- Boundary source: {boundary_metadata.boundary_source if boundary_metadata else 'not loaded'}",
            f"- Boundary year: {boundary_metadata.boundary_year if boundary_metadata else 'n/a'}",
            f"- Boundary format: {boundary_metadata.file_format if boundary_metadata else 'n/a'}",
            f"- Boundary CRS: `{boundary_metadata.crs if boundary_metadata else 'n/a'}`",
            f"- Boundary features loaded: {boundary_metadata.feature_count if boundary_metadata else 0}",
            f"- Sample artifact: `{sample_path}`",
            f"- Requested sample size: {diagnostics.get('requested_sample_size')}",
            f"- Selected sample size: {diagnostics.get('selected_sample_size')}",
            f"- Deterministic seed: {diagnostics.get('seed')}",
            f"- Selection mode: {diagnostics.get('selection_mode')}",
            f"- Stratification targets: `{diagnostics.get('strata')}`",
            f"- Duplicate county/date handling: {diagnostics.get('duplicate_same_county_date_avoidance')}",
            f"- Blocker: `{blocker}`" if blocker else "- Blocker: none",
            "",
            "Selected pilot sample summary:",
            "",
            *summary_lines,
            "",
            "Execution boundary:",
            "",
            "- Live retrieval was not run.",
            "- Compound pairing was not run.",
            "- p95 was not processed.",
            "",
        ]
    )
    safe_write_text(path, text)


def run_pre_retrieval_pipeline(
    *,
    input_csv: Path = RAIN_P99_INPUT,
    output_dir: Path = config.OUTPUT_DIR,
    county_boundaries: Path | None = None,
    sample_size: int = 10,
    seed: int = 20260617,
    close_window_days: int = 1,
    row_limit: int | None = None,
    reuse_grid_events: bool = False,
) -> PipelineRunResult:
    output_dir = Path(output_dir)
    county_boundaries = Path(county_boundaries) if county_boundaries else find_default_county_boundary_file()
    paths = {
        "grid_events": output_dir / "events" / "rainfall_grid_events_p99.jsonl",
        "skipped_rows": output_dir / "diagnostics" / "rainfall_p99_skipped_rows.csv",
        "mapping_lookup_csv": output_dir / "geo" / "rainfall_p99_grid_to_county_lookup.csv",
        "mapping_lookup_jsonl": output_dir / "geo" / "rainfall_p99_grid_to_county_lookup.jsonl",
        "boundary_validation": output_dir / "geo" / "rainfall_p99_boundary_validation.json",
        "county_candidates": output_dir / "events" / "rainfall_county_candidates_p99.jsonl",
        "pilot_sample": output_dir / "pilot" / "rainfall_p99_single_event_sample.jsonl",
        "preflight_report": output_dir / "reports" / "rainfall_p99_preflight_report.md",
        "mapping_report": output_dir / "reports" / "rainfall_p99_mapping_report.md",
        "aggregation_report": output_dir / "reports" / "rainfall_p99_aggregation_report.md",
        "pilot_sample_report": output_dir / "reports" / "rainfall_p99_pilot_sample_report.md",
    }
    if reuse_grid_events and paths["grid_events"].exists():
        preflight = summarize_existing_grid_events(paths["grid_events"], source_dataset=Path(input_csv).stem)
    else:
        preflight = stream_validate_and_normalize_p99(
            Path(input_csv),
            output_dir=output_dir,
            expected_threshold="p99",
            row_limit=row_limit,
        )
        write_preflight_report(paths["preflight_report"], preflight, paths)

    lookup, mapping_blocker, boundary_metadata = build_grid_to_county_lookup(preflight.unique_grid_cells, boundary_file=county_boundaries)
    write_mapping_lookup_csv(paths["mapping_lookup_csv"], lookup)
    write_mapping_lookup_jsonl(paths["mapping_lookup_jsonl"], lookup)
    if boundary_metadata:
        safe_write_text(paths["boundary_validation"], json.dumps(boundary_metadata.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        safe_write_text(paths["boundary_validation"], json.dumps({"loaded": False, "blocker": mapping_blocker}, indent=2, sort_keys=True))
    write_mapping_report(
        paths["mapping_report"],
        preflight=preflight,
        lookup=lookup,
        boundary_file=county_boundaries,
        boundary_metadata=boundary_metadata,
        blocker=mapping_blocker,
        output_paths=paths,
    )

    counts = Counter(item.mapping_status for item in lookup.values())
    if counts.get("mapped_us_land_county", 0) and paths["grid_events"].exists():
        candidates, aggregation_diagnostics = aggregate_county_candidates(
            paths["grid_events"],
            lookup,
            close_window_days=close_window_days,
        )
    else:
        if len(counts) == 1:
            only_status = next(iter(counts))
            skipped_event_status_counts = {only_status: preflight.valid_rows}
        else:
            skipped_event_status_counts = dict(sorted(counts.items()))
        candidates = []
        aggregation_diagnostics = {
            "grid_events_read": preflight.valid_rows,
            "eligible_mapped_grid_events": 0,
            "skipped_grid_events_by_mapping_status": skipped_event_status_counts,
            "county_threshold_groups": 0,
            "candidate_count": 0,
            "close_window_days": close_window_days,
            "aggregation_rule": "same county FIPS, same threshold, overlapping or gap <= close_window_days",
            "live_retrieval_run": False,
            "compound_pairing_run": False,
        }
    write_jsonl(paths["county_candidates"], candidates)
    write_aggregation_report(
        paths["aggregation_report"],
        diagnostics=aggregation_diagnostics,
        candidate_path=paths["county_candidates"],
        boundary_metadata=boundary_metadata,
    )

    sample, sample_diagnostics = select_pilot_sample(candidates, sample_size=sample_size, seed=seed)
    write_jsonl(paths["pilot_sample"], sample)
    blocker = mapping_blocker or (None if sample else "no_eligible_county_candidates_for_sample")
    write_pilot_sample_report(
        paths["pilot_sample_report"],
        sample=sample,
        diagnostics=sample_diagnostics,
        sample_path=paths["pilot_sample"],
        blocker=blocker,
        boundary_metadata=boundary_metadata,
    )

    return PipelineRunResult(
        preflight=preflight,
        mapping_status_counts=counts,
        candidate_count=len(candidates),
        sample_count=len(sample),
        paths=paths,
        blocker=blocker,
        boundary_metadata=boundary_metadata,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the rainfall-only p99 pre-retrieval pipeline stage.")
    parser.add_argument("--input", type=Path, default=RAIN_P99_INPUT, help="p99 rainfall event-window CSV.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--county-boundaries",
        type=Path,
        default=None,
        help="Local US county GeoJSON FeatureCollection. No external geocoding is used.",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--close-window-days", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=None, help="Optional development limit; omit for full p99 preflight.")
    parser.add_argument(
        "--reuse-grid-events",
        action="store_true",
        help="Reuse outputs/events/rainfall_grid_events_p99.jsonl and rerun only mapping, aggregation, and sampling.",
    )
    args = parser.parse_args(argv)

    result = run_pre_retrieval_pipeline(
        input_csv=args.input,
        output_dir=args.output_dir,
        county_boundaries=args.county_boundaries,
        sample_size=args.sample_size,
        seed=args.seed,
        close_window_days=args.close_window_days,
        row_limit=args.row_limit,
        reuse_grid_events=args.reuse_grid_events,
    )
    print(f"rows_read={result.preflight.rows_read}")
    print(f"valid_rows={result.preflight.valid_rows}")
    print(f"skipped_rows={result.preflight.skipped_rows}")
    print(f"mapping_status_counts={dict(sorted(result.mapping_status_counts.items()))}")
    print(f"county_candidates={result.candidate_count}")
    print(f"pilot_sample={result.sample_count}")
    if result.blocker:
        print(f"blocker={result.blocker}")
    print(f"sample_path={result.paths['pilot_sample']}")
    return 0 if result.preflight.rows_read or result.preflight.skipped_reason_counts.get("input_csv_missing", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
