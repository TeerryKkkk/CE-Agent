from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIFORNIA_REPRO_DIR = REPO_ROOT / "outputs" / "california_positive_case_pipeline_reproduction"
DEFAULT_NOAA_CACHE_DIR = REPO_ROOT / "outputs" / "single_event_impact_region_audit" / "source_cache" / "noaa_stormevents"
DEFAULT_USDM_CACHE_DIR = CALIFORNIA_REPRO_DIR / "source_cache" / "usdm"
DEFAULT_OPENFEMA_CACHE_DIR = CALIFORNIA_REPRO_DIR / "source_cache" / "openfema"
NOAA_SOURCE_URL_TEMPLATE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/{name}"
OPENFEMA_DECLARATIONS_API_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
USDM_COUNTY_API_URL = (
    "https://usdmdataservices.unl.edu/api/CountyStatistics/"
    "GetDroughtSeverityStatisticsByAreaPercent"
)
USDM_CACHE_FIELDNAMES = [
    "MapDate",
    "FIPS",
    "County",
    "State",
    "None",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "ValidStart",
    "ValidEnd",
    "StatisticFormatID",
]

WET_EVENT_TYPES = {
    "coastal flood",
    "debris flow",
    "flash flood",
    "flood",
    "heavy rain",
    "high surf",
    "high wind",
    "marine thunderstorm wind",
    "storm surge/tide",
    "strong wind",
    "thunderstorm wind",
    "tropical storm",
    "winter storm",
}
OPENFEMA_RELEVANT_INCIDENT_TERMS = {
    "atmospheric river",
    "flood",
    "flooding",
    "heavy rain",
    "landslide",
    "landslides",
    "mudslide",
    "mudslides",
    "severe storm",
    "storm",
    "straight-line wind",
    "tropical storm",
    "winter storm",
}
OPENFEMA_UNRELATED_INCIDENT_TERMS = {
    "biological",
    "covid",
    "drought",
    "earthquake",
    "fire",
    "pandemic",
    "terror",
    "toxic",
    "volcan",
}


@dataclass(frozen=True)
class OfficialSourceLaneControls:
    enable_noaa_storm_events: bool = True
    enable_usdm_county_statistics: bool = True
    enable_usdm_live_api_fallback: bool = True
    enable_openfema_admin_response: bool = False
    noaa_cache_dir: Path = DEFAULT_NOAA_CACHE_DIR
    usdm_cache_dir: Path = DEFAULT_USDM_CACHE_DIR
    openfema_cache_dir: Path = DEFAULT_OPENFEMA_CACHE_DIR
    noaa_max_matches_per_case: int = 4
    usdm_api_timeout_seconds: float = 20.0
    usdm_write_live_api_cache: bool = True
    usdm_d1_area_percent_threshold: float = 0.0
    openfema_api_timeout_seconds: float = 20.0
    openfema_write_live_api_cache: bool = True
    openfema_same_storm_sequence_days: int = 7


@dataclass(frozen=True)
class NoaaMatch:
    row: dict[str, str]
    source_url: str
    source_name: str
    begin_date: date
    end_date: date
    exact_window: bool
    close_window: bool
    impact_channels: tuple[str, ...]
    severity_score: float


@dataclass(frozen=True)
class StructuredOfficialLaneResult:
    evidence_rows: list[dict[str, Any]]
    retrieval_log_rows: list[dict[str, Any]]
    attempted_lanes: tuple[str, ...]
    support_flags: dict[str, bool]


@dataclass(frozen=True)
class OpenFemaAdminLaneResult:
    evidence_rows: list[dict[str, Any]]
    retrieval_log_rows: list[dict[str, Any]]
    attempted_lanes: tuple[str, ...]
    admin_response_flags: dict[str, bool]
    source_mode: str
    source_check_status: str
    failure_reason: str = ""


@dataclass(frozen=True)
class OpenFemaLookupResult:
    rows: list[dict[str, Any]]
    cache_status: str
    fetch_status: str
    source_check_status: str
    source_mode: str = "local_cache"
    source_url: str = OPENFEMA_DECLARATIONS_API_URL
    failure_reason: str = ""


@dataclass(frozen=True)
class UsdmLookupResult:
    row: dict[str, str] | None
    usdm_date: date | None
    cache_path: Path | None
    source_mode: str
    source_url: str = ""
    failure_reason: str = ""


@dataclass(frozen=True)
class NoaaIndexResult:
    index: dict[str, list[tuple[dict[str, str], str, str]]]
    cache_status: str
    fetch_status: str
    source_check_status: str
    loaded_years: tuple[str, ...]
    missing_years: tuple[str, ...]
    source_mode: str = "local_cache"
    failure_reason: str = ""


@dataclass(frozen=True)
class NoaaLookupResult:
    matches: list[NoaaMatch]
    cache_status: str
    fetch_status: str
    source_check_status: str
    loaded_years: tuple[str, ...]
    missing_years: tuple[str, ...]
    matched_record_count: int
    source_mode: str = "local_cache"
    failure_reason: str = ""


def _candidate_id(case: dict[str, Any]) -> str:
    return str(case.get("raw_candidate_id") or case.get("phase2_3_sample_id") or case.get("candidate_id") or "")


def _fips(case: dict[str, Any]) -> str:
    digits = re.sub(r"\D", "", str(case.get("FIPS") or case.get("county_fips") or ""))
    return digits.zfill(5) if digits else ""


def _county(case: dict[str, Any]) -> str:
    return str(case.get("county") or "")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _openfema_cache_path(controls: OfficialSourceLaneControls, state: str = "CA") -> Path:
    return controls.openfema_cache_dir / f"openfema_disaster_declarations_{state.lower()}.json"


def _openfema_query_url(state: str = "CA", *, top: int = 5000) -> str:
    params = {
        "$filter": f"state eq '{state}'",
        "$orderby": "disasterNumber,designatedArea",
        "$top": str(top),
    }
    return f"{OPENFEMA_DECLARATIONS_API_URL}?{urllib.parse.urlencode(params)}"


def _fetch_openfema_rows_live(
    *,
    state: str = "CA",
    timeout_seconds: float = 20.0,
) -> tuple[list[dict[str, Any]], str, str]:
    source_url = _openfema_query_url(state)
    request = urllib.request.Request(
        source_url,
        headers={"Accept": "application/json", "User-Agent": "CE-Agent OpenFEMA admin response lane"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return [], source_url, f"OpenFEMA live API request failed: {exc}"
    rows = payload.get("DisasterDeclarationsSummaries")
    if not isinstance(rows, list):
        return [], source_url, "OpenFEMA live API response did not contain DisasterDeclarationsSummaries rows."
    return [row for row in rows if isinstance(row, dict)], source_url, ""


def _write_openfema_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump({"DisasterDeclarationsSummaries": rows}, handle, ensure_ascii=True, sort_keys=True)
    os.replace(tmp_path, path)


def openfema_source_rows(
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
    state: str = "CA",
) -> OpenFemaLookupResult:
    cache_path = _openfema_cache_path(controls, state)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("DisasterDeclarationsSummaries")
            if isinstance(rows, list):
                return OpenFemaLookupResult(
                    rows=[row for row in rows if isinstance(row, dict)],
                    cache_status="cache_hit",
                    fetch_status="local_structured_cache_hit",
                    source_check_status="success",
                    source_mode="local_cache",
                    source_url=f"structured-cache://openfema/{state.lower()}",
                )
        except (OSError, json.JSONDecodeError) as exc:
            return OpenFemaLookupResult(
                rows=[],
                cache_status="cache_read_failed",
                fetch_status="cache_read_failed",
                source_check_status="lane_failed",
                source_mode="local_cache",
                source_url=f"structured-cache://openfema/{state.lower()}",
                failure_reason=f"OpenFEMA cache could not be read: {cache_path.name}: {exc}",
            )

    rows, source_url, error = _fetch_openfema_rows_live(
        state=state,
        timeout_seconds=controls.openfema_api_timeout_seconds,
    )
    if error:
        return OpenFemaLookupResult(
            rows=[],
            cache_status="cache_miss",
            fetch_status="live_api_fallback_failed",
            source_check_status="lane_failed",
            source_mode="live_api_fallback_failed",
            source_url=source_url,
            failure_reason=error,
        )
    if controls.openfema_write_live_api_cache:
        _write_openfema_cache(cache_path, rows)
    return OpenFemaLookupResult(
        rows=rows,
        cache_status="cache_miss",
        fetch_status="live_api_fallback_success",
        source_check_status="success",
        source_mode="live_api_fallback",
        source_url=source_url,
    )


def _normalize_county_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("(county)", " county")
    text = re.sub(r"\bcounty\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _openfema_designated_county(row: dict[str, Any]) -> str:
    return str(row.get("designatedArea") or row.get("designated_area") or "").strip()


def _openfema_county_matches(case: dict[str, Any], row: dict[str, Any]) -> bool:
    area = _openfema_designated_county(row)
    if "(county)" not in area.lower():
        return False
    candidate_fips = _fips(case)
    state_code = re.sub(r"\D", "", str(row.get("fipsStateCode") or row.get("stateFipsCode") or ""))
    county_code = re.sub(r"\D", "", str(row.get("fipsCountyCode") or row.get("countyFipsCode") or ""))
    if candidate_fips and state_code and county_code:
        row_fips = f"{state_code.zfill(2)}{county_code.zfill(3)}"
        if row_fips == candidate_fips:
            return True
    return _normalize_county_name(area) == _normalize_county_name(_county(case))


def _openfema_state_matches(row: dict[str, Any], state: str = "CA") -> bool:
    row_state = str(row.get("state") or row.get("stateCode") or "").strip().upper()
    row_fips = str(row.get("fipsStateCode") or row.get("stateFipsCode") or "").strip().zfill(2)
    return row_state == state.upper() or (state.upper() == "CA" and row_fips == "06")


def _openfema_incident_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "incidentType",
            "declarationTitle",
            "incidentTitle",
            "title",
            "incidentDescription",
        )
    ).lower()


def _openfema_incident_relevant(row: dict[str, Any]) -> bool:
    text = _openfema_incident_text(row)
    if any(term in text for term in OPENFEMA_UNRELATED_INCIDENT_TERMS):
        return False
    return any(term in text for term in OPENFEMA_RELEVANT_INCIDENT_TERMS)


def _openfema_administrative_signal(row: dict[str, Any]) -> bool:
    program_fields = (
        "paProgramDeclared",
        "iaProgramDeclared",
        "ihProgramDeclared",
        "hmProgramDeclared",
    )
    if any(str(row.get(field)).strip().lower() in {"true", "1", "yes"} for field in program_fields):
        return True
    declaration_type = str(row.get("declarationType") or "").strip().upper()
    if declaration_type in {"DR", "EM", "FM"}:
        return True
    return bool(str(row.get("disasterNumber") or "").strip() and _openfema_designated_county(row))


def _openfema_date_alignment(
    case: dict[str, Any],
    row: dict[str, Any],
    *,
    same_storm_sequence_days: int,
) -> tuple[bool, str]:
    rain_start = _parse_date(case.get("rain_start"))
    rain_end = _parse_date(case.get("rain_end")) or rain_start
    incident_start = _parse_date(row.get("incidentBeginDate") or row.get("incident_begin_date"))
    incident_end = _parse_date(row.get("incidentEndDate") or row.get("incident_end_date")) or incident_start
    if not rain_start or not rain_end or not incident_start or not incident_end:
        return False, "date_not_evaluable"
    if incident_start <= rain_end and incident_end >= rain_start:
        return True, "exact_window"
    close_start = rain_start - timedelta(days=same_storm_sequence_days)
    close_end = rain_end + timedelta(days=same_storm_sequence_days)
    if incident_start <= close_end and incident_end >= close_start:
        return True, "same_storm_sequence"
    return False, "outside_window"


def _openfema_programs(row: dict[str, Any]) -> str:
    programs: list[str] = []
    if str(row.get("paProgramDeclared")).strip().lower() in {"true", "1", "yes"}:
        programs.append("Public Assistance")
    if str(row.get("iaProgramDeclared")).strip().lower() in {"true", "1", "yes"}:
        programs.append("Individual Assistance")
    if str(row.get("ihProgramDeclared")).strip().lower() in {"true", "1", "yes"}:
        programs.append("Individual and Households Program")
    if str(row.get("hmProgramDeclared")).strip().lower() in {"true", "1", "yes"}:
        programs.append("Hazard Mitigation")
    declaration_type = str(row.get("declarationType") or "").strip().upper()
    if declaration_type:
        programs.append(f"Declaration Type {declaration_type}")
    return "; ".join(programs)


def _openfema_disaster_url(row: dict[str, Any], source_url: str = OPENFEMA_DECLARATIONS_API_URL) -> str:
    disaster_number = str(row.get("disasterNumber") or "").strip()
    if disaster_number:
        filter_text = f"state eq 'CA' and disasterNumber eq {disaster_number}"
        return f"{OPENFEMA_DECLARATIONS_API_URL}?{urllib.parse.urlencode({'$filter': filter_text})}"
    return source_url


def _parse_noaa_datetime(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def previous_or_same_tuesday(day: date) -> date:
    return day - timedelta(days=(day.weekday() - 1) % 7)


def _usdm_query_url(fips: str, map_date: date) -> str:
    return (
        f"{USDM_COUNTY_API_URL}?aoi={fips}&startdate={map_date.isoformat()}&"
        f"enddate={map_date.isoformat()}&statisticsType=1"
    )


def _parse_usdm_csv(text: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


def _find_usdm_row(rows: list[dict[str, str]], fips: str) -> dict[str, str] | None:
    normalized_fips = str(fips).zfill(5)
    return next((row for row in rows if str(row.get("FIPS") or "").zfill(5) == normalized_fips), None)


def _write_usdm_cache(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field for field in USDM_CACHE_FIELDNAMES if field in row]
    fieldnames.extend(field for field in row.keys() if field not in fieldnames)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    os.replace(tmp_path, path)


def _fetch_usdm_live_row(
    fips: str,
    usdm_date: date,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, str] | None, str, str]:
    source_url = _usdm_query_url(fips, usdm_date)
    request = urllib.request.Request(
        source_url,
        headers={"Accept": "text/csv", "User-Agent": "CE-Agent USDM structured lane"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8-sig", errors="replace")
    except (TimeoutError, urllib.error.URLError) as exc:
        return None, source_url, f"USDM live API request failed: {exc}"

    rows = _parse_usdm_csv(text)
    if not rows:
        return None, source_url, "USDM live API returned no CSV rows."
    row = _find_usdm_row(rows, fips)
    if not row:
        return None, source_url, f"USDM live API returned rows, but none matched FIPS {fips}."
    return row, source_url, ""


def _parse_float(value: Any) -> float:
    try:
        return float(str(value or "").strip() or "0")
    except ValueError:
        return 0.0


def _parse_usdm_percent(row: dict[str, str], key: str) -> float:
    text = str(row.get(key) or "").strip()
    if not text:
        raise ValueError(f"USDM row is missing {key} percent.")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"USDM row has invalid {key} percent: {text!r}.") from exc


def _parse_int(value: Any) -> int:
    try:
        return int(float(str(value or "").strip() or "0"))
    except ValueError:
        return 0


def _parse_damage(value: Any) -> float:
    text = str(value or "").strip().upper()
    if not text:
        return 0.0
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([KMB]?)$", text)
    if not match:
        return 0.0
    amount = float(match.group(1))
    multiplier = {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[match.group(2)]
    return amount * multiplier


def _impact_channels_for_noaa(row: dict[str, str]) -> tuple[str, ...]:
    narrative = f"{row.get('EPISODE_NARRATIVE', '')} {row.get('EVENT_NARRATIVE', '')}".lower()
    channels: set[str] = set()
    if _parse_damage(row.get("DAMAGE_PROPERTY", "")) > 0:
        channels.add("property / housing")
    if _parse_damage(row.get("DAMAGE_CROPS", "")) > 0:
        channels.add("agriculture")
    injuries = _parse_int(row.get("INJURIES_DIRECT", "")) + _parse_int(row.get("INJURIES_INDIRECT", ""))
    deaths = _parse_int(row.get("DEATHS_DIRECT", "")) + _parse_int(row.get("DEATHS_INDIRECT", ""))
    if injuries or deaths or re.search(r"\b(rescu|fatal|drown|injur|death|evacuat|shelter)\w*", narrative):
        channels.add("human impact")
    if re.search(
        r"\b(road|route|highway|freeway|interstate|sr\s*\d+|i-\d+|bridge|closure|closed|traffic|vehicle|mudslide|debris flow)\b",
        narrative,
    ):
        channels.add("roads / transport")
    if re.search(r"\b(power|outage|electric|utility|utilities|transmission|substation|gas line|water line)\b", narrative):
        channels.add("energy / utility")
    if re.search(r"\b(levee|dam|water system|wastewater|school|public works|oes|emergency manager|emergency operations|shelter)\b", narrative):
        channels.add("public services / water")
    if re.search(r"\b(crop|orchard|farm|agricultur|livestock|dairy)\b", narrative):
        channels.add("agriculture")
    return tuple(sorted(channels))


def _noaa_severity_score(row: dict[str, str], channels: tuple[str, ...], exact_window: bool) -> float:
    damage = _parse_damage(row.get("DAMAGE_PROPERTY", "")) + _parse_damage(row.get("DAMAGE_CROPS", ""))
    people = (
        _parse_int(row.get("INJURIES_DIRECT", ""))
        + _parse_int(row.get("INJURIES_INDIRECT", ""))
        + 5 * _parse_int(row.get("DEATHS_DIRECT", ""))
        + 5 * _parse_int(row.get("DEATHS_INDIRECT", ""))
    )
    return (100 if exact_window else 0) + len(channels) * 10 + people * 5 + min(damage / 100_000.0, 100.0)


def _noaa_files_for_years(years: tuple[str, ...], cache_dir: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {year: [] for year in years}
    if not cache_dir.exists():
        return files
    for path in sorted(cache_dir.glob("StormEvents_details-ftp_v1.0_d*.csv.gz")):
        match = re.search(r"_d(\d{4})_", path.name)
        if match and match.group(1) in set(years):
            files.setdefault(match.group(1), []).append(path)
    return files


@lru_cache(maxsize=16)
def _load_noaa_index(years: tuple[str, ...], cache_dir_text: str) -> NoaaIndexResult:
    cache_dir = Path(cache_dir_text)
    index: dict[str, list[tuple[dict[str, str], str, str]]] = defaultdict(list)
    if not years:
        return NoaaIndexResult(
            index={},
            cache_status="not_attempted_missing_inputs",
            fetch_status="not_attempted_missing_inputs",
            source_check_status="lane_failed",
            loaded_years=(),
            missing_years=(),
            failure_reason="NOAA lane skipped because wet-event dates were missing.",
        )
    if not cache_dir.exists():
        return NoaaIndexResult(
            index={},
            cache_status="cache_dir_missing",
            fetch_status="cache_dir_missing",
            source_check_status="lane_failed",
            loaded_years=(),
            missing_years=years,
            failure_reason=f"NOAA Storm Events cache directory was missing: {cache_dir}",
        )

    files_by_year = _noaa_files_for_years(years, cache_dir)
    missing_years = tuple(year for year in years if not files_by_year.get(year))
    loaded_years: set[str] = set()
    if missing_years:
        return NoaaIndexResult(
            index={},
            cache_status="year_file_missing",
            fetch_status="year_file_missing",
            source_check_status="lane_failed",
            loaded_years=(),
            missing_years=missing_years,
            failure_reason=f"NOAA Storm Events cache files were missing for year(s): {', '.join(missing_years)}.",
        )

    for year in years:
        for path in files_by_year.get(year, []):
            source_url = NOAA_SOURCE_URL_TEMPLATE.format(name=path.name)
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        if str(row.get("STATE") or "").upper() != "CALIFORNIA":
                            continue
                        event_type = str(row.get("EVENT_TYPE") or "").lower()
                        if event_type not in WET_EVENT_TYPES and not any(
                            term in event_type for term in ("flood", "rain", "storm", "wind", "debris")
                        ):
                            continue
                        if str(row.get("CZ_TYPE") or "").upper() != "C":
                            continue
                        fips = f"{str(row.get('STATE_FIPS') or '').zfill(2)}{str(row.get('CZ_FIPS') or '').zfill(3)}"
                        index[fips].append((row, source_url, path.name))
                loaded_years.add(year)
            except (OSError, EOFError, gzip.BadGzipFile, csv.Error, UnicodeDecodeError) as exc:
                return NoaaIndexResult(
                    index={},
                    cache_status="cache_read_failed",
                    fetch_status="cache_read_failed",
                    source_check_status="lane_failed",
                    loaded_years=tuple(sorted(loaded_years)),
                    missing_years=(),
                    failure_reason=f"NOAA Storm Events cache file could not be read: {path.name}: {exc}",
                )

    return NoaaIndexResult(
        index=dict(index),
        cache_status="cache_hit",
        fetch_status="local_structured_cache_hit",
        source_check_status="success",
        loaded_years=tuple(sorted(loaded_years)),
        missing_years=(),
    )


def match_noaa_storm_events_for_case(
    case: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
) -> list[NoaaMatch]:
    return noaa_lookup_for_case(case, controls=controls).matches


def noaa_lookup_for_case(
    case: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
) -> NoaaLookupResult:
    wet_start = _parse_date(case.get("rain_start"))
    wet_end = _parse_date(case.get("rain_end")) or wet_start
    fips = _fips(case)
    if not wet_start or not wet_end or not fips:
        return NoaaLookupResult(
            matches=[],
            cache_status="not_attempted_missing_inputs",
            fetch_status="not_attempted_missing_inputs",
            source_check_status="lane_failed",
            loaded_years=(),
            missing_years=(),
            matched_record_count=0,
            failure_reason="NOAA lane skipped because rain_start, rain_end, or county FIPS was missing.",
        )
    years = tuple(sorted({str(case.get("rainfall_year") or wet_start.year), str(wet_start.year), str(wet_end.year)}))
    index_result = _load_noaa_index(years, str(controls.noaa_cache_dir))
    if index_result.source_check_status != "success":
        return NoaaLookupResult(
            matches=[],
            cache_status=index_result.cache_status,
            fetch_status=index_result.fetch_status,
            source_check_status=index_result.source_check_status,
            loaded_years=index_result.loaded_years,
            missing_years=index_result.missing_years,
            matched_record_count=0,
            source_mode=index_result.source_mode,
            failure_reason=index_result.failure_reason,
        )
    exact_start = wet_start - timedelta(days=1)
    exact_end = wet_end + timedelta(days=1)
    close_start = wet_start - timedelta(days=7)
    close_end = wet_end + timedelta(days=7)
    matches: list[NoaaMatch] = []
    candidate_rows = index_result.index.get(fips, [])
    for row, source_url, source_name in candidate_rows:
        begin = _parse_noaa_datetime(row.get("BEGIN_DATE_TIME", ""))
        end = _parse_noaa_datetime(row.get("END_DATE_TIME", "")) or begin
        if not begin or not end:
            continue
        begin_date = begin.date()
        end_date = end.date()
        exact_window = begin_date <= exact_end and end_date >= exact_start
        close_window = begin_date <= close_end and end_date >= close_start
        if not close_window:
            continue
        channels = _impact_channels_for_noaa(row)
        matches.append(
            NoaaMatch(
                row=row,
                source_url=source_url,
                source_name=source_name,
                begin_date=begin_date,
                end_date=end_date,
                exact_window=exact_window,
                close_window=close_window,
                impact_channels=channels,
                severity_score=_noaa_severity_score(row, channels, exact_window),
            )
        )
    matches.sort(key=lambda item: item.severity_score, reverse=True)
    return NoaaLookupResult(
        matches=matches[: controls.noaa_max_matches_per_case],
        cache_status=index_result.cache_status,
        fetch_status="local_structured_cache_match" if matches else "local_structured_cache_no_match",
        source_check_status="success",
        loaded_years=index_result.loaded_years,
        missing_years=index_result.missing_years,
        matched_record_count=len(matches),
        source_mode=index_result.source_mode,
    )


def usdm_row_for_case(
    case: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
) -> UsdmLookupResult:
    drought_end = _parse_date(case.get("drought_end_month_end"))
    fips = _fips(case)
    if not drought_end or not fips:
        return UsdmLookupResult(
            row=None,
            usdm_date=None,
            cache_path=None,
            source_mode="not_attempted",
            failure_reason="USDM lane skipped because drought_end_month_end or county FIPS was missing.",
        )
    usdm_date = previous_or_same_tuesday(drought_end)
    path = controls.usdm_cache_dir / f"usdm_{fips}_{usdm_date.isoformat()}.csv"
    cache_failure = ""
    if path.exists():
        rows = _read_csv(path)
        row = _find_usdm_row(rows, fips) or (rows[0] if rows else None)
        if row:
            return UsdmLookupResult(
                row=row,
                usdm_date=usdm_date,
                cache_path=path,
                source_mode="local_cache",
                source_url=f"structured-cache://usdm/{fips}/{usdm_date.isoformat()}",
            )
        cache_failure = f"Local USDM cache file was empty or did not contain FIPS {fips}: {path.name}."
    else:
        cache_failure = f"Local USDM cache file was missing: {path.name}."

    if not controls.enable_usdm_live_api_fallback:
        return UsdmLookupResult(
            row=None,
            usdm_date=usdm_date,
            cache_path=path,
            source_mode="local_cache_miss",
            failure_reason=cache_failure,
        )

    row, source_url, api_error = _fetch_usdm_live_row(
        fips,
        usdm_date,
        timeout_seconds=controls.usdm_api_timeout_seconds,
    )
    if row:
        if controls.usdm_write_live_api_cache:
            _write_usdm_cache(path, row)
        return UsdmLookupResult(
            row=row,
            usdm_date=usdm_date,
            cache_path=path,
            source_mode="live_api_fallback",
            source_url=source_url,
        )
    return UsdmLookupResult(
        row=None,
        usdm_date=usdm_date,
        cache_path=path,
        source_mode="live_api_fallback_failed",
        source_url=source_url,
        failure_reason=f"{cache_failure} Live USDM API fallback failed: {api_error}",
    )


def _base_evidence_row(case: dict[str, Any]) -> dict[str, Any]:
    cid = _candidate_id(case)
    return {
        "candidate_id": cid,
        "independent_case_id": cid,
        "pipeline_case_id_if_any": cid,
        "county": _county(case),
        "source_url": "",
        "source_title": "",
        "source_family": "",
        "evidence_origin": "structured_official_first",
        "fetch_status": "",
        "final_status": "uninitialized",
        "component_supported": "unknown",
        "impact_supported": "",
        "impact_types": "",
        "wet_impact_support": "",
        "drought_impact_support": "",
        "explicit_transition_support": "",
        "location_match": "same_county",
        "time_match": "exact_window",
        "page_type": "structured_official_record",
        "quoted_supporting_spans": "",
        "final_reason": "",
        "retrieval_tier": "structured_official_first",
        "source_lane": "",
        "officiality": "federal_official_structured",
        "county_match": "true",
        "time_alignment": "aligned",
        "component_support_type": "",
        "structured_record_id": "",
        "source_dataset": "",
    }


def usdm_evidence_row(case: dict[str, Any], *, controls: OfficialSourceLaneControls = OfficialSourceLaneControls()) -> dict[str, Any]:
    lookup = usdm_row_for_case(case, controls=controls)
    row = lookup.row
    usdm_date = lookup.usdm_date
    path = lookup.cache_path
    evidence = _base_evidence_row(case)
    evidence.update(
        {
            "source_title": "US Drought Monitor County Statistics",
            "source_family": "usdm_drought_official_structured",
            "source_lane": "usdm_county_statistics",
            "source_dataset": "USDM county statistics",
            "component_support_type": "drought_hazard",
            "time_alignment": "drought_end_week",
            "supports_drought": "",
            "usdm_source_mode": lookup.source_mode,
        }
    )
    if not row:
        public_status = "api_failed" if lookup.source_mode == "live_api_fallback_failed" else "lane_failed"
        evidence.update(
            {
                "source_url": lookup.source_url,
                "quoted_supporting_spans": "USDM county-statistics row could not be retrieved for the candidate drought-end week.",
                "final_status": "lane_failed",
                "final_reason": lookup.failure_reason
                or "Structured USDM lane could not retrieve county/week data.",
                "fetch_status": lookup.source_mode,
                "structured_record_id": str(path.name if path else ""),
                "public_drought_check_status": public_status,
            }
        )
        return evidence

    try:
        d0 = _parse_usdm_percent(row, "D0")
        d1 = _parse_usdm_percent(row, "D1")
        d2 = _parse_usdm_percent(row, "D2")
        d3 = _parse_usdm_percent(row, "D3")
        d4 = _parse_usdm_percent(row, "D4")
    except ValueError as exc:
        evidence.update(
            {
                "source_url": lookup.source_url,
                "quoted_supporting_spans": "USDM county-statistics row was retrieved, but severity percentages could not be parsed.",
                "final_status": "lane_failed",
                "final_reason": str(exc),
                "fetch_status": "parse_failed",
                "structured_record_id": str(path.name if path else ""),
                "public_drought_check_status": "lane_failed",
            }
        )
        return evidence

    source_date = str(row.get("ValidStart") or row.get("MapDate") or (usdm_date.isoformat() if usdm_date else ""))
    supports = d1 > controls.usdm_d1_area_percent_threshold
    live_fallback = lookup.source_mode == "live_api_fallback"
    evidence.update(
        {
            "source_title": (
                "US Drought Monitor County Statistics live API"
                if live_fallback
                else "US Drought Monitor County Statistics cache"
            ),
            "source_url": lookup.source_url if live_fallback else f"structured-cache://usdm/{_fips(case)}/{source_date}",
            "source_date_if_available": source_date,
            "quoted_supporting_spans": (
                f"USDM county statistics for {_county(case)} near drought-window end: "
                f"D0+ {d0:.2f}%, D1+ {d1:.2f}%, D2+ {d2:.2f}%, D3+ {d3:.2f}%, D4 {d4:.2f}%."
            ),
            "final_status": "accepted" if supports else "rejected",
            "component_supported": "drought" if supports else "none",
            "supports_drought": str(supports).lower(),
            "final_reason": (
                "Official county-level USDM structured row supports antecedent drought context."
                if supports
                else "Official USDM structured row was checked, but D1+ is not above the qualifying threshold."
            ),
            "fetch_status": "live_api_fallback_success" if live_fallback else "local_cache_hit",
            "structured_record_id": f"usdm_{_fips(case)}_{source_date}",
            "public_drought_check_status": "live_api_fallback_success" if live_fallback else "success",
            "usdm_d0_area_percent": f"{d0:.2f}",
            "usdm_d1_area_percent": f"{d1:.2f}",
            "usdm_d2_area_percent": f"{d2:.2f}",
            "usdm_d3_area_percent": f"{d3:.2f}",
            "usdm_d4_area_percent": f"{d4:.2f}",
        }
    )
    return evidence


def _noaa_diagnostics(lookup: NoaaLookupResult) -> dict[str, str]:
    return {
        "source_mode": lookup.source_mode,
        "cache_status": lookup.cache_status,
        "source_check_status": lookup.source_check_status,
        "failure_reason": lookup.failure_reason,
        "loaded_years": ";".join(lookup.loaded_years),
        "missing_years": ";".join(lookup.missing_years),
        "matched_record_count": str(lookup.matched_record_count),
    }


def noaa_evidence_row(case: dict[str, Any], match: NoaaMatch, lookup: NoaaLookupResult | None = None) -> dict[str, Any]:
    row = match.row
    evidence = _base_evidence_row(case)
    channels = "; ".join(match.impact_channels)
    narrative = " ".join(
        part.strip() for part in [row.get("EPISODE_NARRATIVE", ""), row.get("EVENT_NARRATIVE", "")] if part.strip()
    )
    narrative = re.sub(r"\s+", " ", narrative)[:700]
    supports_impact = bool(match.impact_channels)
    evidence.update(
        {
            "source_title": "NOAA/NCEI Storm Events Details CSV",
            "source_url": match.source_url,
            "source_family": "noaa_ncei_storm_events_structured",
            "source_date_if_available": match.begin_date.isoformat(),
            "fetch_status": "local_structured_cache_hit",
            "source_mode": lookup.source_mode if lookup else "local_cache",
            "cache_status": lookup.cache_status if lookup else "cache_hit",
            "source_check_status": lookup.source_check_status if lookup else "success",
            "failure_reason": "",
            "loaded_years": ";".join(lookup.loaded_years) if lookup else "",
            "missing_years": ";".join(lookup.missing_years) if lookup else "",
            "matched_record_count": str(lookup.matched_record_count if lookup else 1),
            "final_status": "accepted" if supports_impact else "context_only",
            "component_supported": "wet",
            "impact_supported": str(supports_impact).lower(),
            "impact_types": channels,
            "wet_impact_support": str(supports_impact).lower(),
            "location_match": "same_county",
            "time_match": "exact_window" if match.exact_window else "same_storm_sequence",
            "quoted_supporting_spans": (
                f"NOAA event {row.get('EVENT_ID')} ({row.get('EVENT_TYPE')}) in {row.get('CZ_NAME')} County; "
                f"property={row.get('DAMAGE_PROPERTY') or '0'}, crops={row.get('DAMAGE_CROPS') or '0'}, "
                f"deaths={row.get('DEATHS_DIRECT') or '0'}/{row.get('DEATHS_INDIRECT') or '0'}, "
                f"injuries={row.get('INJURIES_DIRECT') or '0'}/{row.get('INJURIES_INDIRECT') or '0'}. "
                f"Narrative: {narrative}"
            ),
            "final_reason": (
                "Structured NOAA Storm Events row matches candidate county and wet-event window/close storm sequence; impact channels were extracted from damage, casualty, and narrative fields."
                if supports_impact
                else "Structured NOAA Storm Events row matches county/window but contains no direct impact signal under the conservative structured extractor."
            ),
            "source_lane": "noaa_ncei_storm_events",
            "source_dataset": "NOAA/NCEI Storm Events Details CSV",
            "component_support_type": "wet_event_and_impact" if supports_impact else "wet_event_hazard",
            "time_alignment": "exact_window" if match.exact_window else "close_storm_sequence",
            "structured_record_id": str(row.get("EVENT_ID") or ""),
        }
    )
    return evidence


def noaa_no_match_evidence_row(case: dict[str, Any], lookup: NoaaLookupResult | None = None) -> dict[str, Any]:
    lookup = lookup or NoaaLookupResult(
        matches=[],
        cache_status="cache_hit",
        fetch_status="local_structured_cache_no_match",
        source_check_status="success",
        loaded_years=(),
        missing_years=(),
        matched_record_count=0,
    )
    evidence = _base_evidence_row(case)
    evidence.update(
        {
            "source_title": "NOAA/NCEI Storm Events Details CSV",
            "source_family": "noaa_ncei_storm_events_structured",
            "source_lane": "noaa_ncei_storm_events",
            "source_dataset": "NOAA/NCEI Storm Events Details CSV",
            "fetch_status": lookup.fetch_status,
            "final_status": "rejected",
            "component_supported": "none",
            "impact_supported": "false",
            "wet_impact_support": "false",
            **_noaa_diagnostics(lookup),
            "county_match": "false",
            "time_alignment": "no_county_window_match",
            "component_support_type": "wet_event_hazard",
            "quoted_supporting_spans": "NOAA/NCEI Storm Events source data were successfully checked; no same-county storm-event row matched the candidate wet-event window.",
            "final_reason": "Structured NOAA lane successfully checked source data and found no county/window Storm Events row for this case.",
        }
    )
    return evidence


def noaa_lane_failed_evidence_row(case: dict[str, Any], lookup: NoaaLookupResult) -> dict[str, Any]:
    evidence = _base_evidence_row(case)
    evidence.update(
        {
            "source_title": "NOAA/NCEI Storm Events Details CSV",
            "source_family": "noaa_ncei_storm_events_structured",
            "source_lane": "noaa_ncei_storm_events",
            "source_dataset": "NOAA/NCEI Storm Events Details CSV",
            "source_url": NOAA_SOURCE_URL_TEMPLATE.format(name=""),
            "fetch_status": lookup.fetch_status,
            "final_status": "lane_failed",
            "component_supported": "unknown",
            "impact_supported": "",
            "wet_impact_support": "",
            **_noaa_diagnostics(lookup),
            "county_match": "",
            "time_alignment": "not_evaluable",
            "component_support_type": "wet_event_hazard",
            "quoted_supporting_spans": "NOAA/NCEI Storm Events source data could not be fully checked for the candidate wet-event window.",
            "final_reason": lookup.failure_reason or "Structured NOAA lane could not complete source checking.",
        }
    )
    return evidence


def openfema_evidence_row(
    case: dict[str, Any],
    source_row: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
    source_mode: str = "local_cache",
    source_url: str = OPENFEMA_DECLARATIONS_API_URL,
) -> dict[str, Any]:
    evidence = _base_evidence_row(case)
    state_match = _openfema_state_matches(source_row)
    county_match = _openfema_county_matches(case, source_row)
    incident_relevant = _openfema_incident_relevant(source_row)
    date_aligned, time_match = _openfema_date_alignment(
        case,
        source_row,
        same_storm_sequence_days=controls.openfema_same_storm_sequence_days,
    )
    admin_signal = _openfema_administrative_signal(source_row)
    county_area = _openfema_designated_county(source_row)
    programs = _openfema_programs(source_row)
    disaster_number = str(source_row.get("disasterNumber") or "").strip()
    incident_type = str(source_row.get("incidentType") or "").strip()
    declaration_title = str(source_row.get("declarationTitle") or "").strip()
    incident_begin = str(source_row.get("incidentBeginDate") or "").split("T", 1)[0]
    incident_end = str(source_row.get("incidentEndDate") or "").split("T", 1)[0]
    accepted = state_match and county_match and incident_relevant and date_aligned and admin_signal

    if accepted:
        final_status = "accepted"
        admin_status = "admin_material_proxy"
        reason = (
            "Official OpenFEMA county-designated disaster declaration matches California, "
            "the candidate county, the candidate wet/storm window or same-storm sequence, "
            "and an administrative assistance/declaration signal. This is an administrative-response "
            "material-impact proxy, not direct observed damage and not transition/linkage evidence."
        )
    elif not state_match or not county_match or not incident_relevant or not date_aligned:
        final_status = "rejected"
        admin_status = "admin_rejected"
        rejected_reasons = []
        if not state_match:
            rejected_reasons.append("wrong_state")
        if not county_match:
            rejected_reasons.append("wrong_or_countyless_area")
        if not incident_relevant:
            rejected_reasons.append("unrelated_incident_type")
        if not date_aligned:
            rejected_reasons.append("outside_candidate_storm_window")
        reason = "OpenFEMA row is not accepted as an admin proxy: " + "; ".join(rejected_reasons) + "."
    else:
        final_status = "context_only"
        admin_status = "admin_context_only"
        reason = (
            "OpenFEMA row has county/date/incident context, but no explicit assistance/declaration "
            "program flag was available for material admin-proxy support."
        )

    span = (
        f"OpenFEMA disaster {disaster_number or 'unknown'} for {county_area or 'unknown area'}; "
        f"incidentType={incident_type or 'unknown'}; title={declaration_title or 'unknown'}; "
        f"incidentBeginDate={incident_begin or 'unknown'}; incidentEndDate={incident_end or 'unknown'}; "
        f"programs={programs or 'none flagged'}."
    )
    evidence.update(
        {
            "source_title": "FEMA/OpenFEMA Disaster Declarations Summaries",
            "source_url": _openfema_disaster_url(source_row, source_url),
            "source_family": "fema_openfema_admin_response_structured",
            "source_lane": "openfema_admin_response",
            "source_dataset": "OpenFEMA Disaster Declarations Summaries v2",
            "source_date_if_available": incident_begin,
            "fetch_status": "local_structured_cache_hit" if source_mode == "local_cache" else source_mode,
            "source_mode": source_mode,
            "source_check_status": "success",
            "final_status": final_status,
            "component_supported": "none",
            "impact_supported": "false",
            "supports_impact": "false",
            "supports_wet_event": "false",
            "supports_drought": "false",
            "wet_impact_support": "false",
            "drought_impact_support": "false",
            "explicit_transition_support": "false",
            "location_match": "same_county" if county_match else "no_county_match",
            "time_match": time_match,
            "county_match": str(county_match).lower(),
            "time_alignment": time_match,
            "page_type": "structured_official_record",
            "component_support_type": "administrative_response_impact_proxy",
            "structured_record_id": disaster_number,
            "officiality": "federal_official_structured",
            "impact_types": "administrative_response",
            "impact_channel": "administrative_response",
            "admin_response_supported": str(accepted).lower(),
            "admin_response_status": admin_status,
            "admin_response_programs": programs,
            "admin_proxy_material": str(accepted).lower(),
            "openfema_state_match": str(state_match).lower(),
            "openfema_county_match": str(county_match).lower(),
            "openfema_date_aligned": str(date_aligned).lower(),
            "openfema_incident_type_relevant": str(incident_relevant).lower(),
            "openfema_administrative_signal": str(admin_signal).lower(),
            "openfema_designated_area": county_area,
            "openfema_incident_type": incident_type,
            "openfema_declaration_title": declaration_title,
            "openfema_incident_begin_date": incident_begin,
            "openfema_incident_end_date": incident_end,
            "quoted_supporting_spans": span,
            "final_reason": reason,
        }
    )
    return evidence


def openfema_no_match_evidence_row(
    case: dict[str, Any],
    *,
    lookup: OpenFemaLookupResult | None = None,
) -> dict[str, Any]:
    evidence = _base_evidence_row(case)
    lookup = lookup or OpenFemaLookupResult(
        rows=[],
        cache_status="not_attempted",
        fetch_status="not_attempted",
        source_check_status="success",
    )
    evidence.update(
        {
            "source_title": "FEMA/OpenFEMA Disaster Declarations Summaries",
            "source_url": lookup.source_url,
            "source_family": "fema_openfema_admin_response_structured",
            "source_lane": "openfema_admin_response",
            "source_dataset": "OpenFEMA Disaster Declarations Summaries v2",
            "fetch_status": lookup.fetch_status,
            "source_mode": lookup.source_mode,
            "source_check_status": lookup.source_check_status,
            "cache_status": lookup.cache_status,
            "final_status": "rejected",
            "component_supported": "none",
            "impact_supported": "false",
            "supports_impact": "false",
            "supports_wet_event": "false",
            "supports_drought": "false",
            "wet_impact_support": "false",
            "drought_impact_support": "false",
            "explicit_transition_support": "false",
            "location_match": "no_county_window_match",
            "time_match": "not_evaluable",
            "county_match": "false",
            "time_alignment": "not_evaluable",
            "component_support_type": "administrative_response_impact_proxy",
            "impact_types": "administrative_response",
            "impact_channel": "administrative_response",
            "admin_response_supported": "false",
            "admin_response_status": "admin_not_found",
            "admin_proxy_material": "false",
            "quoted_supporting_spans": "OpenFEMA source data were checked; no county-designated wet/storm/flood administrative declaration matched the candidate window.",
            "final_reason": "No accepted OpenFEMA county/window administrative-response row was found for this case.",
        }
    )
    return evidence


def openfema_lane_failed_evidence_row(case: dict[str, Any], lookup: OpenFemaLookupResult) -> dict[str, Any]:
    evidence = openfema_no_match_evidence_row(case, lookup=lookup)
    evidence.update(
        {
            "final_status": "lane_failed",
            "admin_response_status": "admin_lane_failed",
            "location_match": "not_evaluable",
            "county_match": "",
            "time_match": "not_evaluable",
            "quoted_supporting_spans": "OpenFEMA source data could not be checked for this candidate.",
            "final_reason": lookup.failure_reason or "Structured OpenFEMA lane could not complete source checking.",
            "failure_reason": lookup.failure_reason,
        }
    )
    return evidence


def openfema_rows_for_case(
    case: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
    source_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    lookup = OpenFemaLookupResult(
        rows=source_rows,
        cache_status="provided_rows",
        fetch_status="provided_rows",
        source_check_status="success",
        source_mode="provided_rows",
        source_url=OPENFEMA_DECLARATIONS_API_URL,
    ) if source_rows is not None else openfema_source_rows(controls=controls, state="CA")
    if lookup.source_check_status != "success":
        return []
    candidates: list[dict[str, Any]] = []
    for row in lookup.rows:
        if not _openfema_state_matches(row):
            continue
        county_match = _openfema_county_matches(case, row)
        date_aligned, _time_match = _openfema_date_alignment(
            case,
            row,
            same_storm_sequence_days=controls.openfema_same_storm_sequence_days,
        )
        if county_match and date_aligned:
            candidates.append(row)
    return candidates


def run_openfema_admin_response_lane_for_case(
    case: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
    source_rows: list[dict[str, Any]] | None = None,
) -> OpenFemaAdminLaneResult:
    lookup = OpenFemaLookupResult(
        rows=source_rows,
        cache_status="provided_rows",
        fetch_status="provided_rows",
        source_check_status="success",
        source_mode="provided_rows",
        source_url=OPENFEMA_DECLARATIONS_API_URL,
    ) if source_rows is not None else openfema_source_rows(controls=controls, state="CA")

    evidence_rows: list[dict[str, Any]]
    if lookup.source_check_status != "success":
        evidence_rows = [openfema_lane_failed_evidence_row(case, lookup)]
    else:
        matched_rows = openfema_rows_for_case(case, controls=controls, source_rows=lookup.rows)
        evidence_rows = [
            openfema_evidence_row(
                case,
                row,
                controls=controls,
                source_mode=lookup.source_mode,
                source_url=lookup.source_url,
            )
            for row in matched_rows
        ]
        if not evidence_rows:
            evidence_rows = [openfema_no_match_evidence_row(case, lookup=lookup)]

    validate_evidence_row_statuses(evidence_rows)
    retrieval_log_rows = [_log_row_from_evidence(row, rank) for rank, row in enumerate(evidence_rows, start=1)]
    return OpenFemaAdminLaneResult(
        evidence_rows=evidence_rows,
        retrieval_log_rows=retrieval_log_rows,
        attempted_lanes=("openfema_admin_response",),
        admin_response_flags=admin_response_flags_from_evidence_rows(evidence_rows),
        source_mode=lookup.source_mode,
        source_check_status=lookup.source_check_status,
        failure_reason=lookup.failure_reason,
    )


def admin_response_flags_from_evidence_rows(rows: list[dict[str, Any]]) -> dict[str, bool]:
    admin_material_proxy = False
    admin_context_only = False
    for row in rows:
        if str(row.get("source_lane") or "") != "openfema_admin_response":
            continue
        status = str(row.get("final_status") or "").strip()
        admin_status = str(row.get("admin_response_status") or "").strip()
        if status == "accepted" and admin_status == "admin_material_proxy":
            admin_material_proxy = True
        if status == "context_only" or admin_status == "admin_context_only":
            admin_context_only = True
    return {
        "admin_material_proxy": admin_material_proxy,
        "admin_context_only": admin_context_only,
    }


def validate_evidence_row_statuses(rows: list[dict[str, Any]]) -> None:
    invalid = [
        str(row.get("candidate_id") or row.get("independent_case_id") or "<unknown>")
        for row in rows
        if str(row.get("final_status") or "").strip() in {"", "uninitialized"}
    ]
    if invalid:
        raise ValueError(f"evidence_rows_missing_explicit_final_status:{';'.join(invalid)}")


def _log_row_from_evidence(evidence: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "candidate_id": evidence.get("candidate_id", ""),
        "query_text": f"structured:{evidence.get('source_lane', '')}",
        "backend": (
            "structured_official_live_api"
            if evidence.get("usdm_source_mode") == "live_api_fallback"
            else "structured_official_cache"
        ),
        "retrieval_tier": evidence.get("retrieval_tier", "structured_official_first"),
        "source_lane": evidence.get("source_lane", ""),
        "source_family": evidence.get("source_family", ""),
        "officiality": evidence.get("officiality", ""),
        "result_rank": rank,
        "source_url": evidence.get("source_url", ""),
        "source_title": evidence.get("source_title", ""),
        "fetched": "true",
        "fetch_status": evidence.get("fetch_status", ""),
        "duplicate_url": "false",
        "body_unavailable": "false",
        "county_match": evidence.get("county_match", ""),
        "time_alignment": evidence.get("time_alignment", ""),
        "component_support_type": evidence.get("component_support_type", ""),
        "structured_record_id": evidence.get("structured_record_id", ""),
        "source_dataset": evidence.get("source_dataset", ""),
        "source_mode": evidence.get("source_mode", evidence.get("usdm_source_mode", "")),
        "cache_status": evidence.get("cache_status", ""),
        "source_check_status": evidence.get("source_check_status", ""),
        "failure_reason": evidence.get("failure_reason", ""),
        "loaded_years": evidence.get("loaded_years", ""),
        "missing_years": evidence.get("missing_years", ""),
        "matched_record_count": evidence.get("matched_record_count", ""),
        "unmet_gap_before_search": "",
        "domains_or_source_families_targeted": "",
        "fallback_reason": "",
        "official_sources_attempted": "",
        "unmet_gap_before_fallback": "",
        "notes": evidence.get("final_reason", ""),
    }


def support_flags_from_evidence_rows(rows: list[dict[str, Any]]) -> dict[str, bool]:
    drought = False
    wet = False
    impact = False
    for row in rows:
        status = str(row.get("final_status") or row.get("pipeline_decision_label") or "").strip()
        component = str(row.get("component_supported") or "").strip()
        location_match = str(row.get("location_match") or "").strip().lower()
        time_match = str(row.get("time_match") or "").strip().lower()
        county_match = str(row.get("county_match") or row.get("same_county_or_local") or "").strip().lower()
        time_alignment = str(row.get("time_alignment") or row.get("same_window_or_close") or "").strip().lower()
        same_county = location_match in {"same_county", "mapped_city", "mapped_city_or_place"} or county_match == "true"
        same_window = time_match in {"exact_window", "same_storm_sequence", "same_month", "drought_end_week"} or time_alignment in {
            "aligned",
            "exact_window",
            "close_storm_sequence",
            "drought_end_week",
            "true",
        }
        if status != "accepted" or not (same_county and same_window):
            continue
        if component in {"drought", "both"} or str(row.get("supports_drought")).lower() == "true":
            drought = True
        if component in {"wet", "both"} or str(row.get("supports_wet_event")).lower() == "true":
            wet = True
        if str(row.get("impact_supported")).lower() == "true" or str(row.get("supports_impact")).lower() == "true":
            impact = True
    return {"drought": drought, "wet": wet, "impact": impact}


def run_structured_official_lanes_for_case(
    case: dict[str, Any],
    *,
    controls: OfficialSourceLaneControls = OfficialSourceLaneControls(),
) -> StructuredOfficialLaneResult:
    evidence_rows: list[dict[str, Any]] = []
    retrieval_log_rows: list[dict[str, Any]] = []
    attempted: list[str] = []

    if controls.enable_usdm_county_statistics:
        attempted.append("usdm_county_statistics")
        evidence_rows.append(usdm_evidence_row(case, controls=controls))

    if controls.enable_noaa_storm_events:
        attempted.append("noaa_ncei_storm_events")
        noaa_lookup = noaa_lookup_for_case(case, controls=controls)
        if noaa_lookup.source_check_status != "success":
            evidence_rows.append(noaa_lane_failed_evidence_row(case, noaa_lookup))
        elif noaa_lookup.matches:
            evidence_rows.extend(noaa_evidence_row(case, match, noaa_lookup) for match in noaa_lookup.matches)
        else:
            evidence_rows.append(noaa_no_match_evidence_row(case, noaa_lookup))

    if controls.enable_openfema_admin_response:
        attempted.append("openfema_admin_response")
        openfema_result = run_openfema_admin_response_lane_for_case(case, controls=controls)
        evidence_rows.extend(openfema_result.evidence_rows)

    validate_evidence_row_statuses(evidence_rows)

    for rank, row in enumerate(evidence_rows, start=1):
        retrieval_log_rows.append(_log_row_from_evidence(row, rank))

    return StructuredOfficialLaneResult(
        evidence_rows=evidence_rows,
        retrieval_log_rows=retrieval_log_rows,
        attempted_lanes=tuple(attempted),
        support_flags=support_flags_from_evidence_rows(evidence_rows),
    )


def missing_components_from_flags(flags: dict[str, bool]) -> list[str]:
    missing: list[str] = []
    if not flags.get("drought"):
        missing.append("drought")
    if not flags.get("wet"):
        missing.append("rain_flood")
    if not flags.get("impact"):
        missing.append("impact")
    return missing


def official_sources_attempted_text(attempted_lanes: tuple[str, ...] | list[str]) -> str:
    return ";".join(str(item) for item in attempted_lanes if str(item))
