"""Local-only repaired-v2 official adapters and final alignment enrichment.

The adapters in this module have no client, URL-opening, download, refresh, or
cache-writing path.  Package integrity is proven by ``load_official_package``
before any record is exposed.  NOAA truth inventory is constructed before the
separate evidence-presentation ranking function is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .official_package import LoadedOfficialPackage, load_official_package
from .usdm_vector_auxiliary import (
    ALIGNMENT_ADDENDUM_SHA256,
    ALIGNMENT_ADDENDUM_VERSION,
    USDM_VECTOR_AUXILIARY_ADAPTER_VERSION,
    UsdmVectorAuxiliaryAdapter,
    failed_auxiliary_provenance,
)
from .schemas import (
    ADAPTER_VERSION_ENVELOPE,
    AxisCoverageState,
    AxisName,
    AxisValue,
    EvidenceDisposition,
    LaneExecutionRecord,
    LaneExecutionStatus,
    LaneReadiness,
    ReasonCode,
    RuleId,
    SourceRole,
    SpatialAlignmentGrade,
    StructuredEvidenceRecord,
    TemporalAlignmentGrade,
    VersionLineage,
)


USDM_ADAPTER_VERSION = (
    f"{ADAPTER_VERSION_ENVELOPE}.usdm_county_week_vector_auxiliary_local.2"
)
NOAA_ADAPTER_VERSION = f"{ADAPTER_VERSION_ENVELOPE}.noaa_stormevents_local.2"
NOAA_ENHANCEMENT_FALLBACK_VERSION = (
    "ce_agent_noaa_unavailable_enhancement_fallback_v1.0.0"
)
USDM_LANE = "usdm_county_week"
NOAA_LANE = "noaa_stormevents"
LOCAL_COMPLETION_MARKER = "local_read_only_evaluation_complete"
ALL_AXES = tuple(AxisName)


class OfficialAdapterError(ValueError):
    """Raised when local source identity or deterministic mapping fails closed."""


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def previous_or_same_tuesday(value: date) -> date:
    """Return the Tuesday on or immediately before ``value``."""

    return value - timedelta(days=(value.weekday() - 1) % 7)


def _axis_map(*, drought: AxisValue = AxisValue.NO, wet: AxisValue = AxisValue.NO,
              impact: AxisValue = AxisValue.NO,
              attribution: AxisValue = AxisValue.NO,
              linkage: AxisValue = AxisValue.NO) -> Mapping[AxisName, AxisValue]:
    return {
        AxisName.DROUGHT_CORROBORATION: drought,
        AxisName.WET_HAZARD_CORROBORATION: wet,
        AxisName.REALIZED_IMPACT: impact,
        AxisName.HAZARD_TO_IMPACT_ATTRIBUTION: attribution,
        AxisName.DROUGHT_TO_WET_LINKAGE: linkage,
    }


def noaa_unavailable_enhancement_provenance() -> Mapping[str, Any]:
    """Return the exact owner-frozen unavailable decisions as provenance only."""

    return {
        "fallback_version": NOAA_ENHANCEMENT_FALLBACK_VERSION,
        "alignment_addendum_version": ALIGNMENT_ADDENDUM_VERSION,
        "alignment_addendum_sha256": ALIGNMENT_ADDENDUM_SHA256,
        "locations": {
            "decision": "ENHANCEMENT_UNAVAILABLE",
            "package_id": None,
            "package_path": None,
            "adapter_activation": False,
            "acquisition_or_directory_discovery": False,
            "base_fallback": "Step-2 local/county/time alignment remains active",
            "missing_enhancement_is_negative": False,
            "fail_closed_history": (
                "approved 2021 URL returned HTTP 404 three times; "
                "2022-2025 were not attempted"
            ),
            "rule_ids": [
                "NOAA-LOC-UNAVAILABLE-001",
                "NOAA-LOC-FALLBACK-001",
                "NOAA-LOC-NOEXPAND-001",
            ],
        },
        "historical_zones": {
            "decision": "ENHANCEMENT_UNAVAILABLE",
            "package_id": None,
            "package_path": None,
            "adapter_activation": False,
            "acquisition_or_directory_discovery": False,
            "current_zone_substitution": False,
            "base_fallback": "Step-2 zone context/unresolved handling remains active",
            "rule_ids": [
                "NOAA-ZONE-UNAVAILABLE-001",
                "NOAA-ZONE-FALLBACK-001",
                "NOAA-ZONE-NOSUB-001",
            ],
        },
        "scientific_truth_effect": "none",
    }


@dataclass(frozen=True)
class AdapterResult:
    evidence_records: tuple[StructuredEvidenceRecord, ...]
    lane_execution_record: LaneExecutionRecord
    truth_inventory_count: int
    presentation_count: int = 0


class UsdmCoreAdapter:
    """County-authoritative adapter with optional non-authoritative vector provenance."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        vector_manifest_path: Path | None = None,
    ) -> None:
        self.loaded = load_official_package(Path(manifest_path))
        if self.loaded.manifest.source_lane != "usdm_county_statistics":
            raise OfficialAdapterError("wrong_package_lane:usdm")
        self._records = self._load_records()
        self._crosswalk = self._load_crosswalk()
        self._vector: UsdmVectorAuxiliaryAdapter | None = None
        self._vector_failure: Exception | None = None
        self._vector_manifest_path = (
            Path(vector_manifest_path) if vector_manifest_path is not None else None
        )
        if self._vector_manifest_path is not None:
            try:
                self._vector = UsdmVectorAuxiliaryAdapter(
                    self._vector_manifest_path
                )
            except Exception as exc:
                # The enhancement fails closed while valid county support remains usable.
                self._vector_failure = exc

    @property
    def package_id(self) -> str:
        return self.loaded.manifest.package_id

    @property
    def package_hash(self) -> str:
        return self.loaded.manifest.package_manifest_sha256

    def _load_records(self) -> Mapping[str, Mapping[str, Any]]:
        path = self.loaded.root / "normalized_records.jsonl"
        records: dict[str, Mapping[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("historical_query_key") or "")
                if not key:
                    raise OfficialAdapterError(f"usdm_missing_query_key:{line_number}")
                if key in records:
                    if records[key] == row:
                        raise OfficialAdapterError(f"usdm_duplicate_record:{key}")
                    raise OfficialAdapterError(f"usdm_conflicting_record:{key}")
                declared = str(row.get("normalized_record_sha256") or "")
                hash_payload = dict(row)
                hash_payload.pop("normalized_record_sha256", None)
                if declared != _stable_hash(hash_payload):
                    raise OfficialAdapterError(f"usdm_normalized_hash_mismatch:{key}")
                records[key] = row
        return records

    def _load_crosswalk(self) -> Mapping[str, Mapping[str, str]]:
        path = self.loaded.root / "candidate_query_crosswalk.csv"
        rows: dict[str, Mapping[str, str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                candidate_id = str(row.get("candidate_id") or "")
                if not candidate_id or candidate_id in rows:
                    raise OfficialAdapterError(
                        f"usdm_duplicate_or_missing_candidate:{candidate_id}"
                    )
                rows[candidate_id] = dict(row)
        return rows

    def adapt_candidate(
        self,
        *,
        candidate_id: str,
        county_fips: str,
        drought_end: date,
        drought_grid_centers: str | Iterable[str | Sequence[float]] = (),
        candidate_physical_source_hash: str = "",
    ) -> AdapterResult:
        crosswalk = self._crosswalk.get(candidate_id)
        if crosswalk is None:
            return AdapterResult(
                evidence_records=(),
                lane_execution_record=self._execution(
                    candidate_id=candidate_id,
                    status=LaneExecutionStatus.UNAVAILABLE,
                    coverage=AxisCoverageState.UNAVAILABLE,
                    examined=0,
                    matches=0,
                    accepted=0,
                    contextual=0,
                    unresolved=0,
                    readiness=LaneReadiness.UNAVAILABLE,
                ),
                truth_inventory_count=0,
            )
        if len(county_fips) != 5 or not county_fips.isdigit():
            raise OfficialAdapterError("usdm_county_fips_must_preserve_five_digits")
        expected_tuesday = previous_or_same_tuesday(drought_end).isoformat()
        if str(crosswalk["county_fips"]) != county_fips:
            raise OfficialAdapterError(f"usdm_candidate_fips_mismatch:{candidate_id}")
        if str(crosswalk["reference_tuesday"]) != expected_tuesday:
            raise OfficialAdapterError(f"usdm_candidate_reference_week_mismatch:{candidate_id}")
        query_key = f"{county_fips}|{expected_tuesday}"
        if str(crosswalk["historical_query_key"]) != query_key:
            raise OfficialAdapterError(f"usdm_crosswalk_query_mismatch:{candidate_id}")
        source = self._records.get(query_key)
        if source is None:
            return AdapterResult(
                evidence_records=(),
                lane_execution_record=self._execution(
                    candidate_id=candidate_id,
                    status=LaneExecutionStatus.UNAVAILABLE,
                    coverage=AxisCoverageState.UNAVAILABLE,
                    examined=0,
                    matches=0,
                    accepted=0,
                    contextual=0,
                    unresolved=0,
                    readiness=LaneReadiness.UNAVAILABLE,
                ),
                truth_inventory_count=0,
            )
        if str(source["county_fips"]) != county_fips or str(source["map_date"]) != expected_tuesday:
            raise OfficialAdapterError(f"usdm_source_identity_mismatch:{query_key}")

        raw_d1 = str(source.get("d1_plus_percent_raw") or "")
        try:
            d1 = Decimal(raw_d1)
            if not d1.is_finite() or d1 < 0:
                raise InvalidOperation
            parse_status = "parsed"
        except (InvalidOperation, ValueError):
            d1 = None
            parse_status = "blank" if not raw_d1.strip() else "malformed"

        if d1 is None:
            disposition = EvidenceDisposition.UNRESOLVED
            drought_value = AxisValue.UNRESOLVED
            reason_codes = (ReasonCode.SOURCE_MALFORMED,)
            coverage = AxisCoverageState.PARTIAL
            accepted_count = 0
            unresolved_count = 1
        elif d1 > 0:
            disposition = EvidenceDisposition.ACCEPTED
            drought_value = AxisValue.YES
            reason_codes = (ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD,)
            coverage = AxisCoverageState.COMPLETE
            accepted_count = 1
            unresolved_count = 0
        else:
            disposition = EvidenceDisposition.REJECTED_NON_SUPPORT
            drought_value = AxisValue.NO
            reason_codes = (ReasonCode.NO_SUPPORT_IN_SUCCESSFULLY_EVALUATED_SCOPE,)
            coverage = AxisCoverageState.COMPLETE
            accepted_count = 0
            unresolved_count = 0

        provenance = {
            "historical_query_key": query_key,
            "county_fips": county_fips,
            "reference_tuesday": expected_tuesday,
            "raw_unit": source["raw_unit"],
            "parse_status": parse_status,
            "raw_values": {
                key: source[key]
                for key in (
                    "none_percent_raw",
                    "d0_plus_percent_raw",
                    "d1_plus_percent_raw",
                    "d2_plus_percent_raw",
                    "d3_plus_percent_raw",
                    "d4_percent_raw",
                )
            },
            "verification_status": source["verification_status"],
            "recovery_status": source["recovery_status"],
            "normalized_record_sha256": source["normalized_record_sha256"],
        }
        if self._vector_manifest_path is not None:
            if self._vector is None:
                provenance["usdm_vector_auxiliary"] = failed_auxiliary_provenance(
                    error=self._vector_failure or "vector_adapter_unavailable",
                    map_date=expected_tuesday,
                )
            else:
                try:
                    provenance["usdm_vector_auxiliary"] = self._vector.evaluate(
                        map_date=expected_tuesday,
                        grid_centers=drought_grid_centers,
                        candidate_physical_source_hash=(
                            candidate_physical_source_hash
                        ),
                        county_support_value=drought_value,
                    ).to_provenance()
                except Exception as exc:
                    provenance["usdm_vector_auxiliary"] = (
                        failed_auxiliary_provenance(
                            error=exc,
                            map_date=expected_tuesday,
                            package_id=self._vector.package_id,
                            package_hash=self._vector.package_hash,
                        )
                    )
        hash_payload = {
            "candidate_id": candidate_id,
            "package_id": self.package_id,
            "source_record_id": source["source_record_id"],
            "provenance": provenance,
            "drought": drought_value.value,
            "disposition": disposition.value,
        }
        evidence = StructuredEvidenceRecord(
            candidate_id=candidate_id,
            lane=USDM_LANE,
            package_id=self.package_id,
            source_role=SourceRole.USDM_COUNTY_WEEK_DROUGHT_CONTEXT,
            disposition=disposition,
            source_record_id=str(source["source_record_id"]),
            source_event_id="",
            source_file=str(source["raw_file"]),
            source_row="1",
            source_url="",
            source_sha256=str(source["raw_file_sha256"]),
            source_version=self.loaded.manifest.source_product_version,
            source_date=expected_tuesday,
            raw_value_provenance=provenance,
            spatial_grade=SpatialAlignmentGrade.COUNTY_CONTEXT,
            temporal_grade=TemporalAlignmentGrade.PREVIOUS_OR_SAME_TUESDAY,
            axis_assessments=_axis_map(drought=drought_value),
            rule_ids=(
                RuleId.USDM_REFERENCE_WEEK,
                RuleId.USDM_D1_POSITIVE if d1 is not None and d1 > 0
                else RuleId.USDM_D1_ZERO if d1 == 0
                else RuleId.USDM_MISSING,
                RuleId.USDM_AXIS_SCOPE,
            ),
            reason_codes=reason_codes,
            record_hash=_stable_hash(hash_payload),
            lineage=VersionLineage(adapter_version=USDM_ADAPTER_VERSION),
        )
        execution = self._execution(
            candidate_id=candidate_id,
            status=LaneExecutionStatus.COMPLETED_WITH_MATCHES,
            coverage=coverage,
            examined=1,
            matches=1,
            accepted=accepted_count,
            contextual=0,
            unresolved=unresolved_count,
            readiness=LaneReadiness.READY,
        )
        return AdapterResult((evidence,), execution, truth_inventory_count=1)

    def _execution(
        self,
        *,
        candidate_id: str,
        status: LaneExecutionStatus,
        coverage: AxisCoverageState,
        examined: int,
        matches: int,
        accepted: int,
        contextual: int,
        unresolved: int,
        readiness: LaneReadiness,
    ) -> LaneExecutionRecord:
        return LaneExecutionRecord(
            candidate_id=candidate_id,
            lane=USDM_LANE,
            enabled=True,
            required=True,
            source_mode="immutable_local_package_read_only",
            package_id=self.package_id,
            package_sha256=self.package_hash,
            readiness=readiness,
            execution_status=status,
            coverage=coverage,
            records_examined=examined,
            matches_found=matches,
            accepted_count=accepted,
            contextual_count=contextual,
            unresolved_count=unresolved,
            completed_at=LOCAL_COMPLETION_MARKER if status.is_successful_completion else "",
            lineage=VersionLineage(adapter_version=USDM_ADAPTER_VERSION),
        )


@dataclass(frozen=True)
class NoaaCandidate:
    candidate_id: str
    state: str
    county_name: str
    wet_window_start: datetime
    wet_window_end: datetime
    latitude: float | None = None
    longitude: float | None = None
    approved_zone_names: tuple[str, ...] = ()
    local_tolerance_km: float = 5.0

    def __post_init__(self) -> None:
        if self.wet_window_end < self.wet_window_start:
            raise OfficialAdapterError("candidate_wet_window_reversed")
        if self.local_tolerance_km <= 0:
            raise OfficialAdapterError("local_tolerance_must_be_positive")


@dataclass(frozen=True)
class NarrativeGrounding:
    wet_trigger_span: str = ""
    impact_span: str = ""
    explicit_relation_span: str = ""


@dataclass(frozen=True)
class ParsedConsequence:
    raw_value: str
    normalized_value: Decimal | int | None
    unit: str
    parse_status: str
    directness: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_value": self.raw_value,
            "normalized_value": (
                str(self.normalized_value)
                if isinstance(self.normalized_value, Decimal)
                else self.normalized_value
            ),
            "unit": self.unit,
            "parse_status": self.parse_status,
            "directness": self.directness,
        }


@dataclass(frozen=True)
class NoaaInventoryRecord:
    source_row: Mapping[str, str]
    source_record_id: str
    source_event_id: str
    source_file: str
    source_file_sha256: str
    source_row_number: str
    role: SourceRole
    spatial_grade: SpatialAlignmentGrade
    temporal_grade: TemporalAlignmentGrade
    grounding: NarrativeGrounding
    consequence_fields: Mapping[str, ParsedConsequence]


_BROAD_CONTEXT_TYPES = {
    "High Wind",
    "Strong Wind",
    "Thunderstorm Wind",
    "Hail",
    "High Surf",
    "Winter Storm",
    "Winter Weather",
    "Tropical Storm",
    "Hurricane",
    "Tornado",
}


def _narrative(row: Mapping[str, str]) -> str:
    return "\n".join(
        value for value in (row.get("EPISODE_NARRATIVE", ""), row.get("EVENT_NARRATIVE", ""))
        if value
    )


def _valid_exact_span(span: str, narrative: str) -> bool:
    return bool(span and span in narrative)


def _explicit_relation(span: str, narrative: str) -> bool:
    if not _valid_exact_span(span, narrative):
        return False
    lowered = span.casefold()
    relation_terms = ("caused", "resulted in", "led to", "triggered", "due to", "produced")
    wet_terms = ("rain", "flood", "precipitation", "runoff")
    return any(term in lowered for term in relation_terms) and any(
        term in lowered for term in wet_terms
    )


def classify_noaa_event_role(
    event_type: str,
    *,
    narrative: str = "",
    grounding: NarrativeGrounding = NarrativeGrounding(),
) -> SourceRole:
    event = event_type.strip()
    if event == "Heavy Rain":
        return SourceRole.DIRECT_PRECIPITATION_EVENT
    if event in {"Flood", "Flash Flood"}:
        return SourceRole.RAIN_INDUCED_HYDROLOGIC_EVENT
    if event == "Debris Flow":
        return SourceRole.CONDITIONAL_WET_CONSEQUENCE
    if event in {"Coastal Flood", "Storm Surge/Tide"}:
        return SourceRole.COASTAL_OR_ZONE_EVENT
    if event in _BROAD_CONTEXT_TYPES:
        if _explicit_relation(grounding.wet_trigger_span, narrative):
            return SourceRole.RAIN_INDUCED_HYDROLOGIC_EVENT
        return SourceRole.BROADER_STORM_CONTEXT
    if _explicit_relation(grounding.wet_trigger_span, narrative):
        return SourceRole.RAIN_INDUCED_HYDROLOGIC_EVENT
    return SourceRole.IRRELEVANT_EVENT


def _clean_county(value: str) -> str:
    lowered = value.strip().casefold()
    return lowered.removesuffix(" county").strip()


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def assess_noaa_spatial_alignment(
    row: Mapping[str, str], candidate: NoaaCandidate
) -> SpatialAlignmentGrade:
    if (row.get("STATE") or "").strip().casefold() != candidate.state.strip().casefold():
        return SpatialAlignmentGrade.MISMATCH
    cz_type = (row.get("CZ_TYPE") or "").strip().upper()
    cz_name = (row.get("CZ_NAME") or "").strip()
    if cz_type == "C":
        if _clean_county(cz_name) == _clean_county(candidate.county_name):
            return SpatialAlignmentGrade.COUNTY
        return SpatialAlignmentGrade.MISMATCH
    if cz_type == "Z":
        if cz_name.casefold() in {name.casefold() for name in candidate.approved_zone_names}:
            return SpatialAlignmentGrade.ZONE
        return SpatialAlignmentGrade.UNRESOLVED
    if candidate.latitude is not None and candidate.longitude is not None:
        coordinates = [
            (_float(row.get("BEGIN_LAT", "")), _float(row.get("BEGIN_LON", ""))),
            (_float(row.get("END_LAT", "")), _float(row.get("END_LON", ""))),
        ]
        distances = [
            _haversine_km(candidate.latitude, candidate.longitude, lat, lon)
            for lat, lon in coordinates
            if lat is not None and lon is not None
        ]
        if distances and min(distances) <= candidate.local_tolerance_km:
            return SpatialAlignmentGrade.LOCAL
    if cz_name or row.get("BEGIN_LOCATION") or row.get("END_LOCATION"):
        return SpatialAlignmentGrade.CONTEXT
    return SpatialAlignmentGrade.UNRESOLVED


def _parse_noaa_datetime(raw: str) -> datetime | None:
    for form in ("%d-%b-%y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), form)
        except (ValueError, TypeError):
            pass
    return None


def assess_noaa_temporal_alignment(
    row: Mapping[str, str], candidate: NoaaCandidate
) -> TemporalAlignmentGrade:
    begin = _parse_noaa_datetime(row.get("BEGIN_DATE_TIME", ""))
    end = _parse_noaa_datetime(row.get("END_DATE_TIME", ""))
    if begin is None or end is None:
        return TemporalAlignmentGrade.UNRESOLVED
    if begin <= candidate.wet_window_end and end >= candidate.wet_window_start:
        return TemporalAlignmentGrade.CANDIDATE_WINDOW_EXACT_OVERLAP
    distance = min(
        abs((begin - candidate.wet_window_end).total_seconds()),
        abs((candidate.wet_window_start - end).total_seconds()),
    )
    if distance <= 72 * 3600:
        return TemporalAlignmentGrade.NEAR_WINDOW_BROADER_STORM_SEQUENCE
    return TemporalAlignmentGrade.MISMATCH


_UNKNOWN_VALUES = {"UNKNOWN", "UNK", "N/A", "NA", "NULL", "-"}


def parse_noaa_damage(raw: str) -> ParsedConsequence:
    original = str(raw or "")
    text = original.strip().upper().replace(",", "").replace("$", "")
    if not text:
        return ParsedConsequence(original, None, "USD", "blank", "not_applicable")
    if text in _UNKNOWN_VALUES:
        return ParsedConsequence(original, None, "USD", "unknown", "not_applicable")
    multiplier = Decimal("1")
    if text[-1:] in {"K", "M", "B"}:
        multiplier = {"K": Decimal("1000"), "M": Decimal("1000000"), "B": Decimal("1000000000")}[text[-1]]
        text = text[:-1]
    try:
        value = Decimal(text) * multiplier
        if not value.is_finite() or value < 0:
            raise InvalidOperation
    except InvalidOperation:
        return ParsedConsequence(original, None, "USD", "malformed", "not_applicable")
    return ParsedConsequence(original, value, "USD", "positive" if value > 0 else "zero", "not_applicable")


def parse_noaa_casualty(raw: str, *, directness: str) -> ParsedConsequence:
    original = str(raw or "")
    text = original.strip().upper()
    if not text:
        return ParsedConsequence(original, None, "persons", "blank", directness)
    if text in _UNKNOWN_VALUES:
        return ParsedConsequence(original, None, "persons", "unknown", directness)
    try:
        value = int(text)
        if value < 0:
            raise ValueError
    except ValueError:
        return ParsedConsequence(original, None, "persons", "malformed", directness)
    return ParsedConsequence(original, value, "persons", "positive" if value > 0 else "zero", directness)


def parse_noaa_consequences(row: Mapping[str, str]) -> Mapping[str, ParsedConsequence]:
    return {
        "DAMAGE_PROPERTY": parse_noaa_damage(row.get("DAMAGE_PROPERTY", "")),
        "DAMAGE_CROPS": parse_noaa_damage(row.get("DAMAGE_CROPS", "")),
        "INJURIES_DIRECT": parse_noaa_casualty(row.get("INJURIES_DIRECT", ""), directness="direct"),
        "INJURIES_INDIRECT": parse_noaa_casualty(row.get("INJURIES_INDIRECT", ""), directness="indirect"),
        "DEATHS_DIRECT": parse_noaa_casualty(row.get("DEATHS_DIRECT", ""), directness="direct"),
        "DEATHS_INDIRECT": parse_noaa_casualty(row.get("DEATHS_INDIRECT", ""), directness="indirect"),
    }


def _file_hashes(package: LoadedOfficialPackage) -> Mapping[str, str]:
    return {item.relative_path: item.sha256 for item in package.manifest.files}


class NoaaCoreAdapter:
    """Read-only NOAA details adapter with truth/presentation separation."""

    def __init__(self, manifest_path: Path) -> None:
        self.loaded = load_official_package(Path(manifest_path))
        if self.loaded.manifest.source_lane != "noaa_stormevents_details":
            raise OfficialAdapterError("wrong_package_lane:noaa")
        self._hashes = _file_hashes(self.loaded)

    @property
    def package_id(self) -> str:
        return self.loaded.manifest.package_id

    @property
    def package_hash(self) -> str:
        return self.loaded.manifest.package_manifest_sha256

    def iter_source_rows(self) -> Iterator[Mapping[str, str]]:
        for item in self.loaded.manifest.files:
            if not item.relative_path.endswith(".csv.gz"):
                continue
            path = self.loaded.root / item.relative_path
            with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
                for row_number, row in enumerate(csv.DictReader(handle), start=2):
                    enriched = dict(row)
                    enriched["_source_file"] = item.relative_path
                    enriched["_source_file_sha256"] = item.sha256
                    enriched["_source_row_number"] = str(row_number)
                    yield enriched

    def inventory_candidate(
        self,
        *,
        candidate: NoaaCandidate,
        source_rows: Iterable[Mapping[str, str]] | None = None,
        groundings: Mapping[str, NarrativeGrounding] | None = None,
    ) -> tuple[NoaaInventoryRecord, ...]:
        rows = source_rows if source_rows is not None else self.iter_source_rows()
        grounding_map = groundings or {}
        inventory: list[NoaaInventoryRecord] = []
        seen_exact: set[str] = set()
        for ordinal, original in enumerate(rows, start=1):
            row = {str(key): str(value or "") for key, value in original.items()}
            # Candidate matching scope is deliberately broad enough to retain
            # explicit mismatches supplied by controlled inventories.  Full
            # package scans are limited to the candidate state and +/- 72 h.
            if source_rows is None and row.get("STATE", "").casefold() != candidate.state.casefold():
                continue
            temporal = assess_noaa_temporal_alignment(row, candidate)
            if source_rows is None and temporal is TemporalAlignmentGrade.MISMATCH:
                continue
            event_id = row.get("EVENT_ID", "").strip()
            source_file = row.get("_source_file") or "controlled_fixture.csv"
            source_hash = row.get("_source_file_sha256") or _stable_hash(
                {key: value for key, value in row.items() if not key.startswith("_")}
            )
            row_number = row.get("_source_row_number") or f"fixture-{source_hash[:16]}"
            source_record_id = f"{Path(source_file).name}:{row_number}:{event_id or 'no-event-id'}"
            exact_identity = _stable_hash(
                {key: value for key, value in row.items() if key not in {"_source_row_number"}}
            )
            if exact_identity in seen_exact:
                continue
            seen_exact.add(exact_identity)
            grounding = grounding_map.get(source_record_id) or grounding_map.get(event_id) or NarrativeGrounding()
            narrative = _narrative(row)
            role = classify_noaa_event_role(
                row.get("EVENT_TYPE", ""), narrative=narrative, grounding=grounding
            )
            inventory.append(
                NoaaInventoryRecord(
                    source_row=row,
                    source_record_id=source_record_id,
                    source_event_id=event_id,
                    source_file=source_file,
                    source_file_sha256=source_hash,
                    source_row_number=row_number,
                    role=role,
                    spatial_grade=assess_noaa_spatial_alignment(row, candidate),
                    temporal_grade=temporal,
                    grounding=grounding,
                    consequence_fields=parse_noaa_consequences(row),
                )
            )
        return tuple(inventory)

    def adapt_candidate(
        self,
        *,
        candidate: NoaaCandidate,
        source_rows: Iterable[Mapping[str, str]] | None = None,
        groundings: Mapping[str, NarrativeGrounding] | None = None,
    ) -> AdapterResult:
        inventory = self.inventory_candidate(
            candidate=candidate,
            source_rows=source_rows,
            groundings=groundings,
        )
        evidence = tuple(self._map_record(candidate, record) for record in inventory)
        accepted = sum(row.disposition is EvidenceDisposition.ACCEPTED for row in evidence)
        contextual = sum(row.disposition is EvidenceDisposition.CONTEXTUAL for row in evidence)
        unresolved = sum(row.disposition is EvidenceDisposition.UNRESOLVED for row in evidence)
        if not evidence:
            status = LaneExecutionStatus.COMPLETED_ZERO_MATCH
        elif accepted == 0 and contextual > 0 and unresolved == 0:
            status = LaneExecutionStatus.COMPLETED_CONTEXT_ONLY
        else:
            status = LaneExecutionStatus.COMPLETED_WITH_MATCHES
        coverage = AxisCoverageState.PARTIAL if unresolved else AxisCoverageState.COMPLETE
        execution = LaneExecutionRecord(
            candidate_id=candidate.candidate_id,
            lane=NOAA_LANE,
            enabled=True,
            required=True,
            source_mode="immutable_local_package_read_only",
            package_id=self.package_id,
            package_sha256=self.package_hash,
            readiness=LaneReadiness.READY,
            execution_status=status,
            coverage=coverage,
            records_examined=len(inventory),
            matches_found=len(inventory),
            accepted_count=accepted,
            contextual_count=contextual,
            unresolved_count=unresolved,
            completed_at=LOCAL_COMPLETION_MARKER,
            lineage=VersionLineage(adapter_version=NOAA_ADAPTER_VERSION),
        )
        return AdapterResult(evidence, execution, truth_inventory_count=len(inventory))

    def _map_record(
        self, candidate: NoaaCandidate, record: NoaaInventoryRecord
    ) -> StructuredEvidenceRecord:
        spatial_ok = record.spatial_grade in {
            SpatialAlignmentGrade.LOCAL,
            SpatialAlignmentGrade.COUNTY,
            SpatialAlignmentGrade.ZONE,
        }
        temporal_ok = record.temporal_grade in {
            TemporalAlignmentGrade.CANDIDATE_WINDOW_EXACT_OVERLAP,
            TemporalAlignmentGrade.EXACT_TIME_SAME_EVENT_EPISODE,
        }
        narrative = _narrative(record.source_row)
        conditional_relation = _explicit_relation(
            record.grounding.wet_trigger_span, narrative
        )
        role_eligible = record.role in {
            SourceRole.DIRECT_PRECIPITATION_EVENT,
            SourceRole.RAIN_INDUCED_HYDROLOGIC_EVENT,
        } or (
            record.role in {
                SourceRole.CONDITIONAL_WET_CONSEQUENCE,
                SourceRole.COASTAL_OR_ZONE_EVENT,
            }
            and conditional_relation
        )
        explicit_mismatch = (
            record.spatial_grade is SpatialAlignmentGrade.MISMATCH
            or record.temporal_grade is TemporalAlignmentGrade.MISMATCH
        )
        alignment_unresolved = (
            record.spatial_grade is SpatialAlignmentGrade.UNRESOLVED
            or record.temporal_grade is TemporalAlignmentGrade.UNRESOLVED
        )

        if explicit_mismatch:
            disposition = EvidenceDisposition.REJECTED_MISMATCH
            wet = impact = attribution = AxisValue.NO
            reasons = (
                ReasonCode.SPATIAL_MISMATCH
                if record.spatial_grade is SpatialAlignmentGrade.MISMATCH
                else ReasonCode.TEMPORAL_MISMATCH,
            )
        elif alignment_unresolved:
            disposition = EvidenceDisposition.UNRESOLVED
            wet = impact = attribution = AxisValue.UNRESOLVED
            reasons = (ReasonCode.PLACE_TIME_ALIGNMENT_INSUFFICIENT,)
        elif not (role_eligible and spatial_ok and temporal_ok):
            disposition = EvidenceDisposition.CONTEXTUAL
            wet = impact = attribution = AxisValue.NO
            reasons = (ReasonCode.CONTEXT_ONLY,)
        else:
            disposition = EvidenceDisposition.ACCEPTED
            wet = AxisValue.YES
            consequence_states = {
                value.parse_status for value in record.consequence_fields.values()
            }
            structured_positive = "positive" in consequence_states
            grounded_impact = _valid_exact_span(record.grounding.impact_span, narrative)
            grounded_relation = _explicit_relation(
                record.grounding.explicit_relation_span, narrative
            )
            if structured_positive or grounded_impact:
                impact = AxisValue.YES
            elif consequence_states <= {"zero"}:
                impact = AxisValue.NO
            else:
                impact = AxisValue.UNRESOLVED
            if structured_positive or grounded_relation:
                attribution = AxisValue.YES
            elif impact is AxisValue.NO:
                attribution = AxisValue.NO
            elif grounded_impact and not grounded_relation:
                attribution = AxisValue.NO
            else:
                attribution = AxisValue.UNRESOLVED
            reasons = (ReasonCode.ACCEPTED_MINIMUM_SCIENTIFIC_STANDARD,)

        consequence_payload = {
            key: value.to_dict() for key, value in record.consequence_fields.items()
        }
        provenance = {
            "event_id": record.source_event_id,
            "episode_id": record.source_row.get("EPISODE_ID", ""),
            "event_type": record.source_row.get("EVENT_TYPE", ""),
            "role": record.role.value,
            "spatial_grade": record.spatial_grade.value,
            "temporal_grade": record.temporal_grade.value,
            "structured_consequences": consequence_payload,
            "narrative_grounding": {
                "wet_trigger_span": record.grounding.wet_trigger_span,
                "impact_span": record.grounding.impact_span,
                "explicit_relation_span": record.grounding.explicit_relation_span,
                "all_spans_are_exact_source_substrings": all(
                    not span or span in narrative
                    for span in (
                        record.grounding.wet_trigger_span,
                        record.grounding.impact_span,
                        record.grounding.explicit_relation_span,
                    )
                ),
            },
            "alignment_enhancement_fallbacks": (
                noaa_unavailable_enhancement_provenance()
            ),
        }
        axes = _axis_map(wet=wet, impact=impact, attribution=attribution)
        hash_payload = {
            "candidate_id": candidate.candidate_id,
            "source_record_id": record.source_record_id,
            "source_event_id": record.source_event_id,
            "package_id": self.package_id,
            "disposition": disposition.value,
            "axes": {key.value: value.value for key, value in axes.items()},
            "provenance": provenance,
        }
        return StructuredEvidenceRecord(
            candidate_id=candidate.candidate_id,
            lane=NOAA_LANE,
            package_id=self.package_id,
            source_role=record.role,
            disposition=disposition,
            source_record_id=record.source_record_id,
            source_event_id=record.source_event_id,
            source_file=record.source_file,
            source_row=record.source_row_number,
            source_url="",
            source_sha256=record.source_file_sha256,
            source_version=self.loaded.manifest.source_product_version,
            source_date=record.source_row.get("BEGIN_DATE_TIME", ""),
            raw_value_provenance=provenance,
            spatial_grade=record.spatial_grade,
            temporal_grade=record.temporal_grade,
            axis_assessments=axes,
            rule_ids=(
                RuleId.NOAA_EVENT_ROLE,
                RuleId.NOAA_SPATIAL,
                RuleId.NOAA_NO_UNIVERSAL_25KM,
                RuleId.NOAA_STRUCTURED_IMPACT,
                RuleId.NOAA_STRUCTURED_ATTRIBUTION,
                RuleId.NOAA_NARRATIVE,
                RuleId.NOAA_AXIS_SCOPE,
                RuleId.LINKAGE_NON_INFERENCE,
            ),
            reason_codes=reasons,
            record_hash=_stable_hash(hash_payload),
            lineage=VersionLineage(adapter_version=NOAA_ADAPTER_VERSION),
        )


def rank_noaa_for_review(
    records: Sequence[StructuredEvidenceRecord],
    *,
    top_n: int | None = None,
    ranking_key: Callable[[StructuredEvidenceRecord], Any] | None = None,
) -> tuple[StructuredEvidenceRecord, ...]:
    """Return a display-only view; never mutate or filter the truth inventory."""

    key = ranking_key or (
        lambda record: (
            record.disposition is EvidenceDisposition.ACCEPTED,
            record.axis_assessments.get(AxisName.REALIZED_IMPACT) is AxisValue.YES,
            record.source_event_id,
            record.source_record_id,
        )
    )
    ranked = tuple(sorted(records, key=key, reverse=True))
    if top_n is None:
        return ranked
    if top_n < 0:
        raise OfficialAdapterError("presentation_top_n_must_be_nonnegative")
    return ranked[:top_n]


__all__ = [
    "AdapterResult",
    "NOAA_ADAPTER_VERSION",
    "NOAA_ENHANCEMENT_FALLBACK_VERSION",
    "NOAA_LANE",
    "NarrativeGrounding",
    "NoaaCandidate",
    "NoaaCoreAdapter",
    "NoaaInventoryRecord",
    "OfficialAdapterError",
    "ParsedConsequence",
    "USDM_ADAPTER_VERSION",
    "USDM_LANE",
    "noaa_unavailable_enhancement_provenance",
    "UsdmCoreAdapter",
    "assess_noaa_spatial_alignment",
    "assess_noaa_temporal_alignment",
    "classify_noaa_event_role",
    "parse_noaa_casualty",
    "parse_noaa_consequences",
    "parse_noaa_damage",
    "previous_or_same_tuesday",
    "rank_noaa_for_review",
]
