from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from . import config


class SchemaValidationError(ValueError):
    """Raised when an active evidence-validation schema object is invalid."""


class ModelMixin:
    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return {k: convert(v) for k, v in asdict(value).items()}
            if hasattr(value, "model_dump"):
                return value.model_dump(mode=mode)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(self)


HAZARD_TYPES = {
    "agricultural_drought",
    "drought",
    "extreme_rainfall",
    "flood_related_rainfall",
    "extreme_rainfall_flood",
    "flash_flood",
    "inland_flooding",
    "river_flood",
    "winter_storm_flooding",
    "heatwave_deferred",
    "cold_wave_deferred",
    "other",
    "unknown",
}
ACTIVE_HAZARD_TYPES = {
    "agricultural_drought",
    "drought",
    "extreme_rainfall",
    "flood_related_rainfall",
    "extreme_rainfall_flood",
    "flash_flood",
    "inland_flooding",
    "river_flood",
    "winter_storm_flooding",
}
SOURCE_TYPES = {
    "noaa_nws",
    "fema",
    "usgs",
    "state_local_government",
    "extension",
    "credible_media",
    "generic_media",
    "target_county_official",
    "target_city_official",
    "target_province_official",
    "sector_official",
    "national_official",
    "cross_region_official_repost",
    "state_media",
    "local_media",
    "other_media",
    "academic_or_technical_report",
    "unknown",
}
LOCATION_RELEVANCE_LABELS = {"exact", "partial", "broad", "mismatch", "unknown"}
HAZARD_RELEVANCE_LABELS = {"exact", "compatible", "weak", "mismatch", "unknown"}
TEMPORAL_RELEVANCE_LABELS = {"aligned", "partially_aligned", "retrospective", "mismatch", "unknown"}
EVIDENCE_ROLE_LABELS = {"direct", "retrospective", "contextual", "repost", "irrelevant"}
PAGE_ACCEPTANCE_LABELS = {"accepted", "rejected", "needs_review"}
CLAIM_TYPES = {"hazard", "impact", "agriculture_impact", "response", "time", "location", "source_context"}
SUPPORT_STATUS_LABELS = {"span_supported", "weakly_supported", "unsupported", "conflicting", "invalid"}
VALIDATOR_STATUS_LABELS = {"valid", "warning", "invalid", "needs_review"}
SUPPORT_LABELS = {"supported", "weakly_supported", "unsupported", "conflicting"}
ALIGNMENT_LABELS = {"aligned", "partially_aligned", "misaligned", "insufficient_evidence"}
SOURCE_QUALITY_LABELS = {"high", "medium", "low", "insufficient", "unknown"}
COVERAGE_STATUS_LABELS = {"adequate", "limited", "sparse", "failed"}
CONFIDENCE_LABELS = {"high", "medium", "low", "insufficient"}
FINAL_EVENT_STATUS_LABELS = {
    "evidence_supported_impact_event",
    "hazard_only_public_evidence",
    "climate_signal_only_no_public_evidence",
    "misaligned_public_evidence",
    "insufficient_retrieval_coverage",
    "needs_review",
}
COMPOUND_CANDIDATE_TYPES = {
    "unsupported",
    "preconditioned",
    "multivariate",
    "temporally_compounding",
    "spatially_compounding",
    "uncertain",
}
MECHANISM_SUPPORT_LABELS = {"supported", "weak", "missing", "contradicted"}
COMPOUND_DECISIONS = {"candidate", "unsupported", "needs_review"}
DATE_PRECISION_LABELS = {"day", "month", "season", "year", "relative", "unknown"}
EXTRACTION_METHODS = {"llm", "regex", "human", "fixture"}
ALLOWED_UNITS = {
    "mu",
    "hectare",
    "km2",
    "person",
    "household",
    "RMB_yuan",
    "RMB_10k_yuan",
    "RMB_100m_yuan",
    "percent",
    "ton",
    "unknown",
}
AGRICULTURE_FIELDS = {
    "crop_or_farmland_area_affected",
    "crop_or_yield_loss",
    "agricultural_economic_loss",
    "irrigation_or_water_shortage",
    "recovery_or_replanting_action",
    "qualitative_agricultural_impact",
}
HAZARD_FIELDS = {
    "hazard_occurrence",
    "rainfall_or_flood_description",
    "drought_description",
    "hazard_intensity",
    "hazard_location",
    "hazard_time",
}
IMPACT_FIELDS = {
    "affected_population",
    "economic_loss",
    "infrastructure_damage",
    "water_supply_disruption",
    "transport_or_power_disruption",
    "qualitative_impact",
}
RESPONSE_FIELDS = {
    "response_action_present",
    "emergency_response_level",
    "water_supply_or_drought_relief",
    "flood_control_or_drainage",
    "agricultural_recovery_or_replanting",
    "relief_funding_or_materials",
    "qualitative_response",
}
CLAIM_FIELDS_BY_TYPE = {
    "hazard": HAZARD_FIELDS,
    "impact": IMPACT_FIELDS,
    "agriculture_impact": AGRICULTURE_FIELDS,
    "response": RESPONSE_FIELDS,
    "time": {"event_time_mentioned"},
    "location": {"location_mentioned"},
    "source_context": {"source_attribution", "repost_marker", "source_context"},
}


def _require(value: Any, field_name: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
        raise SchemaValidationError(f"{field_name} is required")


def _enum(value: str, allowed: set[str], field_name: str) -> None:
    if value not in allowed:
        raise SchemaValidationError(f"{field_name} has unsupported value: {value}")


def _date(value: str | None, field_name: str, required: bool = False) -> None:
    if value is None:
        if required:
            raise SchemaValidationError(f"{field_name} is required")
        return
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field_name} must be an ISO date string")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} must use YYYY-MM-DD format") from exc


def _datetime_z(value: str | None, field_name: str, required: bool = False) -> None:
    if value is None:
        if required:
            raise SchemaValidationError(f"{field_name} is required")
        return
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field_name} must be an ISO datetime string")
    parse_value = value.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} must be ISO-8601 datetime") from exc


def _date_pair(start: str | None, end: str | None, start_field: str, end_field: str, required: bool = False) -> None:
    _date(start, start_field, required=required)
    _date(end, end_field, required=required)
    if start and end and start > end:
        raise SchemaValidationError(f"{start_field} must be <= {end_field}")


def _url(value: str, field_name: str) -> None:
    _require(value, field_name)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SchemaValidationError(f"{field_name} must be an http(s) URL")


def _list_of_ids(values: list[str], prefix: str, field_name: str, allow_empty: bool = False) -> None:
    if not isinstance(values, list):
        raise SchemaValidationError(f"{field_name} must be a list")
    if not allow_empty and not values:
        raise SchemaValidationError(f"{field_name} must not be empty")
    bad = [value for value in values if not isinstance(value, str) or not value.startswith(prefix)]
    if bad:
        raise SchemaValidationError(f"{field_name} contains IDs without {prefix} prefix: {bad}")


def _non_empty_location(admin_area: dict[str, Any], spatial_footprint: dict[str, Any], input_record_refs: list[str]) -> bool:
    admin_has_value = any(v not in (None, "", [], {}) for v in admin_area.values())
    footprint_has_value = any(v not in (None, "", [], {}) for v in spatial_footprint.values())
    return admin_has_value or footprint_has_value or bool(input_record_refs)


@dataclass
class ClimateSignalEvent(ModelMixin):
    event_id: str
    hazard_type: str
    admin_area: dict[str, Any]
    spatial_footprint: dict[str, Any]
    start_date: str
    end_date: str
    severity_metrics: dict[str, Any]
    source_dataset: str
    input_record_refs: list[str] = field(default_factory=list)
    claim_status: str = "candidate_climate_signal_only"
    schema_version: str = "v1"
    created_by: str = "pipeline"
    run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(self.event_id, "event_id")
        if not self.event_id.startswith("evt_"):
            raise SchemaValidationError("event_id must start with evt_")
        _enum(self.hazard_type, HAZARD_TYPES, "hazard_type")
        _date_pair(self.start_date, self.end_date, "start_date", "end_date", required=True)
        _require(self.source_dataset, "source_dataset")
        if self.claim_status != "candidate_climate_signal_only":
            raise SchemaValidationError("ClimateSignalEvent must remain candidate_climate_signal_only")
        if not isinstance(self.admin_area, dict) or not isinstance(self.spatial_footprint, dict):
            raise SchemaValidationError("admin_area and spatial_footprint must be objects")
        if not _non_empty_location(self.admin_area, self.spatial_footprint, self.input_record_refs):
            raise SchemaValidationError("ClimateSignalEvent needs admin_area, spatial_footprint, or input_record_refs")


@dataclass
class EvidencePage(ModelMixin):
    page_id: str
    event_id: str
    url: str
    canonical_url: str
    title: str
    source_domain: str
    source_type: str
    retrieval: dict[str, Any]
    snapshot: dict[str, Any]
    dates: dict[str, Any]
    audit: dict[str, Any]
    source_name: str | None = None
    schema_version: str = "v1"
    created_by: str = "pipeline"
    run_id: str | None = None

    def __post_init__(self) -> None:
        _require(self.page_id, "page_id")
        if not self.page_id.startswith("pg_"):
            raise SchemaValidationError("page_id must start with pg_")
        _require(self.event_id, "event_id")
        if not self.event_id.startswith("evt_"):
            raise SchemaValidationError("event_id must start with evt_")
        _url(self.url, "url")
        _url(self.canonical_url, "canonical_url")
        _require(self.title, "title")
        _require(self.source_domain, "source_domain")
        _enum(self.source_type, SOURCE_TYPES, "source_type")
        self._validate_retrieval()
        self._validate_snapshot()
        self._validate_dates()
        self._validate_audit()

    def _validate_retrieval(self) -> None:
        if not isinstance(self.retrieval, dict):
            raise SchemaValidationError("retrieval must be an object")
        _datetime_z(self.retrieval.get("retrieved_at"), "retrieval.retrieved_at", required=True)
        client = self.retrieval.get("search_client")
        if client not in {"bocha", "fixture", "manual_fixture", "other", "tavily"}:
            raise SchemaValidationError(f"retrieval.search_client has unsupported value: {client}")

    def _validate_snapshot(self) -> None:
        if not isinstance(self.snapshot, dict):
            raise SchemaValidationError("snapshot must be an object")
        _require(self.snapshot.get("content_hash"), "snapshot.content_hash")
        fetch_status = self.snapshot.get("fetch_status")
        if fetch_status not in {"success", "failed", "partial"}:
            raise SchemaValidationError(f"snapshot.fetch_status has unsupported value: {fetch_status}")
        if self.audit.get("page_acceptance") == "accepted" and not self.snapshot.get("cleaned_text_path"):
            raise SchemaValidationError("accepted EvidencePage requires snapshot.cleaned_text_path")

    def _validate_dates(self) -> None:
        if not isinstance(self.dates, dict):
            raise SchemaValidationError("dates must be an object")
        _date(self.dates.get("publication_date"), "dates.publication_date")
        _date_pair(
            self.dates.get("event_mentioned_start_date"),
            self.dates.get("event_mentioned_end_date"),
            "dates.event_mentioned_start_date",
            "dates.event_mentioned_end_date",
        )
        pub_source = self.dates.get("publication_date_source", "unknown")
        if pub_source not in {"metadata", "page_text", "url", "unknown"}:
            raise SchemaValidationError(f"dates.publication_date_source has unsupported value: {pub_source}")
        event_source = self.dates.get("event_mentioned_date_source", "unknown")
        if event_source not in {"page_text", "title", "unknown"}:
            raise SchemaValidationError(f"dates.event_mentioned_date_source has unsupported value: {event_source}")
        precision = self.dates.get("date_precision", "unknown")
        _enum(precision, DATE_PRECISION_LABELS, "dates.date_precision")
        pub = self.dates.get("publication_date")
        mentioned = self.dates.get("event_mentioned_start_date")
        if pub and mentioned == pub and event_source == "unknown" and not self.dates.get("event_mentioned_raw_text"):
            raise SchemaValidationError("publication_date must not be copied into event-mentioned date without textual support")

    def _validate_audit(self) -> None:
        if not isinstance(self.audit, dict):
            raise SchemaValidationError("audit must be an object")
        _enum(self.audit.get("location_relevance", "unknown"), LOCATION_RELEVANCE_LABELS, "audit.location_relevance")
        _enum(self.audit.get("hazard_relevance", "unknown"), HAZARD_RELEVANCE_LABELS, "audit.hazard_relevance")
        _enum(self.audit.get("temporal_relevance", "unknown"), TEMPORAL_RELEVANCE_LABELS, "audit.temporal_relevance")
        _enum(self.audit.get("evidence_role", "contextual"), EVIDENCE_ROLE_LABELS, "audit.evidence_role")
        _enum(self.audit.get("page_acceptance", "needs_review"), PAGE_ACCEPTANCE_LABELS, "audit.page_acceptance")
        if self.source_type == "cross_region_official_repost" and not self.audit.get("cross_region_repost_flag"):
            raise SchemaValidationError("cross_region_official_repost pages require audit.cross_region_repost_flag=true")
        if self.audit.get("page_acceptance") == "accepted":
            if self.audit.get("location_relevance") not in {"exact", "partial", "broad"}:
                raise SchemaValidationError("accepted page requires relevant location")
            if self.audit.get("hazard_relevance") not in {"exact", "compatible", "weak"}:
                raise SchemaValidationError("accepted page requires relevant hazard")
            role = self.audit.get("evidence_role", "contextual")
            temporal = self.audit.get("temporal_relevance", "unknown")
            if role != "contextual" and temporal in {"mismatch", "unknown"}:
                raise SchemaValidationError("accepted non-contextual page requires non-mismatched temporal relevance")


@dataclass
class EvidenceClaim(ModelMixin):
    claim_id: str
    event_id: str
    page_id: str
    claim_type: str
    field: str
    span: dict[str, Any]
    support_status: str
    value: Any = None
    unit: str | None = None
    normalized_value: Any = None
    normalized_unit: str | None = None
    qualifier: str | None = None
    location_mentioned: str | None = None
    event_time_mentioned: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "v1"
    created_by: str = "pipeline"
    run_id: str | None = None

    def __post_init__(self) -> None:
        _require(self.claim_id, "claim_id")
        if not self.claim_id.startswith("clm_"):
            raise SchemaValidationError("claim_id must start with clm_")
        _require(self.event_id, "event_id")
        _require(self.page_id, "page_id")
        if not self.event_id.startswith("evt_"):
            raise SchemaValidationError("event_id must start with evt_")
        if not self.page_id.startswith("pg_"):
            raise SchemaValidationError("page_id must start with pg_")
        _enum(self.claim_type, CLAIM_TYPES, "claim_type")
        if self.field not in CLAIM_FIELDS_BY_TYPE[self.claim_type]:
            raise SchemaValidationError(f"field {self.field} is not allowed for claim_type {self.claim_type}")
        _enum(self.support_status, SUPPORT_STATUS_LABELS, "support_status")
        self._validate_span()
        self._validate_unit()
        self._validate_event_time()
        self._validate_extraction()
        self._validate_claim_validation()

    def _validate_span(self) -> None:
        if not isinstance(self.span, dict):
            raise SchemaValidationError("span must be an object")
        start = self.span.get("span_start")
        end = self.span.get("span_end")
        text = self.span.get("span_text")
        if not isinstance(start, int) or not isinstance(end, int):
            raise SchemaValidationError("span_start and span_end must be integers")
        if start < 0 or end <= start:
            raise SchemaValidationError("span offsets are invalid")
        _require(text, "span.span_text")
        if len(text) > end - start and self.span.get("offsets_are_normalized") is not True:
            raise SchemaValidationError("span_text length exceeds span offset width")

    def _validate_unit(self) -> None:
        if self.unit is not None:
            _enum(self.unit, ALLOWED_UNITS, "unit")
        if self.normalized_unit is not None:
            _enum(self.normalized_unit, ALLOWED_UNITS, "normalized_unit")

    def _validate_event_time(self) -> None:
        if not isinstance(self.event_time_mentioned, dict):
            raise SchemaValidationError("event_time_mentioned must be an object")
        _date_pair(
            self.event_time_mentioned.get("start_date"),
            self.event_time_mentioned.get("end_date"),
            "event_time_mentioned.start_date",
            "event_time_mentioned.end_date",
        )
        precision = self.event_time_mentioned.get("precision", "unknown")
        _enum(precision, DATE_PRECISION_LABELS, "event_time_mentioned.precision")
        status = self.event_time_mentioned.get("support_status", self.support_status)
        _enum(status, SUPPORT_STATUS_LABELS, "event_time_mentioned.support_status")

    def _validate_extraction(self) -> None:
        if not self.extraction:
            return
        if not isinstance(self.extraction, dict):
            raise SchemaValidationError("extraction must be an object")
        method = self.extraction.get("method")
        if method is not None:
            _enum(method, EXTRACTION_METHODS, "extraction.method")
        _datetime_z(self.extraction.get("extracted_at"), "extraction.extracted_at")

    def _validate_claim_validation(self) -> None:
        if not self.validation:
            return
        if not isinstance(self.validation, dict):
            raise SchemaValidationError("validation must be an object")
        status = self.validation.get("validator_status")
        if status is not None:
            _enum(status, VALIDATOR_STATUS_LABELS, "validation.validator_status")
        if self.support_status in {"span_supported", "weakly_supported"}:
            if self.validation.get("span_offsets_valid") is False:
                raise SchemaValidationError("supporting claims require valid span offsets")
            if self.value is not None and self.validation.get("value_supported_by_span") is False:
                raise SchemaValidationError("numeric or explicit values must be supported by span")
            if self.unit is not None and self.validation.get("unit_supported_by_span") is False:
                raise SchemaValidationError("units must be supported by span")
            if any(self.event_time_mentioned.get(k) for k in ("start_date", "end_date")) and self.validation.get("date_supported_by_span") is False:
                raise SchemaValidationError("event-mentioned dates must be supported by span")


@dataclass
class SingleEventEvidenceNode(ModelMixin):
    event_node_id: str
    event_id: str
    supporting_page_ids: list[str]
    supporting_claim_ids: list[str]
    support_labels: dict[str, str]
    alignment: dict[str, str]
    source_quality: dict[str, Any]
    coverage_status: str
    missing_evidence: list[str]
    confidence: dict[str, Any]
    final_event_status: str
    schema_version: str = "v1"
    created_by: str = "pipeline"
    run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(self.event_node_id, "event_node_id")
        if not self.event_node_id.startswith("enode_"):
            raise SchemaValidationError("event_node_id must start with enode_")
        _require(self.event_id, "event_id")
        if not self.event_id.startswith("evt_"):
            raise SchemaValidationError("event_id must start with evt_")
        _list_of_ids(self.supporting_page_ids, "pg_", "supporting_page_ids", allow_empty=True)
        _list_of_ids(self.supporting_claim_ids, "clm_", "supporting_claim_ids", allow_empty=True)
        for key in ("hazard_support", "impact_support", "agriculture_support", "response_support"):
            _enum(self.support_labels.get(key), SUPPORT_LABELS, f"support_labels.{key}")
        for key in ("time_alignment", "location_alignment", "hazard_alignment"):
            if key in self.alignment:
                _enum(self.alignment[key], ALIGNMENT_LABELS, f"alignment.{key}")
        _enum(self.source_quality.get("label", "unknown"), SOURCE_QUALITY_LABELS, "source_quality.label")
        for key, value in self.source_quality.items():
            if key.endswith("_count") and (not isinstance(value, int) or value < 0):
                raise SchemaValidationError(f"source_quality.{key} must be a non-negative integer")
        _enum(self.coverage_status, COVERAGE_STATUS_LABELS, "coverage_status")
        _enum(self.confidence.get("label"), CONFIDENCE_LABELS, "confidence.label")
        _enum(self.final_event_status, FINAL_EVENT_STATUS_LABELS, "final_event_status")
        if self.coverage_status == "failed" and self.confidence.get("label") != "insufficient":
            raise SchemaValidationError("coverage_status=failed requires confidence.label=insufficient")
        if self.support_labels.get("agriculture_support") == "supported" and not self.supporting_claim_ids:
            raise SchemaValidationError("agriculture_support=supported requires supporting_claim_ids")


@dataclass
class CompoundCandidate(ModelMixin):
    compound_candidate_id: str
    events_involved: list[str]
    supporting_event_node_ids: list[str]
    candidate_type: str
    candidate_generation_method: str
    relations: dict[str, str]
    mechanism_support: dict[str, Any]
    supporting_claim_ids: list[str]
    missing_evidence: list[str]
    confidence: dict[str, Any]
    decision: str
    schema_version: str = "v1"
    created_by: str = "pipeline"
    run_id: str | None = None
    notes: list[str] = field(default_factory=list)
    negative_control_fixture: bool = False

    def __post_init__(self) -> None:
        _require(self.compound_candidate_id, "compound_candidate_id")
        if not self.compound_candidate_id.startswith("cc_"):
            raise SchemaValidationError("compound_candidate_id must start with cc_")
        _list_of_ids(self.events_involved, "evt_", "events_involved")
        if len(self.events_involved) < 2:
            raise SchemaValidationError("events_involved must contain at least two events")
        if not self.negative_control_fixture:
            _list_of_ids(self.supporting_event_node_ids, "enode_", "supporting_event_node_ids")
        else:
            _list_of_ids(self.supporting_event_node_ids, "enode_", "supporting_event_node_ids", allow_empty=True)
        _enum(self.candidate_type, COMPOUND_CANDIDATE_TYPES, "candidate_type")
        mechanism_label = self.mechanism_support.get("label")
        _enum(mechanism_label, MECHANISM_SUPPORT_LABELS, "mechanism_support.label")
        self._downgrade_unsupported_mechanism(mechanism_label)
        _list_of_ids(self.supporting_claim_ids, "clm_", "supporting_claim_ids", allow_empty=True)
        _enum(self.confidence.get("label"), CONFIDENCE_LABELS, "confidence.label")
        _enum(self.decision, COMPOUND_DECISIONS, "decision")
        if self.confidence.get("label") == "high" and (
            self.candidate_type in {"unsupported", "uncertain"} or mechanism_label != "supported"
        ):
            raise SchemaValidationError("high compound confidence requires supported mechanism and supported candidate type")

    def _downgrade_unsupported_mechanism(self, mechanism_label: str) -> None:
        if self.candidate_type == "preconditioned" and mechanism_label in {"missing", "contradicted"}:
            self.candidate_type = "uncertain"
            if not self.missing_evidence:
                self.missing_evidence.append("preconditioning mechanism not supported by accepted claims")
            if self.decision == "candidate":
                self.decision = "needs_review"
            if self.confidence.get("label") == "high":
                self.confidence["label"] = "low"


@dataclass
class AdminCandidate(ModelMixin):
    country: str = "中国"
    province: str | None = None
    city: str | None = None
    county: str | None = None
    admin_code: str | None = None
    match_level: str = "unknown"
    match_confidence: float = 0.0
    aliases: list[str] = field(default_factory=list)


@dataclass
class EventRecord(ModelMixin):
    event_id: str
    target_case_id: str
    hazard_type: str
    hazard_family: str
    era5_indicator: str
    detection_rule_version: str
    source_csv: str
    source_row_ids: list[str]
    selected_from_csv: bool
    fixture_only: bool
    start_date: str
    end_date: str
    peak_date: str | None
    duration_days: int | None
    duration_months: int | None
    representative_point: dict[str, float]
    grid_cells: list[dict[str, Any]]
    severity_metrics: dict[str, Any]
    admin_candidates: list[dict[str, Any]]
    search_aliases: dict[str, list[str]]
    event_time_window_for_retrieval: dict[str, str]
    notes: str | None = None
    selection_method: str = "target_case_bbox_rank_aggregate"
    selection_rank_metrics: dict[str, Any] = field(default_factory=dict)
    retrieval_mode: str = "live_attempt"
    mode_notice: str | None = None

    def __post_init__(self) -> None:
        required = [
            self.event_id,
            self.target_case_id,
            self.hazard_type,
            self.source_csv,
            self.start_date,
            self.end_date,
            self.admin_candidates,
            self.event_time_window_for_retrieval,
        ]
        if any(v in (None, "", []) for v in required):
            raise ValueError("EventRecord is missing required fields")


@dataclass
class SearchIntent(ModelMixin):
    intent_id: str
    event_id: str
    target_admin: str
    target_admin_level: str
    agency_type: str
    hazard_type: str
    evidence_need: str
    query: str
    expected_page_types: list[str]
    reason: str
    must_include_terms: list[str]
    should_include_terms: list[str]
    exclude_terms: list[str]
    time_window: dict[str, str]
    sector: str | None = None
    admin_level: str | None = None
    intent: str | None = None
    expected_fields: list[str] = field(default_factory=list)
    domain_constraint: str | None = None
    document_type: str | None = None
    source_key: str | None = None
    query_role: str = "evidence_discovery"


@dataclass
class SearchResult(ModelMixin):
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    fixture_path: str | None = None


@dataclass
class OfficialSource(ModelMixin):
    source_id: str
    canonical_domain: str
    base_url: str
    source_name_zh: str
    agency_type: str
    source_tier: str
    admin_scope: dict[str, Any]
    parent_source_id: str | None
    validation_status: str
    validation_method: str
    officiality_evidence: list[dict[str, Any]]
    region_evidence: list[dict[str, Any]]
    discovered_by_action_id: str | None
    discovery_query: str | None
    first_seen_at: str
    last_seen_at: str
    last_validated_at: str
    active: bool
    robots_status: str
    allowed_for_main_evidence: bool
    allowed_for_source_discovery_only: bool
    repost_risk: str
    yield_stats: dict[str, int]
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.agency_type not in config.AGENCY_TYPES:
            raise ValueError(f"Unsupported agency_type: {self.agency_type}")
        if self.source_tier not in config.SOURCE_TIERS:
            raise ValueError(f"Unsupported source_tier: {self.source_tier}")
        if self.validation_status != "validated_official":
            self.allowed_for_main_evidence = False


@dataclass
class RetrievalAction(ModelMixin):
    action_id: str
    event_id: str
    iteration: int
    action_type: str
    target_admin: dict[str, Any]
    agency_type: str
    evidence_need: str
    hazard_type: str
    query: str | None
    url: str | None
    reason: str
    planned_by: str
    planner_prompt_version: str
    policy_guard_version: str
    preconditions: dict[str, Any]
    expected_page_types: list[str]
    budget_cost: dict[str, int]
    status: str
    result_summary: dict[str, Any]
    created_at: str
    executed_at: str | None = None

    def __post_init__(self) -> None:
        if self.action_type not in config.ACTION_TYPES:
            raise ValueError(f"Unsupported action_type: {self.action_type}")


@dataclass
class FetchedPage(ModelMixin):
    url: str
    canonical_url: str
    domain: str
    http_status: int
    fetch_time: str
    content_type: str
    content_hash: str
    html: str
    text: str
    title: str
    snapshot_paths: dict[str, str]
    source_result: dict[str, Any]
    error: str | None = None
    fetch_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageAudit(ModelMixin):
    page_audit_id: str
    event_id: str
    action_id: str
    source_id: str | None
    url: str
    canonical_url: str
    domain: str
    http_status: int
    fetch_time: str
    content_hash: str
    snapshot_paths: dict[str, str]
    publication_date_raw: str | None
    publication_date: str | None
    page_title: str
    source_official_status: str
    source_region_match: str
    target_time_relevance: str
    is_cross_region_repost: bool
    repost_source_raw: str | None
    page_type: str
    page_type_confidence: float
    evidence_types_detected: list[str]
    allowed_for_extraction: bool
    allowed_for_main_evidence: bool
    rejection_reasons: list[str]
    auditor_components: dict[str, Any]
    notes: str | None = None
    fetch_metadata: dict[str, Any] = field(default_factory=dict)
    primary_page_type: str | None = None
    page_type_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.page_type not in config.PAGE_TYPES:
            raise ValueError(f"Unsupported page_type: {self.page_type}")
        if self.primary_page_type is None:
            self.primary_page_type = self.page_type
        if not self.page_type_labels:
            self.page_type_labels = [self.page_type]


@dataclass
class EvidenceUnit(ModelMixin):
    evidence_unit_id: str
    page_audit_id: str
    event_id: str
    canonical_url: str
    source_id: str | None
    unit_text: str
    char_start: int
    char_end: int
    unit_type_hint: str
    location_mentions: list[str]
    date_mentions: list[str]
    hazard_mentions: list[str]
    splitter_version: str = "splitter_v1"


@dataclass
class EvidenceRecord(ModelMixin):
    evidence_id: str
    event_id: str
    page_audit_id: str
    evidence_unit_id: str
    canonical_url: str
    source_id: str | None
    source_agency: str
    source_tier: str
    publication_date: str | None
    page_type: str
    evidence_type: str
    raw_fields: dict[str, Any]
    source_spans: list[dict[str, Any]]
    llm_provider: str
    llm_model: str
    extractor_prompt_version: str
    raw_llm_output_path: str | None
    field_validation: dict[str, int]
    created_at: str
    snapshot_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class NormalizedEvidenceRecord(ModelMixin):
    normalized_evidence_id: str
    evidence_id: str
    event_id: str
    canonical_url: str
    source_id: str | None
    page_type: str
    evidence_type: str
    time_span: dict[str, Any] | None
    location_norm: list[dict[str, Any]]
    hazard_norm: dict[str, Any]
    response_norm: dict[str, Any] | None
    population_metrics: list[dict[str, Any]]
    agriculture_metrics: list[dict[str, Any]]
    hydrology_metrics: list[dict[str, Any]]
    economic_metrics: list[dict[str, Any]]
    infrastructure_metrics: list[dict[str, Any]]
    normalization_log: list[dict[str, Any]]
    dedupe_key: str
    is_duplicate_of: str | None
    normalizer_version: str
    created_at: str


@dataclass
class EventEvidenceLink(ModelMixin):
    link_id: str
    event_id: str
    evidence_id: str
    normalized_evidence_id: str
    relationship_label: str
    support_strength: str
    axis_scores: dict[str, int]
    component_scores: dict[str, int]
    total_score: int
    matched_admin_units: list[str]
    matched_dates: list[str]
    penalties: list[str]
    score_version: str
    review_status: str
    created_at: str


@dataclass
class EventDossier(ModelMixin):
    dossier_id: str
    event_id: str
    target_case_id: str
    event_summary: dict[str, Any]
    retrieval_summary: dict[str, Any]
    source_coverage: dict[str, str]
    accepted_evidence_pages: list[Any]
    rejected_pages: list[dict[str, Any]]
    evidence_records: list[str]
    normalized_evidence_records: list[str]
    event_evidence_links: list[str]
    best_source_spans: list[dict[str, Any]]
    support_summary: dict[str, Any]
    remaining_uncertainty: list[str]
    retrieval_failure_notes: list[str]
    generated_at: str
    dossier_builder_version: str
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateChain(ModelMixin):
    chain_id: str
    event_id: str
    chain_type: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    support_level: str
    time_gap_days: int | None
    location_overlap: str
    shared_receptors: list[str]
    requires_independent_validation: bool
    notes: str | None
    chain_assembler_version: str


@dataclass
class EventEvidenceState(ModelMixin):
    event: dict[str, Any]
    known_hazard_evidence: list[str] = field(default_factory=list)
    known_response_evidence: list[str] = field(default_factory=list)
    known_social_impact_evidence: list[str] = field(default_factory=list)
    known_agricultural_impact_evidence: list[str] = field(default_factory=list)
    known_hydrological_evidence: list[str] = field(default_factory=list)
    source_coverage: dict[str, str] = field(default_factory=dict)
    accepted_pages: list[str] = field(default_factory=list)
    rejected_pages: list[dict[str, Any]] = field(default_factory=list)
    budgets_spent: dict[str, int] = field(default_factory=lambda: {"search_queries": 0, "fetches": 0, "llm_calls": 0})
    last_planner_actions: list[str] = field(default_factory=list)
