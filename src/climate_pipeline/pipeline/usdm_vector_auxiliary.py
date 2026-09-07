"""Offline-only USDM vector auxiliary-grade implementation for repair Step 4.

County-week D1+ remains the sole drought-support authority.  This module reads
one immutable local production-support package and reports only bounded grid-
center alignment provenance.  It has no acquisition, HTTP, API, model,
retrieval, guard, cache-write, or fallback-to-another-source path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from rainfall_p99_pipeline import (
    CountyFeature,
    _crs_is_lonlat_wgs84_compatible,
    _parse_dbf_records,
    _parse_shp_polygons,
    _read_shapefile_components,
    point_in_feature,
)

from .official_package import (
    LoadedOfficialPackage,
    OfficialPackageError,
    PackageClassification,
    load_official_package,
)
from .schemas import ADAPTER_VERSION_ENVELOPE, AxisValue


USDM_VECTOR_AUXILIARY_ADAPTER_VERSION = (
    f"{ADAPTER_VERSION_ENVELOPE}.usdm_vector_auxiliary_local.1"
)
USDM_VECTOR_AUXILIARY_COMPUTATION_VERSION = (
    "ce_agent_usdm_vector_auxiliary_computation_v1.0.0"
)
USDM_VECTOR_AUXILIARY_GRADE = (
    "grid_center_aligned_to_usdm_d1plus_polygon"
)
USDM_VECTOR_SUPPORT_LANE = "usdm_vector_auxiliary"
USDM_VECTOR_PACKAGE_SOURCE_LANE = "usdm_vector_geometry_auxiliary"
ALIGNMENT_ADDENDUM_VERSION = "ce_agent_alignment_contract_addendum_v1.0.0"
ALIGNMENT_ADDENDUM_SHA256 = (
    "f8b2cc93efd1b6b26bdc909223220e1c46f37e0805e41532b4de8bbeed2073b6"
)
PARENT_PILOT_PACKAGE_ID = "pilot_draft_usdm_weekly_vectors_frozen40_dev20_v1"
PARENT_PILOT_MANIFEST_SHA256 = (
    "65fd302f3eb88adc5feec2882711cc213315f1cd99dcfc5adcd35858ac5da3d0"
)
USDM_AUXILIARY_RULE_IDS = (
    "USDM-AUX-GRADE-001",
    "USDM-AUX-FALLBACK-001",
    "USDM-AUX-NONNEG-001",
    "USDM-AUX-ISOLATION-001",
    "USDM-AUX-CLAIM-001",
    "USDM-AUX-BIAS-001",
)

_EXPECTED_WEEKS = (
    "2020-12-29",
    "2021-09-28",
    "2021-11-30",
    "2022-07-26",
    "2022-08-30",
    "2022-10-25",
    "2022-11-29",
    "2023-07-25",
    "2023-08-29",
    "2023-12-26",
    "2024-01-30",
    "2024-10-29",
    "2024-11-26",
    "2025-02-25",
)
_DBF_COLUMNS = ("OBJECTID", "DM", "Shape_Leng", "Shape_Area")
_FILE_RE = re.compile(r"^USDM_(\d{8})_M\.zip$")


class UsdmVectorAuxiliaryError(ValueError):
    """Raised when the optional vector enhancement must fail closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _geometry_sha256(rings: Sequence[Sequence[Sequence[float]]]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", len(rings)))
    for ring in rings:
        digest.update(struct.pack("<I", len(ring)))
        for point in ring:
            digest.update(struct.pack("<dd", float(point[0]), float(point[1])))
    return digest.hexdigest()


@dataclass(frozen=True)
class UsdmVectorFeature:
    feature_index: int
    object_id: str
    dm: int
    bbox: tuple[float, float, float, float]
    ring_count: int
    point_count: int
    geometry_sha256: str
    geometry: CountyFeature

    def provenance(self) -> Mapping[str, Any]:
        return {
            "feature_index": self.feature_index,
            "object_id": self.object_id,
            "dm": self.dm,
            "geometry_sha256": self.geometry_sha256,
        }


@dataclass(frozen=True)
class UsdmVectorWeek:
    map_date: str
    raw_relative_path: str
    raw_sha256: str
    raw_byte_size: int
    crs_wkt: str
    bounds: tuple[float, float, float, float]
    features: tuple[UsdmVectorFeature, ...]
    normalized_bundle_sha256: str


@dataclass(frozen=True)
class UsdmVectorAuxiliaryResult:
    status: str
    auxiliary_grade: str | None
    coverage: str
    diagnostic_codes: tuple[str, ...]
    provenance: Mapping[str, Any]

    def to_provenance(self) -> Mapping[str, Any]:
        return {
            "status": self.status,
            "usdm_vector_auxiliary_grade": self.auxiliary_grade,
            "auxiliary_coverage": self.coverage,
            "diagnostic_codes": list(self.diagnostic_codes),
            **dict(self.provenance),
        }


def _validate_rings(
    rings: Sequence[Sequence[Sequence[float]]],
    *,
    file_name: str,
    dm: int,
) -> None:
    if not rings:
        raise UsdmVectorAuxiliaryError(
            f"usdm_vector_geometry_missing:{file_name}:D{dm}"
        )
    for ring_index, ring in enumerate(rings, start=1):
        if len(ring) < 4 or tuple(ring[0][:2]) != tuple(ring[-1][:2]):
            raise UsdmVectorAuxiliaryError(
                f"usdm_vector_invalid_ring:{file_name}:D{dm}:{ring_index}"
            )
        for point in ring:
            lon, lat = float(point[0]), float(point[1])
            if not (
                math.isfinite(lon)
                and math.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                raise UsdmVectorAuxiliaryError(
                    f"usdm_vector_invalid_coordinate:{file_name}:D{dm}"
                )


def _load_week(path: Path, *, package_root: Path) -> UsdmVectorWeek:
    match = _FILE_RE.fullmatch(path.name)
    if not match:
        raise UsdmVectorAuxiliaryError(f"unexpected_usdm_vector_name:{path.name}")
    map_date = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    stem = f"USDM_{match.group(1)}"
    required_members = {
        f"{stem}.shp",
        f"{stem}.shx",
        f"{stem}.dbf",
        f"{stem}.prj",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(sorted(archive.namelist()))
            for member in members:
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise UsdmVectorAuxiliaryError(
                        f"unsafe_usdm_vector_member:{path.name}:{member}"
                    )
            missing = sorted(required_members - set(members))
            if missing:
                raise UsdmVectorAuxiliaryError(
                    f"usdm_vector_components_missing:{path.name}:{','.join(missing)}"
                )
        shp_bytes, dbf_bytes, prj, encoding = _read_shapefile_components(path)
        if not prj or not _crs_is_lonlat_wgs84_compatible(prj):
            raise UsdmVectorAuxiliaryError(
                f"usdm_vector_crs_not_wgs84:{path.name}"
            )
        columns, records = _parse_dbf_records(dbf_bytes, encoding=encoding)
        shape_type, shapes, bounds = _parse_shp_polygons(shp_bytes)
    except UsdmVectorAuxiliaryError:
        raise
    except Exception as exc:
        raise UsdmVectorAuxiliaryError(
            f"usdm_vector_unreadable:{path.name}:{type(exc).__name__}"
        ) from exc
    if shape_type != 5 or tuple(columns) != _DBF_COLUMNS:
        raise UsdmVectorAuxiliaryError(f"usdm_vector_schema_mismatch:{path.name}")
    if len(records) != len(shapes) or not records:
        raise UsdmVectorAuxiliaryError(
            f"usdm_vector_record_geometry_count_mismatch:{path.name}"
        )
    if bounds is None or not all(math.isfinite(float(value)) for value in bounds):
        raise UsdmVectorAuxiliaryError(f"usdm_vector_bounds_invalid:{path.name}")

    features: list[UsdmVectorFeature] = []
    index_rows: list[dict[str, Any]] = []
    seen_dm: set[int] = set()
    raw_sha256 = _sha256_file(path)
    for index, (record, shape) in enumerate(zip(records, shapes), start=1):
        try:
            dm = int(record["DM"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UsdmVectorAuxiliaryError(
                f"usdm_vector_dm_invalid:{path.name}:{index}"
            ) from exc
        if dm not in {0, 1, 2, 3, 4} or dm in seen_dm:
            raise UsdmVectorAuxiliaryError(
                f"usdm_vector_dm_inventory_invalid:{path.name}:D{dm}"
            )
        seen_dm.add(dm)
        rings = shape.get("rings") or []
        bbox_raw = shape.get("bbox")
        if bbox_raw is None:
            raise UsdmVectorAuxiliaryError(
                f"usdm_vector_bbox_missing:{path.name}:D{dm}"
            )
        _validate_rings(rings, file_name=path.name, dm=dm)
        bbox = tuple(float(value) for value in bbox_raw)
        geometry_hash = _geometry_sha256(rings)
        feature = UsdmVectorFeature(
            feature_index=index,
            object_id=str(record.get("OBJECTID") or ""),
            dm=dm,
            bbox=bbox,
            ring_count=len(rings),
            point_count=sum(len(ring) for ring in rings),
            geometry_sha256=geometry_hash,
            geometry=CountyFeature(
                properties=dict(record),
                geometry_type="ShapefilePolygon",
                coordinates=rings,
                bbox=bbox,
            ),
        )
        features.append(feature)
        index_rows.append(
            {
                "normalized_schema_version": (
                    "ce_agent_pilot_usdm_vector_feature_index_v1.0.0"
                ),
                "map_date": map_date,
                "raw_file": path.name,
                "raw_file_sha256": raw_sha256,
                "feature_index": index,
                "object_id": feature.object_id,
                "dm": dm,
                "shape_length_raw": str(record.get("Shape_Leng") or ""),
                "shape_area_raw": str(record.get("Shape_Area") or ""),
                "bbox": list(bbox),
                "ring_count": feature.ring_count,
                "point_count": feature.point_count,
                "geometry_sha256": geometry_hash,
                "crs": "WGS84 geographic longitude/latitude from embedded .prj",
            }
        )
    if seen_dm != {0, 1, 2, 3, 4}:
        raise UsdmVectorAuxiliaryError(
            f"usdm_vector_dm_inventory_incomplete:{path.name}"
        )
    features.sort(key=lambda item: item.dm)
    index_rows.sort(key=lambda item: int(item["dm"]))
    return UsdmVectorWeek(
        map_date=map_date,
        raw_relative_path=path.resolve().relative_to(package_root.resolve()).as_posix(),
        raw_sha256=raw_sha256,
        raw_byte_size=path.stat().st_size,
        crs_wkt=prj,
        bounds=tuple(float(value) for value in bounds),
        features=tuple(features),
        normalized_bundle_sha256=_stable_hash(index_rows),
    )


def parse_grid_centers(
    values: str | Iterable[str | Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    raw_values: Iterable[str | Sequence[float]]
    raw_values = values.split(";") if isinstance(values, str) else values
    result: list[tuple[float, float]] = []
    for value in raw_values:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                lat_text, lon_text = text.split(",", 1)
                lat, lon = float(lat_text), float(lon_text)
            except ValueError as exc:
                raise UsdmVectorAuxiliaryError(
                    f"invalid_candidate_grid_center:{text}"
                ) from exc
        else:
            try:
                lat, lon = float(value[0]), float(value[1])
            except (IndexError, TypeError, ValueError) as exc:
                raise UsdmVectorAuxiliaryError(
                    "invalid_candidate_grid_center"
                ) from exc
        if not (
            math.isfinite(lat)
            and math.isfinite(lon)
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):
            raise UsdmVectorAuxiliaryError("invalid_candidate_grid_center")
        result.append((lat, lon))
    if len(result) != len(set(result)):
        raise UsdmVectorAuxiliaryError("duplicate_candidate_grid_center")
    return tuple(sorted(result))


class UsdmVectorAuxiliaryAdapter:
    """Read and validate one local production-support package."""

    def __init__(self, manifest_path: Path) -> None:
        try:
            self.loaded: LoadedOfficialPackage = load_official_package(
                Path(manifest_path), require_production=True
            )
        except OfficialPackageError as exc:
            raise UsdmVectorAuxiliaryError(str(exc)) from exc
        manifest = self.loaded.manifest
        if (
            manifest.classification is not PackageClassification.PRODUCTION
            or not manifest.production_eligible
            or not manifest.immutable
            or not manifest.read_only
        ):
            raise UsdmVectorAuxiliaryError(
                "usdm_vector_support_not_immutable_production"
            )
        if manifest.source_lane != USDM_VECTOR_PACKAGE_SOURCE_LANE:
            raise UsdmVectorAuxiliaryError("wrong_package_lane:usdm_vector_auxiliary")
        if not manifest.package_id.startswith("production_support_"):
            raise UsdmVectorAuxiliaryError("wrong_package_prefix:usdm_vector_auxiliary")
        self._validate_support_metadata()
        raw_paths = sorted(self.loaded.root.glob("raw/USDM_*_M.zip"))
        weeks = tuple(
            _load_week(path, package_root=self.loaded.root) for path in raw_paths
        )
        self._weeks = {week.map_date: week for week in weeks}
        if tuple(sorted(self._weeks)) != _EXPECTED_WEEKS:
            raise UsdmVectorAuxiliaryError("usdm_vector_week_inventory_mismatch")
        self._validate_normalized_index()

    @property
    def package_id(self) -> str:
        return self.loaded.manifest.package_id

    @property
    def package_hash(self) -> str:
        return self.loaded.manifest.package_manifest_sha256

    @property
    def weeks(self) -> Mapping[str, UsdmVectorWeek]:
        return self._weeks

    def _validate_support_metadata(self) -> None:
        path = self.loaded.root / "production_support_metadata.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UsdmVectorAuxiliaryError(
                "production_support_metadata_unreadable"
            ) from exc
        declared = str(value.get("metadata_sha256") or "")
        payload = dict(value)
        payload.pop("metadata_sha256", None)
        if declared != _stable_hash(payload):
            raise UsdmVectorAuxiliaryError("production_support_metadata_hash_mismatch")
        if value.get("package_id") != self.package_id:
            raise UsdmVectorAuxiliaryError("production_support_metadata_id_mismatch")
        parent = value.get("parent_pilot_package", {})
        if parent != {
            "package_id": PARENT_PILOT_PACKAGE_ID,
            "package_manifest_sha256": PARENT_PILOT_MANIFEST_SHA256,
            "runtime_dependency": False,
        }:
            raise UsdmVectorAuxiliaryError("production_support_parent_mismatch")
        addendum = value.get("alignment_addendum", {})
        if (
            addendum.get("version") != ALIGNMENT_ADDENDUM_VERSION
            or addendum.get("sha256") != ALIGNMENT_ADDENDUM_SHA256
        ):
            raise UsdmVectorAuxiliaryError("production_support_addendum_mismatch")

    def _validate_normalized_index(self) -> None:
        path = self.loaded.root / "normalized/usdm_vector_feature_index_v1.jsonl"
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise UsdmVectorAuxiliaryError(
                            "usdm_vector_normalized_index_not_object"
                        )
                    rows.append(value)
        by_key = {(str(row["map_date"]), int(row["dm"])): row for row in rows}
        if len(rows) != 70 or len(by_key) != 70:
            raise UsdmVectorAuxiliaryError(
                "usdm_vector_normalized_index_inventory_mismatch"
            )
        for map_date, week in self._weeks.items():
            for feature in week.features:
                row = by_key.get((map_date, feature.dm))
                if row is None:
                    raise UsdmVectorAuxiliaryError(
                        "usdm_vector_normalized_index_missing_feature"
                    )
                expected = {
                    "raw_file": Path(week.raw_relative_path).name,
                    "raw_file_sha256": week.raw_sha256,
                    "feature_index": feature.feature_index,
                    "object_id": feature.object_id,
                    "dm": feature.dm,
                    "bbox": list(feature.bbox),
                    "ring_count": feature.ring_count,
                    "point_count": feature.point_count,
                    "geometry_sha256": feature.geometry_sha256,
                }
                if any(row.get(key) != value for key, value in expected.items()):
                    raise UsdmVectorAuxiliaryError(
                        f"usdm_vector_normalized_replay_mismatch:{map_date}:D{feature.dm}"
                    )

    def evaluate(
        self,
        *,
        map_date: str,
        grid_centers: str | Iterable[str | Sequence[float]],
        candidate_physical_source_hash: str,
        county_support_value: AxisValue,
    ) -> UsdmVectorAuxiliaryResult:
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_physical_source_hash):
            raise UsdmVectorAuxiliaryError(
                "candidate_physical_source_hash_required"
            )
        week = self._weeks.get(map_date)
        if week is None:
            return self._unavailable(
                map_date=map_date,
                coverage="unavailable_missing_reference_week",
                diagnostic="vector_reference_week_unavailable",
                candidate_physical_source_hash=candidate_physical_source_hash,
            )
        points = parse_grid_centers(grid_centers)
        if not points:
            return self._unavailable(
                map_date=map_date,
                coverage="unavailable_candidate_grid_lineage",
                diagnostic="candidate_grid_centers_unavailable",
                candidate_physical_source_hash=candidate_physical_source_hash,
            )

        point_rows: list[dict[str, Any]] = []
        used_features: dict[tuple[int, int], Mapping[str, Any]] = {}
        aligned_count = 0
        for lat, lon in points:
            matched: list[Mapping[str, Any]] = []
            boundary_dm: list[int] = []
            for feature in week.features:
                inside, boundary = point_in_feature(lon, lat, feature.geometry)
                if inside:
                    matched.append(feature.provenance())
                    if feature.dm >= 1:
                        used_features[(feature.feature_index, feature.dm)] = (
                            feature.provenance()
                        )
                if boundary:
                    boundary_dm.append(feature.dm)
            d1plus = any(int(item["dm"]) >= 1 for item in matched)
            aligned_count += int(d1plus)
            point_rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "d1plus_aligned": d1plus,
                    "matched_dm": sorted(int(item["dm"]) for item in matched),
                    "boundary_dm": sorted(boundary_dm),
                }
            )
        nonaligned_count = len(points) - aligned_count
        observed_alignment = aligned_count > 0
        diagnostics: list[str] = []
        grade: str | None = None
        if county_support_value is AxisValue.YES and observed_alignment:
            grade = USDM_VECTOR_AUXILIARY_GRADE
        elif county_support_value is AxisValue.YES:
            diagnostics.append("county_support_preserved_vector_nonaligned")
        elif county_support_value is AxisValue.NO and observed_alignment:
            diagnostics.append(
                "county_negative_vector_positive_source_consistency"
            )
        elif county_support_value is AxisValue.UNRESOLVED and observed_alignment:
            diagnostics.append(
                "county_unresolved_vector_alignment_observed_non_authoritative"
            )
        return UsdmVectorAuxiliaryResult(
            status=(
                "grid_center_aligned_to_official_usdm_d1plus_polygon"
                if observed_alignment
                else "explicit_grid_center_nonalignment"
            ),
            auxiliary_grade=grade,
            coverage="complete_for_candidate_grid_centers",
            diagnostic_codes=tuple(diagnostics),
            provenance={
                "observed_d1plus_alignment": observed_alignment,
                "total_grid_center_count": len(points),
                "aligned_grid_center_count": aligned_count,
                "nonaligned_grid_center_count": nonaligned_count,
                "map_date": map_date,
                "source_file": week.raw_relative_path,
                "source_file_sha256": week.raw_sha256,
                "source_file_byte_size": week.raw_byte_size,
                "source_feature_or_polygon_provenance": [
                    used_features[key] for key in sorted(used_features)
                ],
                "input_crs": "EPSG:4326",
                "source_crs": (
                    "embedded WGS84 geographic longitude/latitude .prj"
                ),
                "crs_transformation": (
                    "deterministic identity transform EPSG:4326 to verified WGS84"
                ),
                "crs_wkt_sha256": hashlib.sha256(
                    week.crs_wkt.encode("utf-8")
                ).hexdigest(),
                "package_id": self.package_id,
                "package_manifest_sha256": self.package_hash,
                "candidate_physical_source_hash": candidate_physical_source_hash,
                "point_classification_bundle_sha256": _stable_hash(point_rows),
                "normalized_geometry_bundle_sha256": (
                    week.normalized_bundle_sha256
                ),
                "alignment_addendum_version": ALIGNMENT_ADDENDUM_VERSION,
                "alignment_addendum_sha256": ALIGNMENT_ADDENDUM_SHA256,
                "rule_ids": list(USDM_AUXILIARY_RULE_IDS),
                "adapter_version": USDM_VECTOR_AUXILIARY_ADAPTER_VERSION,
                "computation_version": USDM_VECTOR_AUXILIARY_COMPUTATION_VERSION,
                "support_authority": "county-week D1+ core record only",
                "claim": (
                    "grid center aligned to an official USDM D1+ polygon"
                ),
                "claim_limitations": [
                    "not complete drought-footprint overlap",
                    "not full candidate spatial confirmation",
                    "not drought-episode confirmation",
                    "not independent ground truth",
                ],
                "scientifically_authoritative": False,
                "allowed_uses": [
                    "provenance inspection",
                    "evidence navigation",
                    "display sorting",
                    "sensitivity analysis",
                ],
                "forbidden_cross_axis_contribution": True,
            },
        )

    def _unavailable(
        self,
        *,
        map_date: str,
        coverage: str,
        diagnostic: str,
        candidate_physical_source_hash: str,
    ) -> UsdmVectorAuxiliaryResult:
        return UsdmVectorAuxiliaryResult(
            status="auxiliary_unavailable",
            auxiliary_grade=None,
            coverage=coverage,
            diagnostic_codes=(diagnostic,),
            provenance={
                "observed_d1plus_alignment": None,
                "total_grid_center_count": 0,
                "aligned_grid_center_count": 0,
                "nonaligned_grid_center_count": 0,
                "map_date": map_date,
                "package_id": self.package_id,
                "package_manifest_sha256": self.package_hash,
                "candidate_physical_source_hash": candidate_physical_source_hash,
                "alignment_addendum_version": ALIGNMENT_ADDENDUM_VERSION,
                "alignment_addendum_sha256": ALIGNMENT_ADDENDUM_SHA256,
                "rule_ids": list(USDM_AUXILIARY_RULE_IDS),
                "adapter_version": USDM_VECTOR_AUXILIARY_ADAPTER_VERSION,
                "computation_version": USDM_VECTOR_AUXILIARY_COMPUTATION_VERSION,
                "support_authority": "county-week D1+ core record only",
                "scientifically_authoritative": False,
                "forbidden_cross_axis_contribution": True,
            },
        )


def failed_auxiliary_provenance(
    *,
    error: Exception | str,
    map_date: str,
    package_id: str = "",
    package_hash: str = "",
) -> Mapping[str, Any]:
    """Create a non-authoritative failure diagnostic without changing truth."""

    return {
        "status": "auxiliary_failed",
        "usdm_vector_auxiliary_grade": None,
        "auxiliary_coverage": "failed",
        "diagnostic_codes": ["vector_auxiliary_failure_county_support_preserved"],
        "error": str(error),
        "map_date": map_date,
        "package_id": package_id,
        "package_manifest_sha256": package_hash,
        "alignment_addendum_version": ALIGNMENT_ADDENDUM_VERSION,
        "alignment_addendum_sha256": ALIGNMENT_ADDENDUM_SHA256,
        "rule_ids": list(USDM_AUXILIARY_RULE_IDS),
        "adapter_version": USDM_VECTOR_AUXILIARY_ADAPTER_VERSION,
        "computation_version": USDM_VECTOR_AUXILIARY_COMPUTATION_VERSION,
        "support_authority": "county-week D1+ core record only",
        "scientifically_authoritative": False,
        "forbidden_cross_axis_contribution": True,
    }


__all__ = [
    "ALIGNMENT_ADDENDUM_SHA256",
    "ALIGNMENT_ADDENDUM_VERSION",
    "PARENT_PILOT_MANIFEST_SHA256",
    "PARENT_PILOT_PACKAGE_ID",
    "USDM_AUXILIARY_RULE_IDS",
    "USDM_VECTOR_AUXILIARY_ADAPTER_VERSION",
    "USDM_VECTOR_AUXILIARY_COMPUTATION_VERSION",
    "USDM_VECTOR_AUXILIARY_GRADE",
    "USDM_VECTOR_PACKAGE_SOURCE_LANE",
    "USDM_VECTOR_SUPPORT_LANE",
    "UsdmVectorAuxiliaryAdapter",
    "UsdmVectorAuxiliaryError",
    "UsdmVectorAuxiliaryResult",
    "failed_auxiliary_provenance",
    "parse_grid_centers",
]
