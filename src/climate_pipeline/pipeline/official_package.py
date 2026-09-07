"""Local-only official-source package manifests and integrity verification.

This module is an implementation of the non-scientific W017 package contract.
It does not acquire data, resolve network URLs, or write source caches.  Every
runtime file reference is package-relative and is verified before a package is
returned to an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping


PACKAGE_MANIFEST_SCHEMA_VERSION = (
    "ce_agent_official_source_package_manifest_v1.0.0"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_OLD_WORKSPACE_NAMES = {
    "ce_agent",
    "ce_agent_analysis",
}


class OfficialPackageError(ValueError):
    """Raised when an official package fails closed validation."""


class PackageClassification(str, Enum):
    PRODUCTION = "production"
    PILOT_DRAFT = "pilot_draft"
    NON_PRODUCTION = "non_production"


class PackageFileCountKind(str, Enum):
    CSV_ROWS = "csv_rows"
    JSONL_RECORDS = "jsonl_records"
    FILE_ONLY = "file_only"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OfficialPackageError(f"missing_metadata:{field_name}")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise OfficialPackageError(f"invalid_sha256:{field_name}")


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], kind: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise OfficialPackageError(f"unknown_{kind}_fields:{','.join(unknown)}")


def _tuple_text(values: Iterable[Any], field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not allow_empty and not result:
        raise OfficialPackageError(f"missing_metadata:{field_name}")
    if any(not value.strip() for value in result):
        raise OfficialPackageError(f"blank_value:{field_name}")
    if len(set(result)) != len(result):
        raise OfficialPackageError(f"duplicate_value:{field_name}")
    return result


def _path_mentions_old_workspace(path: Path) -> bool:
    return any(part.casefold() in _OLD_WORKSPACE_NAMES for part in path.parts)


@dataclass(frozen=True)
class CoverageDimension:
    name: str
    expected_values: tuple[str, ...]
    covered_values: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.name, "coverage.name")
        _tuple_text(self.expected_values, f"coverage.{self.name}.expected_values")
        _tuple_text(
            self.covered_values,
            f"coverage.{self.name}.covered_values",
            allow_empty=True,
        )
        unexpected = sorted(set(self.covered_values) - set(self.expected_values))
        if unexpected:
            raise OfficialPackageError(
                f"unexpected_coverage:{self.name}:{','.join(unexpected)}"
            )

    @property
    def complete(self) -> bool:
        return set(self.expected_values) == set(self.covered_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_values": list(self.expected_values),
            "covered_values": list(self.covered_values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoverageDimension":
        allowed = {"name", "expected_values", "covered_values"}
        _reject_unknown(data, allowed, "coverage_dimension")
        try:
            return cls(
                name=str(data["name"]),
                expected_values=tuple(str(v) for v in data["expected_values"]),
                covered_values=tuple(str(v) for v in data["covered_values"]),
            )
        except KeyError as exc:
            raise OfficialPackageError(
                f"missing_metadata:coverage_dimension.{exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class RawToNormalizedLineage:
    raw_field: str
    normalized_field: str
    transformation: str

    def __post_init__(self) -> None:
        _require_text(self.raw_field, "lineage.raw_field")
        _require_text(self.normalized_field, "lineage.normalized_field")
        _require_text(self.transformation, "lineage.transformation")

    def to_dict(self) -> dict[str, str]:
        return {
            "raw_field": self.raw_field,
            "normalized_field": self.normalized_field,
            "transformation": self.transformation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawToNormalizedLineage":
        allowed = {"raw_field", "normalized_field", "transformation"}
        _reject_unknown(data, allowed, "lineage")
        try:
            return cls(
                raw_field=str(data["raw_field"]),
                normalized_field=str(data["normalized_field"]),
                transformation=str(data["transformation"]),
            )
        except KeyError as exc:
            raise OfficialPackageError(
                f"missing_metadata:lineage.{exc.args[0]}"
            ) from exc


@dataclass(frozen=True)
class OfficialPackageFile:
    relative_path: str
    byte_size: int
    sha256: str
    row_or_feature_count: int
    count_kind: PackageFileCountKind
    raw_schema_columns: tuple[str, ...]
    source_record_count: int

    def __post_init__(self) -> None:
        _require_text(self.relative_path, "files.relative_path")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise OfficialPackageError(
                f"runtime_path_not_package_relative:{self.relative_path}"
            )
        if _path_mentions_old_workspace(path):
            raise OfficialPackageError(
                f"old_workspace_runtime_reference:{self.relative_path}"
            )
        if self.byte_size < 0:
            raise OfficialPackageError("invalid_byte_size")
        _require_sha256(self.sha256, f"files.{self.relative_path}.sha256")
        if self.row_or_feature_count < 0 or self.source_record_count < 0:
            raise OfficialPackageError("invalid_record_count")
        _tuple_text(
            self.raw_schema_columns,
            f"files.{self.relative_path}.raw_schema_columns",
            allow_empty=self.count_kind is PackageFileCountKind.FILE_ONLY,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "row_or_feature_count": self.row_or_feature_count,
            "count_kind": self.count_kind.value,
            "raw_schema_columns": list(self.raw_schema_columns),
            "source_record_count": self.source_record_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialPackageFile":
        allowed = {
            "relative_path",
            "byte_size",
            "sha256",
            "row_or_feature_count",
            "count_kind",
            "raw_schema_columns",
            "source_record_count",
        }
        _reject_unknown(data, allowed, "package_file")
        try:
            return cls(
                relative_path=str(data["relative_path"]),
                byte_size=int(data["byte_size"]),
                sha256=str(data["sha256"]),
                row_or_feature_count=int(data["row_or_feature_count"]),
                count_kind=PackageFileCountKind(str(data["count_kind"])),
                raw_schema_columns=tuple(
                    str(value) for value in data["raw_schema_columns"]
                ),
                source_record_count=int(data["source_record_count"]),
            )
        except KeyError as exc:
            raise OfficialPackageError(
                f"missing_metadata:files.{exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, OfficialPackageError):
                raise
            raise OfficialPackageError(f"invalid_package_file:{exc}") from exc


@dataclass(frozen=True)
class OfficialSourcePackageManifest:
    package_manifest_schema_version: str
    package_id: str
    package_type: str
    source_lane: str
    official_provider: str
    official_source_url: str
    historical_source_locator: str
    access_or_download_date: str
    access_date_status: str
    source_product_version: str
    geographic_coverage: tuple[str, ...]
    temporal_coverage: tuple[str, ...]
    files: tuple[OfficialPackageFile, ...]
    normalized_schema_version: str
    raw_to_normalized_lineage: tuple[RawToNormalizedLineage, ...]
    source_record_count: int
    coverage_dimensions: tuple[CoverageDimension, ...]
    known_missingness: tuple[str, ...]
    immutable: bool
    read_only: bool
    classification: PackageClassification
    production_eligible: bool
    package_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.package_manifest_schema_version != PACKAGE_MANIFEST_SCHEMA_VERSION:
            raise OfficialPackageError(
                "incompatible_package_manifest_schema_version:"
                f"{self.package_manifest_schema_version}"
            )
        for field_name in (
            "package_id",
            "package_type",
            "source_lane",
            "official_provider",
            "access_date_status",
            "source_product_version",
            "normalized_schema_version",
        ):
            _require_text(str(getattr(self, field_name)), field_name)
        if not _PACKAGE_ID_RE.fullmatch(self.package_id):
            raise OfficialPackageError(f"invalid_package_id:{self.package_id}")
        if not self.official_source_url and not self.historical_source_locator:
            raise OfficialPackageError("missing_metadata:source_locator")
        if self.official_source_url and not self.official_source_url.startswith(
            ("https://", "http://")
        ):
            raise OfficialPackageError("invalid_official_source_url")
        if self.access_date_status == "known" and not self.access_or_download_date:
            raise OfficialPackageError("missing_metadata:access_or_download_date")
        if self.access_date_status not in {
            "known",
            "unknown_historical_not_recorded",
            "mixed_per_file",
        }:
            raise OfficialPackageError("invalid_access_date_status")
        _tuple_text(self.geographic_coverage, "geographic_coverage")
        _tuple_text(self.temporal_coverage, "temporal_coverage")
        if not self.files:
            raise OfficialPackageError("missing_metadata:files")
        relative_paths = [file.relative_path for file in self.files]
        if len(set(relative_paths)) != len(relative_paths):
            raise OfficialPackageError("duplicate_package_file_path")
        if not self.raw_to_normalized_lineage:
            raise OfficialPackageError("missing_metadata:raw_to_normalized_lineage")
        if self.source_record_count < 0:
            raise OfficialPackageError("invalid_source_record_count")
        if not self.coverage_dimensions:
            raise OfficialPackageError("missing_metadata:coverage_dimensions")
        names = [dimension.name for dimension in self.coverage_dimensions]
        if len(set(names)) != len(names):
            raise OfficialPackageError("duplicate_coverage_dimension")
        _tuple_text(self.known_missingness, "known_missingness", allow_empty=True)
        if self.classification is PackageClassification.PRODUCTION:
            if not self.immutable or not self.read_only or not self.production_eligible:
                raise OfficialPackageError("production_package_not_immutable_read_only")
            incomplete = [
                dimension.name
                for dimension in self.coverage_dimensions
                if not dimension.complete
            ]
            if incomplete:
                raise OfficialPackageError(
                    f"incomplete_coverage:{','.join(sorted(incomplete))}"
                )
        if self.classification is PackageClassification.PILOT_DRAFT:
            if self.production_eligible:
                raise OfficialPackageError("pilot_package_presented_as_production")
            if not self.package_id.startswith("pilot_draft_"):
                raise OfficialPackageError("pilot_package_id_missing_draft_prefix")
        if (
            self.package_id.startswith("pilot_draft_")
            and self.classification is PackageClassification.PRODUCTION
        ):
            raise OfficialPackageError("pilot_package_presented_as_production")
        _require_sha256(
            self.package_manifest_sha256,
            "package_manifest_sha256",
        )

    @property
    def coverage_complete(self) -> bool:
        return all(dimension.complete for dimension in self.coverage_dimensions)

    def to_dict(self, *, include_manifest_hash: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "package_manifest_schema_version": self.package_manifest_schema_version,
            "package_id": self.package_id,
            "package_type": self.package_type,
            "source_lane": self.source_lane,
            "official_provider": self.official_provider,
            "official_source_url": self.official_source_url,
            "historical_source_locator": self.historical_source_locator,
            "access_or_download_date": self.access_or_download_date,
            "access_date_status": self.access_date_status,
            "source_product_version": self.source_product_version,
            "geographic_coverage": list(self.geographic_coverage),
            "temporal_coverage": list(self.temporal_coverage),
            "files": [file.to_dict() for file in self.files],
            "normalized_schema_version": self.normalized_schema_version,
            "raw_to_normalized_lineage": [
                item.to_dict() for item in self.raw_to_normalized_lineage
            ],
            "source_record_count": self.source_record_count,
            "coverage_dimensions": [
                dimension.to_dict() for dimension in self.coverage_dimensions
            ],
            "known_missingness": list(self.known_missingness),
            "immutable": self.immutable,
            "read_only": self.read_only,
            "classification": self.classification.value,
            "production_eligible": self.production_eligible,
        }
        if include_manifest_hash:
            data["package_manifest_sha256"] = self.package_manifest_sha256
        return data

    def computed_manifest_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(include_manifest_hash=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def with_computed_manifest_hash(self) -> "OfficialSourcePackageManifest":
        return replace(
            self,
            package_manifest_sha256=self.computed_manifest_sha256(),
        )

    def validate_manifest_hash(self) -> None:
        actual = self.computed_manifest_sha256()
        if actual != self.package_manifest_sha256:
            raise OfficialPackageError(
                "package_manifest_hash_mismatch:"
                f"declared={self.package_manifest_sha256}:actual={actual}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OfficialSourcePackageManifest":
        allowed = {
            "package_manifest_schema_version",
            "package_id",
            "package_type",
            "source_lane",
            "official_provider",
            "official_source_url",
            "historical_source_locator",
            "access_or_download_date",
            "access_date_status",
            "source_product_version",
            "geographic_coverage",
            "temporal_coverage",
            "files",
            "normalized_schema_version",
            "raw_to_normalized_lineage",
            "source_record_count",
            "coverage_dimensions",
            "known_missingness",
            "immutable",
            "read_only",
            "classification",
            "production_eligible",
            "package_manifest_sha256",
        }
        _reject_unknown(data, allowed, "manifest")
        try:
            manifest = cls(
                package_manifest_schema_version=str(
                    data["package_manifest_schema_version"]
                ),
                package_id=str(data["package_id"]),
                package_type=str(data["package_type"]),
                source_lane=str(data["source_lane"]),
                official_provider=str(data["official_provider"]),
                official_source_url=str(data["official_source_url"]),
                historical_source_locator=str(data["historical_source_locator"]),
                access_or_download_date=str(data["access_or_download_date"]),
                access_date_status=str(data["access_date_status"]),
                source_product_version=str(data["source_product_version"]),
                geographic_coverage=tuple(
                    str(value) for value in data["geographic_coverage"]
                ),
                temporal_coverage=tuple(
                    str(value) for value in data["temporal_coverage"]
                ),
                files=tuple(
                    OfficialPackageFile.from_dict(value) for value in data["files"]
                ),
                normalized_schema_version=str(data["normalized_schema_version"]),
                raw_to_normalized_lineage=tuple(
                    RawToNormalizedLineage.from_dict(value)
                    for value in data["raw_to_normalized_lineage"]
                ),
                source_record_count=int(data["source_record_count"]),
                coverage_dimensions=tuple(
                    CoverageDimension.from_dict(value)
                    for value in data["coverage_dimensions"]
                ),
                known_missingness=tuple(
                    str(value) for value in data["known_missingness"]
                ),
                immutable=bool(data["immutable"]),
                read_only=bool(data["read_only"]),
                classification=PackageClassification(str(data["classification"])),
                production_eligible=bool(data["production_eligible"]),
                package_manifest_sha256=str(data["package_manifest_sha256"]),
            )
        except KeyError as exc:
            raise OfficialPackageError(
                f"missing_metadata:{exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, OfficialPackageError):
                raise
            raise OfficialPackageError(f"invalid_manifest_value:{exc}") from exc
        manifest.validate_manifest_hash()
        return manifest


@dataclass(frozen=True)
class LoadedOfficialPackage:
    root: Path
    manifest_path: Path
    manifest: OfficialSourcePackageManifest


def validate_unique_package_ids(
    manifests: Iterable[OfficialSourcePackageManifest],
) -> None:
    seen: set[str] = set()
    for manifest in manifests:
        if manifest.package_id in seen:
            raise OfficialPackageError(f"duplicate_package_id:{manifest.package_id}")
        seen.add(manifest.package_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_shape(path: Path) -> tuple[int, tuple[str, ...]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        count = sum(1 for _ in reader)
    return count, columns


def _read_jsonl_shape(path: Path) -> tuple[int, tuple[str, ...]]:
    keys: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OfficialPackageError(
                    f"jsonl_record_not_object:{path.name}:{count + 1}"
                )
            keys.update(str(key) for key in value)
            count += 1
    return count, tuple(sorted(keys))


def _is_read_only(path: Path) -> bool:
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return path.stat().st_mode & write_bits == 0


def load_official_package(
    manifest_path: Path,
    *,
    require_production: bool = True,
    enforce_filesystem_read_only: bool = True,
) -> LoadedOfficialPackage:
    """Load and verify a local immutable package without any fallback.

    The loader intentionally accepts only a local :class:`Path`.  It never
    interprets an official URL as a runtime location and never writes a cache.
    """

    manifest_path = Path(manifest_path).resolve()
    if _path_mentions_old_workspace(manifest_path):
        raise OfficialPackageError(
            f"old_workspace_runtime_reference:{manifest_path}"
        )
    if not manifest_path.is_file():
        raise OfficialPackageError(f"package_manifest_missing:{manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialPackageError(f"package_manifest_unreadable:{exc}") from exc
    if not isinstance(raw, dict):
        raise OfficialPackageError("package_manifest_not_object")
    manifest = OfficialSourcePackageManifest.from_dict(raw)
    if require_production and manifest.classification is not PackageClassification.PRODUCTION:
        raise OfficialPackageError(
            f"package_not_production:{manifest.classification.value}"
        )
    root = manifest_path.parent.resolve()
    if _path_mentions_old_workspace(root):
        raise OfficialPackageError(f"old_workspace_runtime_reference:{root}")
    if enforce_filesystem_read_only and not _is_read_only(root):
        raise OfficialPackageError(f"mutable_source_directory:{root}")

    for item in manifest.files:
        path = (root / item.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise OfficialPackageError(
                f"runtime_path_outside_package:{item.relative_path}"
            ) from exc
        if _path_mentions_old_workspace(path):
            raise OfficialPackageError(f"old_workspace_runtime_reference:{path}")
        if not path.is_file():
            raise OfficialPackageError(f"package_file_missing:{item.relative_path}")
        if path.stat().st_size != item.byte_size:
            raise OfficialPackageError(
                f"unexpected_file_mutation:size:{item.relative_path}"
            )
        actual_hash = _sha256_file(path)
        if actual_hash != item.sha256:
            raise OfficialPackageError(
                f"corrupt_file_hash:{item.relative_path}:"
                f"declared={item.sha256}:actual={actual_hash}"
            )
        if enforce_filesystem_read_only and not _is_read_only(path):
            raise OfficialPackageError(f"mutable_source_file:{item.relative_path}")

        if item.count_kind is PackageFileCountKind.CSV_ROWS:
            actual_count, columns = _read_csv_shape(path)
        elif item.count_kind is PackageFileCountKind.JSONL_RECORDS:
            actual_count, columns = _read_jsonl_shape(path)
        else:
            actual_count, columns = item.row_or_feature_count, item.raw_schema_columns
        if actual_count != item.row_or_feature_count:
            raise OfficialPackageError(
                f"inconsistent_row_count:{item.relative_path}:"
                f"declared={item.row_or_feature_count}:actual={actual_count}"
            )
        if tuple(columns) != item.raw_schema_columns:
            raise OfficialPackageError(
                f"raw_schema_inventory_mismatch:{item.relative_path}"
            )

    return LoadedOfficialPackage(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
    )


__all__ = [
    "CoverageDimension",
    "LoadedOfficialPackage",
    "OfficialPackageError",
    "OfficialPackageFile",
    "OfficialSourcePackageManifest",
    "PACKAGE_MANIFEST_SCHEMA_VERSION",
    "PackageClassification",
    "PackageFileCountKind",
    "RawToNormalizedLineage",
    "load_official_package",
    "validate_unique_package_ids",
]
