"""Deterministic, auditable manifest construction for SEED-VIG data files."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import pickle
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy.io

LOGGER = logging.getLogger(__name__)


SUBJECT_SESSIONS: dict[int, list[int]] = {
    1: [0],
    2: [1],
    3: [2],
    4: [3, 4],
    5: [5, 6],
    6: [7],
    7: [8],
    8: [9],
    9: [10],
    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: [15],
    15: [16],
    16: [17],
    17: [18],
    18: [19],
    19: [20],
    20: [21],
    21: [22],
}

SUPPORTED_SUFFIXES: tuple[str, ...] = (
    ".mat",
    ".npy",
    ".npz",
    ".csv",
    ".pkl",
    ".pickle",
)

MANIFEST_FIELDS: tuple[str, ...] = (
    "subject_id",
    "session_id",
    "physical_index",
    "file_path",
    "relative_path",
    "file_name",
    "file_suffix",
    "condition",
    "data_role",
    "sample_count",
    "feature_shape",
    "feature_dtype",
    "perclos_shape",
    "perclos_min",
    "perclos_max",
    "perclos_mean",
    "perclos_std",
    "missing_rate",
    "feature_missing_rate",
    "perclos_missing_rate",
    "file_size_bytes",
    "modified_time",
    "sha256",
    "loader_name",
    "scan_status",
    "error_message",
    "session_resolution_source",
    "subject_resolution_source",
    "feature_key",
    "perclos_key",
    "candidate_arrays",
    "selection_reason",
)

_FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "results",
    "cache",
    "caches",
}
_TEMP_FILE_PREFIXES = ("~$", ".~", ".#")
_FEATURE_KEY_PRIORITY = (
    "de_lds",
    "de_movingave",
    "psd_lds",
    "psd_movingave",
    "features",
    "feature",
    "eeg",
    "x",
)
_LABEL_NAMES = {"perclos", "perclos_label", "perclos_labels"}


def validate_subject_session_mapping(mapping: Mapping[int, Sequence[int]]) -> None:
    """Validate the authoritative subject-session mapping or raise ``ValueError``."""

    errors: list[str] = []
    subject_ids = set(mapping)
    if subject_ids != set(range(1, 22)):
        errors.append(
            f"Subject IDs must be exactly 1..21; got {sorted(subject_ids)}"
        )
    sessions = [int(session) for values in mapping.values() for session in values]
    if len(sessions) != len(set(sessions)):
        duplicates = sorted({value for value in sessions if sessions.count(value) > 1})
        errors.append(f"Duplicate Session IDs are forbidden: {duplicates}")
    if set(sessions) != set(range(23)):
        missing = sorted(set(range(23)) - set(sessions))
        extra = sorted(set(sessions) - set(range(23)))
        errors.append(f"Session IDs must be exactly 0..22; missing={missing}, extra={extra}")
    if list(mapping.get(4, ())) != [3, 4]:
        errors.append("Subject 4 must own Sessions [3, 4]")
    if list(mapping.get(5, ())) != [5, 6]:
        errors.append("Subject 5 must own Sessions [5, 6]")
    if len(subject_ids) != 21 or len(sessions) != 23:
        errors.append(
            f"Expected 21 subjects and 23 sessions; got {len(subject_ids)} and {len(sessions)}"
        )
    if errors:
        raise ValueError("Invalid SUBJECT_SESSIONS mapping: " + "; ".join(errors))


def _build_reverse_mapping(mapping: Mapping[int, Sequence[int]]) -> dict[int, int]:
    validate_subject_session_mapping(mapping)
    return {
        int(session_id): int(subject_id)
        for subject_id, session_ids in mapping.items()
        for session_id in session_ids
    }


SESSION_TO_SUBJECT: dict[int, int] = _build_reverse_mapping(SUBJECT_SESSIONS)


@dataclass
class FileInspection:
    """Normalized inspection result returned by every registered data loader."""

    feature_shape: tuple[int, ...] | None = None
    feature_dtype: str | None = None
    sample_count: int | None = None
    perclos_shape: tuple[int, ...] | None = None
    perclos_min: float | None = None
    perclos_max: float | None = None
    perclos_mean: float | None = None
    perclos_std: float | None = None
    missing_rate: float | None = None
    feature_missing_rate: float | None = None
    perclos_missing_rate: float | None = None
    loader_name: str = "unknown"
    data_role: str = "unknown"
    feature_key: str | None = None
    perclos_key: str | None = None
    candidate_arrays: list[dict[str, Any]] = field(default_factory=list)
    selection_reason: str = ""
    internal_session_id: int | None = None
    internal_subject_id: int | None = None


@dataclass
class ManifestValidationReport:
    """Structured result of validating a manifest."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


Loader = Callable[[Path, bool], FileInspection]


class LoaderRegistry:
    """Per-scan registry mapping file suffixes to inspection callables."""

    def __init__(self) -> None:
        self._loaders: dict[str, Loader] = {}

    def register(self, suffix: str, loader: Loader) -> None:
        """Register or replace a loader for ``suffix``."""

        normalized = suffix.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        self._loaders[normalized] = loader

    def get(self, suffix: str) -> Loader:
        """Return the loader for ``suffix`` or raise a diagnostic error."""

        normalized = suffix.lower()
        if normalized not in self._loaders:
            raise KeyError(
                f"No data loader registered for suffix {normalized!r}; "
                f"available={sorted(self._loaders)}"
            )
        return self._loaders[normalized]

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Return registered suffixes in stable order."""

        return tuple(sorted(self._loaders))

    @classmethod
    def default(cls) -> "LoaderRegistry":
        """Create a registry containing all built-in loaders."""

        registry = cls()
        registry.register(".mat", _inspect_mat)
        registry.register(".npy", _inspect_npy)
        registry.register(".npz", _inspect_npz)
        registry.register(".csv", _inspect_csv)
        registry.register(".pkl", _inspect_pickle)
        registry.register(".pickle", _inspect_pickle)
        return registry


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file SHA256 digest without loading the whole file into memory."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_nonfinite_rate(array: np.ndarray) -> float:
    if array.size == 0:
        return 1.0
    if np.issubdtype(array.dtype, np.number):
        try:
            return float(np.count_nonzero(~np.isfinite(array)) / array.size)
        except TypeError:
            return 0.0
    return 0.0


def _safe_float_stat(array: np.ndarray, operation: str) -> float | None:
    numeric = np.asarray(array, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None
    if operation == "min":
        return float(np.min(finite))
    if operation == "max":
        return float(np.max(finite))
    if operation == "mean":
        return float(np.mean(finite))
    if operation == "std":
        return float(np.std(finite))
    raise ValueError(f"Unsupported statistic operation: {operation}")


def _as_int_scalar(value: Any) -> int | None:
    try:
        array = np.asarray(value).squeeze()
        if array.size == 1:
            return int(array.item())
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def _extract_internal_id(mapping: Mapping[str, Any], names: set[str]) -> int | None:
    for key, value in mapping.items():
        if key.lower() in names:
            parsed = _as_int_scalar(value)
            if parsed is not None:
                return parsed
    return None


def _candidate_priority(name: str) -> tuple[int, str]:
    lowered = name.lower()
    try:
        return (_FEATURE_KEY_PRIORITY.index(lowered), lowered)
    except ValueError:
        return (len(_FEATURE_KEY_PRIORITY), lowered)


def _is_label_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in _LABEL_NAMES or "perclos" in lowered


def _infer_sample_count(shape: tuple[int, ...] | None) -> int | None:
    if not shape or any(int(size) <= 0 for size in shape):
        return None
    if len(shape) == 1:
        return int(shape[0])
    return int(max(shape))


def _inspect_array_metadata(
    entries: Sequence[tuple[str, tuple[int, ...], str]], loader_name: str
) -> FileInspection:
    """Inspect names/shapes/dtypes without allocating arrays or reading payloads."""

    candidates = [
        {
            "name": name,
            "shape": [int(item) for item in shape],
            "dtype": dtype,
            "role": "perclos" if _is_label_name(name) else "feature_candidate",
        }
        for name, shape, dtype in entries
        if shape and int(np.prod(shape, dtype=np.int64)) > 1
    ]
    label_entries = sorted(
        (entry for entry in entries if _is_label_name(entry[0])),
        key=lambda entry: entry[0].lower(),
    )
    feature_entries = sorted(
        (
            entry
            for entry in entries
            if not _is_label_name(entry[0])
            and len(entry[1]) >= 2
            and int(np.prod(entry[1], dtype=np.int64)) > 1
        ),
        key=lambda entry: _candidate_priority(entry[0]),
    )
    feature_entry = feature_entries[0] if feature_entries else None
    label_entry = label_entries[0] if label_entries else None
    feature_shape = feature_entry[1] if feature_entry else None
    perclos_shape = label_entry[1] if label_entry else None
    reasons: list[str] = ["metadata-only scan; no array payload allocated"]
    if feature_entry:
        reasons.append(f"feature selected by configured priority: {feature_entry[0]}")
    if len(feature_entries) > 1:
        reasons.append(
            f"additional feature candidates recorded: {[entry[0] for entry in feature_entries[1:]]}"
        )
    if label_entry:
        reasons.append(f"PERCLOS selected by explicit key match: {label_entry[0]}")
    role = (
        "feature+label"
        if feature_entry and label_entry
        else "feature"
        if feature_entry
        else "perclos"
        if label_entry
        else "unknown"
    )
    return FileInspection(
        feature_shape=feature_shape,
        feature_dtype=feature_entry[2] if feature_entry else None,
        sample_count=_infer_sample_count(perclos_shape)
        or _infer_sample_count(feature_shape),
        perclos_shape=perclos_shape,
        loader_name=loader_name,
        data_role=role,
        feature_key=feature_entry[0] if feature_entry else None,
        perclos_key=label_entry[0] if label_entry else None,
        candidate_arrays=candidates,
        selection_reason="; ".join(reasons),
    )


def _inspect_arrays(
    arrays: Mapping[str, Any], loader_name: str, metadata_only: bool = False
) -> FileInspection:
    numeric_arrays: dict[str, np.ndarray] = {}
    candidates: list[dict[str, Any]] = []
    for name, value in arrays.items():
        if name.startswith("__"):
            continue
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if not np.issubdtype(array.dtype, np.number) or array.ndim == 0:
            continue
        numeric_arrays[name] = array
        candidates.append(
            {
                "name": name,
                "shape": [int(item) for item in array.shape],
                "dtype": str(array.dtype),
                "role": "perclos" if _is_label_name(name) else "feature_candidate",
            }
        )

    label_names = sorted(
        (name for name in numeric_arrays if _is_label_name(name)), key=str.lower
    )
    feature_names = sorted(
        (
            name
            for name, array in numeric_arrays.items()
            if name not in label_names and array.ndim >= 2 and array.size > 1
        ),
        key=_candidate_priority,
    )
    feature_key = feature_names[0] if feature_names else None
    perclos_key = label_names[0] if label_names else None
    feature = numeric_arrays.get(feature_key) if feature_key else None
    perclos = numeric_arrays.get(perclos_key) if perclos_key else None

    reasons: list[str] = []
    if feature_key:
        reasons.append(f"feature selected by configured priority: {feature_key}")
    if len(feature_names) > 1:
        reasons.append(f"additional feature candidates recorded: {feature_names[1:]}")
    if perclos_key:
        reasons.append(f"PERCLOS selected by explicit key match: {perclos_key}")
    if len(label_names) > 1:
        reasons.append(f"multiple label candidates recorded: {label_names}")

    feature_shape = tuple(int(item) for item in feature.shape) if feature is not None else None
    perclos_shape = tuple(int(item) for item in perclos.shape) if perclos is not None else None
    sample_count = _infer_sample_count(perclos_shape) or _infer_sample_count(feature_shape)

    feature_missing = None if feature is None or metadata_only else _safe_nonfinite_rate(feature)
    perclos_missing = None if perclos is None or metadata_only else _safe_nonfinite_rate(perclos)
    weighted_rates: list[tuple[int, float]] = []
    if feature is not None and feature_missing is not None:
        weighted_rates.append((int(feature.size), feature_missing))
    if perclos is not None and perclos_missing is not None:
        weighted_rates.append((int(perclos.size), perclos_missing))
    missing_rate = (
        sum(size * rate for size, rate in weighted_rates)
        / sum(size for size, _ in weighted_rates)
        if weighted_rates
        else None
    )
    role = "feature+label" if feature is not None and perclos is not None else (
        "feature" if feature is not None else "perclos" if perclos is not None else "unknown"
    )
    return FileInspection(
        feature_shape=feature_shape,
        feature_dtype=str(feature.dtype) if feature is not None else None,
        sample_count=sample_count,
        perclos_shape=perclos_shape,
        perclos_min=None if perclos is None or metadata_only else _safe_float_stat(perclos, "min"),
        perclos_max=None if perclos is None or metadata_only else _safe_float_stat(perclos, "max"),
        perclos_mean=None if perclos is None or metadata_only else _safe_float_stat(perclos, "mean"),
        perclos_std=None if perclos is None or metadata_only else _safe_float_stat(perclos, "std"),
        missing_rate=missing_rate,
        feature_missing_rate=feature_missing,
        perclos_missing_rate=perclos_missing,
        loader_name=loader_name,
        data_role=role,
        feature_key=feature_key,
        perclos_key=perclos_key,
        candidate_arrays=candidates,
        selection_reason="; ".join(reasons),
        internal_session_id=_extract_internal_id(
            arrays, {"session", "session_id", "sessionindex", "session_index"}
        ),
        internal_subject_id=_extract_internal_id(
            arrays, {"subject", "subject_id", "subjectindex", "subject_index"}
        ),
    )


def _inspect_mat(path: Path, metadata_only: bool) -> FileInspection:
    try:
        if metadata_only:
            entries = [
                (name, tuple(int(item) for item in shape), matlab_type)
                for name, shape, matlab_type in scipy.io.whosmat(path)
                if matlab_type not in {"char", "cell", "struct"}
            ]
            return _inspect_array_metadata(entries, "scipy.io.whosmat")
        mapping = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        return _inspect_arrays(mapping, "scipy.io.loadmat", False)
    except (NotImplementedError, ValueError, OSError) as scipy_error:
        try:
            with h5py.File(path, "r") as handle:
                datasets = [
                    (name, dataset)
                    for name, dataset in handle.items()
                    if isinstance(dataset, h5py.Dataset)
                ]
                if metadata_only:
                    entries = [
                        (
                            name,
                            tuple(int(item) for item in dataset.shape),
                            str(dataset.dtype),
                        )
                        for name, dataset in datasets
                    ]
                    inspection = _inspect_array_metadata(entries, "h5py[metadata]")
                else:
                    arrays = {name: np.asarray(dataset) for name, dataset in datasets}
                    inspection = _inspect_arrays(arrays, "h5py", False)
            inspection.selection_reason += (
                f"; scipy fallback reason={type(scipy_error).__name__}"
            )
            return inspection
        except (OSError, ValueError, TypeError) as h5_error:
            raise ValueError(
                "MAT inspection failed with both scipy.io.loadmat "
                f"({type(scipy_error).__name__}: {scipy_error}) and h5py "
                f"({type(h5_error).__name__}: {h5_error})"
            ) from h5_error


def _inspect_npy(path: Path, metadata_only: bool) -> FileInspection:
    array = np.load(path, mmap_mode="r" if metadata_only else None, allow_pickle=False)
    return _inspect_arrays({"features": array}, "numpy.load[npy]", metadata_only)


def _inspect_npz(path: Path, metadata_only: bool) -> FileInspection:
    with np.load(path, mmap_mode="r" if metadata_only else None, allow_pickle=False) as data:
        arrays = {name: data[name] for name in sorted(data.files)}
    return _inspect_arrays(arrays, "numpy.load[npz]", metadata_only)


def _inspect_csv(path: Path, metadata_only: bool) -> FileInspection:
    frame = pd.read_csv(path)
    arrays: dict[str, np.ndarray] = {}
    label_columns = [column for column in frame.columns if _is_label_name(str(column))]
    feature_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in label_columns
    ]
    if feature_columns:
        arrays["features"] = frame[feature_columns].to_numpy()
    if label_columns:
        arrays["perclos"] = frame[label_columns[0]].to_numpy()
    return _inspect_arrays(arrays, "pandas.read_csv", metadata_only)


def _inspect_pickle(path: Path, metadata_only: bool) -> FileInspection:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, pd.DataFrame):
        arrays = {str(column): value[column].to_numpy() for column in value.columns}
    elif isinstance(value, Mapping):
        arrays = {str(key): item for key, item in value.items()}
    else:
        arrays = {"features": np.asarray(value)}
    return _inspect_arrays(arrays, "pickle.load", metadata_only)


def inspect_data_file(
    path: str | Path,
    *,
    metadata_only: bool = False,
    registry: LoaderRegistry | None = None,
) -> FileInspection:
    """Inspect one supported file through a loader registry."""

    file_path = Path(path)
    active_registry = registry or LoaderRegistry.default()
    return active_registry.get(file_path.suffix)(file_path, metadata_only)


def _path_is_excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in _FORBIDDEN_DIR_NAMES for part in relative.parts[:-1]):
        return True
    name = path.name
    return name.startswith(_TEMP_FILE_PREFIXES) or name.startswith(".")


def _resolve_session(
    path: Path,
    root: Path,
    inspection: FileInspection,
    physical_fallback: Mapping[Path, int] | None,
) -> tuple[int | None, str]:
    if inspection.internal_session_id in SESSION_TO_SUBJECT:
        return inspection.internal_session_id, "internal_metadata"
    relative = path.relative_to(root)
    for part in relative.parts[:-1]:
        match = re.fullmatch(r"(?i)(?:session|sess|s)[_-]?(\d{1,2})", part)
        if match and int(match.group(1)) in SESSION_TO_SUBJECT:
            return int(match.group(1)), "directory_name"
    if path.stem.isdigit() and int(path.stem) in SESSION_TO_SUBJECT:
        return int(path.stem), "file_name_exact_integer"
    match = re.search(r"(?i)(?:session|sess|s)[_-]?(\d{1,2})", path.stem)
    if match and int(match.group(1)) in SESSION_TO_SUBJECT:
        return int(match.group(1)), "file_name_pattern"
    if physical_fallback and path in physical_fallback:
        return int(physical_fallback[path]), "verified_physical_order"
    return None, "unresolved"


def _physical_order_fallback(files: Sequence[Path], root: Path) -> dict[Path, int]:
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in files:
        relative = path.relative_to(root)
        condition = relative.parts[0] if len(relative.parts) > 1 else "root"
        grouped.setdefault((Path(condition), path.suffix.lower()), []).append(path)
    fallback: dict[Path, int] = {}
    for group_files in grouped.values():
        unresolved = [path for path in group_files if not path.stem.isdigit()]
        if len(group_files) == 23 and len(unresolved) == 23:
            for index, path in enumerate(sorted(group_files, key=lambda item: item.as_posix())):
                fallback[path] = index
    return fallback


def _inspection_fields(inspection: FileInspection) -> dict[str, Any]:
    return {
        "data_role": inspection.data_role,
        "sample_count": inspection.sample_count,
        "feature_shape": json.dumps(inspection.feature_shape) if inspection.feature_shape else None,
        "feature_dtype": inspection.feature_dtype,
        "perclos_shape": json.dumps(inspection.perclos_shape) if inspection.perclos_shape else None,
        "perclos_min": inspection.perclos_min,
        "perclos_max": inspection.perclos_max,
        "perclos_mean": inspection.perclos_mean,
        "perclos_std": inspection.perclos_std,
        "missing_rate": inspection.missing_rate,
        "feature_missing_rate": inspection.feature_missing_rate,
        "perclos_missing_rate": inspection.perclos_missing_rate,
        "loader_name": inspection.loader_name,
        "feature_key": inspection.feature_key,
        "perclos_key": inspection.perclos_key,
        "candidate_arrays": json.dumps(inspection.candidate_arrays, sort_keys=True),
        "selection_reason": inspection.selection_reason,
    }


def build_manifest(
    data_root: str | Path,
    *,
    recursive: bool = True,
    compute_sha256: bool = True,
    metadata_only: bool = False,
    dry_run: bool = False,
    supported_suffixes: Iterable[str] | None = None,
    registry: LoaderRegistry | None = None,
    allow_physical_order_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Recursively scan ``data_root`` and return deterministic manifest rows."""

    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"SEED-VIG data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"SEED-VIG data root is not a directory: {root}")
    active_registry = registry or LoaderRegistry.default()
    suffixes = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in (supported_suffixes or active_registry.suffixes)
    }
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file()
            and path.suffix.lower() in suffixes
            and not _path_is_excluded(path, root)
        ),
        key=lambda item: item.relative_to(root).as_posix().lower(),
    )
    physical_fallback = (
        _physical_order_fallback(files, root) if allow_physical_order_fallback else {}
    )
    rows: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        relative = path.relative_to(root)
        condition = relative.parts[0] if len(relative.parts) > 1 else "root"
        base: dict[str, Any] = {
            field_name: None for field_name in MANIFEST_FIELDS
        }
        base.update(
            {
                "file_path": str(path),
                "relative_path": relative.as_posix(),
                "file_name": path.name,
                "file_suffix": path.suffix.lower(),
                "condition": condition,
                "file_size_bytes": int(stat.st_size),
                "modified_time": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": compute_file_sha256(path) if compute_sha256 and not dry_run else None,
                "scan_status": "dry_run" if dry_run else "ok",
                "error_message": "",
            }
        )
        try:
            inspection = (
                FileInspection(loader_name="dry_run", data_role="uninspected")
                if dry_run
                else inspect_data_file(
                    path, metadata_only=metadata_only, registry=active_registry
                )
            )
            session_id, session_source = _resolve_session(
                path, root, inspection, physical_fallback
            )
            if session_id is None:
                raise ValueError(
                    "Session ID could not be resolved from internal metadata, directory, "
                    f"or file name for {relative.as_posix()}"
                )
            subject_id = SESSION_TO_SUBJECT[session_id]
            if (
                inspection.internal_subject_id is not None
                and inspection.internal_subject_id != subject_id
            ):
                raise ValueError(
                    "Internal subject metadata conflicts with authoritative mapping: "
                    f"internal={inspection.internal_subject_id}, session={session_id}, "
                    f"expected_subject={subject_id}"
                )
            base.update(_inspection_fields(inspection))
            base.update(
                {
                    "session_id": session_id,
                    "physical_index": session_id,
                    "subject_id": subject_id,
                    "session_resolution_source": session_source,
                    "subject_resolution_source": "authoritative_session_mapping",
                }
            )
        except Exception as error:  # one corrupt file must not abort the scan
            LOGGER.warning("Failed to inspect %s: %s", path, error)
            base["scan_status"] = "error"
            base["error_message"] = f"{type(error).__name__}: {error}"
        rows.append(base)
    return rows


scan_dataset = build_manifest


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_manifest(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save CSV, JSONL, summary, and abnormal-file artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": destination / "manifest.csv",
        "jsonl": destination / "manifest.jsonl",
        "summary": destination / "manifest_summary.json",
        "abnormal": destination / "abnormal_files.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Manifest outputs already exist; pass overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    normalized_rows = [
        {field_name: _json_value(row.get(field_name)) for field_name in MANIFEST_FIELDS}
        for row in rows
    ]
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)
    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        for row in normalized_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    abnormal = [row for row in normalized_rows if row.get("scan_status") == "error"]
    with paths["abnormal"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(abnormal)
    summary = _manifest_summary(normalized_rows)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def _manifest_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("scan_status") in {"ok", "dry_run"}]
    subject_values = {_to_int(row.get("subject_id")) for row in successful}
    session_values = {_to_int(row.get("session_id")) for row in successful}
    parsed_subjects = sorted(value for value in subject_values if value is not None)
    parsed_sessions = sorted(value for value in session_values if value is not None)
    return {
        "total_files": len(rows),
        "successful_files": len(successful),
        "error_files": sum(row.get("scan_status") == "error" for row in rows),
        "unique_subjects": len(parsed_subjects),
        "unique_sessions": len(parsed_sessions),
        "subject_ids": parsed_subjects,
        "session_ids": parsed_sessions,
        "suffix_distribution": _count_values(rows, "file_suffix"),
        "condition_distribution": _count_values(rows, "condition"),
        "role_distribution": _count_values(rows, "data_role"),
        "feature_shape_distribution": _count_values(rows, "feature_shape", omit_empty=True),
        "scan_status_distribution": _count_values(rows, "scan_status"),
    }


def _count_values(
    rows: Sequence[Mapping[str, Any]], key: str, *, omit_empty: bool = False
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if omit_empty and (value is None or value == "" or _is_nan_scalar(value)):
            continue
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a saved CSV or JSONL manifest."""

    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if manifest_path.suffix.lower() == ".csv":
        frame = pd.read_csv(manifest_path)
        return [
            {
                str(key): (None if _is_nan_scalar(value) else value)
                for key, value in row.items()
            }
            for row in frame.to_dict(orient="records")
        ]
    raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


def _is_nan_scalar(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _to_int(value: Any) -> int | None:
    if value is None or value == "" or _is_nan_scalar(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "" or _is_nan_scalar(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> ManifestValidationReport:
    """Validate coverage, mapping integrity, file quality, and cross-role alignment."""

    errors: list[str] = []
    warnings: list[str] = []
    if not rows:
        return ManifestValidationReport(False, ["Manifest is empty"], [], {})
    missing_fields = sorted(set(MANIFEST_FIELDS) - set().union(*(set(row) for row in rows)))
    if missing_fields:
        errors.append(f"Manifest fields are missing: {missing_fields}")

    error_rows = [row for row in rows if str(row.get("scan_status")) == "error"]
    if error_rows:
        errors.append(f"Manifest contains {len(error_rows)} file inspection errors")

    good_rows = [row for row in rows if str(row.get("scan_status")) in {"ok", "dry_run"}]
    subjects = sorted(
        {value for row in good_rows if (value := _to_int(row.get("subject_id"))) is not None}
    )
    sessions = sorted(
        {value for row in good_rows if (value := _to_int(row.get("session_id"))) is not None}
    )
    if subjects != list(range(1, 22)):
        errors.append(f"Manifest subjects must be 1..21; got {subjects}")
    if sessions != list(range(23)):
        errors.append(f"Manifest sessions must be 0..22; got {sessions}")

    for row in good_rows:
        session_id = _to_int(row.get("session_id"))
        subject_id = _to_int(row.get("subject_id"))
        if session_id is None or subject_id is None:
            errors.append(f"Unresolved subject/session in {row.get('relative_path')}")
            continue
        expected_subject = SESSION_TO_SUBJECT.get(session_id)
        if expected_subject != subject_id:
            errors.append(
                f"Subject-session conflict in {row.get('relative_path')}: "
                f"session {session_id} maps to {expected_subject}, not {subject_id}"
            )
        size = _to_int(row.get("file_size_bytes"))
        if size is not None and size <= 0:
            errors.append(f"Completely empty file: {row.get('relative_path')}")
        sample_count = _to_int(row.get("sample_count"))
        if sample_count is not None and sample_count <= 0:
            errors.append(f"Non-positive sample_count in {row.get('relative_path')}")
        for field_name in (
            "missing_rate",
            "feature_missing_rate",
            "perclos_missing_rate",
        ):
            rate = _to_float(row.get(field_name))
            if rate is not None and not 0.0 <= rate <= 1.0:
                errors.append(
                    f"{field_name} outside [0, 1] in {row.get('relative_path')}: {rate}"
                )
        minimum = _to_float(row.get("perclos_min"))
        maximum = _to_float(row.get("perclos_max"))
        if minimum is not None and maximum is not None:
            if minimum > maximum:
                errors.append(f"PERCLOS min > max in {row.get('relative_path')}")
            if minimum < 0.0 or maximum > 1.0:
                warnings.append(
                    f"PERCLOS outside expected [0, 1] in {row.get('relative_path')}: "
                    f"[{minimum}, {maximum}]"
                )

    paths = [str(row.get("file_path")) for row in rows]
    duplicate_paths = sorted({path for path in paths if path and paths.count(path) > 1})
    if duplicate_paths:
        errors.append(f"Duplicate file paths: {duplicate_paths}")
    hashes = [str(row.get("sha256")) for row in good_rows if row.get("sha256")]
    duplicate_hashes = sorted({digest for digest in hashes if hashes.count(digest) > 1})
    if duplicate_hashes:
        warnings.append(f"Duplicate SHA256 values detected: {duplicate_hashes}")

    condition_sessions: dict[tuple[str, int], list[str]] = {}
    for row in good_rows:
        session_id = _to_int(row.get("session_id"))
        if session_id is None:
            continue
        key = (str(row.get("condition")), session_id)
        condition_sessions.setdefault(key, []).append(str(row.get("relative_path")))
    duplicated_condition_sessions = {
        f"{condition}:{session}": files
        for (condition, session), files in condition_sessions.items()
        if len(files) > 1
    }
    if duplicated_condition_sessions:
        errors.append(
            "Duplicate files for the same condition/session: "
            + json.dumps(duplicated_condition_sessions, ensure_ascii=False, sort_keys=True)
        )

    session_counts: dict[int, set[int]] = {}
    for row in good_rows:
        session_id = _to_int(row.get("session_id"))
        sample_count = _to_int(row.get("sample_count"))
        if session_id is not None and sample_count is not None:
            session_counts.setdefault(session_id, set()).add(sample_count)
    mismatched_counts = {
        session: sorted(counts) for session, counts in session_counts.items() if len(counts) > 1
    }
    if mismatched_counts:
        errors.append(f"Cross-role sample count mismatch: {mismatched_counts}")

    for subject_id, expected_sessions in ((4, [3, 4]), (5, [5, 6])):
        present_values = {
            _to_int(row.get("session_id"))
            for row in good_rows
            if _to_int(row.get("subject_id")) == subject_id
        }
        present = sorted(value for value in present_values if value is not None)
        if present != expected_sessions:
            errors.append(
                f"Subject {subject_id} dual-session coverage invalid: "
                f"expected={expected_sessions}, got={present}"
            )

    statistics = _manifest_summary(rows)
    statistics.update(
        {
            "duplicate_sha256_count": len(duplicate_hashes),
            "duplicate_condition_session_count": len(duplicated_condition_sessions),
            "cross_role_sample_mismatch_count": len(mismatched_counts),
            "subject_4_sessions": [3, 4] if 3 in sessions and 4 in sessions else [],
            "subject_5_sessions": [5, 6] if 5 in sessions and 6 in sessions else [],
        }
    )
    return ManifestValidationReport(not errors, errors, warnings, statistics)


def save_validation_report(
    report: ManifestValidationReport, path: str | Path
) -> Path:
    """Persist a manifest validation report as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination
