"""Strict SEED-VIG Session loader built on the Phase 1 manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import scipy.io

from .alignment import (
    AlignmentError,
    canonicalize_eeg,
    canonicalize_perclos,
    handle_nonfinite,
    validate_session_alignment,
)
from .manifest import SESSION_TO_SUBJECT


@dataclass(frozen=True)
class SessionRecord:
    """Canonical, immutable representation of one physical SEED-VIG session."""

    subject_id: int
    session_id: int
    condition: str | None
    features_5bands: np.ndarray | None
    features_2hz: np.ndarray | None
    perclos: np.ndarray
    sample_count: int
    channel_count: int
    timestamps: np.ndarray | None
    metadata: dict[str, Any]


def _load_mat_array(path: Path, key: str) -> tuple[np.ndarray, str]:
    try:
        payload = scipy.io.loadmat(path, variable_names=[key])
        if key not in payload:
            raise KeyError(f"MAT key {key!r} not found in {path}")
        return np.asarray(payload[key]), "scipy.io.loadmat"
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"HDF5 MAT key {key!r} not found in {path}")
            # MATLAB v7.3 stores dimensions in reversed HDF5 order.
            return np.asarray(handle[key]).transpose(), "h5py_matlab_v7.3_reversed_axes"


def _select_unique(rows: pd.DataFrame, token: str, session_id: int) -> pd.Series:
    mask = rows["relative_path"].astype(str).str.replace("\\", "/").str.lower().str.contains(token)
    selected = rows.loc[mask]
    if len(selected) != 1:
        raise AlignmentError(
            f"Session {session_id} requires exactly one {token} file; found {len(selected)}"
        )
    return selected.iloc[0]


def _parse_candidate_keys(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item.get("name")) for item in parsed if isinstance(item, dict) and item.get("name")]


def _resolve_key(row: pd.Series, *, label: bool) -> str:
    field = "perclos_key" if label else "feature_key"
    value = str(row.get(field, "")).strip()
    if value and value.lower() != "nan":
        return value
    candidates = _parse_candidate_keys(row.get("candidate_arrays", "[]"))
    matches = [key for key in candidates if ("perclos" in key.lower()) == label]
    if not matches:
        raise AlignmentError(f"No {'PERCLOS' if label else 'feature'} key recorded for {row['file_path']}")
    return matches[0]


def _load_manifest(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = manifest.copy(deep=True) if isinstance(manifest, pd.DataFrame) else pd.read_csv(Path(manifest))
    required = {"subject_id", "session_id", "file_path", "relative_path", "scan_status", "sample_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    status = frame["scan_status"].astype(str).str.lower()
    frame = frame.loc[status.isin({"ok", "success"})].copy()
    frame["subject_id"] = frame["subject_id"].astype(int)
    frame["session_id"] = frame["session_id"].astype(int)
    return frame.sort_values(["session_id", "relative_path"], kind="stable").reset_index(drop=True)


def load_session(
    manifest: str | Path | pd.DataFrame,
    *,
    session_id: int,
    data_root: str | Path | None = None,
    strict: bool = True,
    nonfinite_strategy: str = "error",
    alignment_mode: str = "strict",
) -> SessionRecord:
    """Load and strictly align one session without modifying source files or arrays."""

    if alignment_mode not in {"strict", "truncate"}:
        raise ValueError("alignment_mode must be 'strict' or diagnostic 'truncate'")
    frame = _load_manifest(manifest)
    rows = frame.loc[frame["session_id"].eq(int(session_id))]
    if len(rows) != 3:
        raise AlignmentError(f"Session {session_id} must have exactly 3 successful files; found {len(rows)}")
    expected_subject = SESSION_TO_SUBJECT.get(int(session_id))
    subjects = set(rows["subject_id"].astype(int).tolist())
    if expected_subject is None or subjects != {expected_subject}:
        raise AlignmentError(
            f"Session {session_id} subject mismatch: manifest={sorted(subjects)}, expected={expected_subject}"
        )

    row5 = _select_unique(rows, "eeg_feature_5bands", session_id)
    row2 = _select_unique(rows, "eeg_feature_2hz", session_id)
    rowp = _select_unique(rows, "perclos", session_id)
    root = Path(data_root) if data_root is not None else None

    def resolve(row: pd.Series) -> Path:
        recorded = Path(str(row["file_path"]))
        candidate = recorded if recorded.is_absolute() else (root / str(row["relative_path"]) if root else recorded)
        if not candidate.exists() and root is not None:
            candidate = root / str(row["relative_path"])
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    path5, path2, pathp = resolve(row5), resolve(row2), resolve(rowp)
    raw5, loader5 = _load_mat_array(path5, _resolve_key(row5, label=False))
    raw2, loader2 = _load_mat_array(path2, _resolve_key(row2, label=False))
    rawp, loaderp = _load_mat_array(pathp, _resolve_key(rowp, label=True))
    declared_counts = [int(row5["sample_count"]), int(row2["sample_count"]), int(rowp["sample_count"])]
    if len(set(declared_counts)) != 1:
        raise AlignmentError(f"Manifest sample counts disagree for Session {session_id}: {declared_counts}")
    declared = declared_counts[0]
    eeg5, axis5 = canonicalize_eeg(raw5, sample_count=declared, frequency_count=5)
    eeg2, axis2 = canonicalize_eeg(raw2, sample_count=declared, frequency_count=25)
    perclos = canonicalize_perclos(rawp, sample_count=declared)

    if alignment_mode == "truncate":
        # Canonicalization already requires declared axes; truncation remains a
        # diagnostic marker and is never enabled by experiment configs.
        strict = False
    eeg5, repair5 = handle_nonfinite(eeg5, strategy=nonfinite_strategy, modality="5bands")
    eeg2, repair2 = handle_nonfinite(eeg2, strategy=nonfinite_strategy, modality="2hz")
    perclos, repairp = handle_nonfinite(perclos, strategy=nonfinite_strategy, modality="perclos")
    alignment = validate_session_alignment(
        subject_id=expected_subject,
        session_id=session_id,
        features_5bands=eeg5,
        features_2hz=eeg2,
        perclos=perclos,
        strict=strict,
    )
    condition_values = [str(value) for value in rows.get("condition", pd.Series(dtype=str)).dropna() if str(value)]
    metadata: dict[str, Any] = {
        "alignment": alignment,
        "alignment_mode": alignment_mode,
        "files": {"5bands": str(path5), "2hz": str(path2), "perclos": str(pathp)},
        "loaders": {"5bands": loader5, "2hz": loader2, "perclos": loaderp},
        "axes": {
            "5bands": axis5.__dict__,
            "2hz": axis2.__dict__,
            "perclos": {
                "original_shape": tuple(int(value) for value in rawp.shape),
                "canonical_shape": tuple(int(value) for value in perclos.shape),
                "axis_resolution_source": "declared_sample_count_and_singleton_axis",
            },
        },
        "repairs": [repair5, repair2, repairp],
    }
    return SessionRecord(
        subject_id=expected_subject,
        session_id=int(session_id),
        condition=condition_values[0] if condition_values else None,
        features_5bands=eeg5,
        features_2hz=eeg2,
        perclos=perclos,
        sample_count=declared,
        channel_count=17,
        timestamps=None,
        metadata=metadata,
    )


def load_all_sessions(
    manifest: str | Path | pd.DataFrame,
    *,
    data_root: str | Path | None = None,
    session_ids: Iterable[int] | None = None,
    strict: bool = True,
    nonfinite_strategy: str = "error",
) -> list[SessionRecord]:
    """Load sessions in stable physical Session order."""

    frame = _load_manifest(manifest)
    selected = sorted(set(int(value) for value in (session_ids if session_ids is not None else frame["session_id"])))
    return [
        load_session(
            frame,
            session_id=session_id,
            data_root=data_root,
            strict=strict,
            nonfinite_strategy=nonfinite_strategy,
        )
        for session_id in selected
    ]
