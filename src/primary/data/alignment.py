"""Strict, explicit alignment rules for SEED-VIG session modalities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np


class AlignmentError(ValueError):
    """Raised when modality identity, axes, or time alignment is ambiguous."""


@dataclass(frozen=True)
class AxisResolution:
    """Auditable record of an array's conversion to time-channel-frequency."""

    original_shape: tuple[int, ...]
    canonical_shape: tuple[int, ...]
    axis_permutation: tuple[int, int, int]
    axis_resolution_source: str


def canonicalize_eeg(
    array: np.ndarray,
    *,
    sample_count: int,
    frequency_count: int,
    channel_count: int = 17,
) -> tuple[np.ndarray, AxisResolution]:
    """Convert an explicitly resolvable 3-D EEG array to ``T x C x F``.

    Resolution is based on the declared SEED-VIG dimensions, never on the
    largest dimension or total element count.
    """

    source = np.asarray(array)
    if source.ndim != 3:
        raise AlignmentError(f"EEG array must be 3-D; got shape={source.shape}")
    target = (int(sample_count), int(channel_count), int(frequency_count))
    matches = [
        tuple(int(index) for index in order)
        for order in permutations(range(3))
        if tuple(int(source.shape[index]) for index in order) == target
    ]
    if len(matches) != 1:
        raise AlignmentError(
            "EEG axes are not uniquely resolvable from declared dimensions: "
            f"shape={source.shape}, expected logical dimensions={target}, matches={matches}"
        )
    matched = matches[0]
    order: tuple[int, int, int] = (matched[0], matched[1], matched[2])
    canonical = np.array(np.transpose(source, axes=order), dtype=float, copy=True)
    canonical.setflags(write=False)
    return canonical, AxisResolution(
        original_shape=tuple(int(value) for value in source.shape),
        canonical_shape=tuple(int(value) for value in canonical.shape),
        axis_permutation=order,
        axis_resolution_source="declared_seed_vig_dimensions",
    )


def canonicalize_perclos(array: np.ndarray, *, sample_count: int) -> np.ndarray:
    """Return a copied one-dimensional PERCLOS sequence without vague squeeze."""

    source = np.asarray(array)
    valid_shapes = {(sample_count,), (sample_count, 1), (1, sample_count)}
    if tuple(int(value) for value in source.shape) not in valid_shapes:
        raise AlignmentError(
            f"PERCLOS shape must be one of {sorted(valid_shapes)}; got {source.shape}"
        )
    canonical = np.array(source.reshape(sample_count), dtype=float, copy=True)
    canonical.setflags(write=False)
    return canonical


def handle_nonfinite(
    array: np.ndarray,
    *,
    strategy: str,
    modality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an explicit non-finite policy and return an immutable copy."""

    copied = np.array(array, dtype=float, copy=True)
    mask = ~np.isfinite(copied)
    count = int(np.count_nonzero(mask))
    repair: dict[str, Any] = {
        "modality": modality,
        "nonfinite_count": count,
        "strategy": strategy,
    }
    if count:
        if strategy == "error":
            raise AlignmentError(f"{modality} contains {count} non-finite values")
        if strategy != "zero":
            raise AlignmentError(
                f"Unsupported non-finite strategy {strategy!r}; use 'error' or diagnostic 'zero'"
            )
        copied[mask] = 0.0
        repair["diagnostic_repair"] = True
    copied.setflags(write=False)
    return copied, repair


def validate_session_alignment(
    *,
    subject_id: int,
    session_id: int,
    features_5bands: np.ndarray | None,
    features_2hz: np.ndarray | None,
    perclos: np.ndarray,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate identity-independent shape and time alignment for one session."""

    if subject_id < 1 or session_id < 0:
        raise AlignmentError("Subject and Session identifiers must be non-negative")
    lengths = {"perclos": int(perclos.shape[0])}
    if perclos.ndim != 1:
        raise AlignmentError(f"PERCLOS must be 1-D; got {perclos.shape}")
    if features_5bands is not None:
        if features_5bands.ndim != 3 or features_5bands.shape[1:] != (17, 5):
            raise AlignmentError(
                f"5Bands must have canonical shape T x 17 x 5; got {features_5bands.shape}"
            )
        lengths["5bands"] = int(features_5bands.shape[0])
    if features_2hz is not None:
        if features_2hz.ndim != 3 or features_2hz.shape[1:] != (17, 25):
            raise AlignmentError(
                f"2Hz must have canonical shape T x 17 x 25; got {features_2hz.shape}"
            )
        lengths["2hz"] = int(features_2hz.shape[0])
    if len(set(lengths.values())) != 1:
        message = f"Session {session_id} modality lengths disagree: {lengths}"
        if strict:
            raise AlignmentError(message)
        return {"status": "WARNING", "message": message, "lengths": lengths}
    if not np.all(np.isfinite(perclos)):
        raise AlignmentError(f"Session {session_id} PERCLOS is not finite")
    return {"status": "PASS", "lengths": lengths, "sample_count": next(iter(lengths.values()))}


def align_session_modalities(**kwargs: Any) -> dict[str, Any]:
    """Named alignment entry point used by loaders and diagnostic tooling."""

    return validate_session_alignment(**kwargs)


def validate_time_index(time_index: np.ndarray, *, sample_count: int) -> None:
    """Reject reversed, duplicated, missing, or cross-Session-style time axes."""

    index = np.asarray(time_index)
    if index.ndim != 1 or len(index) != sample_count:
        raise AlignmentError("Time index length does not match the declared Session")
    expected = np.arange(sample_count)
    if not np.array_equal(index, expected):
        raise AlignmentError("Time index must be unique, contiguous, increasing, and Session-local")
