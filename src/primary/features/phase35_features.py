"""Expanded strictly-causal EEG features for Phase 3.5 accuracy research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.seed_vig import SessionRecord
from src.models.handcrafted_features import extract_handcrafted_features

DEFAULT_BAND_NAMES: tuple[str, ...] = ("band_0", "band_1", "band_2", "band_3", "band_4")


@dataclass(frozen=True)
class Phase35FeatureMatrix:
    values: np.ndarray
    names: tuple[str, ...]
    causal: bool = True
    uses_target: bool = False


def _clean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    missing = ~np.isfinite(values)
    return np.where(missing, 0.0, values), missing


def _rolling_statistics(values: np.ndarray, windows: Sequence[int], prefix: str) -> tuple[np.ndarray, list[str]]:
    """Past/current mean, std, slope and normalized deviation for T x D values."""

    rows: list[np.ndarray] = []
    names: list[str] = []
    for window in windows:
        output = np.empty((len(values), values.shape[1] * 4), dtype=np.float32)
        for time_index in range(len(values)):
            start = max(0, time_index - int(window) + 1)
            history = values[start : time_index + 1]
            mean = np.mean(history, axis=0); std = np.std(history, axis=0)
            if len(history) > 1:
                x = np.arange(len(history), dtype=float); centered = x - np.mean(x)
                slope = np.sum(centered[:, None] * (history - np.mean(history, axis=0)), axis=0) / max(float(np.sum(centered**2)), 1e-12)
            else:
                slope = np.zeros(values.shape[1])
            deviation = (history[-1] - mean) / np.where(std > 1e-8, std, 1.0)
            output[time_index] = np.concatenate([mean, std, slope, deviation])
        rows.append(output)
        for statistic in ("mean", "std", "slope", "normalized_deviation"):
            names.extend([f"{prefix}_w{window}_{statistic}_{index}" for index in range(values.shape[1])])
    return np.concatenate(rows, axis=1), names


def _past_autocorrelation(values: np.ndarray, windows: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    output = np.zeros((len(values), len(windows) * 2), dtype=np.float32)
    for time_index in range(len(values)):
        for index, window in enumerate(windows):
            start = max(0, time_index - int(window) + 1); history = values[start : time_index + 1]
            if len(history) > 2 and np.std(history[:-1]) > 1e-8 and np.std(history[1:]) > 1e-8:
                output[time_index, 2 * index] = float(np.corrcoef(history[:-1], history[1:])[0, 1])
            differences = np.diff(history)
            output[time_index, 2 * index + 1] = float(np.abs(np.mean(np.sign(differences)))) if len(differences) else 0.0
    names = [name for window in windows for name in (f"global_w{window}_autocorrelation", f"global_w{window}_trend_consistency")]
    return output, names


def extract_phase35_features(
    record: SessionRecord,
    *,
    feature_set: str,
    windows: Sequence[int] = (5, 10, 20, 40),
    band_names: Sequence[str] = DEFAULT_BAND_NAMES,
) -> Phase35FeatureMatrix:
    """Extract one configured feature set without reading PERCLOS."""

    if record.features_5bands is None or record.features_2hz is None:
        raise ValueError("Phase 3.5 features require both EEG views")
    five, five_missing = _clean(np.asarray(record.features_5bands, dtype=float))
    two, two_missing = _clean(np.asarray(record.features_2hz, dtype=float))
    base, specs = extract_handcrafted_features(five, two, window_size=10, band_names=band_names)
    if feature_set == "base81":
        return Phase35FeatureMatrix(base.astype(np.float32), tuple(item.feature_name for item in specs))
    if feature_set not in {"enhanced", "enhanced_raw5", "enhanced_full"}:
        raise ValueError(f"Unsupported Phase 3.5 feature set: {feature_set}")
    matrices: list[np.ndarray] = [base]
    names: list[str] = [item.feature_name for item in specs]
    band_mean = np.mean(five, axis=1); two_bin_mean = np.mean(two, axis=1)
    grouped_two = two_bin_mean.reshape(len(two), 5, 5).mean(axis=2)
    median = np.median(five, axis=1); mad = np.median(np.abs(five - median[:, None, :]), axis=1)
    q25, q75 = np.quantile(five, 0.25, axis=1), np.quantile(five, 0.75, axis=1)
    robust = np.concatenate([median, mad, q25, q75, q75 - q25], axis=1)
    matrices.append(robust); names.extend([f"five_{stat}_{band}" for stat in ("median", "mad", "q25", "q75", "iqr") for band in band_names])
    log_energy = np.log1p(np.mean(five**2, axis=1)); matrices.append(log_energy)
    names.extend([f"five_log_energy_{band}" for band in band_names])
    cross_agreement = band_mean - grouped_two; matrices.append(cross_agreement)
    names.extend([f"cross_view_index_agreement_{index}" for index in range(5)])
    channel_means = np.concatenate([np.mean(five, axis=2), np.mean(two, axis=2)], axis=1)
    matrices.append(channel_means); names.extend([f"five_channel_mean_{index}" for index in range(17)] + [f"two_channel_mean_{index}" for index in range(17)])
    rolling_five, rolling_five_names = _rolling_statistics(band_mean, windows, "five_band")
    rolling_two, rolling_two_names = _rolling_statistics(grouped_two, windows, "two_group")
    matrices.extend([rolling_five, rolling_two]); names.extend(rolling_five_names + rolling_two_names)
    global_energy = np.mean(np.abs(five), axis=(1, 2)); temporal, temporal_names = _past_autocorrelation(global_energy, windows)
    matrices.append(temporal); names.extend(temporal_names)
    missing_by_channel = np.concatenate([np.mean(five_missing, axis=2), np.mean(two_missing, axis=2)], axis=1)
    matrices.append(missing_by_channel); names.extend([f"five_channel_missing_{index}" for index in range(17)] + [f"two_channel_missing_{index}" for index in range(17)])
    if feature_set in {"enhanced_raw5", "enhanced_full"}:
        matrices.append(five.reshape(len(five), -1)); names.extend([f"five_current_c{channel}_f{frequency}" for channel in range(17) for frequency in range(5)])
    if feature_set == "enhanced_full":
        first = np.diff(five, axis=0, prepend=five[:1]); second = np.diff(first, axis=0, prepend=first[:1])
        matrices.extend([first.reshape(len(five), -1), second.reshape(len(five), -1)])
        names.extend([f"five_first_difference_c{channel}_f{frequency}" for channel in range(17) for frequency in range(5)])
        names.extend([f"five_second_difference_c{channel}_f{frequency}" for channel in range(17) for frequency in range(5)])
    result = np.concatenate(matrices, axis=1).astype(np.float32)
    if result.shape[1] != len(names) or not np.all(np.isfinite(result)):
        raise RuntimeError("Phase 3.5 feature matrix/schema mismatch")
    return Phase35FeatureMatrix(result, tuple(names))


def stack_phase35_features(
    records: Sequence[SessionRecord], *, feature_set: str, windows: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Stack features, labels and provenance for training/evaluation."""

    matrices, targets, subjects, sessions = [], [], [], []
    names: tuple[str, ...] | None = None
    for record in records:
        extracted = extract_phase35_features(record, feature_set=feature_set, windows=windows)
        if names is None: names = extracted.names
        elif names != extracted.names: raise RuntimeError("Phase 3.5 feature order changed across Sessions")
        matrices.append(extracted.values); targets.append(np.asarray(record.perclos, dtype=float))
        subjects.append(np.full(record.sample_count, record.subject_id)); sessions.append(np.full(record.sample_count, record.session_id))
    if names is None: raise ValueError("No records supplied")
    return np.concatenate(matrices), np.concatenate(targets), np.concatenate(subjects), np.concatenate(sessions), names
