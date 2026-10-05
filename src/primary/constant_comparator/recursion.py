"""One shared supplied-alpha recursion kernel for reconstructed Full and C0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .errors import fail


KEYS = ["outer_subject_id", "seed", "session_id", "time_index"]
TARGET_ALIASES = {"target", "target_perclos", "y", "label", "outer_target"}
TARGET_FREE_COLUMNS = [*KEYS, "fast", "slow"]


class AlphaProvider(Protocol):
    def alpha_sequence(self, fast: np.ndarray, slow: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RecursionOutput:
    state: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True)
class ConstantAlphaProvider:
    alpha: float

    def alpha_sequence(self, fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
        del slow
        return np.full(len(fast), float(self.alpha), dtype=np.float64)


@dataclass(frozen=True)
class ArrayAlphaProvider:
    alpha: np.ndarray

    def alpha_sequence(self, fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
        del fast, slow
        return np.asarray(self.alpha, dtype=np.float64).copy()


def state_update(previous: float, observation: float, alpha: float) -> float:
    values = np.asarray([previous, observation, alpha], dtype=np.float64)
    if not np.isfinite(values).all():
        fail("FAIL_CLOSED_INITIALIZATION_RESET_OR_CLIP_INVALID", "nonfinite update operand")
    return float(np.clip(alpha * observation + (1.0 - alpha) * previous, 0.0, 1.0))


def run_session(fast: np.ndarray, slow: np.ndarray, provider: AlphaProvider) -> RecursionOutput:
    fast_values = np.asarray(fast, dtype=np.float64)
    slow_values = np.asarray(slow, dtype=np.float64)
    if fast_values.ndim != 1 or slow_values.ndim != 1 or fast_values.shape != slow_values.shape or len(fast_values) == 0:
        fail("FAIL_CLOSED_KEY_OR_AGGREGATION_INVALID", "fast/slow must be aligned non-empty vectors")
    if not np.isfinite(fast_values).all() or not np.isfinite(slow_values).all():
        fail("FAIL_CLOSED_INITIALIZATION_RESET_OR_CLIP_INVALID", "fast/slow must be finite")
    if np.any((fast_values < 0.0) | (fast_values > 1.0) | (slow_values < 0.0) | (slow_values > 1.0)):
        fail("FAIL_CLOSED_INITIALIZATION_RESET_OR_CLIP_INVALID", "fast/slow must lie in [0,1]")
    alpha = np.asarray(provider.alpha_sequence(fast_values, slow_values), dtype=np.float64)
    if alpha.shape != fast_values.shape or not np.isfinite(alpha).all() or np.any((alpha < 0.0) | (alpha > 1.0)):
        fail("FAIL_CLOSED_INITIALIZATION_RESET_OR_CLIP_INVALID", "supplied alpha must be aligned, finite, and in [0,1]")
    state = np.empty(len(fast_values), dtype=np.float64)
    previous = float(slow_values[0])
    for index in range(len(fast_values)):
        previous = state_update(previous, float(fast_values[index]), float(alpha[index]))
        state[index] = previous
    return RecursionOutput(state=state, alpha=alpha)


def validate_target_free_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        fail("FAIL_CLOSED_OUTER_TARGET_ACCESS_DURING_SELECTION", "trajectory input must be a DataFrame")
    forbidden = TARGET_ALIASES & set(frame.columns)
    if forbidden or set(frame.columns) != set(TARGET_FREE_COLUMNS):
        fail("FAIL_CLOSED_OUTER_TARGET_ACCESS_DURING_SELECTION", f"target-bearing or unexpected trajectory columns: {sorted(set(frame.columns) - set(TARGET_FREE_COLUMNS))}")
    if frame.empty or frame.duplicated(KEYS).any():
        fail("FAIL_CLOSED_KEY_OR_AGGREGATION_INVALID", "empty or duplicate target-free trajectory keys")
    ordered = frame.sort_values(KEYS, kind="stable")
    if not ordered.index.equals(frame.index):
        fail("FAIL_CLOSED_KEY_OR_AGGREGATION_INVALID", "trajectory rows are not in canonical key order")
    if not np.isfinite(frame[[*KEYS, "fast", "slow"]].to_numpy(dtype=np.float64)).all():
        fail("FAIL_CLOSED_INITIALIZATION_RESET_OR_CLIP_INVALID", "nonfinite target-free input")
    for _key, group in frame.groupby(KEYS[:-1], sort=False):
        observed = group["time_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, np.arange(len(group), dtype=np.int64)):
            fail("FAIL_CLOSED_KEY_OR_AGGREGATION_INVALID", "time_index must be contiguous from zero")
    return frame.copy()


ProviderFactory = Callable[[tuple[int, int, int], np.ndarray, np.ndarray], AlphaProvider]


def run_target_free_frame(frame: pd.DataFrame, provider_factory: ProviderFactory) -> pd.DataFrame:
    source = validate_target_free_frame(frame)
    rows: list[pd.DataFrame] = []
    for key, group in source.groupby(KEYS[:-1], sort=False):
        fast = group["fast"].to_numpy(dtype=np.float64)
        slow = group["slow"].to_numpy(dtype=np.float64)
        provider = provider_factory(tuple(map(int, key)), fast, slow)
        output = run_session(fast, slow, provider)
        current = group[KEYS].copy()
        current["state"] = output.state
        current["alpha"] = output.alpha
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def constant_provider_factory(alpha: float) -> ProviderFactory:
    def factory(_key: tuple[int, int, int], _fast: np.ndarray, _slow: np.ndarray) -> AlphaProvider:
        return ConstantAlphaProvider(float(alpha))

    return factory
