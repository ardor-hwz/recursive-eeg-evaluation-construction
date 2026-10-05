"""Strictly causal score-first three-state change-aware tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


STATE_NAMES: tuple[str, ...] = ("outlier", "stable", "transition")


@dataclass(frozen=True)
class ChangeAwareTrackerConfig:
    """Frozen tracker hyperparameters; ordered alpha is a hard invariant."""

    past_window: int = 10
    innovation_scale: float = 0.08
    disagreement_scale: float = 0.08
    slope_scale: float = 0.025
    variance_scale: float = 0.01
    temperature: float = 0.75
    outlier_bias: float = 0.0
    stable_bias: float = 0.0
    transition_bias: float = 0.0
    alpha_outlier: float = 0.04
    alpha_stable: float = 0.20
    alpha_transition: float = 0.75

    def __post_init__(self) -> None:
        if self.past_window < 3:
            raise ValueError("past_window must be at least three")
        positive = (self.innovation_scale, self.disagreement_scale, self.slope_scale,
                    self.variance_scale, self.temperature)
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Tracker scales and temperature must be finite and positive")
        if not 0 <= self.alpha_outlier < self.alpha_stable < self.alpha_transition <= 1:
            raise ValueError("Ordered alpha invariant requires outlier < stable < transition")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ChangeAwareTrackerConfig":
        return cls(**{field: values[field] for field in cls.__dataclass_fields__ if field in values})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChangeAwareTrackerOutput:
    state: np.ndarray
    alpha: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    state_index: np.ndarray
    features: np.ndarray

    @property
    def state_name(self) -> np.ndarray:
        return np.asarray(STATE_NAMES, dtype=object)[self.state_index]


def _squash_nonnegative(value: float, scale: float) -> float:
    return float(1.0 - np.exp(-max(float(value), 0.0) / scale))


def _past_statistics(history: np.ndarray, window: int) -> tuple[float, float, float]:
    """Return slope, slope consistency, and variance using past samples only."""

    values = np.asarray(history[-window:], dtype=float)
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    x = np.arange(len(values), dtype=float); centered = x - np.mean(x)
    slope = float(np.sum(centered * (values - np.mean(values))) / max(float(np.sum(centered**2)), 1e-12))
    differences = np.diff(values); nonzero = np.sign(differences[np.abs(differences) > 1e-12])
    consistency = float(abs(np.mean(nonzero))) if len(nonzero) else 0.0
    return slope, consistency, float(np.var(values))


def causal_past_context(fast: np.ndarray, window: int) -> np.ndarray:
    """Precompute target-free past-only slope/consistency/variance for repeated search."""

    values = np.asarray(fast, dtype=float)
    if values.ndim != 1 or len(values) == 0 or window < 3 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid fast sequence or past window")
    return np.asarray([_past_statistics(values[:index], window) for index in range(len(values))], dtype=float)


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    shifted = np.asarray(scores, dtype=float) / temperature
    shifted -= np.max(shifted); exponent = np.exp(shifted)
    return exponent / np.sum(exponent)


class ChangeAwareTracker:
    """Convert fast/slow proxy sequences into one causal adaptive state."""

    def __init__(self, config: ChangeAwareTrackerConfig) -> None:
        self.config = config

    def track(self, fast: np.ndarray, slow: np.ndarray, *, past_context: np.ndarray | None = None) -> ChangeAwareTrackerOutput:
        fast_values, slow_values = np.asarray(fast, dtype=float), np.asarray(slow, dtype=float)
        if (fast_values.ndim != 1 or slow_values.shape != fast_values.shape or len(fast_values) == 0
                or not np.all(np.isfinite(fast_values)) or not np.all(np.isfinite(slow_values))):
            raise ValueError("fast and slow must be aligned finite non-empty vectors")
        if np.any((fast_values < 0) | (fast_values > 1) | (slow_values < 0) | (slow_values > 1)):
            raise ValueError("fast and slow proxy values must lie in [0,1]")
        count = len(fast_values); context = (causal_past_context(fast_values, self.config.past_window)
                                             if past_context is None else np.asarray(past_context, dtype=float))
        if context.shape != (count, 3) or not np.all(np.isfinite(context)):
            raise ValueError("past_context must be a finite T x 3 causal context")
        count = len(fast_values); state = np.empty(count); alpha = np.empty(count)
        scores = np.empty((count, 3)); probabilities = np.empty((count, 3)); state_index = np.empty(count, dtype=int)
        # fast, slow, fast-slow, innovation, |innovation|, past slope,
        # slope consistency, sign consistency, past variance.
        features = np.empty((count, 9)); previous_state = float(slow_values[0])
        for index in range(count):
            past_slope, slope_consistency, past_variance = context[index]
            innovation = float(fast_values[index] - previous_state); disagreement = float(fast_values[index] - slow_values[index])
            if abs(past_slope) <= 1e-12 or abs(innovation) <= 1e-12:
                sign_consistency = 0.5
            else:
                sign_consistency = 1.0 if np.sign(past_slope) == np.sign(innovation) else 0.0
            innovation_strength = _squash_nonnegative(abs(innovation), self.config.innovation_scale)
            disagreement_strength = _squash_nonnegative(abs(disagreement), self.config.disagreement_scale)
            slope_strength = _squash_nonnegative(abs(past_slope), self.config.slope_scale)
            variance_strength = _squash_nonnegative(past_variance, self.config.variance_scale)
            transition_score = (self.config.transition_bias + 1.4 * innovation_strength + 1.1 * disagreement_strength
                                + 1.0 * slope_strength + 1.0 * slope_consistency + 1.2 * sign_consistency
                                - 0.5 * variance_strength)
            outlier_score = (self.config.outlier_bias + 1.6 * innovation_strength + 1.2 * disagreement_strength
                             + 1.1 * (1.0 - slope_consistency) + 1.1 * (1.0 - sign_consistency)
                             + 0.5 * variance_strength - 0.4 * slope_strength)
            stable_score = (self.config.stable_bias + 1.5 * (1.0 - innovation_strength)
                            + 1.0 * (1.0 - disagreement_strength) + 0.8 * (1.0 - slope_strength)
                            + 0.8 * (1.0 - variance_strength) + 0.3 * slope_consistency)
            current_scores = np.asarray([outlier_score, stable_score, transition_score])
            current_probabilities = _softmax(current_scores, self.config.temperature)
            current_alpha = float(current_probabilities @ np.asarray([
                self.config.alpha_outlier, self.config.alpha_stable, self.config.alpha_transition,
            ]))
            current_state = float(np.clip(current_alpha * fast_values[index] + (1.0 - current_alpha) * previous_state, 0.0, 1.0))
            features[index] = [fast_values[index], slow_values[index], disagreement, innovation, abs(innovation),
                               past_slope, slope_consistency, sign_consistency, past_variance]
            scores[index] = current_scores; probabilities[index] = current_probabilities
            alpha[index] = current_alpha; state[index] = current_state; state_index[index] = int(np.argmax(current_probabilities))
            previous_state = current_state
        return ChangeAwareTrackerOutput(state, alpha, scores, probabilities, state_index, features)
