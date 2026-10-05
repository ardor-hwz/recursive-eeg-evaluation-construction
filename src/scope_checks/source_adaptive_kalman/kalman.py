"""One-dimensional fixed and strictly causal adaptive Kalman filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import SequenceBaseline


class FixedKalmanBaseline(SequenceBaseline):
    """Random-walk Kalman filter using fixed Q and R."""

    def __init__(
        self,
        *,
        process_variance: float,
        observation_variance: float,
        initial_state: float | None = None,
        initial_covariance: float = 1.0,
        epsilon: float = 1e-12,
        **config: Any,
    ) -> None:
        if process_variance < 0 or observation_variance <= 0 or initial_covariance < 0 or epsilon <= 0:
            raise ValueError("Kalman variances/covariance must be non-negative and R/epsilon positive")
        super().__init__(
            name="fixed_kalman", process_variance=float(process_variance),
            observation_variance=float(observation_variance), initial_state=initial_state,
            initial_covariance=float(initial_covariance), epsilon=float(epsilon), **config
        )
        self.q = float(process_variance)
        self.r = float(observation_variance)
        self.initial_state = initial_state
        self.initial_covariance = float(initial_covariance)
        self.epsilon = float(epsilon)
        self._mean: float | None = None
        self._covariance = self.initial_covariance
        self._step = 0

    def reset_session(self, session_id: int) -> None:
        self._session_id = int(session_id)
        self._mean = None if self.initial_state is None else float(self.initial_state)
        self._covariance = self.initial_covariance
        self._step = 0

    def _current_observation_variance(self) -> float:
        return self.r

    def _after_update(self, innovation: float) -> None:
        del innovation

    def predict_step(self, proxy_value: float) -> dict[str, float]:
        observation = float(proxy_value)
        if not np.isfinite(observation):
            raise ValueError("Kalman observation must be finite")
        if self._mean is None:
            self._mean = observation
            self._step = 1
            return {
                "prediction": self._mean,
                "prediction_variance": self._covariance,
                "kalman_gain": 0.0,
                "innovation": 0.0,
                "observation_variance": self._current_observation_variance(),
            }
        prior_mean = self._mean
        prior_covariance = max(self.epsilon, self._covariance + self.q)
        r_used = max(self.epsilon, self._current_observation_variance())
        gain = prior_covariance / (prior_covariance + r_used)
        innovation = observation - prior_mean
        self._mean = prior_mean + gain * innovation
        self._covariance = max(0.0, (1.0 - gain) * prior_covariance)
        self._step += 1
        row = {
            "prediction": float(self._mean),
            "prediction_variance": float(self._covariance),
            "kalman_gain": float(gain),
            "innovation": float(innovation),
            "observation_variance": float(r_used),
        }
        # The current innovation is incorporated only after the current gain.
        self._after_update(innovation)
        return row

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({"mean": self._mean, "covariance": self._covariance, "step": self._step})
        return state


class AdaptiveKalmanBaseline(FixedKalmanBaseline):
    """Kalman filter whose R estimate uses past innovations only."""

    def __init__(
        self,
        *,
        process_variance: float,
        observation_variance: float,
        adaptation_rate: float = 0.05,
        min_observation_variance: float = 1e-6,
        max_observation_variance: float = 1.0,
        innovation_clip: float = 1.0,
        warmup_steps: int = 10,
        adaptation_mode: str = "innovation_ema",
        **config: Any,
    ) -> None:
        if not 0.0 < adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must lie in (0, 1]")
        if not 0 < min_observation_variance <= max_observation_variance:
            raise ValueError("Invalid adaptive observation variance bounds")
        if innovation_clip <= 0 or warmup_steps < 0:
            raise ValueError("innovation_clip must be positive and warmup_steps non-negative")
        if adaptation_mode not in {"innovation_ema", "bounded_adaptation"}:
            raise ValueError("Unsupported causal adaptation mode")
        super().__init__(
            process_variance=process_variance, observation_variance=observation_variance,
            name_override="adaptive_kalman", **config
        )
        self._config.update({
            "name": "adaptive_kalman", "adaptation_rate": adaptation_rate,
            "min_observation_variance": min_observation_variance,
            "max_observation_variance": max_observation_variance,
            "innovation_clip": innovation_clip, "warmup_steps": warmup_steps,
            "adaptation_mode": adaptation_mode,
        })
        self.adaptation_rate = float(adaptation_rate)
        self.min_r = float(min_observation_variance)
        self.max_r = float(max_observation_variance)
        self.innovation_clip = float(innovation_clip)
        self.warmup_steps = int(warmup_steps)
        self._adaptive_r = float(np.clip(observation_variance, self.min_r, self.max_r))
        self._initial_r = self._adaptive_r

    def reset_session(self, session_id: int) -> None:
        super().reset_session(session_id)
        self._adaptive_r = self._initial_r

    def _current_observation_variance(self) -> float:
        return self._adaptive_r

    def _after_update(self, innovation: float) -> None:
        if self._step <= self.warmup_steps:
            return
        clipped = float(np.clip(innovation, -self.innovation_clip, self.innovation_clip))
        estimate = clipped * clipped
        updated = (1.0 - self.adaptation_rate) * self._adaptive_r + self.adaptation_rate * estimate
        self._adaptive_r = float(np.clip(updated, self.min_r, self.max_r))

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["adaptive_observation_variance"] = self._adaptive_r
        return state
