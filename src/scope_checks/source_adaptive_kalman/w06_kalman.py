"""Locked, target-free W06 AdaptiveKalmanBaseline full-prefix replay.

``replay`` uses the existing class directly. ``batch_replay`` evaluates those
same scalar equations in independent parallel streams, always from the physical
recording start. No heuristic-updater state or endpoint is used here.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from src.baselines.kalman import AdaptiveKalmanBaseline


LOCKED_PARAMETERS: dict[str, Any] = {
    "process_variance": 1e-4,
    "observation_variance": 1e-2,
    "adaptation_rate": 0.05,
    "min_observation_variance": 1e-6,
    "max_observation_variance": 1.0,
    "innovation_clip": 1.0,
    "warmup_steps": 10,
    "adaptation_mode": "innovation_ema",
    "initial_covariance": 1.0,
    "epsilon": 1e-12,
}
TRACE_KEYS = ("state", "gain", "covariance", "r_used", "r_after", "step", "innovation")
FLOAT_TOLERANCE = 2e-14


def _inputs(fast: np.ndarray, slow: np.ndarray, *, batch: bool = False) -> tuple[np.ndarray, np.ndarray]:
    fast_array = np.asarray(fast, dtype=np.float64)
    slow_array = np.asarray(slow, dtype=np.float64)
    if fast_array.ndim != (2 if batch else 1) or slow_array.ndim != 1:
        raise ValueError("Expected fast [streams,time] for batch or [time], and slow [time]")
    if slow_array.size == 0 or fast_array.shape[-1] != slow_array.size:
        raise ValueError("Fast/slow must be nonempty and aligned within one physical recording")
    if batch and fast_array.shape[0] == 0:
        raise ValueError("Batch replay requires at least one independent stream")
    if not np.all(np.isfinite(fast_array)) or not np.all(np.isfinite(slow_array)):
        raise ValueError("W06 fast/slow inputs must be finite")
    return fast_array, slow_array


def new_updater(initial_state: float) -> AdaptiveKalmanBaseline:
    """Instantiate the frozen class configuration, explicitly binding slow[0]."""
    return AdaptiveKalmanBaseline(initial_state=float(initial_state), **LOCKED_PARAMETERS)


def _trace_class(model: AdaptiveKalmanBaseline, fast: np.ndarray) -> dict[str, np.ndarray]:
    trace = {key: np.empty(fast.size, dtype=np.int64 if key == "step" else np.float64)
             for key in TRACE_KEYS}
    for index, observation in enumerate(fast):
        row = model.predict_step(float(observation))
        internal = model.state_dict()
        trace["state"][index] = row["prediction"]
        trace["gain"][index] = row["kalman_gain"]
        trace["covariance"][index] = row["prediction_variance"]
        trace["r_used"][index] = row["observation_variance"]
        trace["r_after"][index] = internal["adaptive_observation_variance"]
        trace["step"][index] = internal["step"]
        trace["innovation"][index] = row["innovation"]
    return trace


def replay(fast: np.ndarray, slow: np.ndarray, *, session_id: int = 0) -> dict[str, np.ndarray]:
    """Replay one physical recording through the actual frozen class from t=0.

    Targets are not accepted by this API. Only slow[0] initializes the updater;
    no later slow sample influences its state. Each call resets all recursion.
    """
    fast_array, slow_array = _inputs(fast, slow)
    model = new_updater(float(slow_array[0]))
    model.reset_session(int(session_id))
    return _trace_class(model, fast_array)


def batch_replay(fast_matrix: np.ndarray, slow: np.ndarray) -> dict[str, np.ndarray]:
    """Replay independent [streams,time] perturbations from physical t=0.

    The dedicated scalar Kalman equations below preserve the existing class's
    arithmetic order and post-current-gain adaptive-R update. Complete prefixes
    are evaluated for every stream; no partial-state restoration is performed.
    All returned arrays have [streams,time] shape, including the step counter.
    """
    fast_array, slow_array = _inputs(fast_matrix, slow, batch=True)
    streams, length = fast_array.shape
    trace = {key: np.empty((streams, length), dtype=np.float64)
             for key in TRACE_KEYS if key != "step"}
    trace["step"] = np.broadcast_to(np.arange(1, length + 1, dtype=np.int64), (streams, length))
    mean = np.full(streams, float(slow_array[0]), dtype=np.float64)
    covariance = np.full(streams, 1.0, dtype=np.float64)
    adaptive_r = np.full(streams, 1e-2, dtype=np.float64)
    for index in range(length):
        prior_covariance = np.maximum(1e-12, covariance + 1e-4)
        r_used = np.maximum(1e-12, adaptive_r)
        gain = prior_covariance / (prior_covariance + r_used)
        innovation = fast_array[:, index] - mean
        mean = mean + gain * innovation
        covariance = np.maximum(0.0, (1.0 - gain) * prior_covariance)
        # Class increments _step before _after_update. Steps 1..10 keep R.
        if index + 1 > 10:
            clipped = np.clip(innovation, -1.0, 1.0)
            estimate = clipped * clipped
            updated = (1.0 - 0.05) * adaptive_r + 0.05 * estimate
            adaptive_r = np.clip(updated, 1e-6, 1.0)
        trace["state"][:, index] = mean
        trace["gain"][:, index] = gain
        trace["covariance"][:, index] = covariance
        trace["r_used"][:, index] = r_used
        trace["r_after"][:, index] = adaptive_r
        trace["innovation"][:, index] = innovation
    return trace


def _independent_scalar_reference(fast: np.ndarray, slow: np.ndarray) -> dict[str, np.ndarray]:
    """Independent direct scalar equations: no baseline or batch helper calls."""
    result: dict[str, list[float | int]] = {key: [] for key in TRACE_KEYS}
    mean, posterior_p, old_r = float(slow[0]), 1.0, 0.01
    for index, observation in enumerate(fast):
        prior_p = max(1e-12, posterior_p + 0.0001)
        current_r = max(1e-12, old_r)
        k = prior_p / (prior_p + current_r)
        residual = float(observation) - mean
        mean = mean + k * residual
        posterior_p = max(0.0, (1.0 - k) * prior_p)
        step = index + 1
        if step > 10:
            bounded_residual = min(1.0, max(-1.0, residual))
            old_r = min(1.0, max(1e-6, 0.95 * old_r + 0.05 * (bounded_residual * bounded_residual)))
        for key, value in (("state", mean), ("gain", k), ("covariance", posterior_p),
                           ("r_used", current_r), ("r_after", old_r), ("step", step),
                           ("innovation", residual)):
            result[key].append(value)
    return {key: np.asarray(values, dtype=np.int64 if key == "step" else np.float64)
            for key, values in result.items()}


def _compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray], *, tolerance: float = 0.0) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key in TRACE_KEYS:
        if left[key].shape != right[key].shape:
            raise AssertionError(f"Trace shape mismatch for {key}")
        error = float(np.max(np.abs(left[key] - right[key])))
        if error > (0.0 if key == "step" else tolerance):
            raise AssertionError(f"{key} mismatch: {error} > {tolerance}")
        errors[key] = error
    return errors


def verify_synthetic() -> dict[str, Any]:
    """Run the required eight gates plus batch/class/scalar replay equivalence.

    This function evaluates synthetic arrays only. Scientific result direction
    is absent from every acceptance condition. Failures raise AssertionError.
    """
    checks: dict[str, dict[str, Any]] = {}
    time = np.arange(400, dtype=np.float64)
    fast = 0.4 + 0.17 * np.sin(time / 13.0) + 0.07 * np.cos(time / 5.0)
    fast[0] = 0.83
    slow = 0.13 + 0.0004 * time
    original_fast, original_slow = fast.copy(), slow.copy()
    clean = replay(fast, slow, session_id=701)

    prefix_ends = (1, 10, 11, 73, 211, 399)
    for end in prefix_ends:
        prefix = replay(fast[:end], slow[:end], session_id=701)
        _compare(prefix, {key: value[:end] for key, value in clean.items()})
        future_fast, future_slow = fast.copy(), slow.copy()
        future_fast[end:] = -500.0
        future_slow[end:] = 500.0
        changed = replay(future_fast, future_slow, session_id=701)
        _compare(prefix, {key: value[:end] for key, value in changed.items()})
    checks["future_prefix_equality"] = {"status": "PASS", "prefix_lengths": list(prefix_ends),
                                         "bitwise_equal": True}

    second_fast = -0.3 + 0.08 * np.sin(time[:41] / 3.0)
    second_slow = np.full(41, -0.61)
    model = new_updater(float(second_slow[0]))
    model.reset_session(701)
    _trace_class(model, fast * 20.0)
    model.reset_session(702)
    reset_internal = model.state_dict()
    assert reset_internal["mean"] == second_slow[0]
    assert reset_internal["covariance"] == 1.0
    assert reset_internal["adaptive_observation_variance"] == 0.01
    assert reset_internal["step"] == 0 and reset_internal["session_id"] == 702
    _compare(_trace_class(model, second_fast), replay(second_fast, second_slow, session_id=702))
    _compare(replay(second_fast, second_slow), replay(second_fast, second_slow))
    checks["physical_session_reset"] = {"status": "PASS", "reset_fields": ["mean", "covariance", "adaptive_R", "step"],
                                         "fresh_and_contaminated_then_reset_equal": True}

    target_a = np.zeros(fast.size)
    target_b = np.linspace(-1e9, 1e9, fast.size)
    fixture_a = {"fast": fast, "slow": slow, "target": target_a}
    fixture_b = {"fast": fast, "slow": slow, "target": target_b}
    _compare(replay(fixture_a["fast"], fixture_a["slow"]),
             replay(fixture_b["fast"], fixture_b["slow"]))
    assert "target" not in inspect.signature(replay).parameters
    assert "target" not in inspect.signature(batch_replay).parameters
    class InaccessibleTarget:
        def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
            raise AssertionError("Target firewall accessed")
    inaccessible_fixture = {"fast": fast, "slow": slow, "target": InaccessibleTarget()}
    _compare(clean, replay(inaccessible_fixture["fast"], inaccessible_fixture["slow"]))
    checks["target_invariance"] = {"status": "PASS", "changed_targets_trajectory_equal": True,
                                    "target_not_in_replay_api": True, "inaccessible_target_fixture_passed": True}

    oracle = _independent_scalar_reference(fast, slow)
    scalar_errors = _compare(clean, oracle, tolerance=FLOAT_TOLERANCE)
    prior_mean = np.concatenate(([slow[0]], clean["state"][:-1]))
    state_identity = prior_mean + clean["gain"] * clean["innovation"]
    np.testing.assert_array_equal(clean["state"], state_identity)
    np.testing.assert_array_equal(clean["innovation"], fast - prior_mean)
    checks["state_update_algebra"] = {"status": "PASS", "identity": "state_t=state_before_t+K_t*(fast_t-state_before_t)",
                                        "independent_scalar_max_abs_error": scalar_errors,
                                        "state_identity_bitwise_equal": True}

    prior_p = np.maximum(1e-12, np.concatenate(([1.0], clean["covariance"][:-1])) + 1e-4)
    previous_r = np.concatenate(([0.01], clean["r_after"][:-1]))
    np.testing.assert_array_equal(clean["r_used"], previous_r)
    np.testing.assert_array_equal(clean["gain"], prior_p / (prior_p + previous_r))
    np.testing.assert_array_equal(clean["covariance"], np.maximum(0.0, (1.0 - clean["gain"]) * prior_p))
    expected_r = previous_r.copy()
    clipped = np.clip(clean["innovation"][10:], -1.0, 1.0)
    expected_r[10:] = np.clip(0.95 * previous_r[10:] + 0.05 * (clipped * clipped), 1e-6, 1.0)
    np.testing.assert_array_equal(clean["r_after"], expected_r)
    np.testing.assert_array_equal(clean["step"], np.arange(1, fast.size + 1))
    bound_length = 1000
    bound_slow = np.full(bound_length, 0.13)
    bound_fast = np.stack((np.zeros(bound_length), np.where(np.arange(bound_length) % 2 == 0, 100.0, -100.0)))
    bounded = batch_replay(bound_fast, bound_slow)
    assert float(bounded["r_after"].min()) == 1e-6
    assert np.all((bounded["r_after"] >= 1e-6) & (bounded["r_after"] <= 1.0))
    assert np.all(bounded["covariance"] >= 0.0)
    checks["gain_state_covariance_R_bookkeeping"] = {"status": "PASS", "R_used_is_previous_R_after": True,
                                                      "R_adapted_after_current_gain": True,
                                                      "lower_bound_reached": True,
                                                      "bound_R_min": float(bounded["r_after"].min()),
                                                      "bound_R_max": float(bounded["r_after"].max())}

    k0 = (1.0 + 1e-4) / (1.0 + 1e-4 + 0.01)
    assert clean["gain"][0] == k0
    assert clean["innovation"][0] == fast[0] - slow[0]
    assert clean["state"][0] == slow[0] + k0 * (fast[0] - slow[0])
    assert clean["step"][0] == 1
    assert clean["state"][0] != slow[0] and clean["state"][0] != fast[0]
    checks["slow0_initialization_and_process_t0"] = {"status": "PASS", "initial_state": float(slow[0]),
                                                     "first_fast": float(fast[0]),
                                                     "first_processed_state": float(clean["state"][0]),
                                                     "first_processed_gain": float(clean["gain"][0]),
                                                     "first_processed_step": 1}

    np.testing.assert_array_equal(clean["r_after"][:10], np.full(10, 0.01))
    np.testing.assert_array_equal(clean["r_used"][:11], np.full(11, 0.01))
    assert clean["r_after"][10] != 0.01
    assert clean["r_used"][11] == clean["r_after"][10]
    warmup_model = new_updater(float(slow[0]))
    warmup_model.reset_session(1)
    _trace_class(warmup_model, fast[:27])
    assert warmup_model.state_dict()["step"] == 27
    warmup_model.reset_session(2)
    assert warmup_model.state_dict()["step"] == 0
    warmup_reset_trace = _trace_class(warmup_model, fast[:27])
    _compare(warmup_reset_trace, {key: value[:27] for key, value in clean.items()})
    checks["warmup_counter_and_reset"] = {"status": "PASS", "first_adaptation_processed_step": 11,
                                           "first_adaptation_feature_index": 10,
                                           "first_use_of_adapted_R_feature_index": 11,
                                           "warmup_restarts_on_reset": True}

    _compare(clean, replay(fast, slow, session_id=701))
    _compare(batch_replay(np.stack((fast, fast)), slow), batch_replay(np.stack((fast, fast)), slow))
    np.testing.assert_array_equal(fast, original_fast)
    np.testing.assert_array_equal(slow, original_slow)
    checks["deterministic_execution"] = {"status": "PASS", "bitwise_equal": True, "input_arrays_unchanged": True}

    constructions = {"transient": (0.05, 1), "fixed_sustained": (0.05, 20), "equal_total_dose_sustained": (0.0025, 20)}
    perturbation_roots = (4, 173, 336)
    stream_inputs = [fast.copy()]
    labels = ["clean"]
    for root in perturbation_roots:
        for sign in (-1, 1):
            for construction, (amplitude, duration) in constructions.items():
                perturbed = fast.copy()
                perturbed[root:root + duration] += sign * amplitude
                stream_inputs.append(perturbed)
                labels.append(f"{construction}:root={root}:sign={sign}")
    matrix = np.stack(stream_inputs)
    batched = batch_replay(matrix, slow)
    errors = {key: 0.0 for key in TRACE_KEYS}
    for stream, stream_fast in enumerate(matrix):
        actual = replay(stream_fast, slow)
        batch_trace = {key: value[stream] for key, value in batched.items()}
        for compared in (actual, _independent_scalar_reference(stream_fast, slow)):
            stream_errors = _compare(batch_trace, compared, tolerance=FLOAT_TOLERANCE)
            for key in TRACE_KEYS:
                errors[key] = max(errors[key], stream_errors[key])
    for stream in range(bound_fast.shape[0]):
        bound_errors = _compare({key: value[stream] for key, value in bounded.items()},
                                replay(bound_fast[stream], bound_slow), tolerance=FLOAT_TOLERANCE)
        for key in TRACE_KEYS:
            errors[key] = max(errors[key], bound_errors[key])
    checks["batch_full_prefix_equivalence"] = {"status": "PASS", "streams": len(labels),
                                                "labels": labels, "roots": list(perturbation_roots),
                                                "both_signs": True, "all_physical_constructions": True,
                                                "outlier_and_R_bound_streams": int(bound_fast.shape[0]),
                                                "trace_max_abs_error": errors,
                                                "state_tolerance": FLOAT_TOLERANCE,
                                                "full_prefix_only": True, "snapshot_optimization": False}
    return {"status": "PASS", "scope": "synthetic technical validation only; no development/outer inputs or scientific outcomes",
            "updater_class": "src.baselines.kalman.AdaptiveKalmanBaseline",
            "locked_parameters": dict(LOCKED_PARAMETERS),
            "initialization": "initial_state=slow[0]; fresh complete recursion per physical recording; process t=0",
            "checks": checks, "required_checks_passed": 8, "additional_checks_passed": 1,
            "snapshot_optimization": False}


if __name__ == "__main__":
    import json
    print(json.dumps(verify_synthetic(), indent=2, sort_keys=True, allow_nan=False))
