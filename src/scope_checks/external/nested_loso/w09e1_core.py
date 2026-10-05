"""W09E-1 numerical core: fixed ridge and clean development calibration.

This module has no EEG/RT loader, outer-target reader, perturbation, root gate,
accuracy endpoint, or M1 executor. The caller supplies admitted, aligned fitting
rows and development-only OOF segments. Scientific constants come from the one
supplied W09E lock; unsupported semantics fail closed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import numpy as np
from scipy.linalg import solve
from scipy.special import expit


class ContractError(RuntimeError):
    """Missing, contradictory, unsupported, or incomplete locked semantics."""


class NumericalFailure(RuntimeError):
    """A finite admitted input, solve, or clean replay failed numerically."""


class NonEvaluable(RuntimeError):
    """A locked fitting/calibration participant lacks required support."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{name}: boolean is not a scientific numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}: missing/invalid numeric value") from exc
    _require(np.isfinite(result), f"{name}: nonfinite contract value")
    return result


def _int(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0, f"{name}: expected nonnegative integer")
    return value


def _ids(values: Sequence[Any], name: str) -> list[str]:
    _require(isinstance(values, (list, tuple)), f"{name}: expected exact ID list")
    result = list(values)
    _require(all(type(v) is str and bool(v) for v in result), f"{name}: IDs must be nonempty strings")
    _require(len(set(result)) == len(result), f"{name}: duplicate participant IDs")
    return result


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(json.dumps(list(arr.shape)).encode("ascii"))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate only the W09E-1 selectors this module executes."""
    try:
        _require(contract["schema"] == "paper2.w09e.scientific-contract.v1", "Unsupported contract schema")
        _require(contract["workflow"] == "W09E", "Unsupported workflow")
        _require(contract["protocol_status"] == "FROZEN_PRE_W09E_OUTCOME", "Protocol is not frozen")
        cohort = _ids(contract["cohort"]["group_ids"], "cohort.group_ids")
        split = contract["split"]
        outer_n = _int(split["outer_N"], "split.outer_N")
        dev_n = _int(split["development_N"], "split.development_N")
        fit_n = _int(split["inner_fit_N"], "split.inner_fit_N")
        _require(len(cohort) == outer_n and dev_n == outer_n - 1 and fit_n == dev_n - 1,
                 "Inconsistent nested LOSO cardinality")
        seeds = list(split["development_oof_seeds"])
        _require(bool(seeds) and len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds),
                 "Development seeds must be distinct exact integers")

        normalization = contract["normalization"]
        method_match = re.fullmatch(
            r"per-role/per-feature weighted training mean and population SD; floor SD at ([0-9.eE+-]+) in feature units; zero-variance training feature transforms to zero on all partitions",
            normalization["method"],
        )
        _require(method_match is not None, "Unsupported normalizer or zero-variance rule")
        floor = _float(method_match.group(1), "normalization SD floor")
        _require(floor > 0, "Normalizer floor must be positive")
        _require(normalization["training_weights"] ==
                 "equal participant weight, equal recording within participant, equal eligible row within recording: w=1/(G*S_g*n_gs), sum(w)=1",
                 "Unsupported fitting/normalization hierarchy")
        proxy = contract["proxy"]
        regression_match = re.fullmatch(
            r"minimize sum\(w\*\(logRT-b-X\*beta\)\^2\)\+([0-9.eE+-]+)\*sum\(beta\^2\); intercept unpenalized; fixed penalty, no feature/proxy hyperparameter search",
            proxy["regression"],
        )
        solver_match = re.fullmatch(
            r"float64 weighted centered normal equations, fixed ridge lambda=([0-9.eE+-]+); solve SPD matrix; nonfinite failure halts fold",
            proxy["solver"],
        )
        _require(regression_match is not None and solver_match is not None, "Unsupported ridge equation or solver")
        penalty = _float(regression_match.group(1), "ridge regression penalty")
        _require(penalty > 0 and penalty == _float(solver_match.group(1), "ridge solver penalty"),
                 "Inconsistent/nonpositive fixed ridge penalty")
        _require(proxy["normalization_weights"] ==
                 "same designated fitting groups and eligible rows used for each role; rows require both roles so support is shared",
                 "Unsupported shared role support")
        _require(proxy["fast"] == "independent ridge on trailing 3-s DE-style features predicts log RT; fast=expit(predicted log RT)" and
                 proxy["slow"] == "separate ridge on trailing 30-s features predicts same log RT; slow=expit(predicted log RT)",
                 "Unsupported proxy link")

        exclusions = contract["exclusions"]
        fit_support = re.search(r"fitting person >=([0-9]+) aligned eligible rows", exclusions["participant_support"])
        accuracy_support = re.search(r"for accuracy require >=([0-9]+) post-warm-up paired target rows", exclusions["recording_support"])
        _require(fit_support is not None and accuracy_support is not None, "Missing fitting/accuracy support rules")
        min_fit = int(fit_support.group(1))
        min_accuracy = int(accuracy_support.group(1))
        _require(min_fit > 0 and min_accuracy > 0, "Invalid locked support counts")

        calibration = contract["operational_contract"]["calibration"]
        c0 = calibration["C0"]
        _require(c0["objective"] == "equal_participant_mean_development_OOF_yRMSE" and
                 c0["within_person_recording_weight"] == "concatenate paired rows" and
                 c0["seed_reducer"] == "mean" and c0["tie_choice"] == "smallest_gain",
                 "Unsupported C0 objective/reduction/tie rule")
        warmup = _int(c0["postwarmup_indices_start"], "C0 postwarmup start")
        warmup_match = re.fullmatch(
            r"first ([0-9]+) valid consecutive input updates of each segment retained in trace but excluded from RMSE/MAFD/root eligibility",
            exclusions["warmup"],
        )
        _require(warmup_match is not None and int(warmup_match.group(1)) == warmup,
                 "Inconsistent segment warmup selectors")
        tolerance = _float(c0["tie_tolerance"], "C0 tie tolerance")
        _require(tolerance >= 0, "Negative C0 tie tolerance")
        grid = [_float(v, "C0 grid gain") for v in contract["comparators"]["C0"]["grid"]]
        _require(bool(grid) and len(set(grid)) == len(grid) and all(0 <= a <= 1 for a in grid),
                 "Invalid C0 grid")
        _require(contract["comparators"]["C0"]["update"] ==
                 "state_before_t0=slow[0]; state=alpha*fast+(1-alpha)*previous_state; process t0 once; reset at each segment",
                 "Unsupported constant-gain initialization/recursion")
        cm = calibration["Cmean"]
        _require(cm["within_recording"] == "mean valid postwarmup clean updater gains" and
                 cm["within_person"] == "equal recording mean" and
                 cm["across_people"] == "equal development-person mean" and
                 cm["seeds"] == "mean" and cm["separate_by_updater"] is True and
                 cm["outer_gains_allowed"] is False, "Unsupported Cmean semantics")
        permission = contract["operational_contract"]["replay_permissions"]["development_calibration_replay"]
        _require(permission["stage"] == "W09E-1" and permission["population"] == "development_only" and
                 permission["outer_target_access"] is False and permission["perturbations"] is False and
                 permission["scientific_M1"] is False and permission["C0_development_target_access"] is True,
                 "Unsupported development calibration permission")
        _require(set(contract["updaters"]) == {"heuristic", "adaptive_kalman"}, "Unexpected updater keys")
        heuristic = contract["updaters"]["heuristic"]
        kalman = contract["updaters"]["adaptive_kalman"]
        _require(heuristic["state_initialization"] ==
                 "slow[0] at each feature-valid contiguous segment; process first fast value exactly once",
                 "Unsupported heuristic initialization")
        _require(kalman["state_clipping"] ==
                 "none; innovation bounding is used only for adaptive R and is not input/state clipping" and
                 kalman["slow_after_initialization"] == "unused", "Unsupported Kalman state/context semantics")
        _require(contract["implementation_contract"]["internal_Kalman_session_tag"] ==
                 "Instantiate a fresh object per contiguous segment, call reset_session(0); this internal numeric tag has no identity meaning. Canonical biological/session IDs remain exact strings in every external table.",
                 "Unsupported Kalman reset rule")
        source_bindings = dict(contract["implementation_contract"]["source_code_sha256"])
        _require(len(source_bindings) == 3 and
                 all(type(p) is str and type(h) is str and re.fullmatch(r"[0-9a-f]{64}", h) for p, h in source_bindings.items()),
                 "Invalid updater source bindings")
    except KeyError as exc:
        raise ContractError(f"Missing W09E-1 selector: {exc}") from exc
    return {"cohort": cohort, "outer_n": outer_n, "development_n": dev_n, "inner_fit_n": fit_n,
            "seeds": seeds, "sd_floor": floor, "penalty": penalty, "min_fit_rows": min_fit,
            "min_accuracy_rows": min_accuracy, "warmup": warmup, "tie_tolerance": tolerance,
            "grid": sorted(grid), "source_bindings": source_bindings,
            "heuristic_parameters": dict(heuristic["parameters"]),
            "kalman_parameters": dict(kalman["parameters"])}


def weighted_ridge_fit(
    X: np.ndarray, log_rt: np.ndarray, participants: Sequence[str], recordings: Sequence[str],
    fit_ids: Sequence[str], contract: Mapping[str, Any], *, joint_feature_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one role on already selected joint-feature-valid, target-observed rows.

    The caller must select the designated fitting participants before passing any
    label array. No row is silently removed here. The same row selection must be
    used for both role calls and must be recorded by the caller's lineage ledger.
    """
    q = validate_contract(contract)
    ids = _ids(fit_ids, "fit_ids")
    _require(len(ids) in (q["inner_fit_n"], q["development_n"]) and set(ids) <= set(q["cohort"]),
             "Fitting IDs are not a locked inner/final development set")
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(log_rt, dtype=np.float64)
    p = np.asarray(participants, dtype=object)
    r = np.asarray(recordings, dtype=object)
    _require(x.ndim == 2 and x.shape[1] > 0 and x.shape[0] > 0 and
             y.shape == p.shape == r.shape == (x.shape[0],), "Fitting arrays must align as rows")
    _require(all(type(v) is str and bool(v) for v in p) and all(type(v) is str and bool(v) for v in r),
             "Fitting identities must remain exact strings")
    _require(set(p) == set(ids), "Fitting input contains a forbidden/missing participant")
    if joint_feature_valid is not None:
        joint = np.asarray(joint_feature_valid)
        _require(joint.dtype == np.bool_ and joint.shape == y.shape and np.all(joint),
                 "Fit inputs must already be selected joint-feature-valid rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise NumericalFailure("Admitted fitting features/log RT contain a nonfinite value; no row deletion")

    weights = np.zeros(len(y), dtype=np.float64)
    cells: list[dict[str, Any]] = []
    ownership: dict[str, str] = {}
    for person in ids:
        person_mask = p == person
        person_count = int(np.count_nonzero(person_mask))
        if person_count < q["min_fit_rows"]:
            raise NonEvaluable(f"Fitting participant {person} has {person_count} eligible rows")
        person_recordings = sorted(set(r[person_mask]))
        for recording in person_recordings:
            _require(ownership.setdefault(recording, person) == person, "One recording aliases multiple participants")
            mask = person_mask & (r == recording)
            count = int(np.count_nonzero(mask))
            row_weight = 1.0 / (len(ids) * len(person_recordings) * count)
            weights[mask] = row_weight
            cells.append({"participant_id": person, "recording_id": recording, "eligible_rows": count,
                          "contributing_recordings": len(person_recordings), "row_weight": row_weight,
                          "recording_weight": float(np.sum(weights[mask], dtype=np.float64))})
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
        raise NumericalFailure("Invalid hierarchical training weights")
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if abs(weight_sum - 1.0) > np.finfo(np.float64).eps * max(len(y), 1):
        raise NumericalFailure("Hierarchical training weights fail the locked sum(w)=1 rule")

    # A truly constant column is identified from actual training values, avoiding
    # spurious residual variance from floating-point weighted-mean summation.
    zero_mask = np.ptp(x, axis=0) == 0.0
    mean = np.sum(weights[:, None] * x, axis=0, dtype=np.float64) / weight_sum
    mean[zero_mask] = x[0, zero_mask]
    centered_raw = x - mean
    variance = np.sum(weights[:, None] * centered_raw * centered_raw, axis=0, dtype=np.float64) / weight_sum
    variance[zero_mask] = 0.0
    sd = np.maximum(np.sqrt(variance), q["sd_floor"])
    z = centered_raw / sd
    z[:, zero_mask] = 0.0
    z_mean = np.sum(weights[:, None] * z, axis=0, dtype=np.float64) / weight_sum
    y_mean = float(np.sum(weights * y, dtype=np.float64) / weight_sum)
    z_centered = z - z_mean
    y_centered = y - y_mean
    gram = z_centered.T @ (weights[:, None] * z_centered)
    rhs = z_centered.T @ (weights * y_centered)
    matrix = gram + q["penalty"] * np.eye(x.shape[1], dtype=np.float64)
    if not all(np.all(np.isfinite(a)) for a in (mean, sd, z, matrix, rhs)):
        raise NumericalFailure("Nonfinite normalizer/weighted normal equation")
    try:
        beta = solve(matrix, rhs, assume_a="pos", check_finite=True)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise NumericalFailure("Locked weighted centered SPD ridge solve failed") from exc
    intercept = float(y_mean - z_mean @ beta)
    if not np.all(np.isfinite(beta)) or not np.isfinite(intercept):
        raise NumericalFailure("Nonfinite fitted ridge coefficients/intercept")
    return {"mean": mean, "sd": sd, "zero_mask": zero_mask, "beta": np.asarray(beta, dtype=np.float64),
            "intercept": intercept, "metadata": {
                "fit_ids": ids, "normalizer_fit_ids": ids.copy(), "fit_ids_sha256": _json_hash(ids),
                "normalizer_fit_ids_sha256": _json_hash(ids), "n_rows": len(y), "n_features": x.shape[1],
                "penalty": q["penalty"], "sd_floor": q["sd_floor"], "dtype": "float64",
                "intercept_penalized": False, "weight_sum": weight_sum, "weight_cells": cells,
                "joint_feature_valid_declared": joint_feature_valid is not None,
                "row_selection": "caller-selected joint_feature_valid AND target_observed AND designated fit IDs; no core deletion",
                "training_arrays_sha256": _array_hash(x, y, weights),
                "training_identity_sha256": _json_hash({"participants": p.tolist(), "recordings": r.tolist()}),
                "standardized_weighted_mean": z_mean.tolist(), "weighted_log_rt_mean": y_mean,
                "zero_variance_features": np.flatnonzero(zero_mask).tolist(),
            }}


def predict(model: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    """Apply only the stored fitting-population transform and fixed expit link."""
    x = np.asarray(X, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    sd = np.asarray(model["sd"], dtype=np.float64)
    zero = np.asarray(model["zero_mask"])
    beta = np.asarray(model["beta"], dtype=np.float64)
    _require(x.ndim == 2 and x.shape[1:] == mean.shape == sd.shape == zero.shape == beta.shape,
             "Prediction/model feature dimensions disagree")
    _require(zero.dtype == np.bool_ and np.all(sd > 0), "Invalid stored zero mask/SD")
    intercept = float(model["intercept"])
    if not all(np.all(np.isfinite(a)) for a in (x, mean, sd, beta)) or not np.isfinite(intercept):
        raise NumericalFailure("Nonfinite admitted prediction input or stored model")
    z = np.zeros_like(x, dtype=np.float64)
    active = ~zero
    z[:, active] = (x[:, active] - mean[active]) / sd[active]
    eta = intercept + z @ beta
    prediction = np.asarray(expit(eta), dtype=np.float64)
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(eta)) or not np.all(np.isfinite(prediction)):
        raise NumericalFailure("Nonfinite stored-transform prediction")
    return prediction


_BOUND_CLASSES: dict[str, tuple[Any, Any, Any]] = {}


def _load_source(name: str, path: Path, verified_bytes: bytes) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"Unable to load bound source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Compile the bytes that were verified; do not load an older on-disk pyc.
    exec(compile(verified_bytes, str(path), "exec"), module.__dict__)
    return module


def _bound_classes(q: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    paths: dict[str, Path] = {}
    verified_bytes: dict[str, bytes] = {}
    for path_text, expected_hash in q["source_bindings"].items():
        path = Path(path_text)
        raw = path.read_bytes() if path.is_file() else b""
        if not path.is_file() or hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ContractError(f"Bound updater source SHA-256 mismatch: {path}")
        _require(path.name not in paths, "Duplicate bound updater source filename")
        paths[path.name] = path
        verified_bytes[path.name] = raw
    _require(set(paths) == {"change_aware_tracker.py", "kalman.py", "base.py"}, "Unexpected source binding roles")
    key = _json_hash(q["source_bindings"])
    if key not in _BOUND_CLASSES:
        name = "_w09e1_bound_" + key
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
        # base.py imports the original target-free ProxySequence exchange type.
        # Loading under a private package bypasses src.baselines' registry import;
        # no W06 helper or W06 hard-locked parameter configuration is imported.
        source_root = paths["base.py"].parents[2]
        root_text = str(source_root)
        inserted = root_text not in sys.path
        if inserted:
            sys.path.insert(0, root_text)
        try:
            _load_source(name + ".base", paths["base.py"], verified_bytes["base.py"])
            kalman = _load_source(name + ".kalman", paths["kalman.py"], verified_bytes["kalman.py"])
            tracker = _load_source(name + ".tracker", paths["change_aware_tracker.py"], verified_bytes["change_aware_tracker.py"])
        finally:
            if inserted:
                sys.path.remove(root_text)
        _BOUND_CLASSES[key] = (tracker.ChangeAwareTracker, tracker.ChangeAwareTrackerConfig,
                               kalman.AdaptiveKalmanBaseline)
    return _BOUND_CLASSES[key]


def _proxy_vectors(fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(fast, dtype=np.float64)
    s = np.asarray(slow, dtype=np.float64)
    _require(f.ndim == 1 and f.shape == s.shape and len(f) > 0, "Segment proxies must align as nonempty vectors")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(s)):
        raise NumericalFailure("Nonfinite clean segment proxy")
    _require(np.all((f >= 0) & (f <= 1)) and np.all((s >= 0) & (s <= 1)),
             "Clean expit proxies are outside [0,1]; no shared clipping")
    return f, s


def clean_replay(fast: np.ndarray, slow: np.ndarray, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fresh source-bound updater objects for ONE contiguous clean segment."""
    q = validate_contract(contract)
    f, s = _proxy_vectors(fast, slow)
    tracker_class, tracker_config, kalman_class = _bound_classes(q)
    expected_fields = set(tracker_config.__dataclass_fields__)
    _require(set(q["heuristic_parameters"]) == expected_fields, "Missing/extra heuristic constructor parameters")
    heuristic = tracker_class(tracker_config(**q["heuristic_parameters"]))
    ho = heuristic.track(f, s)
    kwargs = dict(q["kalman_parameters"])
    _require("initial_state" not in kwargs, "Kalman initial_state must be the current segment slow[0]")
    kalman = kalman_class(initial_state=float(s[0]), **kwargs)
    kalman.reset_session(0)
    rows = [kalman.predict_step(float(v)) for v in f]
    result = {
        "heuristic": {"state": np.asarray(ho.state, dtype=np.float64),
                      "gain": np.asarray(ho.alpha, dtype=np.float64),
                      "scores": np.asarray(ho.scores, dtype=np.float64),
                      "probabilities": np.asarray(ho.probabilities, dtype=np.float64),
                      "state_index": np.asarray(ho.state_index, dtype=np.int64),
                      "features": np.asarray(ho.features, dtype=np.float64)},
        "adaptive_kalman": {"state": np.asarray([r["prediction"] for r in rows], dtype=np.float64),
                            "gain": np.asarray([r["kalman_gain"] for r in rows], dtype=np.float64),
                            "covariance": np.asarray([r["prediction_variance"] for r in rows], dtype=np.float64),
                            "innovation": np.asarray([r["innovation"] for r in rows], dtype=np.float64),
                            "observation_variance": np.asarray([r["observation_variance"] for r in rows], dtype=np.float64),
                            "final_state": kalman.state_dict()},
        "metadata": {"input_sha256": _array_hash(f, s), "source_bindings": q["source_bindings"],
                     "parameters_sha256": _json_hash({"heuristic": q["heuristic_parameters"],
                                                      "adaptive_kalman": q["kalman_parameters"]}),
                     "initial_state": float(s[0]), "updates": len(f), "fresh_segment_objects": True,
                     "process_t0_once": True, "shared_input_clip": False, "shared_output_clip": False},
    }
    for updater in ("heuristic", "adaptive_kalman"):
        for key, values in result[updater].items():
            if isinstance(values, np.ndarray) and not np.all(np.isfinite(values)):
                raise NumericalFailure(f"Nonfinite clean {updater} {key}; no segment deletion")
    result["metadata"]["output_sha256"] = {
        updater: _array_hash(result[updater]["state"], result[updater]["gain"])
        for updater in ("heuristic", "adaptive_kalman")
    }
    return result


def constant_replay(fast: np.ndarray, slow: np.ndarray, alpha: float) -> np.ndarray:
    """One segment, slow[0] initializer, each fast observation processed once."""
    f, s = _proxy_vectors(fast, slow)
    gain = _float(alpha, "constant gain")
    _require(0 <= gain <= 1, "Constant gain outside [0,1]")
    state = np.empty(len(f), dtype=np.float64)
    previous = float(s[0])
    for index, value in enumerate(f):
        previous = gain * float(value) + (1.0 - gain) * previous
        state[index] = previous
    if not np.all(np.isfinite(state)):
        raise NumericalFailure("Nonfinite constant-gain clean recursion")
    return state


def _calibration_entries(
    entries: Sequence[Mapping[str, Any]], q: Mapping[str, Any], expected_participants: Sequence[str] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    _require(isinstance(entries, (list, tuple)) and bool(entries), "No development OOF segments")
    if expected_participants is None:
        _require("development_ids" in entries[0], "Explicit development participant set required")
        expected_participants = entries[0]["development_ids"]
    people = _ids(expected_participants, "expected_participants")
    _require(len(people) == q["development_n"] and set(people) <= set(q["cohort"]),
             "Calibration requires the exact locked development18, without outer participant")
    seen: set[tuple[str, str, str, int]] = set()
    recording_owners: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for e in entries:
        try:
            person, recording, segment, seed = (e[k] for k in ("participant_id", "recording_id", "segment_id", "seed"))
            _require(all(type(v) is str and bool(v) for v in (person, recording, segment)),
                     "Calibration identities must remain exact strings")
            _require(person in people and type(seed) is int and seed in q["seeds"],
                     "Calibration contains outer/unexpected participant or seed")
            if "development_ids" in e:
                _require(_ids(e["development_ids"], "entry.development_ids") == people,
                         "Entry development membership disagrees with supplied development set")
            key = (person, recording, segment, seed)
            _require(key not in seen, f"Duplicate development segment/seed: {key}")
            seen.add(key)
            _require(recording_owners.setdefault(recording, person) == person,
                     "Calibration recording aliases multiple participants")
            f, s = _proxy_vectors(e["fast"], e["slow"])
            postwarm = np.asarray(e["postwarm_mask"])
            _require(postwarm.dtype == np.bool_ and postwarm.shape == f.shape,
                     "Postwarm mask must be an aligned boolean vector")
            locked_postwarm = np.arange(len(f)) >= q["warmup"]
            _require(np.array_equal(postwarm, locked_postwarm), "Postwarm mask deviates from locked segment warmup")
            normalized.append({**e, "fast": f, "slow": s, "postwarm_mask": postwarm})
        except KeyError as exc:
            raise ContractError(f"Missing calibration entry field: {exc}") from exc
    _require(set(e["participant_id"] for e in normalized) == set(people), "Missing development participant")
    # Identical deterministic seed cells must retain exactly the same segment
    # identities. Missing a seed segment cannot be masked by another recording.
    for person in people:
        identity_sets = [{(e["recording_id"], e["segment_id"]) for e in normalized
                          if e["participant_id"] == person and e["seed"] == seed} for seed in q["seeds"]]
        _require(bool(identity_sets[0]) and all(v == identity_sets[0] for v in identity_sets),
                 f"Incomplete segment/seed coverage for development participant {person}")
        for recording, segment in identity_sets[0]:
            aliases = [e for e in normalized if e["participant_id"] == person and
                       e["recording_id"] == recording and e["segment_id"] == segment]
            _require(all(np.array_equal(e["fast"], aliases[0]["fast"]) and
                         np.array_equal(e["slow"], aliases[0]["slow"]) and
                         np.array_equal(e["postwarm_mask"], aliases[0]["postwarm_mask"]) for e in aliases),
                     "Deterministic development seed proxies/masks disagree")
    return people, normalized


def choose_c0(
    entries: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], *,
    expected_participants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select one shared gain using only development OOF targets, never gains."""
    q = validate_contract(contract)
    people, normalized = _calibration_entries(entries, q, expected_participants)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for e in normalized:
        _require("target_y" in e, "C0 requires development target_y")
        target = np.asarray(e["target_y"], dtype=np.float64)
        _require(target.shape == e["fast"].shape, "Development target/trajectory alignment disagrees")
        if np.any(np.isinf(target)):
            raise NumericalFailure("Infinite development target; NaN alone marks unobserved reference")
        _require(np.all((target[np.isfinite(target)] >= 0) & (target[np.isfinite(target)] <= 1)),
                 "Development represented target outside fixed [0,1] y map")
        e["target_y"] = target
        e["paired_mask"] = e["postwarm_mask"] & np.isfinite(target)
        grouped.setdefault((e["participant_id"], e["recording_id"], e["seed"]), []).append(e)
    target_aliases: dict[tuple[str, str, str], np.ndarray] = {}
    for e in normalized:
        key = (e["participant_id"], e["recording_id"], e["segment_id"])
        first = target_aliases.setdefault(key, e["target_y"])
        _require(np.array_equal(first, e["target_y"], equal_nan=True),
                 "Development target observations disagree across deterministic seed aliases")
    admitted: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for (person, recording, seed), values in sorted(grouped.items()):
        count = sum(int(np.count_nonzero(e["paired_mask"])) for e in values)
        if count < q["min_accuracy_rows"]:
            excluded.append({"participant_id": person, "recording_id": recording, "seed": seed,
                             "paired_postwarm_rows": count, "reason": "INSUFFICIENT_PAIRED_POSTWARM_TARGET_ROWS",
                             "minimum_required": q["min_accuracy_rows"]})
        else:
            admitted[(person, recording, seed)] = values
    for person in people:
        for seed in q["seeds"]:
            if not any(p == person and s == seed for p, _, s in admitted):
                raise NonEvaluable(f"C0 lacks paired recording support: participant={person}, seed={seed}")
    grid_results: list[dict[str, Any]] = []
    for alpha in q["grid"]:
        metrics: list[dict[str, Any]] = []
        seed_means: list[dict[str, Any]] = []
        for seed in q["seeds"]:
            person_rmses: list[float] = []
            for person in people:
                errors: list[np.ndarray] = []
                recording_counts: dict[str, int] = {}
                for (p, recording, s), values in sorted(admitted.items()):
                    if p == person and s == seed:
                        for e in values:
                            state = constant_replay(e["fast"], e["slow"], alpha)
                            mask = e["paired_mask"]
                            errors.append(state[mask] - e["target_y"][mask])
                            recording_counts[recording] = recording_counts.get(recording, 0) + int(np.count_nonzero(mask))
                concatenated = np.concatenate(errors)
                rmse = float(np.sqrt(np.mean(concatenated * concatenated, dtype=np.float64)))
                if not np.isfinite(rmse):
                    raise NumericalFailure("Nonfinite C0 development RMSE; no participant deletion")
                person_rmses.append(rmse)
                metrics.append({"participant_id": person, "seed": seed, "rmse": rmse,
                                "paired_rows": len(concatenated), "recording_paired_rows": recording_counts})
            seed_means.append({"seed": seed, "equal_person_mean_rmse": float(np.mean(person_rmses, dtype=np.float64))})
        objective = float(np.mean([row["equal_person_mean_rmse"] for row in seed_means], dtype=np.float64))
        grid_results.append({"alpha": alpha, "objective": objective, "person_seed_metrics": metrics,
                             "seed_means": seed_means})
    minimum = min(row["objective"] for row in grid_results)
    tied = [row for row in grid_results if abs(row["objective"] - minimum) <= q["tie_tolerance"]]
    chosen = min(tied, key=lambda row: row["alpha"])
    return {"alpha": chosen["alpha"], "objective": chosen["objective"], "minimum_grid_objective": minimum,
            "grid_results": grid_results, "tied_alphas": [row["alpha"] for row in tied],
            "tie_tolerance": q["tie_tolerance"], "tie_choice": "smallest_gain",
            "excluded_recordings": excluded, "development_ids": people, "seeds": q["seeds"],
            "outer_targets_accessed": False, "selection_population": "development_only_inner_OOF"}


def cmean(
    entries: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], *,
    expected_participants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Updater-specific development gain means; this function never reads target_y."""
    q = validate_contract(contract)
    people, normalized = _calibration_entries(entries, q, expected_participants)
    _bound_classes(q)  # Validate all exact source bindings even when traces are cached.
    parameters_hash = _json_hash({"heuristic": q["heuristic_parameters"], "adaptive_kalman": q["kalman_parameters"]})
    for e in normalized:
        replay = e.get("replay")
        if replay is None:
            replay = clean_replay(e["fast"], e["slow"], contract)
        else:
            try:
                meta = replay["metadata"]
                _require(meta["input_sha256"] == _array_hash(e["fast"], e["slow"]) and
                         meta["source_bindings"] == q["source_bindings"] and
                         meta["parameters_sha256"] == parameters_hash and
                         meta["updates"] == len(e["fast"]) and meta["fresh_segment_objects"] is True and
                         meta["process_t0_once"] is True and meta["shared_input_clip"] is False and
                         meta["shared_output_clip"] is False, "Cached clean replay provenance disagrees")
            except KeyError as exc:
                raise ContractError(f"Missing cached clean replay provenance: {exc}") from exc
        e["replay"] = replay
    result: dict[str, Any] = {}
    for updater in ("heuristic", "adaptive_kalman"):
        grouped: dict[tuple[str, str, int], list[np.ndarray]] = {}
        for e in normalized:
            try:
                gain = np.asarray(e["replay"][updater]["gain"], dtype=np.float64)
                state = np.asarray(e["replay"][updater]["state"], dtype=np.float64)
            except KeyError as exc:
                raise ContractError(f"Missing cached updater output: {exc}") from exc
            _require(gain.shape == state.shape == e["fast"].shape, "Clean gain/state alignment disagrees")
            if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(state)):
                raise NumericalFailure("Nonfinite clean cached updater state/gain; no row deletion")
            _require(np.all((gain >= 0) & (gain <= 1)), "Clean updater gain outside [0,1]")
            try:
                _require(e["replay"]["metadata"]["output_sha256"][updater] == _array_hash(state, gain),
                         "Cached clean replay state/gain digest disagrees")
            except KeyError as exc:
                raise ContractError(f"Missing cached updater state/gain digest: {exc}") from exc
            grouped.setdefault((e["participant_id"], e["recording_id"], e["seed"]), []).append(gain[e["postwarm_mask"]])
        recording_means: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for (person, recording, seed), arrays in sorted(grouped.items()):
            gains = np.concatenate(arrays)
            if len(gains) == 0:
                excluded.append({"participant_id": person, "recording_id": recording, "seed": seed,
                                 "reason": "NO_POSTWARM_GAIN_ROWS", "postwarm_gain_rows": 0})
            else:
                recording_means.append({"participant_id": person, "recording_id": recording, "seed": seed,
                                        "gain_mean": float(np.mean(gains, dtype=np.float64)),
                                        "postwarm_gain_rows": len(gains)})
        person_means: list[dict[str, Any]] = []
        seed_means: list[dict[str, Any]] = []
        for seed in q["seeds"]:
            means: list[float] = []
            for person in people:
                cells = [r for r in recording_means if r["participant_id"] == person and r["seed"] == seed]
                if not cells:
                    raise NonEvaluable(f"Cmean {updater} lacks gain support: participant={person}, seed={seed}")
                mean = float(np.mean([r["gain_mean"] for r in cells], dtype=np.float64))
                means.append(mean)
                person_means.append({"participant_id": person, "seed": seed,
                                     "equal_recording_mean_gain": mean, "contributing_recordings": len(cells)})
            seed_means.append({"seed": seed, "equal_person_mean_gain": float(np.mean(means, dtype=np.float64))})
        alpha = float(np.mean([r["equal_person_mean_gain"] for r in seed_means], dtype=np.float64))
        if not np.isfinite(alpha):
            raise NumericalFailure(f"Nonfinite development Cmean {updater}")
        result[updater] = {"alpha": alpha, "recording_means": recording_means,
                           "person_seed_means": person_means, "seed_means": seed_means,
                           "excluded_recordings": excluded, "development_ids": people, "seeds": q["seeds"],
                           "target_y_accessed": False, "outer_gains_accessed": False,
                           "selection_population": "development_only_inner_OOF"}
    return result
