"""W09E-4 contract-driven, pure reduction and paired BCa surface.

The caller admits sealed trajectories and evaluation targets.  This module has
no file reader, fitter, comparator selector, root selector, or replay routine.
All scientific selectors come from the single supplied W09E protocol lock.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import math
import re

import numpy as np
from scipy.stats import norm


class ContractError(ValueError):
    """Identity, selector, or coverage disagreement requires a halt."""


class NumericalFailure(ContractError):
    """A numerical failure must never remove a root or participant."""


class NonEvaluable(ContractError):
    """An endpoint lacks its complete fixed participant vector."""


def _require(ok, message):
    if not ok:
        raise ContractError(message)


def _integer(value, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             name + " must be an integer, never boolean")
    return int(value)


def _finite(value, name):
    _require(not isinstance(value, (bool, np.bool_)), name + " must be numeric, never boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(name + " must be a scalar number") from exc
    if not math.isfinite(result):
        raise NumericalFailure(name + " must be finite")
    return result


def array_sha256(array):
    """Typed, shape-preserving digest, independent of container serialization."""
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def validate_contract(contract):
    """Read selectors and reject unsupported or internally inconsistent rules."""
    try:
        op = contract["operational_contract"]
        m1 = op["m1"]
        required = op["root_gates"]["common"]["required_keys"]
        ids = tuple(contract["cohort"]["group_ids"])
        stats = contract["statistics"]
        _require(ids and len(set(ids)) == len(ids) and all(isinstance(p, str) and p for p in ids),
                 "Participant IDs must be exact unique strings")
        _require(contract["cohort"]["inference_N"] == len(ids) == stats["N"] and
                 tuple(stats["expected_group_ids"]) == ids, "Fixed cohort/statistics disagreement")
        _require(stats["dtype"] == "float64" and stats["rng"] == "numpy.random.Generator(PCG64)" and
                 stats["quantile_method"] == "linear", "Unsupported statistical selector")
        _require(stats["rank_ties"] == "midrank: (count(draw<estimate)+count(draw<=estimate))/(2*B)" and
                 stats["resample_unit"] == "whole paired participant vector", "Unsupported pairing/rank selector")
        _require(stats["p_values"] is False and op["endpoint_plan"]["p_values"] is False,
                 "P-values are uncommissioned")
        count = _integer(stats["bootstrap_resamples"], "bootstrap_resamples")
        rng_seed = _integer(stats["rng_seed"], "rng_seed")
        confidence = _finite(stats["confidence_level"], "confidence_level")
        _require(count >= 2 and 0 < confidence < 1 and rng_seed >= 0, "Invalid bootstrap settings")
        seeds = tuple(_integer(s, "final seed") for s in required["seeds"])
        signs = tuple(_integer(s, "sign") for s in required["signs"])
        _require(seeds and len(set(seeds)) == len(seeds) and signs == (-1, 1), "Unsupported seed/sign coverage")
        _require(tuple(contract["split"]["final_seeds"]) == seeds and
                 tuple(contract["perturbation"]["signs"]) == signs, "Inherited seed/sign disagreement")
        views = tuple(required["views"])
        shapes = tuple(required["shapes"])
        updaters = tuple(required["updaters"])
        _require(views == ("fixed", "dose_normalized", "equal_total_dose") and
                 shapes == ("transient", "sustained") and
                 updaters == ("heuristic", "adaptive_kalman", "C0"), "Unsupported M1 branch selectors")
        area = m1["area"]
        offset_start = _integer(area["offset_start"], "area offset_start")
        horizon = _integer(area["offset_stop_exclusive"], "area offset_stop_exclusive")
        denominator = _finite(area["denominator"], "area denominator")
        _require(offset_start == 0 and horizon == op["root_gates"]["structural"]["horizon_onset_inclusive_indices"] ==
                 contract["perturbation"]["common_horizon_indices"] and horizon > 0,
                 "Area must use the full onset-inclusive structural horizon")
        _require(denominator == contract["perturbation"]["amplitude"] > 0 and
                 m1["equal_total_dose"]["common_area_denominator"] == denominator and
                 area["absolute_before_sum"] is True and area["abs_before_sign_average"] is True,
                 "All views require the common denominator and absolute-before-sum/sign")
        dose = m1["dose_normalized"]
        divisor = _integer(dose["primitive_sustained_divisor"], "primitive dose divisor")
        _require(divisor == contract["perturbation"]["sustained_length_indices"] > 0 and
                 dose["application_level"] == "primitive_component_before_reduction" and
                 dose["transient_unchanged"] is True and dose["requires_replay"] is False and
                 dose["scale_final_theta_or_CI"] is False, "Unsupported dose-normalized application")
        _require(m1["reducer"] == ["sign", "seed", "root", "recording", "participant"] and
                 m1["weights"] == {"sign": "arithmetic_mean", "seed": "arithmetic_mean",
                    "root": "arithmetic_mean_within_recording", "recording": "equal_recording_mean_within_participant",
                    "participant": "equal_mean_of_exact_19"}, "Unsupported ordered M1 reducer")
        _require(m1["theta"]["coefficients"] == {"updater_sustained": 1, "updater_transient": -1,
                    "C0_sustained": -1, "C0_transient": 1} and
                 m1["D"]["paired_components"] == {"equal_total_dose": 1, "fixed": -1} and
                 m1["D"]["CI_from_direct_participant_vector"] is True, "Unsupported Theta/D definition")
        plan = op["endpoint_plan"]
        _require(tuple(plan["M1_updaters"]) == updaters[:-1] and tuple(plan["M1_views"]) == views and
                 plan["direct_contrasts"] == ["equal_minus_fixed"] and
                 plan["auxiliary_metrics"] == ["y_RMSE", "MAFD"] and
                 plan["auxiliary_contrasts"] == ["updater_minus_C0", "updater_specific_Cmean_minus_C0"],
                 "Unsupported commissioned endpoint plan")
        warmup = _integer(op["calibration"]["C0"]["postwarmup_indices_start"], "warmup")
        _require(contract["exclusions"]["warmup"] ==
                 f"first {warmup} valid consecutive input updates of each segment retained in trace but excluded from RMSE/MAFD/root eligibility",
                 "Warmup selectors disagree")
        support = contract["exclusions"]["recording_support"]
        accuracy_match = re.search(r"for accuracy require >=([0-9]+) post-warm-up paired target rows", support)
        mafd_match = re.search(r"for MAFD >=([0-9]+) consecutive post-warm-up pair", support)
        _require(accuracy_match is not None and mafd_match is not None, "Endpoint support selectors missing")
        _require(contract["metrics"]["rmse_reducer"] ==
                 "concatenate common valid post-warm-up target rows of all recordings per person/seed; compute RMSE, then mean seed cells; disclose recording-length weighting" and
                 contract["metrics"]["variation"] ==
                 "MAFD=sum(abs(adjacent_state_diff))/number_of_consecutive_pairs within recording, pooling supported segment pairs without crossing a boundary; median recordings within person/seed, then median seeds",
                 "Unsupported auxiliary reducer selector")
        return {"ids": ids, "N": len(ids), "seeds": seeds, "signs": signs, "updaters": updaters,
                "views": views, "shapes": shapes, "horizon": horizon, "denominator": denominator,
                "dose_divisor": divisor, "warmup": warmup, "min_accuracy_rows": int(accuracy_match.group(1)),
                "min_mafd_pairs": int(mafd_match.group(1)), "bootstrap_resamples": count,
                "rng_seed": rng_seed, "confidence_level": confidence, "plan": plan, "m1": m1}
    except KeyError as exc:
        raise ContractError(f"Missing contract field: {exc}") from exc


def primitive_areas(clean, perturbed, onset, shape, view, contract):
    q = validate_contract(contract)
    start = _integer(onset, "onset")
    _require(shape in q["shapes"] and view in q["views"] and start >= 0, "Unsupported primitive shape/view/onset")
    a, b = np.asarray(clean, dtype=np.float64), np.asarray(perturbed, dtype=np.float64)
    _require(a.ndim == b.ndim == 1 and a.shape == b.shape and len(a) >= start + q["horizon"],
             "Primitive requires a complete aligned onset-inclusive horizon")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise NumericalFailure("Nonfinite trajectory; no root deletion")
    with np.errstate(over="raise", invalid="raise"):
        try:
            raw = float(np.sum(np.abs(b[start:start+q["horizon"]] - a[start:start+q["horizon"]]),
                               dtype=np.float64) / q["denominator"])
        except FloatingPointError as exc:
            raise NumericalFailure("Primitive area overflow") from exc
    area = raw / q["dose_divisor"] if view == "dose_normalized" and shape == "sustained" else raw
    return {"area_raw": _finite(raw, "raw primitive area"), "area": _finite(area, "primitive area")}


def _root_key(row):
    try:
        key = tuple(row[k] for k in ("participant_id", "recording_id", "segment_id", "root_id"))
    except KeyError as exc:
        raise ContractError(f"Missing root identity field: {exc}") from exc
    _require(all(isinstance(s, str) and s for s in key), "Root identities must be exact nonempty strings")
    try:
        token = json.loads(key[-1])
    except (TypeError, ValueError) as exc:
        raise ContractError("Root ID must preserve composite original identity") from exc
    _require(isinstance(token, list) and len(token) == 3 and token[0] == key[1] and token[1] == key[2] and
             type(token[2]) is int, "Root ID composite disagrees with recording/segment/original onset")
    _require(json.dumps(token, separators=(",", ":")) == key[-1], "Root ID must use canonical compact JSON")
    if "onset_original_trial_index" in row:
        _require(_integer(row["onset_original_trial_index"], "original onset") == token[2],
                 "Original onset disagrees with composite root ID")
    return key


def validate_participant_vector(rows, contract, value_key="value"):
    q = validate_contract(contract)
    indexed = {}
    for row in rows:
        try:
            person, value = row["participant_id"], row[value_key]
        except KeyError as exc:
            raise ContractError(f"Missing participant endpoint field: {exc}") from exc
        _require(person in q["ids"] and person not in indexed, "Duplicate or unexpected participant endpoint")
        indexed[person] = _finite(value, "participant endpoint")
    if set(indexed) != set(q["ids"]):
        raise NonEvaluable("NON_EVALUABLE_COMPLETE_19: participant vector incomplete; no shrink")
    return np.asarray([indexed[p] for p in q["ids"]], dtype=np.float64)


def paired_vector(named_components, coefficients, contract):
    _require(set(named_components) == set(coefficients), "Paired contrast component mismatch")
    columns = [validate_participant_vector(named_components[k], contract) for k in coefficients]
    coeffs = [_finite(coefficients[k], "contrast coefficient") for k in coefficients]
    with np.errstate(over="raise", invalid="raise"):
        try:
            vector = np.sum(np.stack([column * coeff for column, coeff in zip(columns, coeffs)]),
                            axis=0, dtype=np.float64)
        except FloatingPointError as exc:
            raise NumericalFailure("Paired contrast overflow") from exc
    if not np.isfinite(vector).all():
        raise NumericalFailure("Nonfinite paired participant contrast")
    return vector


def _mean(values, name):
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = float(np.mean(values, dtype=np.float64))
        except FloatingPointError as exc:
            raise NumericalFailure(name + " overflow") from exc
    return _finite(result, name)


def reduce_m1_rows(rows, frozenroots, contract):
    """Strict 108-cell coverage per frozen root, followed by ordered means.

    Input order is irrelevant: root order comes only from the frozen ledger;
    signs and seeds come only from the contract.  No outcome selects support.
    """
    q = validate_contract(contract)
    frozen = []
    seenroots = set()
    owners = {}
    for row in frozenroots:
        key = _root_key(row)
        _require(key[0] in q["ids"] and key not in seenroots, "Duplicate/unexpected frozen root")
        _require(owners.setdefault(key[1], key[0]) == key[0], "Frozen recording ownership conflict")
        seenroots.add(key)
        frozen.append(key)
    if {key[0] for key in frozen} != set(q["ids"]):
        raise NonEvaluable("NON_EVALUABLE_COMPLETE_19: a fixed participant has no frozen common root")
    branchkeys = tuple(itertools.product(q["updaters"], q["views"], q["shapes"]))
    cells = {}
    for row in rows:
        root = _root_key(row)
        _require(root in seenroots, "Primitive root differs from frozen common-root membership")
        try:
            branch = row["updater"], row["view"], row["shape"]
            seed, sign = _integer(row["seed"], "seed"), _integer(row["sign"], "sign")
            value = _finite(row["area"], "primitive area")
        except KeyError as exc:
            raise ContractError(f"Missing primitive field: {exc}") from exc
        _require(branch in branchkeys and seed in q["seeds"] and sign in q["signs"],
                 "Unexpected primitive updater/view/shape/seed/sign")
        _require(value >= 0, "Primitive area cannot be negative")
        key = root, branch, seed, sign
        _require(key not in cells, "Duplicate primitive cell")
        cells[key] = value
    expected_count = len(frozen) * len(branchkeys) * len(q["seeds"]) * len(q["signs"])
    _require(len(cells) == expected_count, "Incomplete frozen root/branch/sign/seed primitive coverage")
    root_rows, recording_rows, participant_rows, populations = [], [], [], []
    person_components = {}
    for updater, view, shape in branchkeys:
        byrecord = defaultdict(list)
        record_order = []
        for root in frozen:
            seedmeans = []
            for seed in q["seeds"]:
                keylist = [(root, (updater, view, shape), seed, sign) for sign in q["signs"]]
                _require(all(key in cells for key in keylist), "Incomplete sign/seed coverage at frozen root")
                seedmeans.append(_mean([cells[key] for key in keylist], "root sign mean"))
            area = _mean(seedmeans, "root seed mean")
            identity = dict(zip(("participant_id", "recording_id", "segment_id", "root_id"), root))
            root_rows.append({**identity, "updater": updater, "view": view, "shape": shape, "area": area})
            rec = root[0], root[1]
            if rec not in byrecord:
                record_order.append(rec)
            byrecord[rec].append(area)
        bypeople = defaultdict(list)
        for person, recording in record_order:
            values = byrecord[person, recording]
            area = _mean(values, "recording root mean")
            recording_rows.append({"participant_id": person, "recording_id": recording,
                                   "updater": updater, "view": view, "shape": shape,
                                   "area": area, "n_roots": len(values)})
            bypeople[person].append(area)
        for person in q["ids"]:
            if not bypeople[person]:
                raise NonEvaluable("NON_EVALUABLE_COMPLETE_19: no contributing recording")
            area = _mean(bypeople[person], "participant equal-recording mean")
            person_components[person, updater, view, shape] = area
            participant_rows.append({"participant_id": person, "updater": updater, "view": view,
                                     "shape": shape, "area": area, "n_recordings": len(bypeople[person])})
        populations.append({"updater": updater, "view": view, "shape": shape,
                            "area": _mean([person_components[p, updater, view, shape] for p in q["ids"]],
                                          "component equal-participant mean"), "N": q["N"]})
    endpoint_rows = []
    theta_vectors = {}
    for updater in q["plan"]["M1_updaters"]:
        for view in q["plan"]["M1_views"]:
            components = {}
            for name, branch, shape in (("updater_sustained", updater, "sustained"),
                                       ("updater_transient", updater, "transient"),
                                       ("C0_sustained", "C0", "sustained"),
                                       ("C0_transient", "C0", "transient")):
                components[name] = [{"participant_id": p, "value": person_components[p, branch, view, shape]}
                                    for p in q["ids"]]
            vector = paired_vector(components, q["m1"]["theta"]["coefficients"], contract)
            theta_vectors[updater, view] = vector
            for person, value in zip(q["ids"], vector):
                endpoint_rows.append({"participant_id": person, "endpoint_id": f"Theta|{updater}|{view}",
                                      "endpoint_family": "Theta", "updater": updater, "view": view,
                                      "value": float(value)})
        components = {view: [{"participant_id": p, "value": float(value)}
                             for p, value in zip(q["ids"], theta_vectors[updater, view])]
                      for view in q["m1"]["D"]["paired_components"]}
        vector = paired_vector(components, q["m1"]["D"]["paired_components"], contract)
        for person, value in zip(q["ids"], vector):
            endpoint_rows.append({"participant_id": person, "endpoint_id": f"D|{updater}|equal_minus_fixed",
                                  "endpoint_family": "D", "updater": updater, "view": "equal_minus_fixed",
                                  "value": float(value)})
    return {"root_rows": root_rows, "recording_rows": recording_rows, "participant_rows": participant_rows,
            "component_population_rows": populations, "endpoint_rows": endpoint_rows,
            "n_primitive_cells": len(cells), "n_frozen_roots": len(frozen), "N": q["N"]}


def bootstrap_indices(contract):
    q = validate_contract(contract)
    rng = np.random.Generator(np.random.PCG64(q["rng_seed"]))
    return rng.integers(0, q["N"], size=(q["bootstrap_resamples"], q["N"]), dtype=np.int64)


def bca(vector, contract, indices=None):
    """Mean of the fixed paired vector and its pointwise BCa interval.

    Audit arrays are always retained; unavailable intervals retain a finite
    estimate and an exact reason. There is no percentile fallback.
    """
    q = validate_contract(contract)
    x = np.asarray(vector, dtype=np.float64)
    _require(x.shape == (q["N"],), "BCa needs exactly the fixed participant count")
    if not np.isfinite(x).all():
        raise NumericalFailure("BCa participant vector must be finite; no deletion")
    draws = bootstrap_indices(contract) if indices is None else np.asarray(indices)
    _require(draws.dtype == np.dtype("int64") and draws.shape == (q["bootstrap_resamples"], q["N"]) and
             np.all((draws >= 0) & (draws < q["N"])), "Malformed whole-participant bootstrap indices")
    _require(np.array_equal(draws, bootstrap_indices(contract)), "Bootstrap indices differ from frozen PCG64 stream")
    estimate = _mean(x, "paired endpoint estimate")
    with np.errstate(over="ignore", invalid="ignore"):
        jack = np.asarray([np.mean(np.delete(x, i), dtype=np.float64) for i in range(q["N"])], dtype=np.float64)
        boot = np.mean(x[draws], axis=1, dtype=np.float64)
        centered = np.mean(jack, dtype=np.float64) - jack
        denominator = 6 * np.sum(centered**2, dtype=np.float64)**1.5
    base = {"N": q["N"], "estimate": estimate, "percentile_fallback": False,
            "confidence_level": q["confidence_level"], "interval": contract["statistics"]["interval"],
            "bootstrap_resamples": q["bootstrap_resamples"], "rng_seed": q["rng_seed"],
            "draw_shape": list(draws.shape), "jackknife_size": q["N"], "quantile_method": "linear",
            "bootstrap_index_sha256": array_sha256(draws), "participant_vector_sha256": array_sha256(x),
            "bootstrap_means_sha256": array_sha256(boot), "jackknife_sha256": array_sha256(jack),
            "bootstrap_means": boot, "jackknife": jack, "acceleration": None,
            "bias_rank_midrank": None, "z0": None, "adjusted_probabilities": None,
            "p_value": None, "pointwise_only": True}
    def unavailable(reason):
        return {**base, "interval_status": "UNAVAILABLE", "reason": reason, "lower": None, "upper": None}
    if denominator <= 0 or not math.isfinite(float(denominator)):
        return unavailable("degenerate jackknife acceleration")
    with np.errstate(over="ignore", invalid="ignore"):
        acceleration = float(np.sum(centered**3, dtype=np.float64) / denominator)
    if not math.isfinite(acceleration):
        return unavailable("nonfinite jackknife acceleration")
    base["acceleration"] = acceleration
    if not np.isfinite(boot).all():
        return unavailable("nonfinite bootstrap distribution")
    less = int(np.count_nonzero(boot < estimate))
    leq = int(np.count_nonzero(boot <= estimate))
    rank = (less + leq) / (2 * q["bootstrap_resamples"])
    base.update({"bias_rank_midrank": rank, "bootstrap_count_less": less, "bootstrap_count_less_equal": leq})
    if np.ptp(boot) == 0 or not 0 < rank < 1:
        return unavailable("degenerate bootstrap/bias correction")
    z0 = float(norm.ppf(rank))
    base["z0"] = z0
    tail = (1-q["confidence_level"]) / 2
    za = norm.ppf([tail, 1-tail])
    mapping_denominator = 1-acceleration*(z0+za)
    base["probability_mapping_denominator"] = mapping_denominator.tolist()
    if not np.isfinite(mapping_denominator).all() or np.any(mapping_denominator <= 0):
        return unavailable("singular BCa probability mapping")
    probabilities = norm.cdf(z0+(z0+za)/mapping_denominator)
    if not np.isfinite(probabilities).all() or not 0 < probabilities[0] < probabilities[1] < 1:
        return unavailable("invalid adjusted probabilities")
    base["adjusted_probabilities"] = probabilities.tolist()
    lower, upper = np.quantile(boot, probabilities, method=contract["statistics"]["quantile_method"])
    if not np.isfinite([lower, upper]).all() or lower > upper:
        return unavailable("invalid BCa limits")
    return {**base, "interval_status": "AVAILABLE", "reason": None, "lower": float(lower), "upper": float(upper)}


def auxiliary_metrics(entries, contract, expected_recordings=None):
    """Evaluate all clean context branches on identical endpoint-specific rows.

    Original consecutive indices and per-segment warmup are mandatory. Missing
    targets affect RMSE support only, never MAFD, input resets, or M1 membership.
    """
    q = validate_contract(contract)
    branches = tuple(q["plan"]["M1_updaters"]) + ("C0",) + tuple("Cmean_"+u for u in q["plan"]["M1_updaters"])
    normalized, seen = [], set()
    owners, rec_order = {}, []
    for row in expected_recordings or []:
        person, rec = row["participant_id"], row["recording_id"]
        _require(person in q["ids"] and isinstance(rec, str) and rec and rec not in owners,
                 "Duplicate/unexpected expected auxiliary recording")
        owners[rec] = person
        rec_order.append((person, rec))
    known_expected = set(owners) if expected_recordings is not None else None
    bysegment = defaultdict(dict)
    for e in entries:
        try:
            p, r, segment = e["participant_id"], e["recording_id"], e["segment_id"]
            seed = _integer(e["seed"], "auxiliary seed")
            identity = p, r, segment, seed
            _require(p in q["ids"] and seed in q["seeds"] and identity not in seen and
                     isinstance(r, str) and r and isinstance(segment, str) and segment,
                     "Duplicate/unexpected auxiliary segment/seed identity")
            _require(known_expected is None or r in known_expected, "Unexpected auxiliary recording")
            if r not in owners:
                owners[r] = p
                rec_order.append((p, r))
            _require(owners[r] == p, "Auxiliary recording ownership conflict")
            indices = np.asarray(e["original_indices"])
            _require(indices.ndim == 1 and len(indices) > 0 and indices.dtype.kind in "iu" and
                     np.all(np.diff(indices.astype(np.int64)) == 1),
                     "Auxiliary segment must preserve original consecutive trial indices")
            target = np.asarray(e["target_y"], dtype=np.float64)
            postwarm = np.asarray(e["postwarm_mask"])
            _require(target.shape == indices.shape and postwarm.shape == indices.shape and
                     postwarm.dtype == np.bool_ and
                     np.array_equal(postwarm, np.arange(len(indices)) >= q["warmup"]),
                     "Auxiliary target/mask alignment or locked warmup disagreement")
            if np.isinf(target).any():
                raise NumericalFailure("Infinite auxiliary target; only NaN marks an unobserved label")
            _require(np.all((target[np.isfinite(target)] > 0) & (target[np.isfinite(target)] < 1)),
                     "Observed auxiliary target outside fixed strictly bounded RT representation")
            _require(set(e["states"]) == set(branches), "Auxiliary clean branch coverage mismatch")
            states = {}
            for branch in branches:
                state = np.asarray(e["states"][branch], dtype=np.float64)
                _require(state.shape == indices.shape, "Auxiliary clean state alignment mismatch")
                if not np.isfinite(state).all():
                    raise NumericalFailure("Nonfinite auxiliary clean state; no deletion")
                states[branch] = state
            pair_mask = postwarm[1:] & postwarm[:-1] & (np.diff(indices.astype(np.int64)) == 1)
            item = {"participant_id": p, "recording_id": r, "segment_id": segment, "seed": seed,
                    "original_indices": indices, "target_y": target, "states": states,
                    "paired_mask": postwarm & np.isfinite(target), "pair_mask": pair_mask}
            normalized.append(item)
            seen.add(identity)
            bysegment[p, r, segment][seed] = item
        except KeyError as exc:
            raise ContractError(f"Missing auxiliary entry field: {exc}") from exc
    for identity, aliases in bysegment.items():
        _require(set(aliases) == set(q["seeds"]), "Auxiliary segment lacks complete final seed coverage")
        first = aliases[q["seeds"][0]]
        for seed in q["seeds"][1:]:
            e = aliases[seed]
            _require(np.array_equal(first["original_indices"], e["original_indices"]) and
                     np.array_equal(first["target_y"], e["target_y"], equal_nan=True) and
                     all(np.array_equal(first["states"][b], e["states"][b]) for b in branches),
                     "Deterministic auxiliary seed aliases disagree")
    grouped = defaultdict(list)
    for e in normalized:
        grouped[e["participant_id"], e["recording_id"], e["seed"]].append(e)
    recording_rows, seed_rows, participant_rows = [], [], []
    seedvalues = {}
    unsupported = {metric: [] for metric in q["plan"]["auxiliary_metrics"]}
    for metric in q["plan"]["auxiliary_metrics"]:
        for seed in q["seeds"]:
            accum = defaultdict(list)
            contributing = defaultdict(list)
            for person, recording in rec_order:
                segments = grouped[person, recording, seed]
                row_count = sum(int(e["paired_mask"].sum()) for e in segments)
                pair_count = sum(int(e["pair_mask"].sum()) for e in segments)
                support = row_count >= q["min_accuracy_rows"] if metric == "y_RMSE" else pair_count >= q["min_mafd_pairs"]
                reason = None if support else ("INSUFFICIENT_PAIRED_POSTWARM_TARGET_ROWS" if metric == "y_RMSE"
                                                else "INSUFFICIENT_CONSECUTIVE_POSTWARM_PAIRS")
                for branch in branches:
                    value = None
                    if support:
                        if metric == "y_RMSE":
                            errors = np.concatenate([e["states"][branch][e["paired_mask"]]-e["target_y"][e["paired_mask"]]
                                                     for e in segments])
                            with np.errstate(over="raise", invalid="raise"):
                                try:
                                    value = _finite(np.sqrt(np.mean(errors**2, dtype=np.float64)), "recording y-RMSE")
                                except FloatingPointError as exc:
                                    raise NumericalFailure("Auxiliary y-RMSE overflow") from exc
                            accum[person, branch].append(errors)
                        else:
                            deltas = np.concatenate([np.abs(np.diff(e["states"][branch]))[e["pair_mask"]] for e in segments])
                            value = _mean(deltas, "recording MAFD")
                            accum[person, branch].append(value)
                        contributing[person, branch].append(recording)
                    recording_rows.append({"participant_id": person, "recording_id": recording, "seed": seed,
                                           "metric": metric, "branch": branch, "value": value,
                                           "paired_rows": row_count, "pair_count": pair_count,
                                           "support_eligible": bool(support), "reason": reason})
            for person in q["ids"]:
                for branch in branches:
                    data = accum[person, branch]
                    if not data:
                        unsupported[metric].append({"participant_id": person, "seed": seed, "branch": branch,
                                                    "reason": "NO_SUPPORTED_RECORDING_FOR_PARTICIPANT"})
                        seed_rows.append({"participant_id": person, "seed": seed, "metric": metric,
                                          "branch": branch, "value": None, "n_recordings": 0})
                        continue
                    if metric == "y_RMSE":
                        errors = np.concatenate(data)
                        with np.errstate(over="raise", invalid="raise"):
                            try:
                                value = _finite(np.sqrt(np.mean(errors**2, dtype=np.float64)), "participant seed y-RMSE")
                            except FloatingPointError as exc:
                                raise NumericalFailure("Auxiliary participant y-RMSE overflow") from exc
                        denominator = len(errors)
                    else:
                        value = _finite(np.median(data), "participant seed recording-median MAFD")
                        denominator = len(data)
                    seedvalues[metric, person, branch, seed] = value
                    seed_rows.append({"participant_id": person, "seed": seed, "metric": metric, "branch": branch,
                                      "value": value, "n_recordings": len(contributing[person, branch]),
                                      "denominator": denominator})
        if not unsupported[metric]:
            for person in q["ids"]:
                for branch in branches:
                    values = [seedvalues[metric, person, branch, seed] for seed in q["seeds"]]
                    value = _mean(values, "participant seed-mean y-RMSE") if metric == "y_RMSE" else _finite(np.median(values), "participant seed-median MAFD")
                    participant_rows.append({"participant_id": person, "metric": metric, "branch": branch, "value": value})
    indexed = {(row["metric"], row["branch"], row["participant_id"]): row["value"] for row in participant_rows}
    endpoint_rows, endpoint_status = [], []
    for metric in q["plan"]["auxiliary_metrics"]:
        for updater in q["plan"]["M1_updaters"]:
            for contrast in q["plan"]["auxiliary_contrasts"]:
                eid = f"Aux|{metric}|{updater}|{contrast}"
                endpoint_status.append({"endpoint_id": eid, "metric": metric, "updater": updater,
                                        "contrast": contrast, "status": "NON_EVALUABLE" if unsupported[metric] else "EVALUABLE",
                                        "planned_N": q["N"], "N": None if unsupported[metric] else q["N"],
                                        "unsupported_cells": unsupported[metric]})
                if unsupported[metric]:
                    continue
                branch = updater if contrast == "updater_minus_C0" else "Cmean_"+updater
                components = {name: [{"participant_id": p, "value": indexed[metric, source, p]} for p in q["ids"]]
                              for name, source in (("active", branch), ("C0", "C0"))}
                vector = paired_vector(components, {"active": 1, "C0": -1}, contract)
                for person, value in zip(q["ids"], vector):
                    endpoint_rows.append({"participant_id": person, "endpoint_id": eid, "endpoint_family": "Auxiliary",
                                          "metric": metric, "updater": updater, "contrast": contrast, "value": float(value)})
    return {"recording_rows": recording_rows, "person_seed_rows": seed_rows,
            "participant_rows": participant_rows, "endpoint_rows": endpoint_rows,
            "endpoint_status": endpoint_status, "unsupported": unsupported,
            "N": q["N"], "branches": list(branches), "n_segments": len(bysegment),
            "n_recordings": len(rec_order), "deterministic_seed_aliases": True}
