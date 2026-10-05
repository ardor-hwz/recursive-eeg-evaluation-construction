"""Frozen event endpoints and reducer hierarchy."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contract import ANALYSIS_LENGTH, CONTROLLERS, DELTA0, SEEDS, SHAPES, SIGNS, SUBJECTS
from .errors import fail


def primary_a(clean_state: np.ndarray, perturbed_state: np.ndarray, t0: int) -> float:
    clean = np.asarray(clean_state, dtype=np.float64)
    perturbed = np.asarray(perturbed_state, dtype=np.float64)
    end = t0 + ANALYSIS_LENGTH
    if clean.shape != perturbed.shape or clean.ndim != 1 or end > len(clean):
        fail("FAIL_CLOSED_ENDPOINT_INVALID", "primary endpoint arrays/window are invalid")
    delta_z = perturbed[t0:end] - clean[t0:end]
    if not np.isfinite(delta_z).all():
        fail("FAIL_CLOSED_ENDPOINT_INVALID", "primary state deflection is nonfinite")
    value = float(np.sum(np.abs(delta_z), dtype=np.float64) / DELTA0)
    if not np.isfinite(value):
        fail("FAIL_CLOSED_ENDPOINT_INVALID", "primary endpoint is nonfinite")
    return value


def supportive_b_late(clean_state: np.ndarray, perturbed_state: np.ndarray, t0: int, sign: int) -> float:
    if sign not in SIGNS:
        fail("FAIL_CLOSED_ENDPOINT_INVALID", "supportive endpoint sign is invalid")
    clean = np.asarray(clean_state, dtype=np.float64)
    perturbed = np.asarray(perturbed_state, dtype=np.float64)
    if clean.shape != perturbed.shape or clean.ndim != 1 or t0 + 20 > len(clean):
        fail("FAIL_CLOSED_ENDPOINT_INVALID", "supportive endpoint arrays/window are invalid")
    delta_z = perturbed[t0 + 15 : t0 + 20] - clean[t0 + 15 : t0 + 20]
    value = float(np.sum(float(sign) * delta_z, dtype=np.float64) / (5.0 * DELTA0))
    if not np.isfinite(value):
        fail("FAIL_CLOSED_ENDPOINT_INVALID", "supportive endpoint is nonfinite")
    return value


LEDGER_KEYS = ["outer_subject_id", "session_id", "t0", "seed", "sign", "controller", "shape"]


def reduce_event_ledger(
    ledger: pd.DataFrame,
    event_roots: Iterable[tuple[int, int, int]],
    *,
    expected_subjects: tuple[int, ...] = SUBJECTS,
    expected_seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    required = [*LEDGER_KEYS, "A", "B_late", "primary_classification", "supportive_classification"]
    if not isinstance(ledger, pd.DataFrame) or set(ledger.columns) != set(required):
        fail("FAIL_CLOSED_REDUCER_INVALID", "event ledger column registry changed")
    if ledger.empty or ledger.duplicated(LEDGER_KEYS).any():
        fail("FAIL_CLOSED_REDUCER_INVALID", "event ledger is empty or has duplicate keys")
    if not np.isfinite(ledger["A"].to_numpy(dtype=np.float64)).all():
        fail("FAIL_CLOSED_REDUCER_INVALID", "primary ledger contains nonfinite values")
    roots = sorted(set(tuple(map(int, root)) for root in event_roots))
    expected_keys = {
        (subject, session, t0, seed, sign, controller, shape)
        for subject, session, t0 in roots
        for seed in expected_seeds
        for sign in SIGNS
        for controller in CONTROLLERS
        for shape in SHAPES
    }
    observed_keys = set(
        tuple(value) for value in ledger.loc[:, LEDGER_KEYS].itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        fail("FAIL_CLOSED_REDUCER_INVALID", "event/sign/seed/controller/shape coverage mismatch")
    if set(ledger["primary_classification"]) != {"PRIMARY_CONFIRMATORY_COMPONENT"}:
        fail("FAIL_CLOSED_REDUCER_INVALID", "endpoint classification registry changed")
    if set(ledger["supportive_classification"]) != {"SUPPORTIVE_DESCRIPTIVE_ONLY"}:
        fail("FAIL_CLOSED_REDUCER_INVALID", "supportive classification registry changed")

    sign_mean = (
        ledger.groupby(["outer_subject_id", "session_id", "t0", "seed", "controller", "shape"], sort=True, as_index=False)["A"]
        .mean()
    )
    seed_mean = (
        sign_mean.groupby(["outer_subject_id", "session_id", "t0", "controller", "shape"], sort=True, as_index=False)["A"]
        .mean()
    )
    event_mean = (
        seed_mean.groupby(["outer_subject_id", "session_id", "controller", "shape"], sort=True, as_index=False)["A"]
        .mean()
    )
    subject_mean = (
        event_mean.groupby(["outer_subject_id", "controller", "shape"], sort=True, as_index=False)["A"]
        .mean()
    )
    pivot = subject_mean.pivot(index="outer_subject_id", columns=["controller", "shape"], values="A")
    if tuple(int(value) for value in pivot.index) != expected_subjects:
        fail("FAIL_CLOSED_REDUCER_INVALID", "reducer must yield exactly one row per expected subject")
    expected_columns = {(controller, shape) for controller in CONTROLLERS for shape in SHAPES}
    if set(pivot.columns) != expected_columns:
        fail("FAIL_CLOSED_REDUCER_INVALID", "subject controller/shape aggregate is incomplete")
    theta = (
        (pivot[("Full", "sustained")] - pivot[("Full", "transient")])
        - (pivot[("C0", "sustained")] - pivot[("C0", "transient")])
    ).to_numpy(dtype=np.float64)
    if theta.shape != (len(expected_subjects),) or not np.isfinite(theta).all():
        fail("FAIL_CLOSED_REDUCER_INVALID", "subject interaction vector is invalid")

    sustained = ledger.loc[ledger["shape"].eq("sustained")].copy()
    if not np.isfinite(sustained["B_late"].to_numpy(dtype=np.float64)).all():
        fail("FAIL_CLOSED_REDUCER_INVALID", "supportive ledger contains nonfinite values")
    b_sign = (
        sustained.groupby(["outer_subject_id", "session_id", "t0", "seed", "controller"], sort=True, as_index=False)["B_late"]
        .mean()
    )
    b_seed = b_sign.groupby(["outer_subject_id", "session_id", "t0", "controller"], sort=True, as_index=False)["B_late"].mean()
    b_event = b_seed.groupby(["outer_subject_id", "session_id", "controller"], sort=True, as_index=False)["B_late"].mean()
    b_subject = b_event.groupby(["outer_subject_id", "controller"], sort=True, as_index=False)["B_late"].mean()
    return {
        "subject_primary": subject_mean,
        "theta": theta,
        "subject_supportive": b_subject,
        "inferential_n": len(theta),
        "primary_classification": "CONFIRMATORY",
        "supportive_classification": "SUPPORTIVE_DESCRIPTIVE_ONLY",
    }


def classify_primary(ci_lower: float, ci_upper: float, supportive: object | None = None) -> str:
    del supportive
    if not np.isfinite([ci_lower, ci_upper]).all() or ci_lower > ci_upper:
        fail("FAIL_CLOSED_INFERENCE_INVALID", "primary confidence limits are invalid")
    return "SUPPORTED" if ci_lower > 0.0 or ci_upper < 0.0 else "NOT_SUPPORTED"
