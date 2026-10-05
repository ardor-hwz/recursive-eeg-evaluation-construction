"""Frozen estimation-only 21-subject BCa inference; no hypothesis-test code."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import scipy
from scipy.stats import DegenerateDataWarning, bootstrap

from .errors import C0ContractError, fail


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_830
CONFIDENCE_LEVEL = 0.95


def _finite_subject_vector(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (21,) or not np.isfinite(result).all():
        fail("FAIL_CLOSED_INFERENCE_INVALID", "inference requires exactly 21 finite paired subject contrasts")
    return result


def _validate_jackknife_acceleration(values: np.ndarray) -> None:
    jackknife = np.asarray(
        [np.mean(np.delete(values, index), dtype=np.float64) for index in range(len(values))],
        dtype=np.float64,
    )
    centered = float(np.mean(jackknife, dtype=np.float64)) - jackknife
    denominator_base = float(np.sum(centered**2, dtype=np.float64))
    if not np.isfinite(jackknife).all() or not np.isfinite(denominator_base) or denominator_base <= 0.0:
        fail("FAIL_CLOSED_INFERENCE_INVALID", "leave-one-subject-out jackknife acceleration is invalid")


def bca_mean(values: Sequence[float]) -> dict[str, object]:
    differences = _finite_subject_vector(values)
    _validate_jackknife_acceleration(differences)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = bootstrap(
                (differences,),
                np.mean,
                vectorized=False,
                paired=False,
                n_resamples=BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                method="BCa",
                random_state=generator,
            )
    except C0ContractError:
        raise
    except Exception as exc:
        fail("FAIL_CLOSED_INFERENCE_INVALID", f"SciPy BCa failed: {type(exc).__name__}")
    if any(issubclass(item.category, DegenerateDataWarning) for item in caught):
        fail("FAIL_CLOSED_INFERENCE_INVALID", "SciPy BCa emitted DegenerateDataWarning")
    distribution = np.asarray(result.bootstrap_distribution, dtype=np.float64)
    if not np.isfinite(distribution).all() or np.ptp(distribution) == 0.0:
        fail("FAIL_CLOSED_INFERENCE_INVALID", "bootstrap mean distribution is invalid or degenerate")
    estimate = float(np.mean(differences, dtype=np.float64))
    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    if not np.isfinite([estimate, lower, upper]).all():
        fail("FAIL_CLOSED_INFERENCE_INVALID", "BCa estimate or limit is nonfinite")
    return {
        "schema_version": "bspc.c0.bca_estimate.v1",
        "execution_class": "SYNTHETIC_ONLY",
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "confidence_level": CONFIDENCE_LEVEL,
        "ci_type": "BCa",
        "resampling_unit": "subject_paired_contrast",
        "subject_count": 21,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "rng": "numpy.default_rng.PCG64",
        "seed": BOOTSTRAP_SEED,
        "jackknife": "leave_one_subject_out",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "p_value": None,
        "holm": None,
        "fallback_interval": None,
    }
