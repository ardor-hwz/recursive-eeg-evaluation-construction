"""The sole frozen confirmatory inference path: a 21-subject BCa mean CI."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import scipy
from scipy.stats import DegenerateDataWarning, bootstrap

from .errors import M1LiteContractError, fail


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_905
CONFIDENCE_LEVEL = 0.95
BIT_GENERATOR = "PCG64"


def _subject_vector(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (21,) or not np.isfinite(result).all():
        fail("FAIL_CLOSED_INFERENCE_INVALID", "inference requires exactly 21 finite subject interactions")
    return result


def jackknife_acceleration(values: np.ndarray) -> float:
    jackknife = np.asarray(
        [np.mean(np.delete(values, index), dtype=np.float64) for index in range(len(values))],
        dtype=np.float64,
    )
    mean_jackknife = float(np.mean(jackknife, dtype=np.float64))
    centered = mean_jackknife - jackknife
    denominator_base = float(np.sum(centered**2, dtype=np.float64))
    denominator = 6.0 * denominator_base ** 1.5
    numerator = float(np.sum(centered**3, dtype=np.float64))
    if not np.isfinite(jackknife).all() or not np.isfinite([numerator, denominator]).all() or denominator <= 0.0:
        fail("FAIL_CLOSED_INFERENCE_INVALID", "leave-one-subject-out jackknife acceleration is invalid")
    acceleration = numerator / denominator
    if not np.isfinite(acceleration):
        fail("FAIL_CLOSED_INFERENCE_INVALID", "jackknife acceleration is nonfinite")
    return float(acceleration)


def bca_mean(values: Sequence[float]) -> dict[str, object]:
    theta = _subject_vector(values)
    acceleration = jackknife_acceleration(theta)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = bootstrap(
                (theta,),
                np.mean,
                vectorized=False,
                paired=False,
                n_resamples=BOOTSTRAP_RESAMPLES,
                confidence_level=CONFIDENCE_LEVEL,
                method="BCa",
                random_state=generator,
            )
    except M1LiteContractError:
        raise
    except Exception as exc:
        fail("FAIL_CLOSED_INFERENCE_INVALID", f"SciPy BCa failed: {type(exc).__name__}")
    if any(issubclass(item.category, DegenerateDataWarning) for item in caught):
        fail("FAIL_CLOSED_INFERENCE_INVALID", "SciPy BCa emitted DegenerateDataWarning")
    distribution = np.asarray(result.bootstrap_distribution, dtype=np.float64)
    if distribution.shape != (BOOTSTRAP_RESAMPLES,) or not np.isfinite(distribution).all() or np.ptp(distribution) == 0.0:
        fail("FAIL_CLOSED_INFERENCE_INVALID", "bootstrap distribution is invalid or degenerate")
    estimate = float(np.mean(theta, dtype=np.float64))
    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)
    if not np.isfinite([estimate, lower, upper]).all() or lower > upper:
        fail("FAIL_CLOSED_INFERENCE_INVALID", "BCa estimate or limits are invalid")
    return {
        "schema_version": "bspc.m1_lite.primary_bca.v1",
        "endpoint": "mean_of_subject_Theta",
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "confidence_level": CONFIDENCE_LEVEL,
        "ci_type": "BCa",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bit_generator": BIT_GENERATOR,
        "rng_seed": BOOTSTRAP_SEED,
        "jackknife": "leave_one_subject_out",
        "jackknife_acceleration": acceleration,
        "subject_count": 21,
        "dtype": "float64",
        "scipy_version": scipy.__version__,
        "percentile_fallback_available": False,
        "wilcoxon_performed": False,
    }
