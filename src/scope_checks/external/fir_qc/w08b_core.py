"""Frozen W08B preprocessing. No target numbers, model or replay imports."""
from __future__ import annotations
from dataclasses import dataclass
import csv
import numpy as np
from scipy.signal import oaconvolve
from scipy.fft import rfft

RATE = 500
LENGTHS = {"fast": 1500, "slow": 15000}
MEMORY = 1782
DELAY = 891

@dataclass(frozen=True)
class Trial:
    participant_id: str
    session_id: str
    recording_id: str
    original_trial_index: int
    deviation_sample: int
    target_observed: bool

def projected_trials(path):
    fields = ("group_id", "session_id", "recording_id", "original_trial_index", "deviation_sample", "target_observed")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(name) for name in fields]
        for row in reader:
            person, session, recording, index, anchor, observed = (row[i] for i in indices)
            if observed not in ("True", "False"):
                raise ValueError("Invalid observation flag")
            yield Trial(person, session, recording, int(index), int(anchor), observed == "True")

def channel_indices(raw_names, sidecar_names, canonical, excluded):
    if raw_names != sidecar_names or len(set(raw_names)) != len(raw_names):
        raise ValueError("Raw/sidecar channel identity or uniqueness mismatch")
    if set(raw_names) != set(canonical) | set(excluded):
        raise ValueError("Unexpected or missing channel")
    if len(canonical) != 30 or len(set(canonical)) != 30:
        raise ValueError("Expected 30 distinct scalp channels")
    return [raw_names.index(name) for name in canonical]

def band_masks(length, bands):
    f = np.fft.rfftfreq(length, d=1.0 / RATE)
    return [(f >= b["lower_hz"]) & (f < b["upper_hz_exclusive"]) for b in bands]

def spectrum(window, bands):
    # Exact parent mean/Hann/one-sided PSD/half-open bin integration/log formula.
    x = np.asarray(window, dtype=np.float64)
    length = x.shape[1]
    n = np.arange(length, dtype=np.float64)
    h = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
    transformed = rfft((x - x.mean(axis=1, keepdims=True)) * h, axis=1)
    psd = np.abs(transformed) ** 2 / (RATE * np.sum(h ** 2))
    psd[:, 1:-1] *= 2.0
    power = np.stack([psd[:, mask].sum(axis=1) * RATE / length for mask in band_masks(length, bands)], axis=1)
    values = (0.5 * np.log(2.0 * np.pi * np.e * np.maximum(power, 1e-12))).reshape(-1)
    if values.shape != (150,) or not np.isfinite(values).all():
        raise FloatingPointError("Invalid frozen spectral result")
    return values, psd, power

def extract(recording, sample, branch, bands, coefficients, retain=False):
    length = LENGTHS[branch]
    start, end = sample - length - MEMORY, sample
    if len(coefficients) != MEMORY + 1:
        raise ValueError("Frozen coefficient length mismatch")
    q = dict(raw_start_sample=start, raw_end_sample_exclusive=end,
             required_raw_n_samples=length + MEMORY, raw_n_samples=0,
             output_start_sample=sample - length, output_end_sample_exclusive=sample,
             output_n_samples=0, max_raw_sample_used="", group_delay_samples=DELAY,
             support_valid=False, raw_integrity_valid=False, valid=False, invalid_reason="",
             raw_nonfinite_channel_indices=[], raw_flat_channel_indices=[],
             raw_high_ptp_channel_indices=[], residual_high_channel_indices=[],
             residual_flat_channel_indices=[], residual_nonfinite_channel_indices=[],
             raw_nominal_min_ptp_uv="", raw_nominal_max_ptp_uv="",
             residual_min_ptp_uv="", residual_max_ptp_uv="", qc_channel_indices=[])
    if start < 0:
        q["invalid_reason"] = "insufficient_actual_pre_event_history"
        return None, q, None
    if end > recording.shape[1] or end < 0:
        q["invalid_reason"] = "anchor_out_of_recording_bounds"
        return None, q, None
    # Only the declared finite frame is passed to convolution, never a whole recording.
    frame = np.asarray(recording[:, start:end], dtype=np.float64)
    if frame.shape != (30, length + MEMORY):
        raise AssertionError("Dependency dimensions violated")
    q.update(support_valid=True, raw_n_samples=frame.shape[1], max_raw_sample_used=end - 1)
    finite = np.isfinite(frame).all(axis=1)
    nominal = frame[:, -length:]
    ptp = np.full(30, np.nan)
    ptp[finite] = np.ptp(nominal[finite], axis=1)
    nonfinite = np.flatnonzero(~finite).tolist()
    flat = np.flatnonzero(ptp < 1e-6).tolist()
    q.update(raw_nonfinite_channel_indices=nonfinite, raw_flat_channel_indices=flat,
             raw_high_ptp_channel_indices=np.flatnonzero(ptp > 500.0).tolist(),
             raw_nominal_min_ptp_uv=float(ptp[finite].min()) if finite.any() else "",
             raw_nominal_max_ptp_uv=float(ptp[finite].max()) if finite.any() else "")
    if nonfinite or flat:
        q["invalid_reason"] = "|".join(name for name, bad in (("raw_nonfinite_dependency", nonfinite), ("raw_nominal_flat", flat)) if bad)
        q["qc_channel_indices"] = sorted(set(nonfinite + flat))
        return None, q, None
    q["raw_integrity_valid"] = True
    filtered = oaconvolve(frame, coefficients[None, :], mode="valid", axes=1)
    if filtered.shape != (30, length):
        raise AssertionError("Valid convolution dimensions violated")
    q["output_n_samples"] = length
    finite = np.isfinite(filtered).all(axis=1)
    ptp = np.full(30, np.nan)
    ptp[finite] = np.ptp(filtered[finite], axis=1)
    nonfinite = np.flatnonzero(~finite).tolist()
    high = np.flatnonzero(ptp > 500.0).tolist()
    flat = np.flatnonzero(ptp < 1e-6).tolist()
    q.update(residual_nonfinite_channel_indices=nonfinite, residual_high_channel_indices=high,
             residual_flat_channel_indices=flat, qc_channel_indices=sorted(set(nonfinite + high + flat)),
             residual_min_ptp_uv=float(ptp[finite].min()) if finite.any() else "",
             residual_max_ptp_uv=float(ptp[finite].max()) if finite.any() else "")
    reasons = [name for name, bad in (("residual_nonfinite", nonfinite), ("residual_ptp_gt_500_uv", high), ("residual_ptp_lt_1e-6_uv", flat)) if bad]
    evidence = {"filtered": filtered} if retain else None
    if reasons:
        q["invalid_reason"] = "|".join(reasons)
        return None, q, evidence
    values, psd, power = spectrum(filtered, bands)
    q["valid"] = True
    if retain:
        evidence.update(psd=psd, band_power=power, features=values)
    return values, q, evidence

def structural_segments(trials, feature_valid):
    """Rejected feature rows, recording changes and original-index gaps split segments.

    Target observation does not define an EEG segment. No invalid-row compression.
    """
    segments, active = [], []
    for i, t in enumerate(trials):
        continuous = active and trials[active[-1]].recording_id == t.recording_id and trials[active[-1]].original_trial_index + 1 == t.original_trial_index
        if active and (not feature_valid[i] or not continuous):
            segments.append(active)
            active = []
        if feature_valid[i]:
            active.append(i)
    if active:
        segments.append(active)
    return segments
