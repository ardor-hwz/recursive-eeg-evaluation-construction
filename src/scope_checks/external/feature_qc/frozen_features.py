"""W07A fixed within-window spectral extraction. No targets or event parser."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import numpy as np
from scipy.fft import rfft

RATE = 500
LENGTHS = {"fast": 1500, "slow": 15000}
POWER_FLOOR = 1e-12  # microvolt squared, explicitly frozen in W07A


@dataclass(frozen=True)
class Trial:
    participant_id: str
    session_id: str
    recording_id: str
    original_trial_index: int
    deviation_sample: int
    target_observed: bool


def projected_trials(path):
    """Decode only identity, anchor and observed flag; never access RT columns.

    csv.reader tokenizes the sealed CSV container; unwanted fields are neither
    indexed, converted, stored nor passed to QC. group_id is participant_id.
    """
    columns = ("group_id", "session_id", "recording_id", "original_trial_index",
               "deviation_sample", "target_observed")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(name) for name in columns]
        for raw in reader:
            person, session, recording, index, sample, observed = (raw[i] for i in indices)
            if observed not in ("True", "False"):
                raise ValueError("Invalid target-observation flag")
            yield Trial(person, session, recording, int(index), int(sample), observed == "True")


def channel_indices(raw_names, sidecar_names, canonical, excluded):
    if raw_names != sidecar_names or len(set(raw_names)) != len(raw_names):
        raise ValueError("Raw/sidecar channel identity or uniqueness mismatch")
    if set(raw_names) != set(canonical) | set(excluded):
        raise ValueError("Unexpected or missing channel identity")
    if len(canonical) != 30 or len(set(canonical)) != 30:
        raise ValueError("Expected exactly 30 distinct scalp channels")
    return [raw_names.index(name) for name in canonical]


def band_masks(length, bands):
    frequencies = np.fft.rfftfreq(length, d=1.0 / RATE)
    return [(frequencies >= b["lower_hz"]) &
            (frequencies < b["upper_hz_exclusive"]) for b in bands]


def log_features(power):
    return 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(power, POWER_FLOOR))


def spectrum(window, bands):
    """float64 mean, symmetric Hann, one-sided PSD, rectangular bin sum."""
    x = np.asarray(window, dtype=np.float64)
    length = x.shape[1]
    n = np.arange(length, dtype=np.float64)
    h = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
    transformed = rfft((x - x.mean(axis=1, keepdims=True)) * h, axis=1)
    psd = np.abs(transformed) ** 2 / (RATE * np.sum(h ** 2))
    if length % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    power = np.stack([psd[:, mask].sum(axis=1) * RATE / length
                      for mask in band_masks(length, bands)], axis=1)
    values = log_features(power).reshape(-1)  # channel-major, then band
    if values.shape != (150,) or not np.isfinite(values).all():
        raise FloatingPointError("Invalid frozen spectral result")
    return values, psd, power


def extract(recording, sample, branch, bands):
    """recording is already mapped to canonical channels. Slices are [start,end)."""
    length = LENGTHS[branch]
    start, end = sample - length, sample
    info = {"start_sample": start, "end_sample_exclusive": end,
            "required_n_samples": length, "n_samples": 0,
            "max_sample_used": "", "support_valid": False, "valid": False,
            "invalid_reason": "", "qc_channel_names_indices": [],
            "nonfinite_channel_indices": [], "high_ptp_channel_indices": [],
            "flat_channel_indices": [], "min_ptp_uv": "", "max_ptp_uv": ""}
    if start < 0:
        info["invalid_reason"] = "insufficient_pre_event_support"
        return None, info, None, None
    if end > recording.shape[1] or end < 0:
        info["invalid_reason"] = "anchor_out_of_recording_bounds"
        return None, info, None, None
    window = np.asarray(recording[:, start:end], dtype=np.float64)
    if window.shape != (30, length) or end - 1 >= sample:
        raise AssertionError("Causal support violation")
    info.update(support_valid=True, n_samples=length, max_sample_used=end - 1)
    finite_channels = np.isfinite(window).all(axis=1)
    ptp = np.full(30, np.nan)
    ptp[finite_channels] = np.ptp(window[finite_channels], axis=1)
    nonfinite = np.flatnonzero(~finite_channels).tolist()
    high = np.flatnonzero(ptp > 500.0).tolist()
    flat = np.flatnonzero(ptp < 1e-6).tolist()
    reasons = []
    for name, bad in (("nonfinite_included_channel", nonfinite),
                      ("peak_to_peak_gt_500_uv", high),
                      ("peak_to_peak_lt_1e-6_uv", flat)):
        if bad:
            reasons.append(name)
    info.update(nonfinite_channel_indices=nonfinite, high_ptp_channel_indices=high,
                flat_channel_indices=flat,
                qc_channel_names_indices=sorted(set(nonfinite + high + flat)),
                min_ptp_uv=float(ptp[finite_channels].min()) if finite_channels.any() else "",
                max_ptp_uv=float(ptp[finite_channels].max()) if finite_channels.any() else "")
    if reasons:
        info["invalid_reason"] = "|".join(reasons)
        return None, info, None, None
    values, psd, power = spectrum(window, bands)
    info["valid"] = True
    return values, info, psd, power
