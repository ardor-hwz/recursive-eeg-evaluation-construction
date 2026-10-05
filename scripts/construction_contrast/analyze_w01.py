"""W01: direct paired construction contrasts from existing frozen outer trajectories.

No fitting, candidate selection, controller replay, or upstream regeneration.
Run from any directory with the project's .venv-phase3 Python interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "major_revision_experiments_20260728"
C0SRC = EXP / "c0_release_candidate_20260902_development_selection_lock_serialization_v1/src"
REVIEW = EXP / "reviewer_mechanistic_controls_20260912_v1"
CORE = EXP / "stacking_grouping_core_repair_20260912_v1"
STAGE = EXP / "boundary_stage23_execution_20260924_v1"
PROTOCOL = EXP / "boundary_stage23_contract_20260924_v1_1/STAGE23_PROTOCOL_v1_1.json"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(C0SRC)]

from c0.aggregation import aggregate_subject_metrics  # noqa: E402
from c0.inference import (  # noqa: E402
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL, bca_mean,
)
from src.data.manifest import SUBJECT_SESSIONS  # noqa: E402
from src.utils.serialization import stable_hash  # noqa: E402

KEYS = ["outer_subject_id", "seed", "session_id", "time_index"]
METHODS = ["Full", "C0", "Cmean_primary", "Cmean_pooled"]
ENDPOINTS = ["rmse", "mafd"]
SEEDS = (42, 123, 2026)
EPS = 1e-12
EXPECTED_GROUPS = set(range(1, 22))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path, **kwargs):
    # Preserve archived 17-digit float precision when recovering paired metrics.
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


class Bindings:
    """Check only consumed artifacts against their existing project receipts."""

    def __init__(self):
        self.checked = {}

    def check(self, path, expected):
        path = Path(path).resolve()
        actual = sha(path)
        require(actual == expected, f"Frozen source hash mismatch: {path}")
        self.checked[str(path)] = actual

    def from_map(self, path, mapping):
        key = str(Path(path).resolve())
        require(key in mapping, f"Existing source binding missing: {key}")
        self.check(path, mapping[key])

    def unchanged(self):
        require(all(sha(p) == h for p, h in self.checked.items()), "Consumed frozen input changed during W01")


def validate_trajectory(frame):
    needed = KEYS + ["target_perclos", "fast", "slow"] + METHODS
    require(set(needed).issubset(frame.columns), "Missing trajectory method/key column")
    require(not frame.duplicated(KEYS).any(), "Duplicate paired trajectory key")
    require(len(frame) == 23 * 3 * 885, "Unexpected outer trajectory row count")
    require(set(frame.outer_subject_id) == EXPECTED_GROUPS, "Group membership mismatch")
    require(np.isfinite(frame[needed].to_numpy(float)).all(), "Nonfinite trajectory value")
    require(((frame[needed[4:]] >= 0) & (frame[needed[4:]] <= 1)).all().all(), "Value outside [0,1]")
    for group, sessions in SUBJECT_SESSIONS.items():
        unit = frame.loc[frame.outer_subject_id.eq(group)]
        require(set(unit.session_id) == set(sessions), f"Recording mapping mismatch: group {group}")
        for session in sessions:
            record = unit.loc[unit.session_id.eq(session)]
            require(set(record.seed) == set(SEEDS), "Seed coverage mismatch")
            for seed in SEEDS:
                times = record.loc[record.seed.eq(seed), "time_index"].to_numpy()
                require(np.array_equal(np.sort(times), np.arange(885)), "Recording indices must be exactly 0..884")
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def align_frames(first, second, columns):
    require(first[KEYS].equals(second[KEYS]), "Paired keys do not align exactly")
    errors = {column: float(np.max(np.abs(first[column].to_numpy() - second[column].to_numpy())))
              for column in columns}
    require(max(errors.values(), default=0) < EPS, f"Frozen values differ: {errors}")
    return errors


def group_metrics(frame):
    rows = []
    for method in METHODS:
        metric = aggregate_subject_metrics(
            frame[KEYS + ["target_perclos", method]].rename(columns={method: "state"}),
            expected_seeds=SEEDS,
        )
        metric["method"] = method
        rows.append(metric)
    return pd.concat(rows, ignore_index=True)


def validate_metrics(frame):
    require(not frame.duplicated(["outer_subject_id", "method"]).any(), "Duplicate paired group metric")
    require(set(frame.method) == set(METHODS) and len(frame) == 84, "Incomplete paired method coverage")
    for method in METHODS:
        require(set(frame.loc[frame.method.eq(method), "outer_subject_id"]) == EXPECTED_GROUPS,
                "Paired group membership differs")
    require(np.isfinite(frame[ENDPOINTS].to_numpy()).all(), "Nonfinite paired metric")
    return frame


def metric_error(actual, expected):
    validate_metrics(expected)
    keys = ["method", "outer_subject_id"]
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    require(a[keys].equals(b[keys]), "Archived group metrics misaligned")
    err = float(np.max(np.abs(a[ENDPOINTS].to_numpy() - b[ENDPOINTS].to_numpy())))
    require(err < EPS, f"Archived group aggregation mismatch: {err}")
    return err


def signs(vector):
    x = np.asarray(vector, float)
    return {"positive": int((x > EPS).sum()), "negative": int((x < -EPS).sum()),
            "zero": int((np.abs(x) <= EPS).sum())}


def sign_code(vector):
    x = np.asarray(vector, float)
    return np.where(x > EPS, 1, np.where(x < -EPS, -1, 0))


def support(lower, upper):
    return "positive" if lower > 0 else "negative" if upper < 0 else "includes_zero"


def estimate(vector):
    ci = bca_mean(vector)
    # The imported legacy function labels itself SYNTHETIC_ONLY; do not propagate
    # that metadata to this real frozen-data secondary analysis.
    return {"mean": ci["estimate"], "lower": ci["lower"], "upper": ci["upper"],
            "support": support(ci["lower"], ci["upper"]), "signs": signs(vector)}


def summarize(metrics, layer):
    validate_metrics(metrics)
    indexed = {name: metrics.loc[metrics.method.eq(name)].set_index("outer_subject_id").sort_index()
               for name in METHODS}
    result, paired = [], []
    max_algebra, max_bootstrap = 0.0, 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 21, size=(BOOTSTRAP_RESAMPLES, 21))
    for endpoint in ENDPOINTS:
        full = indexed["Full"][endpoint].to_numpy()
        c0 = indexed["C0"][endpoint].to_numpy()
        d0 = full - c0
        e0 = estimate(d0)
        for method in ("Cmean_primary", "Cmean_pooled"):
            cm = indexed[method][endpoint].to_numpy()
            dm = full - cm
            direct = c0 - cm
            reconstructed = dm - d0
            algebra = float(np.max(np.abs(direct - reconstructed)))
            bootstrap_error = float(np.max(np.abs(
                direct[draws].mean(axis=1) - (dm[draws].mean(axis=1) - d0[draws].mean(axis=1)))))
            require(algebra < EPS and bootstrap_error < EPS, "Construction contrast algebra/pairing failed")
            max_algebra = max(max_algebra, algebra)
            max_bootstrap = max(max_bootstrap, bootstrap_error)
            em, ec = estimate(dm), estimate(direct)
            flips = int(((sign_code(d0) * sign_code(dm)) == -1).sum())
            same = int((sign_code(d0) == sign_code(dm)).sum())
            means_retained = int(sign_code([e0["mean"]])[0]) == int(sign_code([em["mean"]])[0])
            ci_retained = e0["support"] == em["support"]
            majority0 = int(np.sign(e0["signs"]["positive"] - e0["signs"]["negative"]))
            majoritym = int(np.sign(em["signs"]["positive"] - em["signs"]["negative"]))
            if not means_retained or not ci_retained or majority0 != majoritym:
                label = "construction-sensitive"
            elif ec["support"] != "includes_zero":
                label = "magnitude-sensitive"
            else:
                label = "construction-robust"
            # Descriptive labels are qualitative audits, not equivalence tests.
            row = {"analysis_layer": layer, "cmean_definition": method, "endpoint": endpoint,
                   "N": 21, "Full_mean": float(full.mean()), "C0_mean": float(c0.mean()),
                   "Cmean_mean": float(cm.mean()), "mean_direction_retained": bool(means_retained),
                   "ci_support_retained": bool(ci_retained), "group_same_sign": same,
                   "group_majority_retained": bool(majority0 == majoritym),
                   "group_opposite_sign": flips, "group_zero_transition": 21 - same - flips,
                   "classification": label, "max_algebra_error": algebra,
                   "max_shared_draw_error": bootstrap_error}
            for prefix, stats in (("Full_vs_C0", e0), ("Full_vs_Cmean", em), ("construction", ec)):
                for key in ("mean", "lower", "upper", "support"):
                    row[f"{prefix}_{key}"] = stats[key]
                for key, value in stats["signs"].items():
                    row[f"{prefix}_{key}_groups"] = value
            result.append(row)
            for i, group in enumerate(range(1, 22)):
                paired.append({"analysis_layer": layer, "cmean_definition": method, "endpoint": endpoint,
                               "group": group, "Full": full[i], "C0": c0[i], "Cmean": cm[i],
                               "Full_vs_C0": d0[i], "Full_vs_Cmean": dm[i],
                               "construction_contrast": direct[i], "reconstructed_contrast": reconstructed[i]})
    return result, paired, {"max_algebra_error": max_algebra, "max_shared_bootstrap_draw_error": max_bootstrap}


def main(output_dir=HERE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bindings = Bindings()
    original_hashes = load(REVIEW / "reports/OUTPUT_HASHES.json")
    baseline_hashes = load(REVIEW / "reports/BASELINE_BINDINGS.json")
    source_hashes = load(REVIEW / "reports/SOURCE_HASHES.json")
    for module in ("aggregation.py", "inference.py", "recursion.py"):
        bindings.from_map(C0SRC / "c0" / module, source_hashes)
    bindings.from_map(REVIEW / "src/cmean.py", source_hashes)
    bindings.from_map(ROOT / "src/models/change_aware_tracker.py", baseline_hashes)
    original_path = REVIEW / "results/cmean/outer_trajectories.csv"
    bindings.from_map(original_path, original_hashes)
    original = validate_trajectory(read_csv(original_path))
    protocol = load(PROTOCOL)
    for item in (protocol["authoritative_corrected_trajectory"], protocol["corrected_c0_trajectory"]):
        bindings.check(item["path"], item["sha256"])
    # Independently anchor primary Full and C0 to the corrected upstream ledger.
    p0 = read_csv(protocol["authoritative_corrected_trajectory"]["path"],
                  usecols=KEYS + ["target_perclos", "fast", "slow", "tracker"])
    p0 = p0.rename(columns={"tracker": "Full"}).sort_values(KEYS).reset_index(drop=True)
    anchor_errors = align_frames(original, p0, ["target_perclos", "fast", "slow", "Full"])
    c0 = read_csv(protocol["corrected_c0_trajectory"]["path"], usecols=KEYS + ["c0_state"])
    c0 = c0.rename(columns={"c0_state": "C0"}).sort_values(KEYS).reset_index(drop=True)
    anchor_errors.update(align_frames(original, c0, ["C0"]))
    original_metrics = group_metrics(original)
    metric_path = REVIEW / "results/cmean/subject_metrics.csv"
    bindings.from_map(metric_path, original_hashes)
    primary_error = metric_error(original_metrics, read_csv(metric_path))
    effects_path = REVIEW / "reports/TABLE_C_PAIRED_EFFECTS.csv"
    # OUTPUT_HASHES binds result assets; this report is bound by the existing
    # historical-comparison receipt instead (no new source hash assumption).
    historical_receipt = load(STAGE / "HISTORICAL_COMPARISON_COMPLETE.json")
    historical_map_path = STAGE / "historical_comparison_source_hashes.json"
    bindings.from_map(historical_map_path, historical_receipt["outputs"])
    bindings.from_map(effects_path, load(historical_map_path))
    original_effects = read_csv(effects_path)

    receipt = load(STAGE / "CMEAN_COMPLETE.json")
    require(receipt["counts"] == {"folds": 21, "outer_rows": 61065, "paired_effect_rows": 6},
            "Stage Cmean completion counts changed")
    bindings.check(PROTOCOL, receipt["protocol_sha256"])
    gain_receipt_path = STAGE / "CMEAN_DEVELOPMENT_LOCKS_COMPLETE.json"
    bindings.check(gain_receipt_path, receipt["inputs"]["cmean_development_receipt"])
    gains = load(gain_receipt_path)
    stage_summary_path = STAGE / "c0_cmean_summary.csv"
    bindings.from_map(stage_summary_path, receipt["outputs"])
    stage_summary = read_csv(stage_summary_path)
    parts, identities = [], []
    for fold in protocol["folds"]:
        group = fold["fold"]
        path = STAGE / "cmean/outer" / f"outer_subject_{group:02d}.csv"
        bindings.from_map(path, receipt["outputs"])
        frame = read_csv(path).rename(columns={"state": "Full", "c0_state": "C0",
                                             "cmean_primary_state": "Cmean_primary",
                                             "cmean_pooled_state": "Cmean_pooled"})
        require(set(frame.outer_subject_id) == {group}, "Stage fold identity mismatch")
        parts.append(frame)
        original_gain_path = REVIEW / "results/cmean/alpha_locks" / f"outer_{group:02d}.json"
        bindings.from_map(original_gain_path, original_hashes)
        og = load(original_gain_path)
        require(og["outer_fold"] == group and og["outer_information_used"] is False,
                "Original Cmean development boundary changed")
        bindings.check(og["development_source"], og["development_sha256"])
        folder = CORE / "results" if group in {1, 3, 5, 12, 13} else EXP / "results"
        full_lock = folder / f"outer_subject_{group:02d}/selection_lock.json"
        bindings.check(full_lock, og["full_lock_sha256"])
        bindings.from_map(full_lock, baseline_hashes)
        original_lock = load(full_lock)
        tracker = original_lock["tracker"]
        require(tracker["uses_outer"] is False, "Primary Full lock uses outer outcome")
        sg_path = STAGE / "cmean/locks" / f"outer_subject_{group:02d}.json"
        bindings.from_map(sg_path, gains["outputs"])
        sg = load(sg_path)
        require(sg["fold"] == group and sg["outer_information_used"] is False,
                "Stage Cmean development boundary changed")
        require(sg["development_source_sha256"] == og["development_sha256"] and
                sg["stage1b_config_sha256"] == fold["selected_config_sha256"] and
                stable_hash(fold["selected_parameters"]) == fold["selected_config_sha256"],
                "Stage Cmean config/proxy binding mismatch")
        sr = stage_summary.loc[stage_summary.fold.eq(group)]
        require(len(sr) == 1 and abs(float(sr.c0_alpha.iloc[0]) - og["corrected_c0_alpha"]) < EPS,
                "C0 gain changed between frozen strata")
        for layer, candidate, digest, params, primary, pooled in (
            ("primary_corrected_192", tracker["candidate_id"], stable_hash(tracker["parameters"]),
             tracker["parameters"], og["cmean_primary_alpha"], og["full_dev_pooled_mean_alpha"]),
            ("existing_posthoc_stage1b_771", fold["selected_candidate_id"], fold["selected_config_sha256"],
             fold["selected_parameters"], sg["primary_group_balanced_alpha"], sg["pooled_alpha"]),
        ):
            identities.append({"analysis_layer": layer, "group": group, "Full_candidate_id": candidate,
                               "Full_config_sha256": digest, "Full_parameters_json": json.dumps(params, sort_keys=True),
                               "fast_candidate_id": original_lock["proxy_roles"]["fast_candidate_id"],
                               "slow_candidate_id": original_lock["proxy_roles"]["slow_candidate_id"],
                               "Full_source_lock_path": str(full_lock if layer == "primary_corrected_192" else PROTOCOL),
                               "Cmean_gain_lock_path": str(original_gain_path if layer == "primary_corrected_192" else sg_path),
                               "C0_alpha": og["corrected_c0_alpha"], "Cmean_primary_alpha": primary,
                               "Cmean_pooled_alpha": pooled})
    stage = validate_trajectory(pd.concat(parts, ignore_index=True))
    common_errors = align_frames(original, stage, ["target_perclos", "fast", "slow", "C0"])
    stage_metrics = group_metrics(stage)
    archived_stage_metrics = []
    for name, prefix in zip(METHODS, ("full", "c0", "cmean_primary", "cmean_pooled")):
        item = stage_summary[["fold", f"{prefix}_rmse", f"{prefix}_mafd"]].rename(
            columns={"fold": "outer_subject_id", f"{prefix}_rmse": "rmse", f"{prefix}_mafd": "mafd"})
        item["method"] = name
        archived_stage_metrics.append(item)
    stage_error = metric_error(stage_metrics, pd.concat(archived_stage_metrics, ignore_index=True))
    stage_effect_path = STAGE / "c0_cmean_effects.csv"
    bindings.from_map(stage_effect_path, receipt["outputs"])
    stage_effects = read_csv(stage_effect_path)

    rows, paired, audits, metrics_all = [], [], {}, []
    reference_error = 0.0
    for layer, metrics, archived in (
        ("primary_corrected_192", original_metrics, original_effects),
        ("existing_posthoc_stage1b_771", stage_metrics, stage_effects),
    ):
        new_rows, new_paired, audit = summarize(metrics, layer)
        for row in new_rows:
            for method, prefix in (("C0", "Full_vs_C0"), (row["cmean_definition"], "Full_vs_Cmean")):
                label = "Full_minus_" + (method if layer == "primary_corrected_192" else method.lower())
                old = archived.loc[archived.comparison.eq(label) & archived.metric.eq(row["endpoint"])]
                require(len(old) == 1, f"Missing archived contrast: {label}")
                for field, oldfield in (("mean", "estimate"), ("lower", "lower"), ("upper", "upper")):
                    reference_error = max(reference_error, abs(row[f"{prefix}_{field}"] - float(old[oldfield].iloc[0])))
        rows.extend(new_rows)
        paired.extend(new_paired)
        audits[layer] = audit
        metrics = metrics.copy()
        metrics.insert(0, "analysis_layer", layer)
        metrics_all.append(metrics)
    require(reference_error < EPS, f"Frozen Full-versus-control estimates/BCa not reproduced: {reference_error}")
    bindings.unchanged()
    pd.DataFrame(rows).to_csv(output_dir / "W01_result_table.csv", index=False, float_format="%.17g")
    pd.DataFrame(paired).to_csv(output_dir / "W01_paired_group_contrasts.csv", index=False, float_format="%.17g")
    pd.concat(metrics_all).to_csv(output_dir / "W01_group_metrics.csv", index=False, float_format="%.17g")
    pd.DataFrame(identities).to_csv(output_dir / "W01_frozen_config_identity.csv", index=False)
    integrity = {"status": "PASS", "N_per_stratum": 21, "records": 23, "outer_seeds": list(SEEDS),
                 "indexed_rows_per_stratum": len(original), "seed_recordings_per_stratum": 69,
                 "max_primary_metric_reproduction_error": primary_error,
                 "max_stage_metric_reproduction_error": stage_error,
                 "max_archived_effect_and_CI_reproduction_error": reference_error,
                 "primary_corrected_anchor_errors": anchor_errors, "cross_stratum_frozen_input_errors": common_errors,
                 "algebraic_audits": audits, "frozen_inputs_unchanged": True,
                 "consumed_source_hashes": bindings.checked}
    dump(output_dir / "W01_integrity_audit.json", integrity)
    dump(output_dir / "W01_summary.json", {
        "workflow": 1, "N": 21, "independent_unit": "filename-defined analysis group",
        "endpoints": ENDPOINTS, "effect_convention": "Full minus comparator; lower metrics give negative effects",
        "construction_contrast": "(Full-Cmean)-(Full-C0)=C0-Cmean, after group aggregation",
        "primary_scope": "primary_corrected_192 / Cmean_primary",
        "secondary_scope": "frozen pooled-weight sensitivity and separately labeled existing posthoc Stage1b; no between-stratum inference",
        "statistical_rule": {"ci": "pointwise 95% BCa", "resamples": BOOTSTRAP_RESAMPLES,
                             "seed": BOOTSTRAP_SEED, "confidence_level": CONFIDENCE_LEVEL,
                             "p_value": None, "holm": None, "sign_tolerance": EPS,
                             "conditioning": "already selected/fitted frozen procedures"},
        "training": False, "updater_replayed": False, "outer_test": "USED: existing frozen outer predictions and labels for evaluation only",
        "classification_rule": "Qualitative: changed mean direction, CI support or group sign majority => construction-sensitive; otherwise a directional construction CI => magnitude-sensitive; otherwise descriptive construction-robust, never equivalence.",
        "results": rows, "integrity": {k: v for k, v in integrity.items() if k != "consumed_source_hashes"},
    })
    print(pd.DataFrame(rows)[["analysis_layer", "cmean_definition", "endpoint", "construction_mean",
                              "construction_lower", "construction_upper", "classification"]].to_string(index=False))
    print(f"W01 PASS: paired units/aggregation/source estimates/BCa reproduced; max source estimate error={reference_error:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.output_dir)
