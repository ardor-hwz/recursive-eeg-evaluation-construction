"""One-authorized-outer fully nested BSPC trial.

This module imports original implementations read-only and writes exclusively
below ``major_revision_experiments_20260728``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.ema import EMABaseline  # noqa: E402
from src.baselines.kalman import FixedKalmanBaseline  # noqa: E402
from src.data.batch import make_proxy_sequence  # noqa: E402
from src.data.manifest import SUBJECT_SESSIONS  # noqa: E402
from src.data.seed_vig import SessionRecord, load_all_sessions  # noqa: E402
from src.data.splits import SplitSpec, generate_loso_splits, subjects_to_sessions  # noqa: E402
from src.models.change_aware_tracker import (  # noqa: E402
    ChangeAwareTracker,
    ChangeAwareTrackerConfig,
)
from src.models.phase35_statistical import fit_point_candidate  # noqa: E402
from src.search.development_search import FeatureCache  # noqa: E402
from src.search.tracker_search import (  # noqa: E402
    baseline_metrics,
    build_sequences,
    evaluate_tracker_config,
    sample_tracker_configs,
)
from src.training.tracker_development import subject_bootstrap  # noqa: E402
from src.utils.serialization import stable_hash  # noqa: E402

from src_major_revision.provenance_guards import sha256  # noqa: E402
from src_major_revision.validated_lag import LagV2Result, lag_v2  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _inner_splits(subjects: Sequence[int], seed: int, n_splits: int = 5) -> list[SplitSpec]:
    """Subset-safe grouped splits, including outer Subjects 4 or 5."""

    unique = sorted({int(item) for item in subjects})
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(rng.permutation(unique), dtype=int)
    validation_folds = [sorted(map(int, fold)) for fold in np.array_split(shuffled, n_splits)]
    splits = []
    for fold_id, validation in enumerate(validation_folds):
        train = sorted(set(unique) - set(validation))
        split = SplitSpec(
            fold_id=fold_id,
            train_subjects=train,
            validation_subjects=validation,
            test_subjects=[],
            train_sessions=subjects_to_sessions(train),
            validation_sessions=subjects_to_sessions(validation),
            test_sessions=[],
            random_seed=int(seed),
            protocol="GROUP5",
        )
        if set(train) & set(validation):
            raise RuntimeError("Inner Subject overlap")
        for subject in unique:
            expected = set(SUBJECT_SESSIONS[subject])
            in_train = bool(expected & set(split.train_sessions))
            in_validation = bool(expected & set(split.validation_sessions))
            if in_train == in_validation:
                raise RuntimeError(f"Inner Subject {subject} lost or split")
        splits.append(split)
    validation_counts = {
        subject: sum(subject in split.validation_subjects for split in splits)
        for subject in unique
    }
    if set(validation_counts.values()) != {1}:
        raise RuntimeError(f"Invalid inner OOF coverage: {validation_counts}")
    return splits


def _records_by_session(records: Sequence[SessionRecord]) -> dict[int, SessionRecord]:
    return {int(record.session_id): record for record in records}


def _records_for(split_ids: Iterable[int], lookup: Mapping[int, SessionRecord]) -> list[SessionRecord]:
    return [lookup[int(item)] for item in split_ids]


def _model_config(phase35: Mapping[str, Any]) -> dict:
    return {
        **dict(phase35["statistical"]),
        **dict(phase35["ensemble"]),
        "ema_alphas": list(phase35["postprocessing"]["ema_alphas"]),
    }


def _fit_candidate_once(
    candidate: Mapping[str, Any],
    split: SplitSpec,
    lookup: Mapping[int, SessionRecord],
    cache: FeatureCache,
    phase35: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[object, dict[int, np.ndarray]]:
    train_records = _records_for(split.train_sessions, lookup)
    validation_records = _records_for(split.validation_sessions, lookup)
    feature_set = str(candidate["feature_set"])
    train_x, train_y, train_subject, _, names = cache.stack(train_records, feature_set)
    val_x, val_y, val_subject, val_session, _ = cache.stack(validation_records, feature_set)
    model = fit_point_candidate(
        train_x,
        train_y,
        train_subject,
        val_x,
        val_y,
        val_subject,
        val_session,
        candidate=candidate,
        model_config=_model_config(phase35),
        seed=int(seed),
        feature_names=names,
    )
    concatenated = model.predict(val_x, val_session)
    predictions: dict[int, np.ndarray] = {}
    offset = 0
    for record in validation_records:
        stop = offset + record.sample_count
        predictions[record.session_id] = np.asarray(concatenated[offset:stop], dtype=float)
        offset = stop
    return model, predictions


def _subject_metrics(
    records: Sequence[SessionRecord],
    predictions: Mapping[tuple[int, int], np.ndarray],
    seeds: Sequence[int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for subject in sorted({record.subject_id for record in records}):
        subject_records = sorted(
            [record for record in records if record.subject_id == subject],
            key=lambda item: item.session_id,
        )
        seed_rmse: list[float] = []
        seed_lag: list[float] = []
        seed_mafd: list[float] = []
        for seed in seeds:
            target = np.concatenate([record.perclos for record in subject_records])
            predicted = np.concatenate(
                [predictions[(int(seed), record.session_id)] for record in subject_records]
            )
            seed_rmse.append(float(np.sqrt(np.mean((target - predicted) ** 2))))
            session_lags = []
            session_mafd = []
            for record in subject_records:
                current = predictions[(int(seed), record.session_id)]
                lag = lag_v2(record.perclos, current)
                if lag.absolute_lag is not None:
                    session_lags.append(float(lag.absolute_lag))
                session_mafd.append(float(np.mean(np.abs(np.diff(current)))))
            seed_lag.append(float(np.median(session_lags)) if session_lags else float("nan"))
            seed_mafd.append(float(np.median(session_mafd)))
        rows.append(
            {
                "subject_id": int(subject),
                "rmse": float(np.mean(seed_rmse)),
                "absolute_session_lag": float(np.nanmedian(seed_lag)),
                "mafd": float(np.median(seed_mafd)),
            }
        )
    return rows


def _summarize_candidate(
    candidate_id: str,
    records: Sequence[SessionRecord],
    predictions: Mapping[tuple[int, int], np.ndarray],
    seeds: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subjects = _subject_metrics(records, predictions, seeds)
    rmse = np.asarray([float(row["rmse"]) for row in subjects])
    summary = {
        "candidate_id": candidate_id,
        "subject_count": int(len(subjects)),
        "mean_subject_rmse": float(np.mean(rmse)),
        "rmse_standard_error": float(np.std(rmse, ddof=1) / np.sqrt(len(rmse))),
        "median_subject_absolute_session_lag": float(
            np.median([float(row["absolute_session_lag"]) for row in subjects])
        ),
        "median_subject_mafd": float(np.median([float(row["mafd"]) for row in subjects])),
    }
    return summary, subjects


def _candidate_search_and_roles(
    *,
    inner_records: Sequence[SessionRecord],
    inner_subjects: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    phase35: Mapping[str, Any],
    outer_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    lookup = _records_by_session(inner_records)
    cache = FeatureCache(inner_records, phase35["features"]["windows"])
    seeds = [int(item) for item in protocol["subjects"]["inner_seeds"]]
    all_predictions: dict[str, dict[tuple[int, int], np.ndarray]] = {
        str(candidate["id"]): {} for candidate in candidates
    }
    timings = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        cid = str(candidate["id"])
        for seed in seeds:
            for split in _inner_splits(inner_subjects, seed):
                start = time.perf_counter()
                _, predicted = _fit_candidate_once(
                    candidate,
                    split,
                    lookup,
                    cache,
                    phase35,
                    seed=seed * 100 + split.fold_id,
                )
                for session_id, values in predicted.items():
                    key = (seed, session_id)
                    if key in all_predictions[cid]:
                        raise RuntimeError(f"Duplicate OOF prediction: {cid}, {key}")
                    all_predictions[cid][key] = values
                seconds = time.perf_counter() - start
                timings.append(
                    {
                        "candidate_id": cid,
                        "seed": seed,
                        "fold_id": split.fold_id,
                        "seconds": seconds,
                    }
                )
                print(
                    f"[proxy-search] {candidate_index}/{len(candidates)} {cid} "
                    f"seed={seed} fold={split.fold_id} {seconds:.1f}s",
                    flush=True,
                )
    expected = len(seeds) * len(inner_records)
    summaries = []
    subject_rows = []
    for candidate in candidates:
        cid = str(candidate["id"])
        if len(all_predictions[cid]) != expected:
            raise RuntimeError(
                f"Incomplete OOF predictions for {cid}: {len(all_predictions[cid])}/{expected}"
            )
        summary, subjects = _summarize_candidate(
            cid, inner_records, all_predictions[cid], seeds
        )
        summaries.append(summary)
        subject_rows.extend({"candidate_id": cid, **row} for row in subjects)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["mean_subject_rmse", "candidate_id"], kind="stable"
    )
    best = summary_frame.iloc[0]
    threshold = float(best["mean_subject_rmse"] + best["rmse_standard_error"])
    summary_frame["within_one_standard_error"] = (
        summary_frame["mean_subject_rmse"] <= threshold
    )
    acceptable = summary_frame.loc[summary_frame["within_one_standard_error"]].copy()
    if len(acceptable) < int(protocol["proxy_selection"]["minimum_acceptable_candidates"]):
        raise RuntimeError(
            f"One-SE pool has {len(acceptable)} candidate(s); protocol requires at least two"
        )
    fast_row = acceptable.sort_values(
        [
            "median_subject_absolute_session_lag",
            "mean_subject_rmse",
            "candidate_id",
        ],
        kind="stable",
    ).iloc[0]
    slow_row = acceptable.loc[
        acceptable["candidate_id"].ne(str(fast_row["candidate_id"]))
    ].sort_values(
        ["median_subject_mafd", "mean_subject_rmse", "candidate_id"],
        kind="stable",
    ).iloc[0]
    by_id = {str(candidate["id"]): candidate for candidate in candidates}
    fast_candidate = by_id[str(fast_row["candidate_id"])]
    slow_candidate = by_id[str(slow_row["candidate_id"])]
    proxy_dir = outer_root / "inner_selection" / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(proxy_dir / "candidate_summary.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(
        proxy_dir / "candidate_per_subject_metrics.csv", index=False
    )
    pd.DataFrame(timings).to_csv(proxy_dir / "candidate_runtime.csv", index=False)
    role_lock = {
        "best_rmse_candidate": str(best["candidate_id"]),
        "best_rmse": float(best["mean_subject_rmse"]),
        "best_rmse_standard_error": float(best["rmse_standard_error"]),
        "one_standard_error_threshold": threshold,
        "acceptable_candidate_ids": acceptable["candidate_id"].astype(str).tolist(),
        "fast_candidate_id": str(fast_row["candidate_id"]),
        "fast_selection_metrics": fast_row.to_dict(),
        "slow_candidate_id": str(slow_row["candidate_id"]),
        "slow_selection_metrics": slow_row.to_dict(),
        "uses_outer": False,
    }
    _json(proxy_dir / "proxy_role_selection.json", role_lock)
    return fast_candidate, slow_candidate, role_lock


def _fit_role_ensemble_oof(
    *,
    role: str,
    candidate: Mapping[str, Any],
    split: SplitSpec,
    lookup: Mapping[int, SessionRecord],
    cache: FeatureCache,
    phase35: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], list[int]]:
    train_records = _records_for(split.train_sessions, lookup)
    validation_records = _records_for(split.validation_sessions, lookup)
    feature_set = str(candidate["feature_set"])
    train_x, train_y, train_subject, _, names = cache.stack(train_records, feature_set)
    val_x, val_y, val_subject, val_session, _ = cache.stack(validation_records, feature_set)
    member_predictions = []
    member_seeds = []
    for member_index in range(int(phase35["statistical"]["bootstrap_members"])):
        member_seed = (
            split.random_seed * 10000 + split.fold_id * 100 + member_index + 1
        )
        boot_x, boot_y, boot_subject = subject_bootstrap(
            train_x, train_y, train_subject, member_seed
        )
        model = fit_point_candidate(
            boot_x,
            boot_y,
            boot_subject,
            val_x,
            val_y,
            val_subject,
            val_session,
            candidate=candidate,
            model_config=_model_config(phase35),
            seed=member_seed,
            feature_names=names,
        )
        member_predictions.append(model.predict(val_x, val_session))
        member_seeds.append(member_seed)
    averaged = np.mean(np.asarray(member_predictions), axis=0)
    predictions = {}
    offset = 0
    for record in validation_records:
        stop = offset + record.sample_count
        predictions[record.session_id] = averaged[offset:stop]
        offset = stop
    print(
        f"[tracker-oof] role={role} candidate={candidate['id']} "
        f"seed={split.random_seed} fold={split.fold_id}",
        flush=True,
    )
    return predictions, member_seeds


def _tracker_oof_and_selection(
    *,
    inner_records: Sequence[SessionRecord],
    inner_subjects: Sequence[int],
    fast_candidate: Mapping[str, Any],
    slow_candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
    phase35: Mapping[str, Any],
    tracker_source_config: Mapping[str, Any],
    outer_root: Path,
) -> tuple[ChangeAwareTrackerConfig, pd.DataFrame, dict[str, Any]]:
    lookup = _records_by_session(inner_records)
    cache = FeatureCache(inner_records, phase35["features"]["windows"])
    frames = []
    for seed in [int(item) for item in protocol["subjects"]["inner_seeds"]]:
        for split in _inner_splits(inner_subjects, seed):
            fast, fast_seeds = _fit_role_ensemble_oof(
                role="fast",
                candidate=fast_candidate,
                split=split,
                lookup=lookup,
                cache=cache,
                phase35=phase35,
            )
            slow, slow_seeds = _fit_role_ensemble_oof(
                role="slow",
                candidate=slow_candidate,
                split=split,
                lookup=lookup,
                cache=cache,
                phase35=phase35,
            )
            for session_id in split.validation_sessions:
                record = lookup[session_id]
                frames.append(
                    pd.DataFrame(
                        {
                            "protocol": "GROUP5",
                            "split_role": "development_validation_oof",
                            "seed": seed,
                            "fold_id": split.fold_id,
                            "subject_id": record.subject_id,
                            "session_id": record.session_id,
                            "time_index": np.arange(record.sample_count),
                            "target_perclos": record.perclos,
                            "fast": fast[session_id],
                            "slow": slow[session_id],
                            "fast_candidate": str(fast_candidate["id"]),
                            "slow_candidate": str(slow_candidate["id"]),
                            "fast_member_seeds": json.dumps(fast_seeds),
                            "slow_member_seeds": json.dumps(slow_seeds),
                            "train_subjects": json.dumps(split.train_subjects),
                            "validation_subjects": json.dumps(split.validation_subjects),
                            "test_subject_count": 0,
                        }
                    )
                )
    frame = pd.concat(frames, ignore_index=True)
    tracker_dir = outer_root / "inner_selection" / "tracker"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tracker_dir / "tracker_selection_oof.csv", index=False)
    sequences = build_sequences(
        frame, [int(item) for item in tracker_source_config["tracker"]["past_windows"]]
    )
    fast_baseline = baseline_metrics(sequences, "fast")
    candidates = sample_tracker_configs(tracker_source_config)
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        row = evaluate_tracker_config(
            candidate,
            sequences,
            fast_baseline,
            tracker_source_config["search"],
        )
        rows.append(row)
        if index % 32 == 0:
            print(f"[tracker-search] {index}/{len(candidates)}", flush=True)
    ranking = pd.DataFrame(rows).sort_values(
        ["eligible", "selection_score", "rmse", "ccc", "candidate_id"],
        ascending=[False, True, True, False, True],
        kind="stable",
    )
    ranking.to_csv(tracker_dir / "tracker_candidate_ranking.csv", index=False)
    eligible = ranking.loc[ranking["eligible"].astype(bool)]
    if eligible.empty:
        raise RuntimeError("No tracker candidate satisfies the pre-frozen eligibility constraints")
    winner = eligible.iloc[0]
    field_names = list(ChangeAwareTrackerConfig.__dataclass_fields__)
    parameters = {
        name: winner[name].item() if hasattr(winner[name], "item") else winner[name]
        for name in field_names
    }
    selected = ChangeAwareTrackerConfig.from_mapping(parameters)
    lock = {
        "candidate_id": str(winner["candidate_id"]),
        "eligible_candidate_count": int(len(eligible)),
        "candidate_count": int(len(ranking)),
        "parameters": selected.to_dict(),
        "metrics": {
            name: float(winner[name])
            for name in (
                "rmse",
                "mae",
                "pcc",
                "ccc",
                "rmse_delta_vs_fast",
                "ccc_gain_vs_fast",
                "maximum_seed_rmse_delta",
            )
        },
        "uses_outer": False,
    }
    _json(tracker_dir / "tracker_selection.json", lock)
    return selected, frame, lock


def _ema(fast: np.ndarray, session_id: int, alpha: float) -> np.ndarray:
    proxy = make_proxy_sequence(
        subject_id=0,
        session_id=session_id,
        values=fast,
        source="nested_inner_fast",
        metadata={},
    )
    return np.asarray(
        EMABaseline(alpha=alpha, initialization="first_observation")
        .predict_sequence(proxy)
        .mean,
        dtype=float,
    )


def _kalman(fast: np.ndarray, session_id: int, q: float, r: float) -> np.ndarray:
    proxy = make_proxy_sequence(
        subject_id=0,
        session_id=session_id,
        values=fast,
        source="nested_inner_fast",
        metadata={},
    )
    return np.asarray(
        FixedKalmanBaseline(
            process_variance=q,
            observation_variance=r,
            initial_covariance=1.0,
            epsilon=1e-12,
        )
        .predict_sequence(proxy)
        .mean,
        dtype=float,
    )


def _comparator_selection(
    *,
    oof: pd.DataFrame,
    tracker_config: ChangeAwareTrackerConfig,
    comparator_lock: Mapping[str, Any],
    outer_root: Path,
) -> dict[str, Any]:
    records = []
    for (seed, subject, session), group in oof.groupby(
        ["seed", "subject_id", "session_id"], sort=True
    ):
        ordered = group.sort_values("time_index", kind="stable")
        target = ordered["target_perclos"].to_numpy(float)
        fast = ordered["fast"].to_numpy(float)
        slow = ordered["slow"].to_numpy(float)
        records.append(
            SessionRecord(
                int(subject),
                int(session),
                None,
                None,
                None,
                target,
                len(target),
                17,
                None,
                {"seed": int(seed), "fast": fast, "slow": slow},
            )
        )
    seeds = sorted({int(record.metadata["seed"]) for record in records})
    method_predictions: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    method_family: dict[str, str] = {}
    method_params: dict[str, dict] = {}
    tracker_id = "full_change_aware_tracker"
    method_predictions[tracker_id] = {}
    method_family[tracker_id] = "tracker"
    method_params[tracker_id] = tracker_config.to_dict()
    for record in records:
        seed = int(record.metadata["seed"])
        output = ChangeAwareTracker(tracker_config).track(
            record.metadata["fast"], record.metadata["slow"]
        )
        method_predictions[tracker_id][(seed, record.session_id)] = output.state
    for alpha in comparator_lock["ema"]["alpha_grid"]:
        method_id = f"ema_alpha_{float(alpha):.12g}"
        method_predictions[method_id] = {}
        method_family[method_id] = "ema"
        method_params[method_id] = {"alpha": float(alpha)}
        for record in records:
            seed = int(record.metadata["seed"])
            method_predictions[method_id][(seed, record.session_id)] = _ema(
                record.metadata["fast"], record.session_id, float(alpha)
            )
    fixed = comparator_lock["fixed_kalman_confirmatory"]
    for q in fixed["q_grid"]:
        for r in fixed["r_grid"]:
            method_id = f"kalman_q_{float(q):.12g}_r_{float(r):.12g}"
            method_predictions[method_id] = {}
            method_family[method_id] = "fixed_kalman"
            method_params[method_id] = {"q": float(q), "r": float(r)}
            for record in records:
                seed = int(record.metadata["seed"])
                method_predictions[method_id][(seed, record.session_id)] = _kalman(
                    record.metadata["fast"], record.session_id, float(q), float(r)
                )
    summaries = []
    subject_rows = []
    for method_id, predictions in method_predictions.items():
        summary, subjects = _summarize_candidate(method_id, records, predictions, seeds)
        summary["family"] = method_family[method_id]
        summary["parameters"] = json.dumps(method_params[method_id], sort_keys=True)
        summaries.append(summary)
        subject_rows.extend({"operating_point_id": method_id, **row} for row in subjects)
    summary_frame = pd.DataFrame(summaries)
    reference = summary_frame.loc[summary_frame["candidate_id"].eq(tracker_id)].iloc[0]
    selected: dict[str, Any] = {"reference": reference.to_dict(), "families": {}}
    for family in ("ema", "fixed_kalman"):
        candidates = summary_frame.loc[summary_frame["family"].eq(family)].copy()
        best = candidates.sort_values(
            [
                "mean_subject_rmse",
                "median_subject_absolute_session_lag",
                "median_subject_mafd",
                "candidate_id",
            ],
            kind="stable",
        ).iloc[0]
        candidates["mafd_distance"] = (
            candidates["median_subject_mafd"] - float(reference["median_subject_mafd"])
        ).abs()
        mafd = candidates.sort_values(
            ["mafd_distance", "mean_subject_rmse", "candidate_id"], kind="stable"
        ).iloc[0]
        candidates["lag_distance"] = (
            candidates["median_subject_absolute_session_lag"]
            - float(reference["median_subject_absolute_session_lag"])
        ).abs()
        lag = candidates.sort_values(
            ["lag_distance", "mean_subject_rmse", "candidate_id"], kind="stable"
        ).iloc[0]
        mafd_relative = float(mafd["mafd_distance"]) / max(
            abs(float(reference["median_subject_mafd"])), 1e-12
        )
        selected["families"][family] = {
            "best_rmse": best.to_dict(),
            "matched_mafd": {
                **mafd.to_dict(),
                "absolute_distance": float(mafd["mafd_distance"]),
                "relative_distance": mafd_relative,
                "within_tolerance": bool(
                    mafd_relative
                    <= float(
                        comparator_lock["matching"]["mafd_relative_tolerance"]
                    )
                ),
            },
            "matched_lag": {
                **lag.to_dict(),
                "absolute_distance": float(lag["lag_distance"]),
                "within_tolerance": bool(
                    float(lag["lag_distance"])
                    <= float(comparator_lock["matching"]["lag_absolute_tolerance"])
                ),
            },
        }
    comparator_dir = outer_root / "inner_selection" / "comparators"
    comparator_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(comparator_dir / "operating_point_summary.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(
        comparator_dir / "operating_point_per_subject_metrics.csv", index=False
    )
    _json(comparator_dir / "comparator_selection.json", selected)
    return selected


def _fit_outer_role(
    *,
    role: str,
    candidate: Mapping[str, Any],
    split: SplitSpec,
    all_records: Sequence[SessionRecord],
    cache: FeatureCache,
    phase35: Mapping[str, Any],
    checkpoint_dir: Path,
) -> dict[int, np.ndarray]:
    lookup = _records_by_session(all_records)
    train_records = _records_for(split.train_sessions, lookup)
    validation_records = _records_for(split.validation_sessions, lookup)
    test_records = _records_for(split.test_sessions, lookup)
    feature_set = str(candidate["feature_set"])
    train_x, train_y, train_subject, _, names = cache.stack(train_records, feature_set)
    val_x, val_y, val_subject, val_session, _ = cache.stack(validation_records, feature_set)
    test_x, _, _, test_session, _ = cache.stack(test_records, feature_set)
    member_predictions = []
    for member_index in range(int(phase35["statistical"]["bootstrap_members"])):
        member_seed = split.random_seed * 100 + member_index + 1
        boot_x, boot_y, boot_subject = subject_bootstrap(
            train_x, train_y, train_subject, member_seed
        )
        model = fit_point_candidate(
            boot_x,
            boot_y,
            boot_subject,
            val_x,
            val_y,
            val_subject,
            val_session,
            candidate=candidate,
            model_config=_model_config(phase35),
            seed=member_seed,
            feature_names=names,
        )
        checkpoint_path = (
            checkpoint_dir
            / role
            / str(split.random_seed)
            / f"member_{member_index}_seed_{member_seed}.pkl"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("wb") as handle:
            pickle.dump(
                {
                    "model": model,
                    "role": role,
                    "candidate": dict(candidate),
                    "outer_subject": split.test_subjects[0],
                    "train_subjects": split.train_subjects,
                    "validation_subjects": split.validation_subjects,
                    "member_seed": member_seed,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        member_predictions.append(model.predict(test_x, test_session))
    mean = np.mean(np.asarray(member_predictions), axis=0)
    predictions = {}
    offset = 0
    for record in test_records:
        stop = offset + record.sample_count
        predictions[record.session_id] = mean[offset:stop]
        offset = stop
    return predictions


def _selected_method_parameters(selection: Mapping[str, Any]) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = {}
    for family, family_selection in selection["families"].items():
        output[family] = {}
        for role in ("best_rmse", "matched_mafd", "matched_lag"):
            row = family_selection[role]
            output[family][role] = {
                "operating_point_id": str(row["candidate_id"]),
                "parameters": json.loads(row["parameters"]),
            }
    return output


def _outer_evaluation(
    *,
    outer_subject: int,
    inner_records: Sequence[SessionRecord],
    fast_candidate: Mapping[str, Any],
    slow_candidate: Mapping[str, Any],
    tracker_config: ChangeAwareTrackerConfig,
    comparator_selection: Mapping[str, Any],
    protocol: Mapping[str, Any],
    phase35: Mapping[str, Any],
    selection_lock_path: Path,
    outer_root: Path,
) -> dict[str, Any]:
    if not selection_lock_path.exists():
        raise RuntimeError("Outer data cannot be loaded before selection lock exists")
    frozen_hash = sha256(selection_lock_path)
    outer_session_ids = list(SUBJECT_SESSIONS[int(outer_subject)])
    outer_records = load_all_sessions(
        PROJECT_ROOT / "results" / "data_audit" / "manifest.csv",
        data_root=Path(protocol["paths"]["data_root"]),
        session_ids=outer_session_ids,
        strict=True,
    )
    all_records = list(inner_records) + list(outer_records)
    cache = FeatureCache(all_records, phase35["features"]["windows"])
    selected = _selected_method_parameters(comparator_selection)
    trajectory_frames = []
    lag_rows = []
    runtime_rows = []
    for seed in [int(item) for item in protocol["subjects"]["outer_final_seeds"]]:
        split = next(
            item
            for item in generate_loso_splits(
                protocol["subjects"]["all"],
                seed=seed,
                validation_fraction=float(protocol["subjects"]["outer_validation_fraction"]),
            )
            if item.test_subjects == [outer_subject]
        )
        if set(split.train_subjects + split.validation_subjects) != set(
            protocol["subjects"]["all"]
        ) - {outer_subject}:
            raise RuntimeError("Outer final fit does not use exactly the remaining Subjects")
        start = time.perf_counter()
        fast = _fit_outer_role(
            role="fast",
            candidate=fast_candidate,
            split=split,
            all_records=all_records,
            cache=cache,
            phase35=phase35,
            checkpoint_dir=outer_root / "checkpoints",
        )
        slow = _fit_outer_role(
            role="slow",
            candidate=slow_candidate,
            split=split,
            all_records=all_records,
            cache=cache,
            phase35=phase35,
            checkpoint_dir=outer_root / "checkpoints",
        )
        runtime_rows.append(
            {
                "seed": seed,
                "outer_proxy_fit_seconds": time.perf_counter() - start,
            }
        )
        for record in outer_records:
            fast_values = fast[record.session_id]
            slow_values = slow[record.session_id]
            outputs: dict[str, np.ndarray] = {
                "fast": fast_values,
                "slow": slow_values,
                "tracker": ChangeAwareTracker(tracker_config)
                .track(fast_values, slow_values)
                .state,
            }
            for family, roles in selected.items():
                for role, spec in roles.items():
                    name = f"{family}_{role}"
                    parameters = spec["parameters"]
                    if family == "ema":
                        outputs[name] = _ema(
                            fast_values, record.session_id, float(parameters["alpha"])
                        )
                    else:
                        outputs[name] = _kalman(
                            fast_values,
                            record.session_id,
                            float(parameters["q"]),
                            float(parameters["r"]),
                        )
            payload: dict[str, Any] = {
                "outer_subject_id": outer_subject,
                "seed": seed,
                "session_id": record.session_id,
                "time_index": np.arange(record.sample_count),
                "target_perclos": record.perclos,
                "fast_candidate": str(fast_candidate["id"]),
                "slow_candidate": str(slow_candidate["id"]),
                "selection_lock_sha256": frozen_hash,
            }
            payload.update(outputs)
            trajectory_frames.append(pd.DataFrame(payload))
            for method, values in outputs.items():
                result = lag_v2(record.perclos, values)
                lag_rows.append(
                    {
                        "outer_subject_id": outer_subject,
                        "seed": seed,
                        "session_id": record.session_id,
                        "method": method,
                        "signed_lag": result.signed_lag,
                        "absolute_lag": result.absolute_lag,
                        "ambiguous": result.ambiguous,
                        "boundary_hit": result.boundary_hit,
                        "maximum_correlation": result.maximum_correlation,
                        "peak_widths": json.dumps(
                            [item.width for item in result.peak_components]
                        ),
                    }
                )
    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    output_dir = outer_root / "outer_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output_dir / "outer_subject_1_trajectories.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(
        output_dir / "outer_subject_1_lag_v2_diagnostics.csv", index=False
    )
    pd.DataFrame(runtime_rows).to_csv(
        output_dir / "outer_subject_1_runtime.csv", index=False
    )
    methods = [
        item
        for item in trajectories.columns
        if item
        not in {
            "outer_subject_id",
            "seed",
            "session_id",
            "time_index",
            "target_perclos",
            "fast_candidate",
            "slow_candidate",
            "selection_lock_sha256",
        }
    ]
    metric_rows = []
    for method in methods:
        seed_rmse = []
        seed_mafd = []
        seed_abs_lag = []
        for seed, seed_frame in trajectories.groupby("seed", sort=True):
            target = seed_frame["target_perclos"].to_numpy(float)
            prediction = seed_frame[method].to_numpy(float)
            seed_rmse.append(float(np.sqrt(np.mean((target - prediction) ** 2))))
            session_mafd = []
            session_lag = []
            for _session, group in seed_frame.groupby("session_id", sort=True):
                values = group.sort_values("time_index")[method].to_numpy(float)
                truth = group.sort_values("time_index")["target_perclos"].to_numpy(float)
                session_mafd.append(float(np.mean(np.abs(np.diff(values)))))
                result = lag_v2(truth, values)
                if result.absolute_lag is not None:
                    session_lag.append(float(result.absolute_lag))
            seed_mafd.append(float(np.median(session_mafd)))
            seed_abs_lag.append(float(np.median(session_lag)))
        metric_rows.append(
            {
                "outer_subject_id": outer_subject,
                "method": method,
                "rmse_mean_over_seeds": float(np.mean(seed_rmse)),
                "mafd_median_sessions_then_seeds": float(np.median(seed_mafd)),
                "absolute_lag_median_sessions_then_seeds": float(
                    np.median(seed_abs_lag)
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "outer_subject_1_subject_metrics.csv", index=False)
    return {
        "outer_loaded_after_selection_lock": True,
        "selection_lock_sha256_at_outer_load": frozen_hash,
        "outer_subject_id": outer_subject,
        "outer_session_ids": outer_session_ids,
        "outer_seed_count": int(trajectories["seed"].nunique()),
        "trajectory_row_count": int(len(trajectories)),
        "all_outputs_finite": bool(
            np.all(np.isfinite(trajectories[methods].to_numpy(float)))
        ),
        "method_count": len(methods),
        "metrics": metric_rows,
        "runtime": runtime_rows,
    }


def run_one_outer(outer_subject: int = 1) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = EXPERIMENT_ROOT / "configs" / "nested_subject_evaluation_protocol.json"
    comparator_path = EXPERIMENT_ROOT / "configs" / "comparator_grid_lock.json"
    protocol = _load_json(protocol_path)
    comparator = _load_json(comparator_path)
    if outer_subject not in protocol["authorization"]["authorized_outer_subjects"]:
        raise PermissionError(f"Outer Subject {outer_subject} is not authorized")
    if protocol["authorization"]["full_21_subject_run"]:
        raise RuntimeError("Smoke-test runner refuses a full-run authorization flag")
    outer_root = EXPERIMENT_ROOT / "results" / f"outer_subject_{outer_subject:02d}"
    if (outer_root / "TRIAL_COMPLETE.json").exists():
        raise FileExistsError("Trial is already complete; refusing to overwrite it")
    outer_root.mkdir(parents=True, exist_ok=True)
    phase35 = _load_yaml(PROJECT_ROOT / "configs" / "phase35.yaml")
    tracker_source = _load_yaml(PROJECT_ROOT / "configs" / "change_aware_tracker.yaml")
    search_space = _load_yaml(PROJECT_ROOT / "configs" / "phase35_search_space.yaml")
    eligible = set(protocol["proxy_selection"]["eligible_candidate_ids"])
    candidates = [
        item for item in search_space["candidates"] if str(item["id"]) in eligible
    ]
    if {str(item["id"]) for item in candidates} != eligible:
        raise RuntimeError("Eligible proxy candidate lock does not match source search space")
    all_subjects = [int(item) for item in protocol["subjects"]["all"]]
    inner_subjects = sorted(set(all_subjects) - {int(outer_subject)})
    inner_session_ids = subjects_to_sessions(inner_subjects)
    outer_session_ids = subjects_to_sessions([outer_subject])
    inner_subject_hash = hashlib.sha256(
        json.dumps(inner_subjects, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    initial_manifest = {
        "outer_subject_id": outer_subject,
        "outer_session_ids_declared_but_not_loaded": outer_session_ids,
        "inner_subject_ids": inner_subjects,
        "inner_session_ids": inner_session_ids,
        "inner_subject_set_sha256": inner_subject_hash,
        "created_utc": _utc(),
    }
    _json(outer_root / "inner_selection" / "inner_subject_manifest.json", initial_manifest)
    inner_records = load_all_sessions(
        PROJECT_ROOT / "results" / "data_audit" / "manifest.csv",
        data_root=Path(protocol["paths"]["data_root"]),
        session_ids=inner_session_ids,
        strict=True,
    )
    if {record.subject_id for record in inner_records} != set(inner_subjects):
        raise RuntimeError("Inner-only loader coverage mismatch")
    if any(record.subject_id == outer_subject for record in inner_records):
        raise RuntimeError("Outer Subject entered inner loader")
    fast_candidate, slow_candidate, proxy_lock = _candidate_search_and_roles(
        inner_records=inner_records,
        inner_subjects=inner_subjects,
        candidates=candidates,
        protocol=protocol,
        phase35=phase35,
        outer_root=outer_root,
    )
    tracker_config, tracker_oof, tracker_lock = _tracker_oof_and_selection(
        inner_records=inner_records,
        inner_subjects=inner_subjects,
        fast_candidate=fast_candidate,
        slow_candidate=slow_candidate,
        protocol=protocol,
        phase35=phase35,
        tracker_source_config=tracker_source,
        outer_root=outer_root,
    )
    comparator_selection = _comparator_selection(
        oof=tracker_oof,
        tracker_config=tracker_config,
        comparator_lock=comparator,
        outer_root=outer_root,
    )
    config_paths = sorted((EXPERIMENT_ROOT / "configs").glob("*.json"))
    source_paths = [
        PROJECT_ROOT / "configs" / "phase35.yaml",
        PROJECT_ROOT / "configs" / "phase35_search_space.yaml",
        PROJECT_ROOT / "configs" / "change_aware_tracker.yaml",
        PROJECT_ROOT / "configs" / "baselines.yaml",
        PROJECT_ROOT / "configs" / "supplementary_common_input_tracking.yaml",
    ]
    selection_lock = {
        "schema_version": "bspc.outer_selection_lock.v1",
        "frozen_utc": _utc(),
        "outer_subject_id": outer_subject,
        "outer_session_ids_not_loaded_during_selection": outer_session_ids,
        "inner_subject_ids": inner_subjects,
        "inner_session_ids": inner_session_ids,
        "inner_subject_set_sha256": inner_subject_hash,
        "protocol_config_sha256": {path.name: sha256(path) for path in config_paths},
        "source_config_sha256": {str(path): sha256(path) for path in source_paths},
        "proxy_roles": proxy_lock,
        "tracker": tracker_lock,
        "comparators": comparator_selection,
        "outer_target_rows_used_for_selection": 0,
        "outer_data_loaded_for_selection": False,
        "immutable": True,
    }
    selection_lock["stable_selection_sha256"] = stable_hash(
        {key: value for key, value in selection_lock.items() if key != "frozen_utc"}
    )
    selection_lock_path = outer_root / "selection_lock.json"
    if selection_lock_path.exists():
        raise FileExistsError("Selection lock already exists; refusing to overwrite")
    _json(selection_lock_path, selection_lock)
    outer_result = _outer_evaluation(
        outer_subject=outer_subject,
        inner_records=inner_records,
        fast_candidate=fast_candidate,
        slow_candidate=slow_candidate,
        tracker_config=tracker_config,
        comparator_selection=comparator_selection,
        protocol=protocol,
        phase35=phase35,
        selection_lock_path=selection_lock_path,
        outer_root=outer_root,
    )
    completed = {
        "schema_version": "bspc.one_outer_trial.v1",
        "completed_utc": _utc(),
        "status": "PASS" if outer_result["all_outputs_finite"] else "FAIL",
        "full_21_subject_run_started": False,
        "outer_result": outer_result,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_lock_path": str(selection_lock_path),
        "selection_lock_sha256": sha256(selection_lock_path),
    }
    _json(outer_root / "TRIAL_COMPLETE.json", completed)
    return completed

