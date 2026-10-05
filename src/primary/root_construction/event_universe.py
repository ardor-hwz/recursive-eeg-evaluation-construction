"""Independent target-free construction of the frozen 13,654-root universe."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .bindings import VerifiedBindings, load_target_free_inputs
from .contract import (
    ANALYSIS_LENGTH,
    DELTA0,
    EXPECTED_EVENT_ROOTS,
    EXPECTED_EXCLUDED_ROOTS,
    EXPECTED_STRUCTURAL_ROOTS,
    SEEDS,
    SUBJECTS,
    SUSTAINED_LENGTH,
    T0_MAX,
    T0_MIN,
)
from .errors import fail
from .hashing import content_sha256, load_json, verify_content_hash


def _validate_session_inventory(inputs: Mapping[int, pd.DataFrame]) -> tuple[dict[int, list[int]], list[list[int]]]:
    if set(inputs) != set(SUBJECTS):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "subject inventory is not exactly 1..21")
    subject_sessions: dict[int, list[int]] = {}
    all_keys: list[list[int]] = []
    session_owner: dict[int, int] = {}
    seed_session_count = 0
    row_count = 0
    for subject in SUBJECTS:
        frame = inputs[subject]
        sessions = sorted(int(value) for value in frame["session_id"].unique())
        subject_sessions[subject] = sessions
        for session in sessions:
            if session in session_owner and session_owner[session] != subject:
                fail("FAIL_CLOSED_EVENT_UNIVERSE", "physical session belongs to multiple subjects")
            session_owner[session] = subject
            for seed in SEEDS:
                group = frame.loc[
                    frame["seed"].astype(int).eq(seed) & frame["session_id"].astype(int).eq(session)
                ]
                observed = group["time_index"].to_numpy(dtype=np.int64)
                if len(group) != 885 or not np.array_equal(observed, np.arange(885, dtype=np.int64)):
                    fail("FAIL_CLOSED_EVENT_UNIVERSE", "seed-session must contain indices 0..884 exactly")
                seed_session_count += 1
                row_count += len(group)
                all_keys.extend([[subject, seed, session, int(t)] for t in observed])
    if sorted(session_owner) != list(range(23)) or len(session_owner) != 23:
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "physical-session inventory must be exactly 0..22")
    if seed_session_count != 69 or row_count != 61_065:
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "frozen seed-session/row accounting mismatch")
    if len(subject_sessions[4]) != 2 or len(subject_sessions[5]) != 2:
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "subjects 4 and 5 must retain two physical sessions")
    if any(len(subject_sessions[s]) != 1 for s in SUBJECTS if s not in (4, 5)):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "unexpected subject/session mapping")
    return subject_sessions, all_keys


def build_event_universe(
    inputs: Mapping[int, pd.DataFrame],
    *,
    protocol_semantic_sha256: str,
    bindings_content_sha256: str,
    input_projection_sha256: Mapping[int, str],
) -> dict[str, Any]:
    subject_sessions, all_keys = _validate_session_inventory(inputs)
    eligible: list[list[int]] = []
    excluded: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        frame = inputs[subject]
        for session in subject_sessions[subject]:
            by_seed: dict[int, np.ndarray] = {}
            for seed in SEEDS:
                group = frame.loc[
                    frame["seed"].astype(int).eq(seed) & frame["session_id"].astype(int).eq(session)
                ]
                fast = group["fast"].to_numpy(dtype=np.float64)
                slow = group["slow"].to_numpy(dtype=np.float64)
                if not np.isfinite(fast).all() or not np.isfinite(slow).all():
                    fail("FAIL_CLOSED_EVENT_UNIVERSE", "nonfinite target-free input")
                if np.any((fast < 0.0) | (fast > 1.0) | (slow < 0.0) | (slow > 1.0)):
                    fail("FAIL_CLOSED_EVENT_UNIVERSE", "proxy input outside [0,1]")
                by_seed[seed] = fast
            for t0 in range(T0_MIN, T0_MAX + 1):
                violations: list[dict[str, Any]] = []
                for seed in SEEDS:
                    block = by_seed[seed][t0 : t0 + SUSTAINED_LENGTH]
                    bad = np.flatnonzero((block < DELTA0) | (block > 1.0 - DELTA0))
                    if len(bad):
                        violations.append(
                            {
                                "seed": seed,
                                "active_offsets": [int(value) for value in bad],
                                "minimum_fast": float(np.min(block)),
                                "maximum_fast": float(np.max(block)),
                                "reason": "BOTH_SIGN_NO_CLIPPING_HEADROOM_FAILED",
                            }
                        )
                root = [subject, session, t0]
                if violations:
                    excluded.append({"root": root, "violations": violations})
                else:
                    eligible.append(root)
    if len(eligible) != EXPECTED_EVENT_ROOTS or len(excluded) != EXPECTED_EXCLUDED_ROOTS:
        fail(
            "FAIL_CLOSED_EVENT_UNIVERSE_COUNT",
            f"expected {EXPECTED_EVENT_ROOTS} eligible and {EXPECTED_EXCLUDED_ROOTS} excluded roots",
        )
    session_counts = Counter(root[1] for root in eligible)
    subject_counts = Counter(root[0] for root in eligible)
    expected_session_counts = {session: (542 if session == 22 else 596) for session in range(23)}
    expected_subject_counts = {subject: (1192 if subject in (4, 5) else 542 if subject == 21 else 596) for subject in SUBJECTS}
    if dict(sorted(session_counts.items())) != expected_session_counts:
        fail("FAIL_CLOSED_EVENT_UNIVERSE_COUNT", "per-session event counts differ from the frozen universe")
    if dict(sorted(subject_counts.items())) != expected_subject_counts:
        fail("FAIL_CLOSED_EVENT_UNIVERSE_COUNT", "per-subject event counts differ from the frozen universe")
    event_hash = content_sha256(eligible)
    excluded_hash = content_sha256(excluded)
    manifest: dict[str, Any] = {
        "schema_version": "bspc.m1_lite.event_universe.v1",
        "classification": "TARGET_FREE_PREEXECUTION_IDENTITY_ONLY",
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "source_bindings_content_sha256": bindings_content_sha256,
        "identity_fields": ["outer_subject_id", "physical_session_id", "t0"],
        "eligibility": {
            "t0_inclusive": [T0_MIN, T0_MAX],
            "all_seeds": list(SEEDS),
            "both_signs_required": True,
            "active_support_length": SUSTAINED_LENGTH,
            "headroom_inclusive": [DELTA0, 1.0 - DELTA0],
            "input_clipping": "PROHIBITED",
            "complete_analysis_length": ANALYSIS_LENGTH,
            "event_overlap": "ALLOWED",
            "event_grid": "NONE_ALL_ELIGIBLE_ROOTS_USED",
            "target_or_controller_output_used": False,
        },
        "accounting": {
            "subject_count": 21,
            "physical_session_count": 23,
            "seed_session_count": 69,
            "target_free_row_count": 61_065,
            "sequence_length_per_seed_session": 885,
            "structural_root_count": EXPECTED_STRUCTURAL_ROOTS,
            "eligible_root_count": len(eligible),
            "excluded_headroom_root_count": len(excluded),
            "per_session_eligible_counts": {str(k): v for k, v in sorted(session_counts.items())},
            "per_subject_eligible_counts": {str(k): v for k, v in sorted(subject_counts.items())},
        },
        "subject_sessions": {str(k): value for k, value in subject_sessions.items()},
        "target_free_input_projection_sha256": {
            str(subject): input_projection_sha256[subject] for subject in SUBJECTS
        },
        "ordered_key_sha256": content_sha256(all_keys),
        "event_universe_sha256": event_hash,
        "excluded_roots_sha256": excluded_hash,
        "eligible_roots": eligible,
        "excluded_structural_roots": excluded,
        "manifest_content_sha256": "",
    }
    manifest["manifest_content_sha256"] = content_sha256({k: v for k, v in manifest.items() if k != "manifest_content_sha256"})
    return manifest


def validate_event_manifest(payload: Mapping[str, Any], expected_roots: list[list[int]] | None = None) -> None:
    if not isinstance(payload, dict):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "event manifest must be an object")
    verify_content_hash(payload, "manifest_content_sha256", code="FAIL_CLOSED_EVENT_UNIVERSE")
    roots = payload.get("eligible_roots")
    excluded = payload.get("excluded_structural_roots")
    if not isinstance(roots, list) or not isinstance(excluded, list):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "event root lists are absent")
    tuples = [tuple(root) if isinstance(root, list) else () for root in roots]
    if len(tuples) != EXPECTED_EVENT_ROOTS:
        fail("FAIL_CLOSED_EVENT_UNIVERSE_COUNT", "event universe count mismatch")
    if len(set(tuples)) != len(tuples):
        fail("FAIL_CLOSED_EVENT_KEY_DUPLICATE", "duplicate event root")
    if tuples != sorted(tuples):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "event roots are not in canonical order")
    for root in tuples:
        if len(root) != 3 or root[0] not in SUBJECTS or not 0 <= root[1] <= 22 or not T0_MIN <= root[2] <= T0_MAX:
            fail("FAIL_CLOSED_EVENT_ROOT_INVALID", "invalid subject/session/t0 event root")
    if content_sha256(roots) != payload.get("event_universe_sha256"):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "event-universe hash mismatch")
    if len(excluded) != EXPECTED_EXCLUDED_ROOTS or content_sha256(excluded) != payload.get("excluded_roots_sha256"):
        fail("FAIL_CLOSED_EVENT_UNIVERSE_COUNT", "excluded-root registry mismatch")
    accounting = payload.get("accounting", {})
    if accounting.get("eligible_root_count") != EXPECTED_EVENT_ROOTS or accounting.get("structural_root_count") != EXPECTED_STRUCTURAL_ROOTS:
        fail("FAIL_CLOSED_EVENT_UNIVERSE_COUNT", "event accounting mismatch")
    if expected_roots is not None and roots != expected_roots:
        missing = set(map(tuple, expected_roots)) - set(tuples)
        extra = set(tuples) - set(map(tuple, expected_roots))
        code = "FAIL_CLOSED_EVENT_ROOT_MISSING" if missing else "FAIL_CLOSED_EVENT_ROOT_EXTRA"
        fail(code, f"event identity mismatch: missing={len(missing)} extra={len(extra)}")


def load_and_verify_event_manifest(bindings: VerifiedBindings) -> dict[str, Any]:
    path = bindings.freeze_root / "M1_LITE_EVENT_UNIVERSE_MANIFEST.json"
    stored = load_json(path, code="FAIL_CLOSED_EVENT_UNIVERSE")
    validate_event_manifest(stored)
    if stored.get("event_universe_sha256") != bindings.protocol.payload["event_universe"].get("event_universe_sha256"):
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "event manifest is not the universe bound by the protocol")
    inputs = load_target_free_inputs(bindings)
    rebuilt = build_event_universe(
        inputs,
        protocol_semantic_sha256=bindings.protocol.semantic_sha256,
        bindings_content_sha256=str(bindings.payload["bindings_content_sha256"]),
        input_projection_sha256=bindings.input_projection_sha256,
    )
    validate_event_manifest(stored, expected_roots=rebuilt["eligible_roots"])
    if stored != rebuilt:
        fail("FAIL_CLOSED_EVENT_UNIVERSE", "serialized event manifest differs from independent reconstruction")
    return stored
