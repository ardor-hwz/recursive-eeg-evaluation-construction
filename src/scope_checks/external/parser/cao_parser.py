"""Frozen Cao event parser. No reference counts, sorting, repair, or imputation.

Input is the admitted canonical BIDS TSV, whose sample column is zero-based.
Every source row and every deviation trial survives into a ledger. Response
offsets and unknown 255 events have no role in numerical RT eligibility.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

RATE_HZ = 500
DEVIATIONS = frozenset((251, 252))
SEMANTICS = {251: "deviation_onset_left", 252: "deviation_onset_right",
             253: "response_onset", 254: "response_offset", 255: "unknown"}


def integer_token(token: str) -> tuple[int | None, str | None]:
    """Accept numeric spelling only when exactly finite and integral."""
    try:
        value = Decimal(token)
    except (InvalidOperation, ValueError):
        return None, "not_numeric"
    if not value.is_finite():
        return None, "nonfinite"
    if value != value.to_integral_value():
        return None, "nonintegral"
    return int(value), None


def parse_events(source: bytes, n_samples: int) -> list[dict[str, Any]]:
    """Preserve file order, source strings and physical line numbers verbatim."""
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer from raw pnts")
    text = source.decode("utf-8-sig", errors="strict")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    header = next(reader)
    required = {"onset", "duration", "trial_type", "value", "sample"}
    if not required.issubset(header) or len(set(header)) != len(header):
        raise ValueError("Missing required or duplicate canonical TSV columns")
    lines = text.splitlines(keepends=True)
    events = []
    for index, fields in enumerate(reader):
        # Canonical release has one physical line per event. No multiline rescue.
        line_number = reader.line_num
        flags: list[str] = []
        raw = dict(zip(header, fields))
        if len(fields) != len(header):
            flags.append("malformed_row_width")
        if line_number != index + 2:
            flags.append("noncanonical_multiline_record")
        code, code_error = integer_token(raw.get("value", ""))
        sample, sample_error = integer_token(raw.get("sample", ""))
        if code_error:
            flags.append("event_code_" + code_error)
        elif code not in SEMANTICS:
            flags.append("unrecognized_event_code")
        if sample_error:
            flags.append("sample_" + sample_error)
        elif not 0 <= sample < n_samples:
            flags.append("sample_out_of_recording_bounds")
        try:
            onset = Decimal(raw.get("onset", ""))
            if not onset.is_finite():
                flags.append("onset_nonfinite")
            elif sample is not None and onset != Decimal(sample) / RATE_HZ:
                flags.append("onset_sample_clock_mismatch")
        except (InvalidOperation, ValueError):
            flags.append("onset_not_numeric")
        # Value is authoritative; text disagreement is flagged, never relabelled.
        if code in SEMANTICS and raw.get("trial_type") != SEMANTICS[code]:
            flags.append("trial_type_value_mismatch")
        events.append({
            "source_event_index": index, "source_line": line_number,
            "code": code, "sample": sample,
            "raw_fields": raw, "raw_values": fields,
            "raw_source_line": lines[line_number - 1] if line_number <= len(lines) else "",
            "frozen_semantics": SEMANTICS.get(code, "unrecognized"),
            "flags": flags,
        })

    # Detect, but never sort or remove, duplicate source events.
    by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    deviations_by_sample: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        if event["code"] is not None and event["sample"] is not None:
            by_key[(event["code"], event["sample"])].append(event)
        if event["code"] in DEVIATIONS and event["sample"] is not None:
            deviations_by_sample[event["sample"]].append(event)
    for duplicates in by_key.values():
        if len(duplicates) > 1:
            for event in duplicates:
                event["flags"].append("duplicate_code_sample")
    for duplicates in deviations_by_sample.values():
        if len(duplicates) > 1:
            for event in duplicates:
                event["flags"].append("ambiguous_deviation_same_sample")
    for previous, current in zip(events, events[1:]):
        if previous["sample"] is not None and current["sample"] is not None:
            if current["sample"] < previous["sample"]:
                previous["flags"].append("source_order_regression")
                current["flags"].append("source_order_regression")
    return events


def reconstruct_trials(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each deviation owns only source rows up to, excluding, the next deviation.

    Exactly one 253 in that interval is required. Ambiguity never resolves to
    selecting the first or last response. A globally untrustworthy source order
    or unknown event grammar produces unobserved rows, not a repaired sequence.
    """
    structural = {"malformed_row_width", "noncanonical_multiline_record",
                  "event_code_not_numeric", "event_code_nonfinite",
                  "event_code_nonintegral", "unrecognized_event_code",
                  "source_order_regression"}
    recording_flags = sorted({flag for event in events for flag in event["flags"]
                              if flag in structural})
    starts = [index for index, event in enumerate(events) if event["code"] in DEVIATIONS]
    trials = []
    covered_responses: set[int] = set()
    for trial_index, start in enumerate(starts):
        end = starts[trial_index + 1] if trial_index + 1 < len(starts) else len(events)
        deviation = events[start]
        interval = events[start + 1:end]
        responses = [event for event in interval if event["code"] == 253]
        offsets = [event for event in interval if event["code"] == 254]
        unknowns = [event for event in interval if event["code"] == 255]
        covered_responses.update(event["source_event_index"] for event in responses)
        reasons = ["recording_structure:" + flag for flag in recording_flags]
        reasons.extend("deviation:" + flag for flag in deviation["flags"])
        if not responses:
            reasons.append("missing_response_before_next_deviation")
        elif len(responses) > 1:
            reasons.append("ambiguous_multiple_responses_before_next_deviation")
            for event in responses:
                if "ambiguous_response_membership" not in event["flags"]:
                    event["flags"].append("ambiguous_response_membership")
        response = responses[0] if len(responses) == 1 else None
        delta = None
        if response is not None:
            reasons.extend("response:" + flag for flag in response["flags"])
            if deviation["sample"] is not None and response["sample"] is not None:
                delta = response["sample"] - deviation["sample"]
                if delta <= 0:
                    reasons.append("nonpositive_RT_sample_difference")
            else:
                reasons.append("RT_sample_difference_unavailable")
        metadata_flags = []
        if not offsets:
            metadata_flags.append("response_offset_254_missing")
        elif len(offsets) > 1:
            metadata_flags.append("multiple_response_offsets_254")
        if response is not None and any(event["source_event_index"] <
                                        response["source_event_index"] for event in offsets):
            metadata_flags.append("response_offset_before_response_onset")
        if unknowns:
            metadata_flags.append("unknown_255_retained")
        # Descriptive 254/255 flags do not gate RT. These rows cannot be targets.
        for event in offsets + unknowns:
            metadata_flags.extend("metadata_event:" + flag for flag in event["flags"])
        observed = not reasons and response is not None and delta is not None and delta > 0
        trials.append({
            "original_trial_index": trial_index,
            "deviation_event_index": deviation["source_event_index"],
            "deviation_source_line": deviation["source_line"],
            "deviation_code": deviation["code"], "deviation_sample": deviation["sample"],
            "deviation_value_verbatim": deviation["raw_fields"].get("value", ""),
            "deviation_sample_verbatim": deviation["raw_fields"].get("sample", ""),
            "next_deviation_event_index": end if end < len(events) else None,
            "response_candidates": [event["source_event_index"] for event in responses],
            "response_candidate_samples": [event["sample"] for event in responses],
            "response_event_index": response["source_event_index"] if response else None,
            "response_source_line": response["source_line"] if response else None,
            "response_sample": response["sample"] if response else None,
            "response_value_verbatim": response["raw_fields"].get("value", "") if response else "",
            "response_sample_verbatim": response["raw_fields"].get("sample", "") if response else "",
            "response_offset_event_indices": [event["source_event_index"] for event in offsets],
            "unknown_255_event_indices": [event["source_event_index"] for event in unknowns],
            "sample_difference": delta,
            "native_RT_seconds": delta / RATE_HZ if observed else None,
            "target_support_start_sample": deviation["sample"],
            "target_support_end_sample": response["sample"] if observed else None,
            "target_available_at_sample": response["sample"] if observed else None,
            "target_observed": observed, "exclusion_reason": sorted(set(reasons)),
            "descriptive_flags": sorted(set(metadata_flags)),
        })
    for event in events:
        if event["code"] == 253 and event["source_event_index"] not in covered_responses:
            if "orphan_response_no_preceding_deviation" not in event["flags"]:
                event["flags"].append("orphan_response_no_preceding_deviation")
        if starts and event["source_event_index"] < starts[0] and event["code"] == 254:
            if "orphan_offset_no_preceding_deviation" not in event["flags"]:
                event["flags"].append("orphan_offset_no_preceding_deviation")
    return trials


def parse_recording(source: bytes, n_samples: int) -> tuple[list[dict], list[dict]]:
    events = parse_events(source, n_samples)
    return events, reconstruct_trials(events)
