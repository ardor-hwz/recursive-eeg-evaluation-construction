"""Atomic, NumPy-safe result serialization."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and NumPy scalars to JSON values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text via same-directory temporary file and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write pretty, standards-compliant JSON."""

    return atomic_write_text(path, json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    """Hash canonical JSON for configs and split metadata."""

    encoded = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
