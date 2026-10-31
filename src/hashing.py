"""Stable content hashes for configurable objects."""

import hashlib
import json
from typing import Any


def stable_hash(cls: type, config: dict[str, Any]) -> str:
    """SHA-256 hex digest of a class's import path plus a JSON-serializable config.

    Key order in `config` doesn't matter. Raises TypeError if `config` isn't
    JSON-serializable.
    """
    payload = {"class": f"{cls.__module__}.{cls.__qualname__}", "config": config}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
