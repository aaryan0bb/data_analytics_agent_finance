from __future__ import annotations
import json
import hashlib
from typing import Any, Mapping


def _normalize(obj: Any) -> Any:
    """Recursively convert to a JSON-serializable structure with sorted keys.

    - Mappings -> dict with sorted keys
    - Iterables (list/tuple/set) -> list of normalized items
    - Pydantic/Dataclass-like with .dict() -> normalize that
    - Other objects -> return as-is (json.dumps will fail if not serializable)
    """
    if isinstance(obj, Mapping):
        return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize(x) for x in obj]
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _normalize(obj.dict())  # type: ignore[attr-defined]
        except Exception:
            pass
    return obj


def canonical_params(params: dict) -> str:
    """Return a canonical JSON string for params (stable key order, no spaces)."""
    norm = _normalize(params)
    return json.dumps(norm, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def params_signature(tool_name: str, params: dict) -> str:
    """Hash a (tool_name, canonical_params) pair to a stable signature."""
    blob = f"{tool_name}:{canonical_params(params)}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

