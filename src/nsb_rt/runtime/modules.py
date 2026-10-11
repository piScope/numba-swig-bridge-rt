"""Resolve explicit extension ownership without importing target libraries."""

from collections.abc import Mapping
from typing import Any


def resolve_modules(modules: Mapping[str, Any], records: list[dict]) -> dict[str, Any]:
    if not isinstance(modules, Mapping):
        raise TypeError(
            "bridge registration requires a mapping of Python module names to modules"
        )
    required = set()
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid bridge class record")
        owner = record.get("owner_module")
        cxx_type = record.get("cxx_type")
        if not isinstance(owner, str) or not owner:
            raise ValueError(f"class {cxx_type} has no owning module")
        if not isinstance(cxx_type, str) or cxx_type in seen:
            raise ValueError(f"invalid or duplicate bridge class {cxx_type!r}")
        seen.add(cxx_type)
        required.add(owner)
    missing = required.difference(modules)
    if missing:
        raise ValueError(f"missing owning module {sorted(missing)[0]!r}")
    return {name: modules[name] for name in sorted(required)}
