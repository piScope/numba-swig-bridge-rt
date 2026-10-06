"""Installed build interface for downstream SWIG wrapper packages."""

import json
from pathlib import Path

from ._version import __version__, STATE_ABI_VERSION
from .resources import include_dir, cpp_dir


def get_build_info() -> dict:
    """Resolve resource paths at query time, independently of installation location."""
    metadata = Path(__file__).with_name("build_info.json")
    build = json.loads(metadata.read_text()) if metadata.is_file() else {}
    return {
        "version": __version__, "abi_version": STATE_ABI_VERSION,
        "include_dir": str(include_dir()), "cpp_dir": str(cpp_dir()),
        "header": str(cpp_dir() / "nsb_director_state.hpp"),
        "sources": [str(cpp_dir() / "nsb_director_state.cpp")],
        "state_interface": str(include_dir() / "nsb-director-state.i"),
        "state_module": "nsb_director_state_bindings",
        "build": build,
    }
