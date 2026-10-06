"""Paths to build resources shipped with the framework package."""

from pathlib import Path


def include_dir() -> Path:
    """Return the directory containing the SWIG annotation interfaces."""
    return Path(__file__).resolve().parent / "include"


def cpp_dir() -> Path:
    """Return the directory containing common C++ director-state support."""
    return Path(__file__).resolve().parent / "cpp"
