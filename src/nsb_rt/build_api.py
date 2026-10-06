"""Stable build-resource API for downstream SWIG bridge packages."""

from .build_info import get_build_info
from .resources import cpp_dir, include_dir

__all__ = ["get_build_info", "cpp_dir", "include_dir"]
