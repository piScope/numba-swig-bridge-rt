"""Public runtime API for generated numba-swig-bridge integrations."""

from ._version import __version__
from .runtime.arrays import as_cpointer

__all__ = [
    "director",
    "override",
    "NumbaSwigBridgeFallbackWarning",
    "upcast",
    "dynamic_cast",
    "require_not_none",
    "own_this",
    "borrowed_from",
    "is_slot_occupied",
    "as_cpointer",
    "register_state_backend",
    "state_backend",
    "StateBackend",
    "__version__",
]


def __getattr__(name):
    if name in {"register_state_backend", "state_backend", "StateBackend"}:
        from . import backends

        value = getattr(backends, name)
    elif name in __all__:
        if name in {
            "upcast",
            "dynamic_cast",
            "require_not_none",
            "own_this",
            "borrowed_from",
        }:
            from .runtime import ordinary as implementation
        else:
            from .runtime import director as implementation
        value = getattr(implementation, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
