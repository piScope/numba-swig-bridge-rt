"""Registration contract for downstream-owned director-state backends.

Each registered module owns one concrete ``StateBlock`` implementation.  The
runtime sees it only through the stable C ABI exported by its generated SWIG
state module.  This keeps generated director helpers independent of both the
SWIG ``StateHandle`` Python type and the downstream C++ implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from llvmlite import binding

from ._version import STATE_ABI_VERSION


# These are the C ABI entry points supplied by the StateHandle implementation.
# Their registered LLVM names are backend-qualified, so two downstream state
# backends can safely coexist in one process.
STATE_ACCESSORS = (
    "float_at", "double_at", "int_at", "int64_at", "bool_at",
    "object_address_at", "object_meminfo_at", "set_object_native",
    "set_object_borrowed",
    "array_data_at", "array_shape_at", "array_stride_at", "set_array_native",
)
LEASE_OPERATIONS = ("lease_address", "lease_is_alive", "lease_close", "lease_release")
_FUNCTIONS = (*STATE_ACCESSORS, *LEASE_OPERATIONS)


@dataclass(frozen=True)
class StateBackend:
    """One downstream provider of opaque director state.

    ``module`` is retained strongly because later native lowering uses function
    addresses exported by that extension.  A backend identifier is stable in
    generated descriptors; it is not a per-director state token.
    """

    identifier: str
    module: ModuleType
    state_abi: int

    def function_name(self, operation: str) -> str:
        """Return the backend module's exported, unqualified C ABI name."""
        if operation not in _FUNCTIONS:
            raise ValueError(f"unsupported native state operation {operation!r}")
        return f"nsb_director_state_{operation}"

    def symbol_name(self, operation: str) -> str:
        """Return the LLVM symbol registered for this backend operation."""
        if operation not in STATE_ACCESSORS:
            raise ValueError(f"{operation!r} is not a lowering accessor")
        return f"nsb_rt_{self.identifier}_state_{operation}"

    def function_address(self, operation: str) -> int:
        """Return one stable C ABI address exported by the backend module."""
        resolver = getattr(self.module, "nsb_director_state_function_address", None)
        if not callable(resolver):
            raise RuntimeError(
                f"state backend {self.identifier!r} does not export "
                "nsb_director_state_function_address"
            )
        address = int(resolver(self.function_name(operation)))
        if address == 0:
            raise RuntimeError(
                f"state backend {self.identifier!r} did not export "
                f"{self.function_name(operation)}"
            )
        return address


_BACKENDS: dict[str, StateBackend] = {}


def register_state_backend(identifier: str, module: ModuleType, *, state_abi: int) -> StateBackend:
    """Register one downstream state provider, rejecting incompatible reuse."""
    if not isinstance(identifier, str) or not identifier.isidentifier():
        raise ValueError("state backend identifier must be a Python identifier")
    if not isinstance(module, ModuleType):
        raise TypeError("state backend module must be a Python module")
    if state_abi != STATE_ABI_VERSION:
        raise RuntimeError(
            f"state backend {identifier!r} has ABI {state_abi}; "
            f"runtime requires {STATE_ABI_VERSION}"
        )
    existing = _BACKENDS.get(identifier)
    if existing is not None:
        if existing.module is module and existing.state_abi == state_abi:
            return existing
        raise RuntimeError(f"state backend {identifier!r} is already registered by a different module or ABI")
    module_abi = getattr(module, "NUMBA_SWIG_BRIDGE_STATE_ABI_VERSION", None)
    if module_abi != state_abi:
        raise RuntimeError(
            f"state backend {identifier!r} reports ABI {module_abi!r}; "
            f"registration requested ABI {state_abi}"
        )
    if not isinstance(getattr(module, "StateHandle", None), type):
        raise RuntimeError(f"state backend {identifier!r} does not export StateHandle")
    backend = StateBackend(identifier, module, state_abi)
    # Register each lowering function under an identifier-qualified LLVM
    # name.  A cfunc compiled for backend ``mfem`` can consequently never
    # be rebound to another downstream library's state implementation.
    for operation in STATE_ACCESSORS:
        binding.add_symbol(backend.symbol_name(operation), backend.function_address(operation))
    _BACKENDS[identifier] = backend
    return backend


def state_backend(identifier: str) -> StateBackend:
    """Return a registered backend or raise a clear configuration error."""
    try:
        return _BACKENDS[identifier]
    except KeyError as error:
        raise RuntimeError(f"state backend {identifier!r} is not registered") from error
