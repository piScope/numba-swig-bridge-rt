"""Typed native state slots shared by generated directors.

Scalar state preserves the C++ ABI category (float, double, int, int64, bool). Object
state holds only a borrowed C++ address plus ownership provenance and a
monotonic generation. Python object assignment keeps the proxy alive; a C++
replacement invalidates the cached proxy at the next Python access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from llvmlite import ir
from numba import types
from numba.core.runtime import _nrt_python
from numba.core import cgutils
from numba.extending import intrinsic
from numba.np.arrayobj import populate_array

from .modules import resolve_modules

from nsb_rt.backends import StateBackend


PYTHON_ASSIGNED = 1
CPP_OWNED = 2
NATIVE_OWNED = 3
SCALAR_KINDS = ("float", "double", "int", "int64", "bool")


try:
    NRT_MEMINFO_RELEASE_ADDRESS = int(_nrt_python.c_helpers["MemInfo_release"])
except (AttributeError, KeyError) as error:
    raise RuntimeError("Numba does not expose the MemInfo_release runtime helper required by native director state") from error
if NRT_MEMINFO_RELEASE_ADDRESS == 0:
    raise RuntimeError("Numba exported a null MemInfo_release runtime helper")


_SCALAR_ABI = {
    "float": (types.float32, ir.FloatType(), "float_at"),
    "double": (types.float64, ir.DoubleType(), "double_at"),
    "int": (types.intc, ir.IntType(32), "int_at"),
    "int64": (types.int64, ir.IntType(64), "int64_at"),
    # C++ exports bool as int to avoid assuming a direct C++ bool-return ABI.
    "bool": (types.boolean, ir.IntType(32), "bool_at"),
}


def scalar_numba_type(kind: str):
    try:
        return _SCALAR_ABI[kind][0]
    except KeyError as error:
        raise ValueError(f"unsupported native state scalar kind {kind!r}") from error


def scalar_kind_from_annotation(annotation: object) -> str | None:
    for kind, (numba_type, _, _) in _SCALAR_ABI.items():
        if annotation == numba_type:
            return kind
    return None


def scalar_at(kind: str, backend: StateBackend):
    """Build a typed intrinsic reading one scalar state slot."""

    try:
        numba_type, llvm_type, operation = _SCALAR_ABI[kind]
    except KeyError as error:
        raise ValueError(f"unsupported native state scalar kind {kind!r}") from error

    @intrinsic
    def reader(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        signature = numba_type(state, slot)

        def codegen(context, builder, sig, args):
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(llvm_type, [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name(operation),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            value = builder.call(function, [args[0], native_slot])
            if kind == "bool":
                return builder.icmp_unsigned("!=", value, ir.Constant(value.type, 0))
            return value

        return signature, codegen

    return reader


def state_object_address_at(backend: StateBackend):
    """Build an intrinsic reading one backend-qualified object address."""

    @intrinsic
    def reader(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        signature = types.uintp(state, slot)

        def codegen(context, builder, sig, args):
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("object_address_at"),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            return builder.call(function, [args[0], native_slot])

        return signature, codegen

    return reader


def state_object_meminfo_at(backend: StateBackend):
    """Build an intrinsic reading one backend-qualified object NRT token."""

    @intrinsic
    def reader(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        meminfo_type = types.MemInfoPointer(types.voidptr)
        signature = meminfo_type(state, slot)

        def codegen(context, builder, sig, args):
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("object_meminfo_at"),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            address = builder.call(function, [args[0], native_slot])
            return builder.inttoptr(address, context.get_value_type(meminfo_type))

        return signature, codegen

    return reader



def state_object_at(backend: StateBackend, object_type):
    """Return a borrowed object view which retains a native state token, if any."""

    @intrinsic
    def reader(typingctx, state, slot):
        from . import ordinary

        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        result_type = ordinary.BorrowedReturnBridgeObjectType(
            object_type.bridge_id, object_type.cxx_type
        )
        signature = result_type(state, slot)

        def codegen(context, builder, sig, args):
            address_function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("object_address_at"),
            )
            meminfo_function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("object_meminfo_at"),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            address = builder.call(address_function, [args[0], native_slot])
            meminfo_address = builder.call(meminfo_function, [args[0], native_slot])
            meminfo = builder.inttoptr(meminfo_address, context.get_value_type(types.MemInfoPointer(types.voidptr)))
            context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), meminfo)
            result = cgutils.create_struct_proxy(result_type)(context, builder)
            result.meminfo = meminfo
            result.pointer = builder.inttoptr(address, ir.IntType(8).as_pointer())
            return result._getvalue()

        return signature, codegen

    return reader



def state_array_data_at(backend: StateBackend, dtype):
    """Return the typed data pointer for one one-dimensional state array."""
    pointer_type = types.CPointer(dtype)

    @intrinsic
    def reader(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        signature = pointer_type(state, slot)

        def codegen(context, builder, sig, args):
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("array_data_at"),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            address = builder.call(function, [args[0], native_slot])
            return builder.inttoptr(address, context.get_value_type(pointer_type))

        return signature, codegen

    return reader


def state_array_address_at(backend: StateBackend):
    """Return an untyped backend-qualified array address for presence checks."""

    @intrinsic
    def reader(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        signature = types.uintp(state, slot)

        def codegen(context, builder, sig, args):
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("array_data_at"),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            return builder.call(function, [args[0], native_slot])

        return signature, codegen

    return reader


def _array_dimension_at(backend: StateBackend, operation: str):
    @intrinsic
    def reader(typingctx, state, slot, dimension):
        if state != types.voidptr or not isinstance(slot, types.Integer) or not isinstance(dimension, types.Integer):
            return None
        signature = types.intp(state, slot, dimension)

        def codegen(context, builder, sig, args):
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.intp.bitwidth), [args[0].type, ir.IntType(32), ir.IntType(32)]),
                name=backend.symbol_name(operation),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            native_dimension = context.cast(builder, args[2], sig.args[2], types.intc)
            return builder.call(function, [args[0], native_slot, native_dimension])

        return signature, codegen

    return reader


def state_array_shape_at(backend: StateBackend):
    return _array_dimension_at(backend, "array_shape_at")


def state_array_stride_at(backend: StateBackend):
    return _array_dimension_at(backend, "array_stride_at")


def state_array_at(backend: StateBackend, array_type: types.Array):
    """Build an array view from the state slot's preserved NumPy metadata."""
    pointer_type = types.CPointer(array_type.dtype)

    @intrinsic
    def reader(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        signature = array_type(state, slot)

        def codegen(context, builder, sig, args):
            address_function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.uintp.bitwidth), [args[0].type, ir.IntType(32)]),
                name=backend.symbol_name("array_data_at"),
            )
            extent_function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.intp.bitwidth), [args[0].type, ir.IntType(32), ir.IntType(32)]),
                name=backend.symbol_name("array_shape_at"),
            )
            stride_function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(types.intp.bitwidth), [args[0].type, ir.IntType(32), ir.IntType(32)]),
                name=backend.symbol_name("array_stride_at"),
            )
            native_slot = context.cast(builder, args[1], sig.args[1], types.intc)
            data = builder.inttoptr(builder.call(address_function, [args[0], native_slot]), context.get_value_type(pointer_type))
            dimensions = [ir.Constant(ir.IntType(32), index) for index in range(array_type.ndim)]
            shape = [builder.call(extent_function, [args[0], native_slot, index]) for index in dimensions]
            strides = [builder.call(stride_function, [args[0], native_slot, index]) for index in dimensions]
            result = context.make_array(array_type)(context, builder)
            itemsize = ir.Constant(ir.IntType(types.intp.bitwidth), context.get_abi_sizeof(context.get_data_type(array_type.dtype)))
            populate_array(result, data, cgutils.pack_array(builder, shape), cgutils.pack_array(builder, strides), itemsize, meminfo=None)
            return result._getvalue()

        return signature, codegen

    return reader


@dataclass(frozen=True)
class DeclaredObjectBridge:
    """Proxy/address helpers for one ordinary bridge class."""

    cxx_type: str
    python_proxy: str
    helper_module: Any
    borrowed_proxy_helper: str

    def proxy_to_address(self, proxy: object) -> int:
        pointer = getattr(proxy, "_nsb_pointer", None)
        if not callable(pointer):
            raise TypeError(f"state field for {self.cxx_type} requires a {self.python_proxy} proxy")
        address = int(pointer())
        if address == 0:
            raise ValueError(f"state field for {self.cxx_type} received a null C++ address")
        return address

    def address_to_proxy(self, address: int) -> object:
        if address == 0:
            raise ReferenceError(f"state field for {self.cxx_type} has no live C++ object")
        return getattr(self.helper_module, self.borrowed_proxy_helper)(int(address))


def object_bridges_from_descriptor(modules: Mapping[str, Any], descriptor_path: str | Path) -> dict[str, DeclaredObjectBridge]:
    descriptor = json.loads(Path(descriptor_path).read_text())
    if descriptor.get("kind") != "numba_swig_ordinary_bridge_descriptor":
        raise ValueError("ordinary descriptor is invalid")
    records = descriptor.get("classes")
    if not isinstance(records, list):
        raise ValueError("ordinary descriptor has no classes")
    module_map = resolve_modules(modules, records)
    bridges: dict[str, DeclaredObjectBridge] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        target_module = module_map[record["owner_module"]]
        cxx_type = record.get("cxx_type")
        proxy = record.get("python_proxy")
        helper = record.get("borrowed_proxy_helper")
        if not all(isinstance(value, str) and value for value in (cxx_type, proxy, helper)):
            raise ValueError("ordinary descriptor has incomplete object bridge metadata")
        if not isinstance(getattr(target_module, proxy, None), type) or not callable(getattr(target_module, helper, None)):
            raise RuntimeError(f"target module has no object bridge helpers for {cxx_type}")
        bridges[cxx_type] = DeclaredObjectBridge(cxx_type, proxy, target_module, helper)
    return bridges


class DirectorStateProperty:
    def __init__(self, *, kind: str | None = None, slot: int, bridge: DeclaredObjectBridge | None = None) -> None:
        self.kind = kind
        self.slot = slot
        self.bridge = bridge
        self.name = ""

    @classmethod
    def scalar(cls, kind: str, slot: int):
        return cls(kind=kind, slot=slot)

    @classmethod
    def object(cls, slot: int, bridge: DeclaredObjectBridge):
        return cls(slot=slot, bridge=bridge)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @property
    def is_object(self) -> bool:
        return self.bridge is not None

    def __get__(self, instance: object | None, owner: type | None = None) -> object:
        if instance is None:
            return self
        state = instance._require_nsb_director_state()
        if not self.is_object:
            assert self.kind is not None
            return _python_scalar_get(state, self.kind, self.slot)
        address = int(state.ObjectAddress(self.slot))
        generation = int(state.ObjectGeneration(self.slot))
        if address == 0:
            instance._object_cache.pop(self.slot, None)
            instance._object_keepalives.pop(self.slot, None)
            raise ReferenceError(f"native state object field {self.name!r} has no live C++ owner")
        cached = instance._object_cache.get(self.slot)
        if cached is not None and cached[0] == generation:
            return cached[1]
        assert self.bridge is not None
        proxy = self.bridge.address_to_proxy(address)
        proxy.thisown = False
        instance._object_cache[self.slot] = (generation, proxy)
        if int(state.ObjectOwnershipCode(self.slot)) != PYTHON_ASSIGNED:
            instance._object_keepalives.pop(self.slot, None)
        return proxy

    def __set__(self, instance: object, value: object) -> None:
        state = instance._require_nsb_director_state()
        if not self.is_object:
            assert self.kind is not None
            _python_scalar_set(state, self.kind, self.slot, value)
            return
        assert self.bridge is not None
        address = self.bridge.proxy_to_address(value)
        state.SetObjectFromPython(self.slot, address)
        generation = int(state.ObjectGeneration(self.slot))
        instance._object_cache[self.slot] = (generation, value)
        instance._object_keepalives[self.slot] = value


def _python_scalar_get(state: Any, kind: str, slot: int) -> object:
    return {
        "float": lambda: float(state.Float(slot)),
        "double": lambda: float(state.Double(slot)),
        "int": lambda: int(state.Int(slot)),
        "int64": lambda: int(state.Int64(slot)),
        "bool": lambda: bool(state.Bool(slot)),
    }[kind]()


def _python_scalar_set(state: Any, kind: str, slot: int, value: object) -> None:
    if kind == "float":
        state.SetFloat(slot, float(value))
    elif kind == "double":
        state.SetDouble(slot, float(value))
    elif kind == "int":
        state.SetInt(slot, int(value))
    elif kind == "int64":
        state.SetInt64(slot, int(value))
    elif kind == "bool":
        state.SetBool(slot, bool(value))
    else:
        raise ValueError(f"unsupported native state scalar kind {kind!r}")


def make_state_facade(
    name: str,
    *,
    backend: StateBackend,
    scalar_fields: Mapping[str, str],
    object_fields: Mapping[str, DeclaredObjectBridge],
    array_fields: Mapping[str, types.Array] = {},
) -> type:
    """Create a Python state facade backed by one registered state provider."""

    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("state facade name must be a Python identifier")
    if not isinstance(backend, StateBackend):
        raise TypeError("state facade requires a registered StateBackend")
    scalar_fields = dict(scalar_fields)
    object_fields = dict(object_fields)
    array_fields = dict(array_fields)
    names = tuple(scalar_fields) + tuple(object_fields) + tuple(array_fields)
    if any(not isinstance(field, str) or not field.isidentifier() for field in names):
        raise ValueError("state field names must be Python identifiers")
    if len(set(names)) != len(names):
        raise ValueError("state field names must be unique")
    if any(kind not in SCALAR_KINDS for kind in scalar_fields.values()):
        raise ValueError("state scalar fields have an unsupported kind")
    if any(not isinstance(bridge, DeclaredObjectBridge) for bridge in object_fields.values()):
        raise TypeError("object state fields require a declared ordinary bridge type")
    if any(not isinstance(array, types.Array) or array.ndim not in {1, 2, 3} or not array.mutable
           for array in array_fields.values()):
        raise TypeError("array state fields require writable one-, two-, or three-dimensional Numba arrays")

    counters = {kind: 0 for kind in SCALAR_KINDS}
    scalar_slots: dict[str, tuple[str, int]] = {}
    for field, kind in scalar_fields.items():
        scalar_slots[field] = (kind, counters[kind])
        counters[kind] += 1
    object_slots = {field: index for index, field in enumerate(object_fields)}
    array_slots = {field: index for index, field in enumerate(array_fields)}

    def _initialize_nsb_director_state(self) -> None:
        if getattr(self, "_nsb_director_state", None) is not None:
            raise RuntimeError("native state facade is already initialized")
        self._nsb_director_state = backend.module.StateHandle(
            counters["double"], counters["int"], counters["int64"], counters["bool"], len(object_fields), counters["float"], len(array_fields),
            NRT_MEMINFO_RELEASE_ADDRESS,
        )
        self._object_cache: dict[int, tuple[int, object]] = {}
        self._object_keepalives: dict[int, object] = {}
        self._array_keepalives: dict[int, object] = {}

    def _require_nsb_director_state(self):
        state = getattr(self, "_nsb_director_state", None)
        if state is None or not state.IsAlive():
            raise RuntimeError("native state facade is closed")
        return state

    def close(self) -> None:
        state = getattr(self, "_nsb_director_state", None)
        if state is not None:
            state.Close()
        self._object_cache.clear()
        self._object_keepalives.clear()
        self._array_keepalives.clear()
        self._nsb_director_state = None

    @property
    def is_alive(self) -> bool:
        state = getattr(self, "_nsb_director_state", None)
        return bool(state is not None and state.IsAlive())

    namespace: dict[str, Any] = {
        "_initialize_nsb_director_state": _initialize_nsb_director_state,
        "_require_nsb_director_state": _require_nsb_director_state,
        "close": close,
        "is_alive": is_alive,
        "_scalar_slots": scalar_slots,
        "_object_slots": object_slots,
        "_object_bridges": object_fields,
        "_array_slots": array_slots,
        "_array_types": array_fields,
        "_nsb_state_backend": backend,
    }
    namespace.update(
        {field: DirectorStateProperty.scalar(kind, slot) for field, (kind, slot) in scalar_slots.items()}
    )
    namespace.update(
        {field: DirectorStateProperty.object(slot, object_fields[field]) for field, slot in object_slots.items()}
    )
    for field, slot in array_slots.items():
        array_type = array_fields[field]
        def getter(self, field=field, slot=slot):
            return self._array_keepalives[slot]
        def setter(self, value, field=field, slot=slot, array_type=array_type):
            import numpy as np
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(array_type.dtype.name):
                raise TypeError(f"array state field {field!r} requires dtype {array_type.dtype}")
            if value.ndim != array_type.ndim or not value.flags.writeable:
                raise TypeError(f"array state field {field!r} requires a writable array with {array_type.ndim} dimensions")
            if array_type.layout == "C" and not value.flags.c_contiguous:
                raise TypeError(f"array state field {field!r} requires C-contiguous storage")
            if array_type.layout == "F" and not value.flags.f_contiguous:
                raise TypeError(f"array state field {field!r} requires Fortran-contiguous storage")
            shape = tuple(value.shape) + (0,) * (3 - value.ndim)
            strides = tuple(value.strides) + (0,) * (3 - value.ndim)
            self._require_nsb_director_state().SetArrayFromPython(slot, int(value.ctypes.data), value.ndim, *shape, *strides)
            self._array_keepalives[slot] = value
        namespace[field] = property(getter, setter)
    return type(name, (), namespace)
