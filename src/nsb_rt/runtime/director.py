"""Generic runtime for generated Numba directors.

The ordinary bridge is installed first.  A director callback therefore sees the
same typed borrowed C++ objects as ordinary ``@njit`` code.  Native state owns
scalar slots and borrowed object addresses on the C++ side; Python merely
provides generation-aware proxy views of those slots.
"""

from __future__ import annotations

from dataclasses import dataclass
import atexit
import ctypes
import dis
import json
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
import warnings
import re
import weakref

from numba import cfunc, njit, types
from numba.core import cgutils
from llvmlite import ir
from numba.core.errors import TypingError
from numba.core.datamodel import models
from numba.extending import intrinsic, make_attribute_wrapper, overload, overload_attribute, lower_setattr, register_model

from . import director_state as state_runtime
from . import ordinary as ordinary_runtime
from .state_inference import InferenceState, infer_object_fields
from .modules import resolve_modules
from nsb_rt.backends import StateBackend, state_backend


_PY_INCREF_ADDRESS = int(ctypes.cast(ctypes.pythonapi.Py_IncRef, ctypes.c_void_p).value or 0)
_PY_DECREF_ADDRESS = int(ctypes.cast(ctypes.pythonapi.Py_DecRef, ctypes.c_void_p).value or 0)


class NumbaSwigBridgeFallbackWarning(UserWarning):
    """A selected non-pure virtual uses its inherited C++ implementation."""


@dataclass(frozen=True)
class DirectorMethod:
    identity: str
    cxx_name: str
    parameter_types: tuple[str, ...]
    abstract: bool
    parameters: tuple[dict[str, Any], ...]
    returns: dict[str, Any]

    @property
    def selector(self) -> tuple[str, tuple[str, ...]]:
        return self.cxx_name, self.parameter_types


@dataclass
class DirectorBinding:
    bridge_id: str
    cxx_type: str
    python_facade: str
    target_module: Any
    base_proxy: type
    borrowed_proxy_helper: Callable[[int], object]
    methods: tuple[DirectorMethod, ...]
    object_bridges: dict[str, state_runtime.DeclaredObjectBridge]
    nsb_factory: Callable[..., object]
    helper_module: Any
    state_backend: StateBackend
    constructors: tuple[tuple[dict[str, Any], ...], ...] = ()

    def method_for_selector(self, selector: tuple[str, tuple[str, ...]]) -> DirectorMethod:
        matches = [method for method in self.methods if method.selector == selector]
        if len(matches) != 1:
            raise KeyError(selector)
        return matches[0]


_BINDINGS: dict[tuple[str, str], DirectorBinding] = {}
_LIVE_DIRECTORS: weakref.WeakSet = weakref.WeakSet()


def _close_live_directors_at_exit() -> None:
    """Release native state before Numba shuts down its NRT runtime."""

    for instance in tuple(_LIVE_DIRECTORS):
        try:
            instance.close()
        except Exception:
            pass
    helpers = {binding.helper_module for binding in _BINDINGS.values()}
    for helper in helpers:
        try:
            helper.nsb_deactivate_external_directors()
        except Exception:
            pass


atexit.register(_close_live_directors_at_exit)


def _nsb_factory_name(cxx_type: str) -> str:
    return "create_nsb_director_" + re.sub(
        r"[^A-Za-z0-9_]", "_", cxx_type.replace("::", "_")
    ).strip("_")


def _load_descriptor(path: str | Path) -> dict[str, Any]:
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get("kind") != "numba_swig_director_descriptor" or descriptor.get("schema_version") != 2:
        raise ValueError("invalid Numba director descriptor")
    if not isinstance(descriptor.get("bridge_id"), str) or not isinstance(descriptor.get("directors"), list):
        raise ValueError("director descriptor is incomplete")
    return descriptor


def _ordinary_classes(path: str | Path) -> dict[str, dict[str, Any]]:
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get("kind") != "numba_swig_ordinary_bridge_descriptor":
        raise ValueError("ordinary descriptor is invalid")
    classes = descriptor.get("classes")
    if not isinstance(classes, list):
        raise ValueError("ordinary descriptor has no class list")
    return {
        entry["cxx_type"]: entry
        for entry in classes
        if isinstance(entry, dict) and isinstance(entry.get("cxx_type"), str)
    }


def install_director_bindings(
    modules: Mapping[str, Any], director_descriptor_path: str | Path, ordinary_descriptor_path: str | Path,
    *, helper_module: Any,
) -> dict[str, DirectorBinding]:
    """Install generated director metadata after ordinary object types exist."""

    descriptor = _load_descriptor(director_descriptor_path)
    backend_id = descriptor.get("state_backend")
    if not isinstance(backend_id, str) or not backend_id:
        raise ValueError("director descriptor has no state backend identifier")
    backend = state_backend(backend_id)
    if helper_module.nsb_director_state_abi_version() != backend.state_abi:
        raise RuntimeError(
            f"generated director helper has state ABI "
            f"{helper_module.nsb_director_state_abi_version()}; backend "
            f"{backend_id!r} requires {backend.state_abi}"
        )
    ordinary_descriptor = ordinary_runtime._load_descriptor(ordinary_descriptor_path)
    if ordinary_descriptor["bridge_id"] != descriptor["bridge_id"]:
        raise ValueError("ordinary and director descriptors have different bridge identities")
    module_map = resolve_modules(modules, descriptor["directors"])
    ordinary = _ordinary_classes(ordinary_descriptor_path)
    object_bridges = state_runtime.object_bridges_from_descriptor(modules, ordinary_descriptor_path)
    bridge_id = descriptor["bridge_id"]
    installed: dict[str, DirectorBinding] = {}
    for record in descriptor["directors"]:
        if not isinstance(record, dict):
            raise ValueError("director descriptor contains an invalid record")
        target_module = module_map[record["owner_module"]]
        cxx_type = record.get("cxx_type")
        facade = record.get("python_facade")
        proxy_name = record.get("python_proxy")
        method_records = record.get("methods")
        ordinary_record = ordinary.get(cxx_type)
        if not all(isinstance(value, str) for value in (cxx_type, facade, proxy_name)):
            raise ValueError("director record has incomplete type metadata")
        if not isinstance(method_records, list) or ordinary_record is None:
            raise ValueError(f"director {cxx_type} has no matching ordinary bridge record")
        if ordinary_record["owner_module"] != record["owner_module"]:
            raise ValueError(f"director {cxx_type} has inconsistent module ownership")
        helper_name = ordinary_record.get("borrowed_proxy_helper")
        if not isinstance(helper_name, str):
            raise ValueError(f"ordinary bridge record for {cxx_type} has no proxy helper")
        try:
            proxy = getattr(target_module, proxy_name)
            borrowed_helper = getattr(target_module, helper_name)
            factory = getattr(helper_module, _nsb_factory_name(cxx_type))
        except AttributeError as error:
            raise RuntimeError(f"native bridge has no generated helper or factory for {cxx_type}") from error
        methods: list[DirectorMethod] = []
        for method in method_records:
            if not isinstance(method, dict):
                raise ValueError(f"director {cxx_type} has an invalid method")
            identity = method.get("identity")
            cxx_name = method.get("cxx_name")
            parameters = method.get("parameters")
            returns = method.get("return")
            if not isinstance(identity, str) or not isinstance(cxx_name, str):
                raise ValueError(f"director {cxx_type} method has no identity")
            if not isinstance(parameters, list) or not isinstance(returns, dict):
                raise ValueError(f"director method {identity} is incomplete")
            parameter_types = tuple(
                (parameter.get("cxx_type") if parameter.get("kind") == "object"
                 else parameter.get("declared_cxx_type", parameter.get("cxx_type")))
                for parameter in parameters if isinstance(parameter, dict)
            )
            if len(parameter_types) != len(parameters) or not all(isinstance(item, str) for item in parameter_types):
                raise ValueError(f"director method {identity} has invalid parameter metadata")
            methods.append(DirectorMethod(
                identity, cxx_name, parameter_types, method.get("abstract") is True,
                tuple(dict(item) for item in parameters), dict(returns),
            ))
        raw_constructors = record.get("constructors")
        if raw_constructors is None:
            # Descriptors generated before constructor overload support carry
            # one constructor directly. Keep them loadable.
            raw_constructors = [{"parameters": record.get("constructor_parameters", [])}]
        if not isinstance(raw_constructors, list) or not raw_constructors:
            raise ValueError(f"director {cxx_type} has no constructor metadata")
        constructors: list[tuple[dict[str, Any], ...]] = []
        for constructor in raw_constructors:
            parameters = constructor.get("parameters") if isinstance(constructor, dict) else None
            if not isinstance(parameters, list) or not all(isinstance(item, dict) for item in parameters):
                raise ValueError(f"director {cxx_type} has invalid constructor metadata")
            constructors.append(tuple(dict(item) for item in parameters))
        binding = DirectorBinding(
            bridge_id, cxx_type, facade, target_module, proxy, borrowed_helper,
            tuple(sorted(methods, key=lambda item: item.identity)), dict(object_bridges), factory, helper_module, backend,
            tuple(constructors),
        )
        key = (bridge_id, cxx_type)
        current = _BINDINGS.get(key)
        if current is not None and current != binding:
            raise RuntimeError(f"director binding already exists for {bridge_id}:{cxx_type}")
        _BINDINGS[key] = binding
        installed[cxx_type] = binding
    return installed


def director_binding(bridge_id: str, cxx_type: str) -> DirectorBinding:
    try:
        return _BINDINGS[(bridge_id, cxx_type)]
    except KeyError as error:
        raise ValueError(f"no installed director binding for {bridge_id}:{cxx_type}") from error


def _normalize_stateaccess(value: object) -> tuple[str, ...] | str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("stateaccess must be a comma-separated string, '*', or None")
    if value == "":
        return ()
    if value.strip() == "*":
        return "*"
    names = tuple(part.strip() for part in value.split(","))
    if any(not name for name in names):
        raise TypeError("stateaccess contains an empty field name")
    if "*" in names:
        raise TypeError("stateaccess='*' must not be combined with field names")
    if any(not name.isidentifier() for name in names):
        raise TypeError("stateaccess field names must be Python identifiers")
    if len(set(names)) != len(names):
        raise TypeError("stateaccess field names must be unique")
    return names


def override(*arguments, stateaccess=None):
    """Select a C++ virtual override; callback arguments cannot have defaults."""

    access = _normalize_stateaccess(stateaccess)

    def validate(function):
        defaults = [name for name, parameter in inspect.signature(function).parameters.items()
                    if parameter.default is not inspect.Parameter.empty]
        if defaults:
            raise TypeError(
                "@nsb.override does not accept default arguments "
                f"({', '.join(defaults)}). Native calls use the C++ defaults "
                "and supply every callback argument explicitly."
            )

    if len(arguments) == 1 and callable(arguments[0]):
        function = arguments[0]
        validate(function)
        function.__numba_swig_override_selector__ = None
        function.__numba_swig_override_stateaccess__ = access
        return function
    if not arguments:
        if stateaccess is None:
            raise TypeError("@override expects a C++ method name and parameter types")
        selector = None
    elif not isinstance(arguments[0], str) or any(
        not isinstance(argument, str) for argument in arguments[1:]
    ):
        raise TypeError("@override expects a C++ method name and parameter types")
    else:
        selector = (arguments[0], tuple(arguments[1:]))

    def decorate(function):
        if not callable(function):
            raise TypeError("@override can decorate only a callable")
        validate(function)
        function.__numba_swig_override_selector__ = selector
        function.__numba_swig_override_stateaccess__ = access
        return function

    return decorate


class _StateViewType(types.Type):
    """Compiled callback view of one typed director-state schema."""

    def __init__(
        self,
        backend: StateBackend,
        scalar_slots: Mapping[str, tuple[str, int]],
        object_slots: Mapping[str, tuple[int, ordinary_runtime.BridgeObjectType]],
        array_slots: Mapping[str, tuple[int, types.Array]],
        cached_slots: Mapping[str, types.Type] | None = None,
        borrowed_fields=(),
    ) -> None:
        self.backend = backend
        self.scalar_slots = dict(scalar_slots)
        self.object_slots = dict(object_slots)
        self.array_slots = dict(array_slots)
        self.cached_slots = dict(cached_slots or {})
        self.borrowed_fields = frozenset(borrowed_fields)
        self._key = (
            self.backend.identifier,
            tuple(self.scalar_slots.items()),
            tuple((name, slot, object_type.key) for name, (slot, object_type) in self.object_slots.items()),
            tuple((name, slot, array_type) for name, (slot, array_type) in self.array_slots.items()),
            tuple(self.cached_slots.items()),
            tuple(sorted(self.borrowed_fields)),
        )
        super().__init__(name=f"numba_swig_bridge.state{self._key!r}")

    @property
    def key(self):
        return self._key


@register_model(_StateViewType)
class _StateViewModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [
            ("opaque", types.voidptr),
            *((f"cached_{index}", value_type)
              for index, value_type in enumerate(fe_type.cached_slots.values())),
        ])


make_attribute_wrapper(_StateViewType, "opaque", "opaque")


def is_slot_occupied(state, field):
    """Test an object or array director-state field without reading it.

    This is available only inside a Numba director override. ``field`` must be
    a literal declared state-field name, for example
    ``nsb.is_slot_occupied(self, "workspace")``.
    """

    raise TypeError("nsb.is_slot_occupied() is available inside Numba director overrides")


@overload(is_slot_occupied)
def _overload_is_slot_occupied(state, field):
    if isinstance(state, InferenceState) and isinstance(field, types.StringLiteral):
        if field.literal_value not in state.fields:
            raise TypingError(f"undeclared state field {field.literal_value!r}")
        def analysis_occupied(state, field):
            return True
        return analysis_occupied
    if not isinstance(state, _StateViewType) or not isinstance(field, types.StringLiteral):
        return None
    name = field.literal_value
    if name in state.object_slots:
        slot, _ = state.object_slots[name]
        reader = state_runtime.state_object_address_at(state.backend)

        def implementation(state, field):
            return reader(state.opaque, slot) != 0

        return implementation
    if name in state.array_slots:
        slot, _ = state.array_slots[name]
        reader = state_runtime.state_array_address_at(state.backend)

        def implementation(state, field):
            return reader(state.opaque, slot) != 0

        return implementation
    if name in state.scalar_slots:
        def implementation(state, field):
            return True

        return implementation
    raise TypingError(f"{state.name} has no declared state field {name!r}")


def _state_view_constructor(state_type: _StateViewType):
    cached_types = tuple(state_type.cached_slots.values())

    if not cached_types:
        @intrinsic
        def empty_constructor(typingctx, opaque):
            if opaque != types.voidptr:
                return None
            signature = state_type(opaque)

            def codegen(context, builder, sig, args):
                result = cgutils.create_struct_proxy(state_type)(context, builder)
                result.opaque = args[0]
                return result._getvalue()

            return signature, codegen

        return empty_constructor

    @intrinsic
    def constructor(typingctx, opaque, cached):
        if opaque != types.voidptr or not isinstance(cached, types.BaseTuple):
            return None
        if tuple(cached) != cached_types:
            return None
        signature = state_type(opaque, cached)

        def codegen(context, builder, sig, args):
            result = cgutils.create_struct_proxy(state_type)(context, builder)
            result.opaque = args[0]
            values = cgutils.unpack_tuple(builder, args[1], len(cached_types))
            for index, value in enumerate(values):
                setattr(result, f"cached_{index}", value)
            return result._getvalue()

        return signature, codegen

    return constructor


def _install_state_attributes(state_type: _StateViewType) -> None:
    def scalar_lower(field):
        def lower(attribute):
            if not isinstance(attribute, _StateViewType) or field not in attribute.scalar_slots:
                return None
            if field in attribute.cached_slots:
                index = tuple(attribute.cached_slots).index(field)

                @intrinsic
                def cached_reader(typingctx, value):
                    if value != attribute:
                        return None
                    signature = attribute.cached_slots[field](value)

                    def codegen(context, builder, sig, args):
                        proxy = cgutils.create_struct_proxy(attribute)(context, builder, value=args[0])
                        return getattr(proxy, f"cached_{index}")

                    return signature, codegen

                def implementation(attribute):
                    return cached_reader(attribute)

                return implementation
            kind, slot = attribute.scalar_slots[field]
            reader = state_runtime.scalar_at(kind, attribute.backend)

            def implementation(attribute):
                return reader(attribute.opaque, slot)

            return implementation

        return lower

    for name, (kind, slot) in state_type.scalar_slots.items():
        del kind, slot
        overload_attribute(_StateViewType, name)(scalar_lower(name))

    def object_lower(field):
        def lower(attribute):
            if not isinstance(attribute, _StateViewType) or field not in attribute.object_slots:
                return None
            if field in attribute.cached_slots:
                index = tuple(attribute.cached_slots).index(field)

                @intrinsic
                def cached_reader(typingctx, value):
                    if value != attribute:
                        return None
                    signature = attribute.cached_slots[field](value)

                    def codegen(context, builder, sig, args):
                        proxy = cgutils.create_struct_proxy(attribute)(context, builder, value=args[0])
                        return getattr(proxy, f"cached_{index}")

                    return signature, codegen

                def implementation(attribute):
                    return cached_reader(attribute)

                return implementation
            slot, object_type = attribute.object_slots[field]
            reader = state_runtime.state_object_at(
                attribute.backend, object_type, borrowed=field in attribute.borrowed_fields,
            )

            def implementation(attribute):
                return reader(attribute.opaque, slot)

            return implementation

        return lower

    for name, (slot, object_type) in state_type.object_slots.items():
        del slot, object_type
        overload_attribute(_StateViewType, name)(object_lower(name))

    def object_setter(field):
        @lower_setattr(_StateViewType, field)
        def lower(context, builder, sig, args):
            slot, object_type = sig.args[0].object_slots[field]
            backend = sig.args[0].backend
            state = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
            if field in sig.args[0].borrowed_fields:
                function = cgutils.get_or_insert_function(
                    builder.module, ir.FunctionType(ir.IntType(32), [
                        state.opaque.type, ir.IntType(32), ir.IntType(types.uintp.bitwidth),
                    ]), name=backend.symbol_name("set_object_borrowed"),
                )
                status = builder.call(function, [state.opaque, ir.Constant(ir.IntType(32), slot),
                    builder.ptrtoint(args[1], ir.IntType(types.uintp.bitwidth))])
                ordinary_runtime._raise_on_failure(context, builder, status)
                return context.get_dummy_value()
            mpi_type = getattr(sys.modules.get("nsb_rt.runtime.mpi"), "comm_type", None)
            if object_type == mpi_type:
                if sig.args[1] != mpi_type:
                    from numba.core.errors import TypingError
                    raise TypingError("communicator state requires a communicator value")
                # StateBlock consumes a reference; the RHS retains its own.
                context.nrt.incref(builder, sig.args[1], args[1])
            value = cgutils.create_struct_proxy(sig.args[1])(context, builder, value=args[1])
            # Constructed bridge objects transfer their result ownership to
            # StateBlock directly; the setter must not add another reference.
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(32), [
                    state.opaque.type, ir.IntType(32), ir.IntType(types.uintp.bitwidth), ir.IntType(8).as_pointer(),
                ]),
                name=backend.symbol_name("set_object_native"),
            )
            status = builder.call(function, [
                state.opaque, ir.Constant(ir.IntType(32), slot),
                builder.ptrtoint(value.pointer, ir.IntType(types.uintp.bitwidth)),
                builder.bitcast(value.meminfo, ir.IntType(8).as_pointer()),
            ])
            if object_type == mpi_type:
                with builder.if_then(builder.icmp_unsigned("!=", status, ir.Constant(status.type, 0))):
                    context.nrt.decref(builder, sig.args[1], args[1])
            ordinary_runtime._raise_on_failure(context, builder, status)
            return context.get_dummy_value()

        return lower

    for name in state_type.object_slots:
        object_setter(name)

    def array_lower(field):
        def lower(attribute):
            if not isinstance(attribute, _StateViewType) or field not in attribute.array_slots:
                return None
            if field in attribute.cached_slots:
                index = tuple(attribute.cached_slots).index(field)

                @intrinsic
                def cached_reader(typingctx, value):
                    if value != attribute:
                        return None
                    signature = attribute.cached_slots[field](value)

                    def codegen(context, builder, sig, args):
                        proxy = cgutils.create_struct_proxy(attribute)(context, builder, value=args[0])
                        return getattr(proxy, f"cached_{index}")

                    return signature, codegen

                def implementation(attribute):
                    return cached_reader(attribute)

                return implementation
            slot, array_type = attribute.array_slots[field]
            reader = state_runtime.state_array_at(attribute.backend, array_type)
            def implementation(attribute):
                return reader(attribute.opaque, slot)
            return implementation
        return lower
    for name in state_type.array_slots:
        overload_attribute(_StateViewType, name)(array_lower(name))

    def array_setter(field):
        slot, array_type = state_type.array_slots[field]

        @lower_setattr(_StateViewType, field)
        def lower(context, builder, sig, args):
            value = context.make_array(sig.args[1])(context, builder, value=args[1])
            state = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
            # StateBlock keeps this array after the assignment expression has
            # released its temporary NRT ownership.
            context.nrt.incref(builder, sig.args[1], args[1])
            shape = list(cgutils.unpack_tuple(builder, value.shape, array_type.ndim))
            strides = list(cgutils.unpack_tuple(builder, value.strides, array_type.ndim))
            zero = ir.Constant(ir.IntType(types.intp.bitwidth), 0)
            shape.extend([zero] * (3 - array_type.ndim))
            strides.extend([zero] * (3 - array_type.ndim))
            function = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(32), [
                    state.opaque.type, ir.IntType(32), ir.IntType(types.uintp.bitwidth), ir.IntType(8).as_pointer(),
                    ir.IntType(32), *([ir.IntType(types.intp.bitwidth)] * 6),
                ]),
                name=state_type.backend.symbol_name("set_array_native"),
            )
            status = builder.call(function, [
                state.opaque, ir.Constant(ir.IntType(32), slot),
                builder.ptrtoint(value.data, ir.IntType(types.uintp.bitwidth)),
                builder.bitcast(value.meminfo, ir.IntType(8).as_pointer()),
                ir.Constant(ir.IntType(32), array_type.ndim), *shape, *strides,
            ])
            ordinary_runtime._raise_on_failure(context, builder, status)
            return context.get_dummy_value()

        return lower

    for name in state_type.array_slots:
        array_setter(name)


def _scalar_type(parameter: dict[str, Any]):
    mapping = {
        "float": types.float32,
        "double": types.float64,
        "int": types.intc,
        "int64": types.int64,
        "bool": types.boolean,
        "enum": types.intc,
    }
    try:
        return mapping[parameter["kind"]]
    except KeyError as error:
        raise RuntimeError(f"unsupported scalar callback parameter {parameter}") from error


def _pointer_type(parameter: dict[str, Any]):
    dtype = parameter.get("numpy_dtype")
    mapping = {"float32": types.float32, "float64": types.float64, "int32": types.int32, "bool": types.boolean}
    try:
        return types.CPointer(mapping[dtype])
    except KeyError as error:
        raise RuntimeError(f"unsupported pointer callback dtype {dtype!r}") from error


def _callback_type(binding: DirectorBinding, parameter: dict[str, Any]):
    kind = parameter["kind"]
    if kind == "object":
        ordinary_runtime.bridge_object_type(binding.bridge_id, parameter["cxx_type"], const=parameter.get("const", False))
        return types.voidptr
    if kind == "pointer":
        return _pointer_type(parameter)
    return _scalar_type(parameter)


def _callback_result_type(method: DirectorMethod):
    """Return the C callback output-slot type for one generated virtual result."""

    kind = method.returns.get("kind")
    mapping = {
        "void": types.void,
        "float": types.float32,
        "double": types.float64,
        "int": types.intc,
        "int64": types.int64,
        "enum": types.intc,
        # Native C++ converts the int result to bool.  This avoids relying on
        # a cross-language bool return ABI for the C callback itself.
        "bool": types.intc,
        "object": types.voidptr,
    }
    try:
        return mapping[kind]
    except KeyError as error:
        raise RuntimeError(f"unsupported Numba director callback return {method.returns}") from error


def _compile_callback(
    binding: DirectorBinding, method: DirectorMethod, override,
    state_constructor, live_state_constructor, cached_fields: tuple[str, ...],
):
    compiled = njit(cache=False)(override)
    cfunc_types = [types.voidptr, *(_callback_type(binding, parameter) for parameter in method.parameters)]
    returns_void = method.returns.get("kind") == "void"
    if not returns_void:
        cfunc_types.append(types.CPointer(_callback_result_type(method)))
    signature = types.intc(*cfunc_types)
    names = [f"arg{index}" for index in range(len(method.parameters))]
    converted: list[str] = []
    object_constructors: dict[str, Any] = {}
    result_address = None
    for index, (parameter, name) in enumerate(zip(method.parameters, names)):
        if parameter["kind"] == "object":
            constructor_name = f"_object_constructor_{index}"
            object_constructors[constructor_name] = ordinary_runtime.object_from_address(
                ordinary_runtime.bridge_object_type(binding.bridge_id, parameter["cxx_type"], const=parameter.get("const", False), force_const=True)
            )
            converted.append(f"{constructor_name}({name})")
        else:
            converted.append(name)
    if method.returns.get("kind") == "object":
        result_address = ordinary_runtime.director_result_address(binding.bridge_id, method.returns)
    parameter_text = ", ".join(["opaque_state", *names])
    if cached_fields:
        cached_arguments = ", ".join(f"live_state.{name}" for name in cached_fields)
        state_expression = f"_state_constructor(opaque_state, ({cached_arguments},))"
        state_setup = "    live_state = _live_state_constructor(opaque_state)\n"
    else:
        state_expression = "_state_constructor(opaque_state)"
        state_setup = ""
    call_text = ", ".join([state_expression, *converted])
    if returns_void:
        callback_statement = f'''{state_setup}    try:
        _compiled({call_text})
        return 1
    except Exception:
        return 0'''
    else:
        result_expression = (
            f"_result_address(_compiled({call_text}))"
            if result_address is not None else f"_compiled({call_text})"
        )
        callback_statement = f'''{state_setup}    try:
        result[0] = {result_expression}
        return 1
    except Exception:
        return 0'''
    source = "\n".join([
        "@cfunc(_signature, cache=False)",
        f"def callback({parameter_text}{', result' if not returns_void else ''}):",
        callback_statement,
    ])
    namespace = {
        "cfunc": cfunc,
        "_signature": signature,
        "_compiled": compiled,
        "_state_constructor": state_constructor,
        "_live_state_constructor": live_state_constructor,
        "_result_address": result_address,
        **object_constructors,
    }
    exec(source, namespace)
    return compiled, namespace["callback"]


def _state_field_usage(function: Callable[..., Any], fields: set[str]) -> tuple[set[str], set[str]]:
    """Conservatively find statically named state reads and assignments.

    Numba supports calls to already-jitted helpers, so follow referenced Python
    functions and dispatcher ``py_func`` objects that receive the state, as
    well as nested code objects. Direct reads are tied to the state receiver.
    Assignments are deliberately conservative because a false negative would
    make a later same-callback read stale.
    """

    reads: set[str] = set()
    writes: set[str] = set()
    seen: set[int] = set()

    def visit_code(code, namespace):
        if id(code) in seen:
            return
        seen.add(id(code))
        instructions = tuple(dis.get_instructions(code))
        state_names = {code.co_varnames[0]} if code.co_argcount else set()
        changed = True
        while changed:
            changed = False
            for previous, instruction in zip(instructions, instructions[1:]):
                if (previous.opname in {"LOAD_FAST", "LOAD_DEREF"}
                        and previous.argval in state_names
                        and instruction.opname in {"STORE_FAST", "STORE_DEREF"}
                        and instruction.argval not in state_names):
                    state_names.add(instruction.argval)
                    changed = True
        for index, instruction in enumerate(instructions):
            if (instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
                    and instruction.argval in fields and index
                    and instructions[index - 1].opname in {"LOAD_FAST", "LOAD_DEREF"}
                    and instructions[index - 1].argval in state_names):
                reads.add(instruction.argval)
            elif instruction.opname == "STORE_ATTR" and instruction.argval in fields:
                writes.add(instruction.argval)
        for constant in code.co_consts:
            if inspect.iscode(constant):
                visit_code(constant, namespace)
        for name in code.co_names:
            candidate = namespace.get(name)
            candidate = getattr(candidate, "py_func", candidate)
            passes_state = False
            for index, instruction in enumerate(instructions):
                if instruction.opname == "LOAD_GLOBAL" and instruction.argval == name:
                    for argument in instructions[index + 1:index + 20]:
                        if argument.opname == "CALL":
                            break
                        if (argument.opname in {"LOAD_FAST", "LOAD_DEREF"}
                                and argument.argval in state_names):
                            passes_state = True
                            break
                if passes_state:
                    break
            if inspect.isfunction(candidate) and passes_state:
                visit_code(candidate.__code__, candidate.__globals__)

    visit_code(function.__code__, function.__globals__)
    return reads, writes


def _collect_overrides(user_class: type, binding: DirectorBinding) -> dict[str, Callable[..., Any]]:
    marked = [
        value for value in user_class.__dict__.values()
        if callable(value) and hasattr(value, "__numba_swig_override_selector__")
    ]
    available = ", ".join(
        f"{method.cxx_name}({', '.join(method.parameter_types)})" for method in binding.methods
    )
    result: dict[str, Callable[..., Any]] = {}
    for function in marked:
        selector = function.__numba_swig_override_selector__
        if selector is None:
            # Automatic director selection can expose several virtuals. A bare
            # decorator remains concise when the Python method spells one
            # unique C++ virtual name exactly, for example ``def Mult(...)``.
            # Overloads and differently named Python methods still need an
            # explicit selector.
            name_matches = [
                candidate for candidate in binding.methods
                if candidate.cxx_name == function.__name__
            ]
            if len(name_matches) == 1:
                method = name_matches[0]
            elif len(binding.methods) == 1:
                method = binding.methods[0]
            else:
                raise TypeError("bare @override is ambiguous; use @override(" + available + ")")
        else:
            try:
                method = binding.method_for_selector(selector)
            except KeyError as error:
                raise TypeError(
                    f"@override{selector!r} is not selected on {binding.cxx_type}; "
                    f"supported selected methods: {available}"
                ) from error
        if method.identity in result:
            raise TypeError(f"duplicate @override for {method.identity}")
        result[method.identity] = function
    missing = [method.identity for method in binding.methods if method.abstract and method.identity not in result]
    if missing:
        raise TypeError(f"{user_class.__name__} is missing required @override: {', '.join(missing)}")
    return result


def _validate_state(
    state: Mapping[str, Any], binding: DirectorBinding
) -> tuple[dict[str, str], dict[str, state_runtime.DeclaredObjectBridge], dict[str, types.Array], dict[str, DirectorBinding]]:
    """Resolve scalar, ordinary-object, and nested-director state declarations.

    ``state={"buffer": fixture.Buffer}`` stores one borrowed Buffer address.
    ``state={"inner": other_binding}`` stores the same address but additionally
    asks the runtime to retain a Numba director dependency edge.
    """

    if not isinstance(state, Mapping) or not state:
        raise TypeError("state must be a non-empty mapping of supported native fields")
    scalar_fields: dict[str, str] = {}
    object_fields: dict[str, state_runtime.DeclaredObjectBridge] = {}
    array_fields: dict[str, types.Array] = {}
    nested_fields: dict[str, DirectorBinding] = {}
    proxies = {
        getattr(bridge.helper_module, bridge.python_proxy): bridge
        for bridge in binding.object_bridges.values()
    }
    for name, annotation in state.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError("state field names must be Python identifiers")
        scalar_kind = state_runtime.scalar_kind_from_annotation(annotation)
        if scalar_kind is not None:
            scalar_fields[name] = scalar_kind
        elif isinstance(annotation, types.Array):
            array_fields[name] = annotation
        elif (mpi_runtime := sys.modules.get("nsb_rt.runtime.mpi")) is not None and mpi_runtime.is_communicator_annotation(annotation):
            object_fields[name] = mpi_runtime.communicator_state_bridge()
        elif isinstance(annotation, state_runtime.DeclaredObjectBridge):
            if annotation.cxx_type not in binding.object_bridges:
                raise TypeError(f"state field {name!r} uses an object outside this bridge descriptor")
            object_fields[name] = annotation
        elif isinstance(annotation, DirectorBinding):
            if annotation.bridge_id != binding.bridge_id:
                raise TypeError(f"nested director state field {name!r} comes from another bridge")
            try:
                object_fields[name] = binding.object_bridges[annotation.cxx_type]
            except KeyError as error:
                raise TypeError(f"nested director state field {name!r} has no ordinary bridge type") from error
            nested_fields[name] = annotation
        elif annotation in proxies:
            object_fields[name] = proxies[annotation]
        else:
            raise TypeError(
                f"state field {name!r} must be numba.types.float32/float64/intc/int64/boolean, "
                "a generated SWIG bridge proxy class, or a DirectorBinding"
            )
    return scalar_fields, object_fields, array_fields, nested_fields


class _NestedDirectorStateProperty(state_runtime.DirectorStateProperty):
    """An object state slot whose value must be a matching decorated director."""

    def __init__(self, slot: int, bridge: state_runtime.DeclaredObjectBridge, expected: DirectorBinding) -> None:
        super().__init__(slot=slot, bridge=bridge)
        self.expected = expected

    def __set__(self, instance: object, value: object) -> None:
        if (getattr(instance, "_nsb_director", None) is not None
                and not getattr(instance, "_nsb_initializing", False)):
            raise RuntimeError("nested director state cannot be replaced after the outer director is live")
        if getattr(value, "_numba_swig_bridge_binding", None) is not self.expected:
            raise TypeError(
                f"nested director state field {self.name!r} requires a director for {self.expected.cxx_type}"
            )
        value._require_nsb_director()
        super().__set__(instance, value.as_base_proxy())
        instance._nested_directors[self.name] = value


def _binding_for_base(base: type) -> DirectorBinding:
    """Find the installed director record owned by one SWIG proxy class."""

    matches = [binding for binding in _BINDINGS.values() if binding.base_proxy is base]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise TypeError(
            "the direct base has no installed Numba director binding; "
            "import its registration module or pass binding= explicitly"
        )
    raise TypeError(
        "the direct base has multiple Numba director bindings; pass binding= explicitly"
    )


def director(*, binding: DirectorBinding | None = None, state: Mapping[str, Any],
             base: type | None = None, fallback: str = "warn"):
    """Decorate a Python class as a selected native C++ director.

    A direct subclass of a registered SWIG director base can omit ``binding``.
    The binding is then selected from that sole direct base.  Composition-style
    directors still provide both ``base`` and ``binding`` explicitly.
    """

    if binding is not None and not isinstance(binding, DirectorBinding):
        raise TypeError("binding must be a generated director binding")
    if fallback not in {"warn", "silent", "error"}:
        raise ValueError("fallback must be 'warn', 'silent', or 'error'")
    configured_binding = binding
    configured_base = base

    def decorate(user_class: type) -> type:
        if not isinstance(user_class, type):
            raise TypeError("@director can decorate only a class")
        resolved_binding = configured_binding
        resolved_base = configured_base
        inferred_base = resolved_base is None
        if resolved_binding is None:
            if resolved_base is not None:
                raise TypeError("a director without binding= must directly inherit its registered SWIG base")
            direct_bases = user_class.__bases__
            if len(direct_bases) != 1:
                raise TypeError("a director without binding= must directly inherit one registered SWIG base")
            resolved_base = direct_bases[0]
            resolved_binding = _binding_for_base(resolved_base)
        elif resolved_base is None:
            resolved_base = resolved_binding.base_proxy
        if resolved_base is not resolved_binding.base_proxy:
            raise TypeError("base and binding must come from the same generated director record")

        scalar_fields, object_fields, array_fields, nested_fields = _validate_state(state, resolved_binding)
        binding = resolved_binding
        base = resolved_base
        subclass_style = issubclass(user_class, base)
        if inferred_base and not subclass_style:
            raise TypeError("a director without base= must subclass its binding's SWIG base")
        if subclass_style and user_class.__bases__ != (base,):
            raise TypeError("a subclass director must directly inherit only its binding's SWIG base")
        overrides = _collect_overrides(user_class, binding)
        policies = infer_object_fields(binding, overrides, scalar_fields, object_fields, array_fields) if object_fields else {}
        borrowed_fields = {name for name, (mode, _) in policies.items() if mode == "borrowed"}
        fallbacks = [method for method in binding.methods if not method.abstract and method.identity not in overrides]
        if fallbacks:
            names = ", ".join(method.identity for method in fallbacks)
            message = f"{user_class.__name__} does not override {names}; the C++ base implementation will be used"
            if fallback == "error":
                raise TypeError(message)
            if fallback == "warn":
                warnings.warn(message, NumbaSwigBridgeFallbackWarning, stacklevel=2)

        state_facade = state_runtime.make_state_facade(
            binding.python_facade, backend=binding.state_backend,
            scalar_fields=scalar_fields, object_fields=object_fields, array_fields=array_fields
        )
        state_facade._borrowed_fields = frozenset(borrowed_fields)
        state_facade._const_borrowed_fields = frozenset(
            name for name, (mode, const) in policies.items() if mode == "borrowed" and const
        )
        scalar_slots = dict(state_facade._scalar_slots)
        object_slots = {
            name: (state_facade._object_slots[name], bridge.numba_state_type if hasattr(bridge, "numba_state_type") else ordinary_runtime.bridge_object_type(
                binding.bridge_id, bridge.cxx_type, const=policies.get(name, (None, False))[1], force_const=True))
            for name, bridge in object_fields.items()
        }
        array_slots = {
            name: (state_facade._array_slots[name], array_fields[name]) for name in array_fields
        }
        all_fields = set(scalar_slots) | set(object_slots) | set(array_slots)
        compiled: dict[str, Any] = {}
        callbacks: dict[str, Any] = {}
        for method in binding.methods:
            override = overrides.get(method.identity)
            if override is not None:
                access = override.__numba_swig_override_stateaccess__
                if access is None:
                    authorized = all_fields
                    cached_names: tuple[str, ...] = ()
                else:
                    authorized = all_fields if access == "*" else set(access)
                    unknown = authorized - all_fields
                    if unknown:
                        names = ", ".join(repr(name) for name in sorted(unknown))
                        raise TypeError(f"stateaccess contains undeclared state field(s): {names}")
                    reads, writes = _state_field_usage(override, all_fields)
                    cache_candidates = reads - writes
                    order = tuple(state) if access == "*" else access
                    cached_names = tuple(name for name in order if name in cache_candidates)

                restricted_scalars = {
                    name: value for name, value in scalar_slots.items() if name in authorized
                }
                restricted_objects = {
                    name: value for name, value in object_slots.items() if name in authorized
                }
                restricted_arrays = {
                    name: value for name, value in array_slots.items() if name in authorized
                }
                live_state_type = _StateViewType(
                    binding.state_backend, restricted_scalars, restricted_objects, restricted_arrays,
                    borrowed_fields=borrowed_fields,
                )
                cached_slots = {}
                for name in cached_names:
                    if name in restricted_scalars:
                        cached_slots[name] = state_runtime.scalar_numba_type(restricted_scalars[name][0])
                    elif name in restricted_objects:
                        object_type = restricted_objects[name][1]
                        cached_slots[name] = (object_type if name in borrowed_fields or object_type == getattr(sys.modules.get("nsb_rt.runtime.mpi"), "comm_type", None) else
                            ordinary_runtime.BorrowedReturnBridgeObjectType(object_type.bridge_id, object_type.cxx_type))
                    else:
                        cached_slots[name] = restricted_arrays[name][1]
                state_type = _StateViewType(
                    binding.state_backend, restricted_scalars, restricted_objects, restricted_arrays, cached_slots,
                    borrowed_fields=borrowed_fields,
                )
                _install_state_attributes(live_state_type)
                if state_type != live_state_type:
                    _install_state_attributes(state_type)
                compiled[method.identity], callbacks[method.identity] = _compile_callback(
                    binding, method, override, _state_view_constructor(state_type),
                    _state_view_constructor(live_state_type), cached_names,
                )
        user_init = user_class.__dict__.get("__init__")

        def _callback_addresses() -> tuple[int, ...]:
            return tuple(
                int(callbacks[method.identity].address) if method.identity in callbacks else 0
                for method in binding.methods
            )

        constructor_signatures = tuple(
            inspect.Signature([
                inspect.Parameter(parameter.get("name") or f"arg{index}",
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for index, parameter in enumerate(constructor)
            ])
            for constructor in binding.constructors
        )

        def _bind_constructor(*args, **kwargs):
            matches = []
            for signature in constructor_signatures:
                try:
                    matches.append(signature.bind(*args, **kwargs))
                except TypeError:
                    pass
            if not matches:
                raise TypeError(
                    f"{binding.cxx_type} has no constructor accepting the supplied arguments"
                )
            # SWIG resolves overloads which share the same Python argument
            # shape (for example, ``int`` versus ``double``). Here we only
            # normalize positional/keyword arguments for that factory call.
            return matches[0].args

        def _construct_native(self, arguments):
            if self._nsb_director is not None:
                raise RuntimeError("C++ director base is already initialized")
            state = self._require_nsb_director_state()
            lease_operations = (
                "lease_address", "lease_is_alive", "lease_close", "lease_release",
                "object_address_at",
            )
            native_operations = tuple(
                binding.state_backend.function_address(operation)
                for operation in lease_operations
            )
            lease = int(state.AcquireLease())
            if lease == 0:
                raise RuntimeError("native state backend could not acquire a director lease")
            director = binding.nsb_factory(
                *arguments, *_callback_addresses(), lease, *native_operations
            )
            if director is None or not director.IsAlive():
                raise RuntimeError("generated Numba director factory rejected its constructor arguments, state or callbacks")
            self._nsb_director = director
            if subclass_style:
                proxy = binding.borrowed_proxy_helper(int(director.BaseAddress()))
                proxy.thisown = False
                self._nsb_swig_this = proxy.this
            return director

        def _base_init(self, *args, **kwargs):
            if not getattr(self, "_nsb_initializing", False):
                raise RuntimeError("C++ base initialization is only allowed during director construction")
            _construct_native(self, _bind_constructor(*args, **kwargs))

        def __init__(self, *args, **kwargs):
            if subclass_style:
                base_arguments = None
            else:
                base_arguments = kwargs.pop("base_args", ())
                if not isinstance(base_arguments, (tuple, list)):
                    raise TypeError("base_args must be a tuple or list of C++ constructor arguments")
                base_arguments = _bind_constructor(*base_arguments)
            self._initialize_nsb_director_state()
            self._nsb_director = None
            self._nsb_initializing = True
            self._nsb_callbacks = callbacks
            self._nested_directors: dict[str, object] = {}
            self._retained_nested: list[object] = []
            self._nested_keepalives: list[object] = []
            self._nested_owner_count = 0
            self._close_requested = False
            self._state_closed = False
            try:
                if user_init is not None:
                    user_init(self, *args, **kwargs)
                elif subclass_style:
                    _base_init(self, *args, **kwargs)
                elif args or kwargs:
                    raise TypeError(f"{user_class.__name__} accepts no constructor arguments")
                if subclass_style:
                    if self._nsb_director is None:
                        raise RuntimeError("director initializer must call super().__init__()")
                    director = self._nsb_director
                else:
                    director = _construct_native(self, base_arguments)
                for field, expected in nested_fields.items():
                    try:
                        inner = self._nested_directors[field]
                    except KeyError as error:
                        raise RuntimeError(f"outer director did not assign its declared nested state {field!r}") from error
                    if getattr(inner, "_numba_swig_bridge_binding", None) is not expected:
                        raise TypeError(f"nested state {field!r} does not match its declared director binding")
                    retained = binding.helper_module.retain_nsb_director_dependency(
                        director,
                        inner._require_nsb_director(),
                        state_facade._object_slots[field],
                    )
                    if not retained:
                        raise RuntimeError(f"generated Numba director rejected nested state field {field!r}")
                    inner._retain_for_outer()
                    self._retained_nested.append(inner)
                    self._nested_keepalives.extend([
                        inner, inner._nsb_callbacks, inner._numba_compiled_overrides,
                        inner._object_keepalives,
                    ])
                _LIVE_DIRECTORS.add(self)
            except BaseException:
                self.close()
                raise
            finally:
                self._nsb_initializing = False

        def _require_nsb_director(self):
            director = getattr(self, "_nsb_director", None)
            if director is None or not director.IsAlive():
                raise RuntimeError("Numba director is closed")
            return director

        def _profile_helper(self, identity):
            self._require_nsb_director()
            matches = [item for item in binding.methods
                       if item.identity == identity or item.cxx_name == identity]
            if len(matches) != 1:
                raise ValueError(f"{binding.cxx_type} has no director callback {identity!r}")
            method = matches[0]
            symbol = "".join(
                character if character.isalnum() or character == "_" else "_"
                for character in method.identity
            ).strip("_")
            return symbol, binding.helper_module

        def enable_callback_profile(self, identity, enabled=True):
            """Enable or disable native inclusive timing for one callback."""
            symbol, helper = _profile_helper(self, identity)
            getattr(helper, f"nsb_director_profile_enable_{symbol}")(bool(enabled))

        def reset_callback_profile(self, identity):
            """Clear the native accumulated timing and call count for one callback."""
            symbol, helper = _profile_helper(self, identity)
            getattr(helper, f"nsb_director_profile_reset_{symbol}")()

        def callback_profile(self, identity):
            """Return native callback count and inclusive elapsed seconds."""
            symbol, helper = _profile_helper(self, identity)
            calls = int(getattr(helper, f"nsb_director_profile_calls_{symbol}")())
            nanoseconds = int(getattr(helper, f"nsb_director_profile_nanoseconds_{symbol}")())
            return {"calls": calls, "seconds": nanoseconds / 1_000_000_000.0}

        def _retain_for_outer(self) -> None:
            self._nested_owner_count += 1

        def _release_outer_owner(self) -> None:
            if self._nested_owner_count <= 0:
                raise RuntimeError("nested director owner count is already zero")
            self._nested_owner_count -= 1
            if self._nested_owner_count == 0 and self._close_requested:
                _finalize_close(self)

        def _finalize_close(self) -> None:
            """Release a closed director after every C++ owner edge is gone."""
            if self._state_closed:
                return
            for inner in tuple(self._retained_nested):
                inner._release_outer_owner()
            self._retained_nested.clear()
            self._nested_directors.clear()
            self._nested_keepalives.clear()
            state_facade.close(self)
            self._state_closed = True
            self._nsb_callbacks = {}
            _LIVE_DIRECTORS.discard(self)

        def as_base_proxy(self):
            address = int(self._require_nsb_director().BaseAddress())
            if address == 0:
                raise RuntimeError("Numba director is closed")
            if subclass_style:
                return self
            proxy = binding.borrowed_proxy_helper(address)
            proxy.thisown = False
            return proxy

        def close(self):
            if getattr(self, "_close_requested", False):
                return
            self._close_requested = True
            director = getattr(self, "_nsb_director", None)
            transferred_live = False
            if director is not None:
                transferred_live = not director.OwnsDirector() and director.IsAlive()
                director.Close()
            self._nsb_director = None
            if transferred_live:
                _LIVE_DIRECTORS.discard(self)
            elif getattr(self, "_nested_owner_count", 0) == 0:
                _finalize_close(self)

        def get_thisown(self):
            return bool(self._require_nsb_director().OwnsDirector())

        def set_thisown(self, value):
            director = self._require_nsb_director()
            if bool(value):
                if not director.OwnsDirector():
                    raise ValueError(
                        "re-adopting externally owned Numba directors is not supported"
                    )
                return
            director.ReleaseOwnership(id(self), _PY_INCREF_ADDRESS, _PY_DECREF_ADDRESS)

        @property
        def is_alive(self):
            director = getattr(self, "_nsb_director", None)
            return bool(director is not None and director.IsAlive() and state_facade.is_alive.__get__(self))

        namespace = {
            "__doc__": user_class.__doc__,
            "__module__": user_class.__module__,
            "__init__": __init__,
            "_require_nsb_director": _require_nsb_director,
            "enable_callback_profile": enable_callback_profile,
            "reset_callback_profile": reset_callback_profile,
            "callback_profile": callback_profile,
            "_retain_for_outer": _retain_for_outer,
            "_release_outer_owner": _release_outer_owner,
            "_finalize_close": _finalize_close,
            "as_base_proxy": as_base_proxy,
            "close": close,
            "is_alive": is_alive,
            "thisown": property(get_thisown, set_thisown),
            "_numba_swig_bridge_binding": binding,
            "_numba_callbacks": callbacks,
            "_numba_compiled_overrides": compiled,
            "_override_sources": overrides,
        }
        for field, expected in nested_fields.items():
            descriptor = _NestedDirectorStateProperty(
                state_facade._object_slots[field], object_fields[field], expected
            )
            descriptor.__set_name__(state_facade, field)
            namespace[field] = descriptor
        if subclass_style:
            # Preserve class identity and __class__ cells so both forms of super()
            # reach this per-director initializer without changing the SWIG base.
            adapter = type(binding.python_facade + "Base", (base, state_facade),
                           {"__init__": _base_init})

            def swig_this(self):
                self._require_nsb_director()
                return self._nsb_swig_this

            def native_pointer(self):
                return int(self._require_nsb_director().BaseAddress())

            namespace["_nsb_pointer"] = native_pointer
            namespace["this"] = property(swig_this)
            user_class.__bases__ = (adapter,)
            for name, value in namespace.items():
                setattr(user_class, name, value)
            return user_class
        facade = type(binding.python_facade, (state_facade,), namespace)
        facade.__qualname__ = user_class.__qualname__
        return facade

    return decorate
