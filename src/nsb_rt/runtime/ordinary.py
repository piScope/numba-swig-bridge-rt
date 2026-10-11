"""Generic Numba runtime for generated ordinary-method descriptors.

The descriptor supplies target-specific proxy names, C++ type spellings,
adapter symbols, and checked lowering records. This module owns no target class
or method spelling. C ABI adapters compiled into the target SWIG extension
perform every native C++ call.
"""

from __future__ import annotations

import json
import operator
import functools
import inspect
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .modules import resolve_modules

from llvmlite import binding, ir
from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.typeconv import Conversion
from numba.core.typing import signature
from numba.core.typing.templates import infer_global
from numba.extending import (
    NativeValue,
    box,
    models,
    intrinsic,
    overload,
    overload_method,
    overload_attribute,
    lower_setattr,
    register_model,
    typeof_impl,
    unbox,
    type_callable,
    lower_builtin,
    lower_cast,
)


class BridgeObjectType(types.Type):
    """A typed borrowed pointer to one selected C++ class."""

    def __init__(self, bridge_id: str, cxx_type: str) -> None:
        self.bridge_id = bridge_id
        self.cxx_type = cxx_type
        super().__init__(name=f"numba_swig_bridge.borrowed[{bridge_id}:{cxx_type}]")

    @property
    def key(self) -> tuple[str, str]:
        return self.bridge_id, self.cxx_type

    def can_convert_to(self, typingctx, other):
        if (type(self) is BridgeObjectType and isinstance(other, ConstBridgeObjectType)
                and self.key == other.key):
            return Conversion.safe

    def unify(self, typingctx, other):
        if (type(self) in (BridgeObjectType, ConstBridgeObjectType)
                and type(other) in (BridgeObjectType, ConstBridgeObjectType)
                and self.key == other.key):
            return bridge_object_type(self.bridge_id, self.cxx_type,
                const=isinstance(self, ConstBridgeObjectType) or isinstance(other, ConstBridgeObjectType),
                force_const=True)


class ConstBridgeObjectType(BridgeObjectType):
    """A const borrowed view preserving C++ callback and state constness."""

    def __init__(self, bridge_id, cxx_type):
        super().__init__(bridge_id, cxx_type)
        self.name = f"numba_swig_bridge.const[{bridge_id}:{cxx_type}]"


@box(ConstBridgeObjectType)
def _box_const(typ, value, c):
    raise TypingError("const bridge views cannot escape as writable Python proxies")


@lower_cast(BridgeObjectType, ConstBridgeObjectType)
def _lower_borrowed_to_const(context, builder, fromty, toty, value):
    if fromty.key != toty.key or isinstance(fromty, (OwnedBridgeObjectType, BorrowedReturnBridgeObjectType)):
        raise TypingError("const conversion requires matching borrowed object types")
    return value


class OwnedBridgeObjectType(BridgeObjectType):
    """A frame-owned selected C++ object held by Numba NRT storage."""

    def __init__(self, bridge_id: str, cxx_type: str) -> None:
        super().__init__(bridge_id, cxx_type)
        self.name = f"numba_swig_bridge.owned[{bridge_id}:{cxx_type}]"


class OwnedReturnBridgeObjectType(OwnedBridgeObjectType):
    """A frame-owned object explicitly transferred to Python by ``own_this``."""

    def __init__(self, bridge_id: str, cxx_type: str) -> None:
        super().__init__(bridge_id, cxx_type)
        self.name = f"numba_swig_bridge.owned-return[{bridge_id}:{cxx_type}]"


class BorrowedReturnBridgeObjectType(BridgeObjectType):
    """A borrowed result that retains an NRT-owned parent until it is boxed."""

    def __init__(self, bridge_id: str, cxx_type: str) -> None:
        super().__init__(bridge_id, cxx_type)
        self.name = f"numba_swig_bridge.borrowed-return[{bridge_id}:{cxx_type}]"


@lower_cast(OwnedBridgeObjectType, BorrowedReturnBridgeObjectType)
def _lower_owned_to_borrowed_return(context, builder, fromty, toty, value):
    """Give a state assignment a retained handle to a constructed object."""
    if fromty.key != toty.key:
        raise TypingError(f"cannot assign {fromty.cxx_type} to state field {toty.cxx_type}")
    source = cgutils.create_struct_proxy(fromty)(context, builder, value=value)
    storage = context.nrt.meminfo_data(builder, source.meminfo)
    pointer = builder.load(
        builder.bitcast(storage, ir.IntType(8).as_pointer().as_pointer())
    )
    context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), source.meminfo)
    result = cgutils.create_struct_proxy(toty)(context, builder)
    result.meminfo = source.meminfo
    result.pointer = pointer
    return result._getvalue()


@register_model(ConstBridgeObjectType)
@register_model(BridgeObjectType)
class BorrowedPointerModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(8).as_pointer())


@register_model(OwnedBridgeObjectType)
class OwnedPointerModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("meminfo", types.MemInfoPointer(types.voidptr))])


@register_model(OwnedReturnBridgeObjectType)
class OwnedReturnPointerModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [
            ("meminfo", types.MemInfoPointer(types.voidptr)),
            ("pointer", types.voidptr),
        ])


@register_model(BorrowedReturnBridgeObjectType)
class BorrowedReturnPointerModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [
            ("meminfo", types.MemInfoPointer(types.voidptr)),
            ("pointer", types.voidptr),
        ])


def _unbox_borrowed_pointer(typ, obj, c):
    # A closed director can raise while resolving its pointer. Preserve that
    # exception rather than passing a NULL PyObject to the next Python C API.
    address_slot = cgutils.alloca_once_value(c.builder, ir.Constant(ir.IntType(64), 0))
    method = c.pyapi.object_getattr_string(obj, "_nsb_pointer")
    with c.builder.if_then(cgutils.is_not_null(c.builder, method)):
        pointer_object = c.pyapi.call_function_objargs(method, ())
        with c.builder.if_then(cgutils.is_not_null(c.builder, pointer_object)):
            address = c.pyapi.long_as_ulonglong(pointer_object)
            c.builder.store(address, address_slot)
            c.pyapi.decref(pointer_object)
        c.pyapi.decref(method)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    native_pointer = c.builder.inttoptr(c.builder.load(address_slot), c.context.get_value_type(typ))
    return NativeValue(native_pointer, is_error=is_error)


@unbox(BridgeObjectType)
def unbox_bridge_object(typ, obj, c):
    return _unbox_borrowed_pointer(typ, obj, c)


@unbox(OwnedBridgeObjectType)
def unbox_owned_bridge_object(typ, obj, c):
    # Dispatchers create Python wrappers even for helpers called only from
    # compiled code. A Python proxy cannot supply an NRT-owned argument.
    c.pyapi.err_set_string("PyExc_TypeError", "frame-owned bridge arguments require a compiled caller")
    return NativeValue(c.context.get_constant_null(typ), is_error=cgutils.true_bit)


@unbox(BorrowedReturnBridgeObjectType)
def unbox_retained_borrow_bridge_object(typ, obj, c):
    # This specialization carries a native owner handle as well as a pointer.
    # A Python proxy uses the ordinary pointer specialization instead; it cannot
    # supply this value. Compiled callers pass the native structure directly.
    c.pyapi.err_set_string("PyExc_TypeError", "retained-borrow bridge arguments require a compiled caller")
    return NativeValue(c.context.get_constant_null(typ), is_error=cgutils.true_bit)


@dataclass(frozen=True)
class BridgeMethod:
    bridge_id: str
    owner: str
    owner_module: str
    function_address_helper: str
    identity: str
    symbol: str
    python_name: str
    const: bool
    returns: dict[str, Any]
    parameters: tuple[dict[str, Any], ...]
    abi_parameters: tuple[dict[str, Any], ...]


_OBJECT_TYPES: dict[tuple[str, str], BridgeObjectType] = {}
_MEMBERS = {}
_INSTALLED_MEMBER_NAMES = set()
_CONST_TYPES = {}
_OWNED_TYPES: dict[tuple[str, str], OwnedBridgeObjectType] = {}
_OWNED_RETURN_TYPES: dict[tuple[str, str], OwnedReturnBridgeObjectType] = {}
_BORROWED_RETURN_TYPES: dict[tuple[str, str], BorrowedReturnBridgeObjectType] = {}
_CONSTRUCTORS: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
_DESTROY_SYMBOLS: dict[tuple[str, str], str] = {}
_METHODS_BY_NAME: dict[str, list[BridgeMethod]] = {}
_INSTALLED_METHOD_NAMES: set[str] = set()
_INSTALLED_DESCRIPTORS: set[tuple] = set()
_OBJECT_OWNERS: dict[tuple[str, str], tuple[str, type]] = {}
_PROXY_TYPES: dict[type, BridgeObjectType] = {}
_BORROWED_PROXY_HELPERS: dict[tuple[str, str], Any] = {}
_OWNED_PROXY_HELPERS: dict[tuple[str, str], Any | None] = {}
_CLASS_INFO = {}
_CASTS = {}
_KEEPALIVE_BY_ID: dict[int, tuple[weakref.ReferenceType[object], object]] = {}


def _attach_keepalive(proxy: object, owner: object) -> None:
    """Keep ``owner`` alive for exactly as long as a boxed proxy remains live."""
    try:
        setattr(proxy, "_nsb_keepalive", owner)
        return
    except (AttributeError, TypeError):
        pass
    key = id(proxy)

    def released(_reference, *, key=key):
        _KEEPALIVE_BY_ID.pop(key, None)

    try:
        reference = weakref.ref(proxy, released)
    except TypeError as error:
        raise TypeError(
            "the SWIG proxy does not support NSB keep-alive attachment; "
            "enable proxy attributes or weak references"
        ) from error
    _KEEPALIVE_BY_ID[key] = (reference, owner)


def _proxy_from_address(key: tuple[str, str], address: int, owning: bool,
                        keepalive: object | None = None) -> object:
    """Create the normal SWIG proxy for a native address at a Python boundary."""
    if address == 0:
        raise ReferenceError(f"cannot box a null {key[1]} bridge object")
    helper = _OWNED_PROXY_HELPERS[key] if owning else _BORROWED_PROXY_HELPERS[key]
    if helper is None:
        raise RuntimeError(
            "this bridge was generated by an older numba-swig-bridge; "
            "regenerate its SWIG wrappers before using nsb.own_this()"
        )
    proxy = helper(address)
    # The generated borrowed helper has no ownership transfer and the generated
    # owned helper has SWIG's %newobject ownership. Set this explicitly as a
    # guard against wrapper-specific defaults.
    proxy.thisown = owning
    if keepalive is not None:
        _attach_keepalive(proxy, keepalive)
    return proxy


def _box_object_proxy(c, key: tuple[str, str], pointer, *, owning: bool,
                      keepalive=None):
    """Emit Python boxing for one native C++ object address."""
    function = c.pyapi.unserialize(c.pyapi.serialize_object(_proxy_from_address))
    key_object = c.pyapi.unserialize(c.pyapi.serialize_object(key))
    address = c.pyapi.long_from_ulonglong(
        c.builder.ptrtoint(pointer, ir.IntType(64))
    )
    ownership = c.pyapi.bool_from_bool(ir.Constant(ir.IntType(1), owning))
    arguments = (key_object, address, ownership)
    if keepalive is not None:
        arguments += (keepalive,)
    result = c.pyapi.call_function_objargs(function, arguments)
    c.pyapi.decref(function)
    c.pyapi.decref(key_object)
    c.pyapi.decref(address)
    c.pyapi.decref(ownership)
    return result


@box(BridgeObjectType)
def box_borrowed_bridge_object(typ, value, c):
    """Box ordinary pointer/reference results as normal non-owning SWIG proxies."""
    return _box_object_proxy(c, typ.key, value, owning=False)


def _reject_unmarked_owned_return() -> None:
    raise TypeError(
        "an NSB frame-owned C++ object cannot return to Python implicitly; "
        "return nsb.own_this(obj) to transfer its ownership"
    )


@box(OwnedBridgeObjectType)
def box_owned_bridge_object(typ, value, c):
    """Reject accidental Python escape of an NRT-owned C++ object."""
    function = c.pyapi.unserialize(c.pyapi.serialize_object(_reject_unmarked_owned_return))
    result = c.pyapi.call_function_objargs(function, ())
    c.pyapi.decref(function)
    return result


@box(OwnedReturnBridgeObjectType)
def box_owned_return_bridge_object(typ, value, c):
    """Transfer an explicitly marked frame-owned object into a SWIG proxy."""
    result = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=value)
    boxed = _box_object_proxy(c, typ.key, result.pointer, owning=True)
    # The generated NRT destructor deletes the C++ pointer stored in this
    # payload. Clearing it transfers that responsibility to the owning SWIG
    # proxy and prevents a second delete when the frame releases ``meminfo``.
    storage = c.context.nrt.meminfo_data(c.builder, result.meminfo)
    c.builder.store(
        ir.Constant(ir.IntType(8).as_pointer(), None),
        c.builder.bitcast(storage, ir.IntType(8).as_pointer().as_pointer()),
    )
    return boxed


@box(BorrowedReturnBridgeObjectType)
def box_borrowed_return_bridge_object(typ, value, c):
    """Box a child while retaining the NRT ``MemInfo`` of its declared owner."""
    result = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=value)
    keepalive = c.box(types.MemInfoPointer(types.voidptr), result.meminfo)
    boxed = _box_object_proxy(c, typ.key, result.pointer, owning=False, keepalive=keepalive)
    c.pyapi.decref(keepalive)
    return boxed


def _automatic_path(bridge_id, source, target):
    path = []
    seen = set()
    while source != target:
        if source in seen:
            return None
        seen.add(source)
        bases = _CLASS_INFO.get((bridge_id, source), {}).get("public_bases", [])
        if len(bases) != 1:
            return None
        base = bases[0]
        conversion = _CASTS.get((bridge_id, source, base, "upcast"))
        if conversion is None or not conversion["flags_value"] & 2:
            return None
        path.append(conversion)
        source = base
    return path


def _cast_pointer(builder, record, value, kind):
    function = cgutils.get_or_insert_function(
        builder.module, ir.FunctionType(ir.IntType(8).as_pointer(),
                                        [ir.IntType(8).as_pointer()]), name=record[kind])
    return builder.call(function, [value])


def _convert_pointer(context, builder, typ, target, value):
    if isinstance(typ, (OwnedReturnBridgeObjectType, BorrowedReturnBridgeObjectType)):
        owner = cgutils.create_struct_proxy(typ)(context, builder, value=value)
        value = owner.pointer
    elif isinstance(typ, OwnedBridgeObjectType):
        owner = cgutils.create_struct_proxy(typ)(context, builder, value=value)
        data = owner.meminfo
        # The NRT payload holds one C++ pointer written by the generated factory.
        function = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(8).as_pointer(), [data.type]), name="NRT_MemInfo_data_fast")
        storage = builder.call(function, [data])
        value = builder.load(builder.bitcast(storage, ir.IntType(8).as_pointer().as_pointer()))
    path = _automatic_path(typ.bridge_id, typ.cxx_type, target)
    if path is None:
        raise TypingError(f"no automatic conversion from {typ.cxx_type} to {target}")
    for record in path:
        value = _cast_pointer(builder, record, value, "upcast")
    return value


def upcast(value, destination):
    """Borrow a public, unambiguous base view inside @njit; retain the owner."""
    raise TypeError("upcast is available inside Numba-compiled functions")


def dynamic_cast(value, destination):
    """Borrow a checked derived view inside @njit; failed casts raise TypeError."""
    raise TypeError("dynamic_cast is available inside Numba-compiled functions")


@intrinsic
def require_not_none(typingctx, value):
    """Unwrap an optional borrowed object inside compiled code.

    This is useful where a C++ API documents a non-null result even though its
    pointer declaration must be represented generically as ``Optional[T]``.
    A null result raises ``ReferenceError`` instead of being dereferenced.
    """

    if not isinstance(value, types.Optional):
        return signature(value, value), lambda context, builder, sig, args: args[0]
    inner = value.type

    def codegen(context, builder, sig, args):
        optional = context.make_helper(builder, sig.args[0], args[0])
        valid = cgutils.as_bool_bit(builder, optional.valid)
        with builder.if_then(builder.not_(valid), likely=False):
            context.call_conv.return_user_exc(
                builder, ReferenceError, ("C++ bridge returned a null object pointer",)
            )
        return optional.data

    return inner(value), codegen


def own_this(value):
    """Transfer an NSB-created C++ object to an owning SWIG proxy on return.

    This marker is valid only inside Numba-compiled code.  It changes no C++
    pointer at the call site; the final boxer transfers the object by clearing
    its NRT destructor payload after it has created the owning SWIG proxy.
    """
    raise TypeError("nsb.own_this() is available inside Numba-compiled functions")


@intrinsic
def _own_this(typingctx, value):
    if not isinstance(value, OwnedBridgeObjectType) or isinstance(value, OwnedReturnBridgeObjectType):
        raise TypingError("nsb.own_this() requires an NSB frame-owned constructed object")
    target = _OWNED_RETURN_TYPES.setdefault(
        value.key, OwnedReturnBridgeObjectType(value.bridge_id, value.cxx_type)
    )
    result_signature = signature(target, value)

    def codegen(context, builder, sig, args):
        source = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
        storage = context.nrt.meminfo_data(builder, source.meminfo)
        pointer = builder.load(
            builder.bitcast(storage, ir.IntType(8).as_pointer().as_pointer())
        )
        # The marker and its source can coexist until the final Python-boundary
        # boxer runs, so give the marker its own NRT reference.
        context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), source.meminfo)
        result = cgutils.create_struct_proxy(target)(context, builder)
        result.meminfo = source.meminfo
        result.pointer = pointer
        return result._getvalue()

    return result_signature, codegen


@overload(own_this)
def _overload_own_this(value):
    if not isinstance(value, OwnedBridgeObjectType) or isinstance(value, OwnedReturnBridgeObjectType):
        return None

    def implementation(value):
        return _own_this(value)

    return implementation


_MISSING = object()


def borrowed_from(obj=_MISSING, owner=_MISSING, *, owner_arg=_MISSING):
    """Declare the owner of a borrowed result.

    ``borrowed_from(obj, owner)`` is the nopython form for an NRT-owned
    ``owner``.  ``borrowed_from(owner_arg=...)`` returns a Python decorator
    that retains the indicated SWIG argument after the compiled dispatcher has
    boxed its ordinary non-owning result.
    """
    if owner_arg is _MISSING:
        raise TypeError(
            "nsb.borrowed_from(obj, owner) is available inside Numba-compiled code; "
            "use nsb.borrowed_from(owner_arg=...) as a decorator"
        )
    if obj is not _MISSING or owner is not _MISSING:
        raise TypeError("nsb.borrowed_from(owner_arg=...) accepts no positional arguments")
    if not isinstance(owner_arg, (int, str)):
        raise TypeError("owner_arg must be an argument index or name")

    def decorate(dispatcher):
        signature_object = inspect.signature(getattr(dispatcher, "py_func", dispatcher))

        @functools.wraps(dispatcher)
        def call(*args, **kwargs):
            bound = signature_object.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                retained_owner = (args[owner_arg] if isinstance(owner_arg, int)
                                  else bound.arguments[owner_arg])
            except (IndexError, KeyError) as error:
                raise TypeError(
                    f"nsb.borrowed_from could not resolve owner argument {owner_arg!r}"
                ) from error
            result = dispatcher(*args, **kwargs)
            if result is not None:
                _attach_keepalive(result, retained_owner)
            return result

        return call

    return decorate


@intrinsic
def _borrowed_from_owned(typingctx, value, owner):
    if isinstance(value, ConstBridgeObjectType):
        raise TypingError("const bridge views cannot be converted to mutable borrowed results")
    if not isinstance(value, BridgeObjectType) or isinstance(value, OwnedBridgeObjectType):
        raise TypingError("nsb.borrowed_from() requires a borrowed bridge result")
    if isinstance(owner, BorrowedReturnBridgeObjectType):
        owner_meminfo_type = owner
    elif isinstance(owner, OwnedBridgeObjectType) and not isinstance(owner, OwnedReturnBridgeObjectType):
        owner_meminfo_type = owner
    else:
        raise TypingError(
            "nsb.borrowed_from() inside compiled code requires an NSB frame-owned owner; "
            "use @nsb.borrowed_from(owner_arg=...) for a Python/SWIG owner"
        )
    if value.bridge_id != owner.bridge_id:
        raise TypingError("nsb.borrowed_from() requires child and owner from the same bridge")
    target = _BORROWED_RETURN_TYPES.setdefault(
        value.key, BorrowedReturnBridgeObjectType(value.bridge_id, value.cxx_type)
    )
    result_signature = signature(target, value, owner)

    def codegen(context, builder, sig, args):
        owner_value = cgutils.create_struct_proxy(owner_meminfo_type)(
            context, builder, value=args[1]
        )
        context.nrt.incref(
            builder, types.MemInfoPointer(types.voidptr), owner_value.meminfo
        )
        result = cgutils.create_struct_proxy(target)(context, builder)
        result.meminfo = owner_value.meminfo
        result.pointer = args[0]
        return result._getvalue()

    return result_signature, codegen


@overload(borrowed_from)
def _overload_borrowed_from(value, owner=None, owner_arg=None):
    if isinstance(owner, types.Omitted):
        return None
    if not isinstance(value, BridgeObjectType):
        return None

    def implementation(value, owner=None, owner_arg=None):
        return _borrowed_from_owned(value, owner)

    return implementation


def _cast_implementation(value, destination, kind):
    if not isinstance(value, BridgeObjectType):
        return None
    if isinstance(destination, types.TypeRef):
        target = destination.instance_type
    elif isinstance(destination, types.Function):
        target = _PROXY_TYPES.get(destination.typing_key)
    else:
        return None
    if not isinstance(target, BridgeObjectType) or target.bridge_id != value.bridge_id:
        raise TypingError("casts require selected classes in the same bridge")
    if isinstance(value, ConstBridgeObjectType):
        target = _CONST_TYPES.setdefault(target.key, ConstBridgeObjectType(*target.key))
    record = None
    if target.cxx_type != value.cxx_type:
        record = _CASTS.get((value.bridge_id, value.cxx_type, target.cxx_type, kind))
        required = 1 if kind == "upcast" else 4
        if record is None or not record["flags_value"] & required:
            raise TypingError(f"unsupported {kind} from {value.cxx_type} to {target.cxx_type}")

    @intrinsic
    def caller(typingctx, obj):
        def codegen(context, builder, sig, args):
            result = args[0] if record is None else _cast_pointer(builder, record, args[0], kind)
            with builder.if_then(cgutils.is_null(builder, result), likely=False):
                context.call_conv.return_user_exc(builder, TypeError, ("C++ bridge cast failed",))
            return result
        return target(obj), codegen

    def implementation(value, destination):
        return caller(value)
    return implementation


@overload(upcast)
def _overload_upcast(value, destination):
    return _cast_implementation(value, destination, "upcast")


@overload(dynamic_cast)
def _overload_dynamic_cast(value, destination):
    return _cast_implementation(value, destination, "dynamic_cast")


def bridge_object_type(bridge_id: str, cxx_type: str, *, const=False, force_const=False) -> BridgeObjectType:
    """Return a registered borrowed-object Numba type.

    Native-director callbacks use this public runtime hook to turn one C++
    object address into the exact same typed value ordinary methods accept.
    """

    try:
        if const and (force_const or _MEMBERS.get((bridge_id, cxx_type))):
            return _CONST_TYPES.setdefault((bridge_id, cxx_type), ConstBridgeObjectType(bridge_id, cxx_type))
        return _OBJECT_TYPES[(bridge_id, cxx_type)]
    except KeyError as error:
        raise RuntimeError(f"no installed bridge type for {bridge_id}:{cxx_type}") from error


def object_from_address(object_type: BridgeObjectType):
    """Return an intrinsic that borrows an object at a native address."""

    @intrinsic
    def constructor(typingctx, address):
        if address not in {types.voidptr, types.uintp}:
            return None
        signature = object_type(address)

        def codegen(context, builder, sig, args):
            if sig.args[0] == types.uintp:
                return builder.inttoptr(args[0], context.get_value_type(object_type))
            return args[0]

        return signature, codegen

    return constructor


def director_result_address(bridge_id: str, specification: dict[str, Any]):
    """Return an intrinsic that converts a borrowed director result to ``void*``.

    A director virtual returning a selected C++ ``T&`` or ``T*`` keeps native
    ownership semantics: it gives C++ a borrowed address, never an NRT-owned
    object.  The intrinsic applies the same public upcast path as ordinary
    method arguments. A nullable C++ pointer accepts either a borrowed object
    or ``Optional[borrowed object]``; a reference rejects Optional at typing
    time so a null reference cannot be manufactured in Numba.
    """
    if specification.get("kind") != "object":
        raise ValueError("director object result requires an object descriptor")
    target = bridge_object_type(bridge_id, specification["cxx_type"])
    nullable = specification.get("passing") == "pointer"

    @intrinsic
    def converter(typingctx, value):
        value_type = value
        optional = isinstance(value_type, types.Optional)
        source = value_type.type if optional else value_type
        if isinstance(source, OwnedBridgeObjectType):
            raise TypingError(
                "a director object result must be borrowed; store a constructed object "
                "in director state before returning a pointer or reference to it"
            )
        if isinstance(source, ConstBridgeObjectType) and not specification.get("const", False):
            raise TypingError("cannot return a const bridge view as a mutable director result")
        if not isinstance(source, BridgeObjectType) or source.bridge_id != bridge_id:
            raise TypingError("director object result must be a selected borrowed bridge object")
        if optional and not nullable:
            raise TypingError("a C++ reference director result cannot be Optional")
        if _automatic_path(bridge_id, source.cxx_type, target.cxx_type) is None:
            raise TypingError(
                f"no automatic conversion from {source.cxx_type} to director result {target.cxx_type}"
            )

        signature = types.voidptr(value)

        def codegen(context, builder, sig, args):
            if not optional:
                return _convert_pointer(context, builder, source, target.cxx_type, args[0])
            source_value = context.make_helper(builder, value_type, args[0])
            result = cgutils.alloca_once_value(builder, cgutils.get_null_value(ir.IntType(8).as_pointer()))
            with builder.if_then(cgutils.as_bool_bit(builder, source_value.valid), likely=True):
                pointer = _convert_pointer(context, builder, source, target.cxx_type, source_value.data)
                builder.store(pointer, result)
            return builder.load(result)

        return signature, codegen

    return converter


def _numba_scalar_type(specification: dict[str, Any]):
    kind = specification["kind"]
    if kind == "mpi_comm":
        from .mpi import comm_type
        return comm_type
    if kind == "void":
        return types.none
    if kind == "float":
        return types.float32
    if kind == "double":
        return types.float64
    if kind == "int":
        return types.intc
    if kind == "int64":
        return types.int64
    if kind == "bool":
        return types.boolean
    raise RuntimeError(f"unsupported scalar ABI kind {kind!r}")


def _numba_result_type(bridge_id: str, specification: dict[str, Any]):
    if specification["kind"] == "object":
        object_type = bridge_object_type(bridge_id, specification["cxx_type"], const=specification.get("const", False), force_const=specification.get("preserve_const", False))
        if specification.get("nullable"):
            return types.Optional(object_type)
        return object_type
    if specification["kind"] == "scalar_pointer":
        return types.CPointer(_array_dtype(specification["numpy_dtype"]))
    return _numba_scalar_type(specification)


def _llvm_scalar_value_type(specification: dict[str, Any]):
    kind = specification["kind"]
    if kind == "void":
        return ir.VoidType()
    if kind == "float":
        return ir.FloatType()
    if kind == "double":
        return ir.DoubleType()
    if kind == "int":
        return ir.IntType(32)
    if kind == "int64":
        return ir.IntType(64)
    if kind == "bool":
        return ir.IntType(1)
    raise RuntimeError(f"unsupported LLVM scalar ABI kind {kind!r}")


def _llvm_abi_type(specification: dict[str, Any]):
    kind = specification["kind"]
    if kind in {"object", "array_data", "scalar_pointer", "mpi_comm"}:
        return ir.IntType(8).as_pointer()
    if kind == "array_count":
        return ir.IntType(64)
    return _llvm_scalar_value_type(specification)


def _llvm_result_type(specification: dict[str, Any]):
    if specification["kind"] in {"object", "mpi_comm"}:
        return ir.IntType(8).as_pointer()
    if specification["kind"] == "scalar_pointer":
        return _array_view_llvm_dtype(specification["numpy_dtype"]).as_pointer()
    if specification["kind"] == "bool":
        return ir.IntType(32)
    return _llvm_scalar_value_type(specification)


def _array_dtype(name: str):
    mapping = {
        "float32": types.float32,
        "float64": types.float64,
        "int32": types.int32,
        "bool": types.boolean,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise RuntimeError(f"unsupported generated array dtype {name!r}") from error


def _array_view_llvm_dtype(name: str):
    mapping = {
        "float32": ir.FloatType(),
        "float64": ir.DoubleType(),
        "int32": ir.IntType(32),
        # Numba stores boolean array elements as bytes.
        "bool": ir.IntType(8),
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise RuntimeError(f"unsupported generated array-view dtype {name!r}") from error


def _array_view_itemsize(name: str) -> int:
    return {"float32": 4, "float64": 8, "int32": 4, "bool": 1}[name]


def _matches(specification: dict[str, Any], value_type: types.Type, bridge_id=None) -> bool:
    kind = specification["kind"]
    if kind == "object":
        return (
            isinstance(value_type, BridgeObjectType)
            and value_type.bridge_id == bridge_id
            and (not isinstance(value_type, ConstBridgeObjectType) or specification.get("const", False))
            and _automatic_path(bridge_id, value_type.cxx_type, specification["cxx_type"]) is not None
        )
    if kind == "mpi_comm":
        from .mpi import CommunicatorType
        return isinstance(value_type, CommunicatorType)
    if kind == "array":
        return (
            isinstance(value_type, types.Array)
            and value_type.dtype == _array_dtype(specification["numpy_dtype"])
            and value_type.ndim == specification["rank"]
            and value_type.layout == specification["layout"]
        )
    if kind == "scalar_pointer":
        return isinstance(value_type, types.CPointer) and value_type.dtype == _array_dtype(specification["numpy_dtype"])
    if kind in {"float", "double"}:
        return isinstance(value_type, types.Float)
    if kind in {"int", "int64"}:
        return isinstance(value_type, types.Integer) and value_type != types.boolean
    if kind == "bool":
        return value_type == types.boolean
    return False


def _select_constructor(constructors: tuple[dict[str, Any], ...],
                        arguments: tuple[types.Type | None, ...], bridge_id: str):
    """Return the unique native constructor matching Numba call arguments.

    ``type_callable`` invokes a generated Python typer with only the supplied
    arguments.  Its remaining default positions therefore remain ordinary
    Python ``None``, rather than becoming ``types.Omitted``.  Ignore both
    representations so a shorter valid overload is not mistaken for a call
    with the class's maximum constructor arity.
    """

    supplied = tuple(
        argument for argument in arguments
        if argument is not None and not isinstance(argument, types.Omitted)
    )
    matches = [
        constructor for constructor in constructors
        if len(constructor["parameters"]) == len(supplied)
        and all(
            _matches(parameter, argument, bridge_id)
            for parameter, argument in zip(constructor["parameters"], supplied)
        )
    ]
    if len(matches) > 1:
        raise TypingError("constructors are ambiguous")
    return matches[0] if matches else None


def _display_parameter(specification: dict[str, Any]) -> str:
    kind = specification["kind"]
    if kind == "object":
        const = "const " if specification.get("const") else ""
        suffix = "&" if specification.get("passing") == "reference" else "*"
        return f"{const}{specification['cxx_type']}{suffix}"
    if kind == "array":
        return (
            f"numpy.ndarray[{specification['numpy_dtype']}, "
            f"{specification['rank']}d, {specification['layout']}]"
        )
    if kind == "scalar_pointer":
        return f"CPointer({specification['numpy_dtype']})"
    return specification["cxx_type"]


def _display_method(method: BridgeMethod) -> str:
    parameters = ", ".join(_display_parameter(item) for item in method.parameters)
    return f"{method.owner}::{method.python_name}({parameters})"


def _raise_on_failure(context, builder, status):
    failed = builder.icmp_unsigned("!=", status, ir.Constant(status.type, 0))
    with builder.if_then(failed, likely=False):
        context.call_conv.return_user_exc(
            builder,
            RuntimeError,
            ("generated C++ ordinary-object bridge adapter failed",),
        )


def _array_data_and_count(context, builder, array_type, value):
    """Read a checked Numba array's address and 1-D extent without Python."""

    array = context.make_array(array_type)(context, builder, value=value)
    data = builder.bitcast(array.data, ir.IntType(8).as_pointer())
    shape = cgutils.unpack_tuple(builder, array.shape, array_type.ndim)
    count = context.cast(builder, shape[0], types.intp, types.int64)
    return data, count


def _adapter_signature(method: BridgeMethod, argument_types: tuple[types.Type, ...]):
    if len(argument_types) != len(method.parameters) + 1:
        return None
    result_specification = method.returns
    if isinstance(argument_types[0], ConstBridgeObjectType):
        result_specification = {**method.returns, "preserve_const": True}
    if method.returns["kind"] == "array_view_1d":
        if not isinstance(argument_types[0], BridgeObjectType):
            return None
        dtype_name = method.returns["dtype"]
        array_type = types.Array(
            _array_dtype(dtype_name), 1, "C",
            readonly=method.returns["mutable"] == "false",
        )
        llvm_dtype = _array_view_llvm_dtype(dtype_name)
        itemsize = _array_view_itemsize(dtype_name)
        call_signature = signature(array_type, *argument_types)

        def codegen(context, builder, sig, arguments):
            adapter = cgutils.get_or_insert_function(
                builder.module,
                ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer(),
                                                  llvm_dtype.as_pointer().as_pointer(),
                                                  ir.IntType(64).as_pointer()]),
                name=method.symbol,
            )
            data_slot = cgutils.alloca_once(builder, llvm_dtype.as_pointer())
            size_slot = cgutils.alloca_once(builder, ir.IntType(64))
            self_pointer = _convert_pointer(context, builder, sig.args[0], method.owner, arguments[0])
            status = builder.call(adapter, [self_pointer, data_slot, size_slot])
            _raise_on_failure(context, builder, status)
            size = builder.load(size_slot)
            result = context.make_array(array_type)(context, builder)
            result.data = builder.load(data_slot)
            result.shape = cgutils.pack_array(builder, [size])
            result.strides = cgutils.pack_array(builder, [ir.Constant(ir.IntType(64), itemsize)])
            result.itemsize = ir.Constant(ir.IntType(64), itemsize)
            result.meminfo = cgutils.get_null_value(result.meminfo.type)
            result.parent = cgutils.get_null_value(result.parent.type)
            result.nitems = size
            return result._getvalue()

        return call_signature.replace(pysig=None), codegen
    if method.returns["kind"] == "array_view_2d":
        array_type = types.Array(types.float64, 2, "F", readonly=method.returns["mutable"] == "false")
        call_signature = signature(array_type, *argument_types)
        def codegen(context, builder, sig, arguments):
            adapter = cgutils.get_or_insert_function(builder.module, ir.FunctionType(
                ir.IntType(32), [ir.IntType(8).as_pointer(), ir.DoubleType().as_pointer().as_pointer(),
                                 ir.IntType(64).as_pointer(), ir.IntType(64).as_pointer()]), name=method.symbol)
            data_slot = cgutils.alloca_once(builder, ir.DoubleType().as_pointer())
            rows_slot, cols_slot = cgutils.alloca_once(builder, ir.IntType(64)), cgutils.alloca_once(builder, ir.IntType(64))
            status = builder.call(adapter, [_convert_pointer(context, builder, sig.args[0], method.owner, arguments[0]), data_slot, rows_slot, cols_slot])
            _raise_on_failure(context, builder, status)
            rows, cols = builder.load(rows_slot), builder.load(cols_slot)
            result = context.make_array(array_type)(context, builder)
            result.data = builder.load(data_slot)
            result.shape = cgutils.pack_array(builder, [rows, cols])
            result.strides = cgutils.pack_array(builder, [ir.Constant(ir.IntType(64), 8), builder.mul(rows, ir.Constant(ir.IntType(64), 8))])
            result.itemsize = ir.Constant(ir.IntType(64), 8)
            result.meminfo = cgutils.get_null_value(result.meminfo.type)
            result.parent = cgutils.get_null_value(result.parent.type)
            result.nitems = builder.mul(rows, cols)
            return result._getvalue()
        return call_signature.replace(pysig=None), codegen
    if method.returns["kind"] == "array_view_3d":
        array_type = types.Array(types.float64, 3, "A", readonly=method.returns["mutable"] == "false")
        call_signature = signature(array_type, *argument_types)
        def codegen(context, builder, sig, arguments):
            adapter = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer(), ir.DoubleType().as_pointer().as_pointer(), ir.IntType(64).as_pointer(), ir.IntType(64).as_pointer(), ir.IntType(64).as_pointer()]), name=method.symbol)
            data = cgutils.alloca_once(builder, ir.DoubleType().as_pointer()); depth = cgutils.alloca_once(builder, ir.IntType(64)); rows = cgutils.alloca_once(builder, ir.IntType(64)); cols = cgutils.alloca_once(builder, ir.IntType(64))
            status = builder.call(adapter, [_convert_pointer(context, builder, sig.args[0], method.owner, arguments[0]), data, depth, rows, cols]); _raise_on_failure(context, builder, status)
            k, i, j = builder.load(depth), builder.load(rows), builder.load(cols); eight = ir.Constant(ir.IntType(64), 8)
            result = context.make_array(array_type)(context, builder); result.data = builder.load(data)
            result.shape = cgutils.pack_array(builder, [k, i, j]); result.strides = cgutils.pack_array(builder, [builder.mul(builder.mul(i, j), eight), eight, builder.mul(i, eight)])
            result.itemsize = eight; result.meminfo = cgutils.get_null_value(result.meminfo.type); result.parent = cgutils.get_null_value(result.parent.type); result.nitems = builder.mul(k, builder.mul(i, j)); return result._getvalue()
        return call_signature.replace(pysig=None), codegen
    return_type = _numba_result_type(method.bridge_id, result_specification)
    call_signature = signature(return_type, *argument_types)

    def codegen(context, builder, sig, arguments):
        abi_argument_types = [ir.IntType(8).as_pointer()]
        for parameter in method.abi_parameters:
            abi_argument_types.append(_llvm_abi_type(parameter))
        has_result = method.returns["kind"] != "void"
        if has_result:
            abi_argument_types.append(_llvm_result_type(method.returns).as_pointer())
        adapter = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(ir.IntType(32), abi_argument_types),
            name=method.symbol,
        )

        array_values: dict[int, tuple[Any, Any]] = {}
        lowered = [_convert_pointer(context, builder, sig.args[0], method.owner, arguments[0])]
        for parameter in method.abi_parameters:
            kind = parameter["kind"]
            if kind in {"array_data", "array_count"}:
                array_argument = parameter["array_argument"]
                if array_argument not in array_values:
                    numba_index = array_argument + 1
                    array_values[array_argument] = _array_data_and_count(
                        context,
                        builder,
                        sig.args[numba_index],
                        arguments[numba_index],
                    )
                data, count = array_values[array_argument]
                lowered.append(data if kind == "array_data" else count)
                continue
            argument_index = parameter["argument_index"] + 1
            if kind == "object":
                lowered.append(_convert_pointer(context, builder, sig.args[argument_index],
                                                parameter["cxx_type"], arguments[argument_index]))
                continue
            if kind == "mpi_comm":
                holder = cgutils.create_struct_proxy(sig.args[argument_index])(context, builder, value=arguments[argument_index])
                lowered.append(holder.pointer)
                continue
            if kind == "scalar_pointer":
                lowered.append(builder.bitcast(arguments[argument_index], ir.IntType(8).as_pointer()))
                continue
            expected = _numba_scalar_type(parameter)
            lowered.append(
                context.cast(builder, arguments[argument_index], sig.args[argument_index], expected)
            )

        if has_result:
            # The C ABI always writes the raw borrowed pointer.  For nullable
            # pointer returns that raw null/non-null value is then packaged as
            # Numba's Optional[BridgeObjectType].
            result_type = _numba_result_type(method.bridge_id, result_specification)
            slot_type = (
                bridge_object_type(method.bridge_id, method.returns["cxx_type"])
                if method.returns["kind"] == "object"
                else result_type
            )
            if method.returns["kind"] == "mpi_comm":
                from .mpi import allocate_native_holder
                result, native_slot = allocate_native_holder(context, builder)
                status = builder.call(adapter, [*lowered, native_slot])
                with builder.if_then(builder.icmp_unsigned("!=", status, ir.Constant(status.type, 0))):
                    context.nrt.decref(builder, result_type, result._getvalue())
                _raise_on_failure(context, builder, status)
                result.pointer = builder.load(native_slot)
                return result._getvalue()
            slot = cgutils.alloca_once(
                builder,
                context.get_value_type(slot_type)
                if method.returns["kind"] != "bool"
                else ir.IntType(32),
            )
            status = builder.call(adapter, [*lowered, slot])
            _raise_on_failure(context, builder, status)
            result = builder.load(slot)
            if method.returns["kind"] == "bool":
                return builder.icmp_unsigned("!=", result, ir.Constant(result.type, 0))
            if method.returns["kind"] == "object" and method.returns.get("nullable"):
                optional = context.make_helper(builder, result_type)
                optional.valid = cgutils.is_not_null(builder, result)
                optional.data = result
                return optional._getvalue()
            return result
        status = builder.call(adapter, lowered)
        _raise_on_failure(context, builder, status)
        return context.get_dummy_value()

    return call_signature.replace(pysig=None), codegen


def _make_adapter_caller(method: BridgeMethod):
    """Create an exact-arity intrinsic around one generated C ABI adapter."""

    arguments = [f"arg{index}" for index in range(len(method.parameters))]
    parameter_text = ", ".join(["typingctx", "value", *arguments])
    argument_tuple = ", ".join(["value", *arguments])
    tuple_text = f"({argument_tuple},)" if len(arguments) == 0 else f"({argument_tuple})"
    source = "\n".join(
        [
            "@_intrinsic",
            f"def generated_caller({parameter_text}):",
            f"    return _adapter_signature(_method, {tuple_text})",
            "",
        ]
    )
    namespace: dict[str, Any] = {
        "_intrinsic": intrinsic,
        "_adapter_signature": _adapter_signature,
        "_method": method,
    }
    exec(source, namespace)
    return namespace["generated_caller"]


def _make_implementation(caller, arity: int, returns_void: bool):
    """Create an exact-arity implementation accepted by Numba overloads."""

    arguments = [f"arg{index}" for index in range(arity)]
    parameter_text = ", ".join(["value", *arguments])
    call_text = ", ".join(["value", *arguments])
    if returns_void:
        source = f"def implementation({parameter_text}):\n    _caller({call_text})\n    return None\n"
    else:
        source = f"def implementation({parameter_text}):\n    return _caller({call_text})\n"
    namespace: dict[str, Any] = {"_caller": caller}
    exec(source, namespace)
    return namespace["implementation"]


def _resolve_method(
    python_name: str, receiver: types.Type, argument_types: tuple[types.Type, ...]
):
    available = _METHODS_BY_NAME.get(python_name, [])
    on_receiver = [
        method
        for method in available
        if (
            isinstance(receiver, BridgeObjectType)
            and method.bridge_id == receiver.bridge_id
            and method.owner == receiver.cxx_type
        )
    ]
    if not on_receiver and isinstance(receiver, BridgeObjectType):
        current = receiver.cxx_type
        seen = set()
        while current not in seen:
            seen.add(current)
            info = _CLASS_INFO.get((receiver.bridge_id, current), {})
            # A declaration on this class hides base overloads, even if unsupported.
            if python_name in info.get("declared_method_names", []):
                break
            bases = info.get("public_bases", [])
            if len(bases) != 1 or _automatic_path(receiver.bridge_id, current, bases[0]) is None:
                break
            current = bases[0]
            on_receiver = [method for method in available
                           if method.bridge_id == receiver.bridge_id and method.owner == current]
            if on_receiver:
                break
    if not on_receiver:
        return None
    candidates = [
        method
        for method in on_receiver
        if len(method.parameters) == len(argument_types)
        and (not isinstance(receiver, ConstBridgeObjectType) or method.const)
        and all(
            _matches(parameter, argument, receiver.bridge_id)
            for parameter, argument in zip(method.parameters, argument_types)
        )
    ]
    if len(candidates) > 1:
        # Prefer exact floating precision when float/double overloads coexist.
        # Keep equal-cost candidates ambiguous rather than selecting by order.
        costs = [sum(parameter["kind"] in {"float", "double"}
                     and argument != _numba_scalar_type(parameter)
                     for parameter, argument in zip(method.parameters, argument_types))
                 for method in candidates]
        best = min(costs)
        candidates = [method for method, cost in zip(candidates, costs) if cost == best]
    if len(candidates) > 1:
        # A SWIG proxy carries a mutable C++ address.  Therefore ordinary C++
        # overload resolution selects the non-const member when candidates
        # differ only by the receiver's const qualification.  Keep multiple
        # non-const candidates ambiguous for the caller to resolve.
        mutable = [method for method in candidates if not method.const]
        if mutable:
            candidates = mutable
    if len(candidates) == 1:
        return candidates[0]
    supported = ", ".join(_display_method(method) for method in on_receiver)
    received = ", ".join(str(argument) for argument in argument_types)
    if len(candidates) > 1:
        raise TypingError(
            f"Numba bridge call {receiver.cxx_type}.{python_name}({received}) is ambiguous; "
            f"generated candidates: {supported}"
        )
    raise TypingError(
        f"Numba bridge for {receiver.cxx_type} supports {supported}; "
        f"received {python_name}({received})"
    )


def _install_method_template(python_name: str, arity: int) -> None:
    # One template per name handles every arity, including hidden base overloads.
    if python_name in _INSTALLED_METHOD_NAMES:
        return

    @overload_method(BridgeObjectType, python_name)
    def lower_generated_method(value, *args):
        if len(args) == 1 and isinstance(args[0], (types.StarArgTuple, types.StarArgUniTuple)):
            args = tuple(args[0])
        method = _resolve_method(python_name, value, args)
        if method is None:
            return None
        caller = _make_adapter_caller(method)
        call = ", ".join(["value", *[f"args[{index}]" for index in range(len(args))]])
        statement = f"_caller({call})\n    return None" if method.returns['kind'] == 'void' else f"return _caller({call})"
        namespace = {"_caller": caller}
        exec(f"def implementation(value, *args):\n    {statement}\n", namespace)
        return namespace['implementation']

    _INSTALLED_METHOD_NAMES.add(python_name)


# Index syntax uses the same checked adapters as SWIG's scalar accessors.
@overload(operator.getitem)
def _bridge_getitem(value, arg0):
    method = _resolve_method("__getitem__", value, (arg0,))
    if method is not None:
        return _make_implementation(_make_adapter_caller(method), 1, False)


@overload(operator.setitem)
def _bridge_setitem(value, arg0, arg1):
    method = _resolve_method("__setitem__", value, (arg0, arg1))
    if method is not None:
        return _make_implementation(_make_adapter_caller(method), 2, True)


def _install_member_template(name):
    if name in _INSTALLED_MEMBER_NAMES:
        return

    @overload_attribute(BridgeObjectType, name)
    def get_member(value):
        member = _MEMBERS.get(value.key, {}).get(name)
        if member is None:
            return None
        method = _resolve_method(member["get"], value, ())
        return _make_implementation(_make_adapter_caller(method), 0, False)

    @lower_setattr(BridgeObjectType, name)
    def set_member(context, builder, sig, args):
        receiver, assigned = sig.args
        member = _MEMBERS.get(receiver.key, {}).get(name)
        if member is None:
            raise TypingError(f"unselected C++ member {receiver.cxx_type}.{name}")
        if not member["writable"] or isinstance(receiver, ConstBridgeObjectType):
            raise TypingError(f"C++ member {receiver.cxx_type}.{name} is read-only")
        method = _resolve_method(member["set"], receiver, (assigned,))
        call_signature, codegen = _adapter_signature(method, (receiver, assigned))
        return codegen(context, builder, call_signature, args)

    _INSTALLED_MEMBER_NAMES.add(name)


def _load_descriptor(path: str | Path) -> dict[str, Any]:
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get("schema_version") != 4:
        raise RuntimeError("unsupported ordinary bridge descriptor schema")
    if descriptor.get("kind") != "numba_swig_ordinary_bridge_descriptor":
        raise RuntimeError("invalid ordinary bridge descriptor kind")
    if not isinstance(descriptor.get("bridge_id"), str):
        raise RuntimeError("ordinary bridge descriptor has no bridge identity")
    if not isinstance(descriptor.get("classes"), list):
        raise RuntimeError("ordinary bridge descriptor has no class list")
    return descriptor


def _install_symbols(target_module: object, methods: list[BridgeMethod]) -> None:
    for method in methods:
        resolver = getattr(target_module, method.function_address_helper, None)
        if not callable(resolver):
            raise RuntimeError(f"owning module {method.owner_module} has no generated "
                               f"{method.function_address_helper} helper")
        address = resolver(method.symbol)
        if not isinstance(address, int) or address == 0:
            raise RuntimeError(f"target SWIG module has no adapter symbol {method.symbol!r}")
        binding.add_symbol(method.symbol, address)


def _register_proxy_type(proxy_type: type, bridge_type: BridgeObjectType, *, constructible: bool = False) -> None:
    def typeof_proxy(value, context):
        del value, context
        return bridge_type

    typeof_impl.register(proxy_type)(typeof_proxy)
    if not constructible:
        infer_global(proxy_type, types.TypeRef(bridge_type))


def _register_owned_constructors(proxy_type: type, bridge_id: str, cxx_type: str) -> None:
    key = (bridge_id, cxx_type)
    constructors = _CONSTRUCTORS[key]
    owned_type = _OWNED_TYPES.setdefault(key, OwnedBridgeObjectType(bridge_id, cxx_type))

    @type_callable(proxy_type)
    def type_constructor(context):
        maximum_arity = max(len(constructor["parameters"]) for constructor in constructors)

        def choose(arguments):
            try:
                constructor = _select_constructor(constructors, arguments, bridge_id)
            except TypingError as error:
                raise TypingError(f"constructors for {cxx_type} are ambiguous") from error
            if constructor is None:
                return None
            return owned_type

        names = [f"arg{index}=None" for index in range(maximum_arity)]
        arguments = ", ".join(f"arg{index}" for index in range(maximum_arity))
        source = f"def typer({', '.join(names)}):\n    return choose(({arguments}{',' if maximum_arity == 1 else ''}))\n"
        namespace = {"choose": choose}
        exec(source, namespace)
        return namespace["typer"]

    def make_lower_constructor(arity):
        @lower_builtin(proxy_type, *([types.Any] * arity))
        def lower_constructor(context, builder, sig, arguments):
            constructor = _select_constructor(constructors, sig.args, bridge_id)
            if constructor is None:
                raise TypingError(f"constructors for {cxx_type} have no unique native lowering")
            destroy = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer(), ir.IntType(types.intp.bitwidth), ir.IntType(8).as_pointer()]),
            name=_DESTROY_SYMBOLS[key],
        )
            meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, 8), destroy
        )
            storage = context.nrt.meminfo_data(builder, meminfo)
            abi_types = []
            lowered = []
            for parameter, argument_type, argument in zip(constructor["parameters"], sig.args, arguments):
                abi_types.append(_llvm_abi_type(parameter))
                if parameter["kind"] == "mpi_comm":
                    holder = cgutils.create_struct_proxy(argument_type)(context, builder, value=argument)
                    lowered.append(holder.pointer)
                elif parameter["kind"] == "scalar_pointer":
                    lowered.append(builder.bitcast(argument, ir.IntType(8).as_pointer()))
                else:
                    lowered.append(context.cast(builder, argument, argument_type, _numba_scalar_type(parameter)))
            adapter = cgutils.get_or_insert_function(
            builder.module,
            ir.FunctionType(ir.IntType(32), [*abi_types, ir.IntType(8).as_pointer().as_pointer()]),
            name=constructor["symbol"],
        )
            status = builder.call(adapter, [*lowered, builder.bitcast(storage, ir.IntType(8).as_pointer().as_pointer())])
            _raise_on_failure(context, builder, status)
            result = cgutils.create_struct_proxy(owned_type)(context, builder)
            result.meminfo = meminfo
            return result._getvalue()
        return lower_constructor

    for arity in {len(constructor["parameters"]) for constructor in constructors}:
        make_lower_constructor(arity)


def install_bridge(modules: Mapping[str, Any], descriptor_path: str | Path) -> None:
    """Install one bridge using each class's owning SWIG extension."""

    descriptor = _load_descriptor(descriptor_path)
    module_map = resolve_modules(modules, descriptor["classes"])
    bridge_id = descriptor["bridge_id"]
    descriptor_key = (bridge_id, str(Path(descriptor_path).resolve()),
                      tuple((name, id(module)) for name, module in module_map.items()))
    if descriptor_key in _INSTALLED_DESCRIPTORS:
        return

    pending_types: dict[str, BridgeObjectType] = {}
    pending_classes: list[dict[str, Any]] = []
    for class_entry in descriptor["classes"]:
        if not isinstance(class_entry, dict):
            raise RuntimeError("ordinary bridge descriptor has an invalid class record")
        cxx_type = class_entry.get("cxx_type")
        proxy_name = class_entry.get("python_proxy")
        methods = class_entry.get("methods")
        if not isinstance(cxx_type, str) or not isinstance(proxy_name, str) or not isinstance(methods, list):
            raise RuntimeError("ordinary bridge descriptor has incomplete class metadata")
        owner_name = class_entry["owner_module"]
        proxy_type = getattr(module_map[owner_name], proxy_name, None)
        if not isinstance(proxy_type, type):
            raise RuntimeError(f"owning module {owner_name} has no proxy {proxy_name!r}")
        object_key = (bridge_id, cxx_type)
        previous_owner = _OBJECT_OWNERS.get(object_key)
        if previous_owner is not None and previous_owner != (owner_name, proxy_type):
            raise RuntimeError(f"bridge class {bridge_id}:{cxx_type} already has another owner")
        existing = _OBJECT_TYPES.get(object_key)
        bridge_type = existing if existing is not None else BridgeObjectType(bridge_id, cxx_type)
        pending_types[cxx_type] = bridge_type
        pending_classes.append(class_entry)

    all_methods: list[BridgeMethod] = []
    for class_entry in pending_classes:
        owner = class_entry["cxx_type"]
        for record in class_entry["methods"]:
            if not isinstance(record, dict):
                raise RuntimeError(f"ordinary bridge class {owner} has an invalid method")
            try:
                method = BridgeMethod(
                    bridge_id=bridge_id,
                    owner=owner,
                    owner_module=class_entry["owner_module"],
                    function_address_helper=class_entry["function_address_helper"],
                    identity=str(record["identity"]),
                    symbol=str(record["symbol"]),
                    python_name=str(record["python_name"]),
                    const=record.get("const") is True,
                    returns=dict(record["return"]),
                    parameters=tuple(dict(item) for item in record["parameters"]),
                    abi_parameters=tuple(dict(item) for item in record["abi_parameters"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"ordinary bridge class {owner} has incomplete method metadata") from error
            all_methods.append(method)

    for module_name, target_module in module_map.items():
        _install_symbols(target_module, [method for method in all_methods
                                        if method.owner_module == module_name])
        for class_entry in pending_classes:
            if class_entry["owner_module"] != module_name:
                continue
            resolver = getattr(target_module, class_entry["function_address_helper"])
            key = (bridge_id, class_entry["cxx_type"])
            _CONSTRUCTORS[key] = tuple(class_entry.get("constructors", []))
            destroy_symbol = class_entry.get("destroy_symbol")
            if not _CONSTRUCTORS[key] or not isinstance(destroy_symbol, str):
                continue
            _DESTROY_SYMBOLS[key] = destroy_symbol
            for constructor in _CONSTRUCTORS[key]:
                binding.add_symbol(constructor["symbol"], resolver(constructor["symbol"]))
            binding.add_symbol(destroy_symbol, resolver(destroy_symbol))
    for record in descriptor.get("conversions", []):
        module = module_map[record["owner_module"]]
        flags = getattr(module, record["flags"])()
        installed = {**record, "flags_value": flags}
        resolver = getattr(module, record["function_address_helper"])
        for kind, source, target in [("upcast", record["derived"], record["base"]),
                                     ("dynamic_cast", record["base"], record["derived"])]:
            address = resolver(record[kind])
            if not address:
                raise RuntimeError(f"missing C++ cast helper {record[kind]}")
            binding.add_symbol(record[kind], address)
            _CASTS[(bridge_id, source, target, kind)] = installed
    for class_entry in pending_classes:
        cxx_type = class_entry["cxx_type"]
        proxy_name = class_entry["python_proxy"]
        module_name = class_entry["owner_module"]
        target_module = module_map[module_name]
        _CLASS_INFO[(bridge_id, cxx_type)] = class_entry
        try:
            proxy_type = getattr(target_module, proxy_name)
        except AttributeError as error:
            raise RuntimeError(
                f"target SWIG module {module_name} has no proxy {proxy_name!r}"
            ) from error
        object_key = (bridge_id, cxx_type)
        if object_key not in _OBJECT_TYPES:
            _register_proxy_type(proxy_type, pending_types[cxx_type], constructible=bool(_CONSTRUCTORS.get(object_key)))
            _OBJECT_TYPES[object_key] = pending_types[cxx_type]
            _OBJECT_OWNERS[object_key] = (module_name, proxy_type)
            _PROXY_TYPES[proxy_type] = pending_types[cxx_type]
            _BORROWED_PROXY_HELPERS[object_key] = getattr(
                target_module, class_entry["borrowed_proxy_helper"]
            )
            owned_helper_name = class_entry.get("owned_proxy_helper")
            _OWNED_PROXY_HELPERS[object_key] = (
                getattr(target_module, owned_helper_name, None)
                if isinstance(owned_helper_name, str) else None
            )
            if _CONSTRUCTORS.get(object_key):
                _register_owned_constructors(proxy_type, bridge_id, cxx_type)

    for entry in pending_classes:
        _MEMBERS[(bridge_id, entry["cxx_type"])] = {
            member["python_name"]: member for member in entry.get("members", [])
        }
        for member in entry.get("members", []):
            _install_member_template(member["python_name"])
    for method in all_methods:
        entries = _METHODS_BY_NAME.setdefault(method.python_name, [])
        if method not in entries:
            entries.append(method)
        _install_method_template(method.python_name, len(method.parameters))
    _INSTALLED_DESCRIPTORS.add(descriptor_key)
