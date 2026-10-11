"""Optional communicator registration; importing nsb_rt does not import MPI."""
from llvmlite import ir
from llvmlite import binding as llvm
from numba import types
from numba.core import cgutils
from numba.core.typing.templates import AttributeTemplate
from numba.extending import (NativeValue, models, register_model, typeof_impl,
                            unbox, box, infer_getattr, lower_getattr)


class CommunicatorType(types.Type):
    def __init__(self):
        super().__init__(name="nsb MPI communicator")


comm_type = CommunicatorType()
_provider = None


@register_model(CommunicatorType)
class CommunicatorModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [
            ("meminfo", types.MemInfoPointer(types.voidptr)),
            ("pointer", types.voidptr),
            ("owner", types.pyobject),
        ])


def register_mpi_provider(provider):
    """Register one artifact module compiled with the consumer's MPI library."""
    global _provider
    if _provider is not None:
        if _provider is provider:
            return comm_type
        raise RuntimeError("an MPI communicator provider is already registered")
    from mpi4py import MPI
    if provider.library_version().strip() != MPI.Get_library_version().strip():
        raise RuntimeError("MPI provider and mpi4py report different MPI libraries")
    addresses = provider.addresses()
    if set(addresses) != {"rank", "size", "destroy", "owner"} or not all(addresses.values()):
        raise ValueError("invalid MPI communicator provider addresses")
    for name, address in addresses.items():
        llvm.add_symbol("nsb_mpi_comm_" + name, int(address))
    _provider = provider

    @typeof_impl.register(MPI.Comm)
    def typeof_comm(value, context):
        return comm_type

    return comm_type


@unbox(CommunicatorType)
def unbox_comm(typ, obj, c):
    result = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    result.meminfo = cgutils.get_null_value(result.meminfo.type)
    result.pointer = cgutils.get_null_value(result.pointer.type)
    result.owner = obj
    factory = c.pyapi.unserialize(c.pyapi.serialize_object(_provider.make_holder))
    holder = c.pyapi.call_function_objargs(factory, (obj,))
    c.pyapi.decref(factory)
    with c.builder.if_then(cgutils.is_not_null(c.builder, holder)):
        getter = c.pyapi.unserialize(c.pyapi.serialize_object(_provider.holder_address))
        address_object = c.pyapi.call_function_objargs(getter, (holder,))
        c.pyapi.decref(getter)
        with c.builder.if_then(cgutils.is_not_null(c.builder, address_object)):
            address = c.pyapi.long_as_voidptr(address_object)
            result.pointer = address
            result.meminfo = c.pyapi.nrt_meminfo_new_from_pyobject(address, holder)
            c.pyapi.decref(address_object)
        c.pyapi.decref(holder)
    return NativeValue(result._getvalue(), is_error=cgutils.is_not_null(c.builder, c.pyapi.err_occurred()))


@box(CommunicatorType)
def box_comm(typ, value, c):
    result = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=value)
    output = cgutils.alloca_once_value(c.builder, c.pyapi.get_null_object())
    with c.builder.if_else(cgutils.is_not_null(c.builder, result.owner)) as (python, native):
        with python:
            c.pyapi.incref(result.owner)
            c.builder.store(result.owner, output)
        with native:
            c.pyapi.err_set_string("PyExc_TypeError", "native communicator returns to Python are unsupported; use it inside compiled code")
    c.context.nrt.decref(c.builder, typ, value)
    return c.builder.load(output)


def allocate_native_holder(context, builder):
    """Own a native wrapper only; the application retains the MPI resource."""
    pointer = ir.IntType(8).as_pointer()
    destroy = cgutils.get_or_insert_function(
        builder.module, ir.FunctionType(ir.VoidType(), [pointer, ir.IntType(types.uintp.bitwidth), pointer]),
        "nsb_mpi_comm_destroy")
    result = cgutils.create_struct_proxy(comm_type)(context, builder)
    result.meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, types.uintp.bitwidth // 8), destroy)
    storage = context.nrt.meminfo_data(builder, result.meminfo)
    slot = builder.bitcast(storage, pointer.as_pointer())
    builder.store(cgutils.get_null_value(pointer), slot)
    result.pointer = cgutils.get_null_value(pointer)
    result.owner = cgutils.get_null_value(context.get_value_type(types.pyobject))
    return result, slot


@infer_getattr
class CommunicatorAttributes(AttributeTemplate):
    key = CommunicatorType

    def resolve_rank(self, typ):
        return types.intc

    def resolve_size(self, typ):
        return types.intc


def _query(context, builder, typ, value, attribute):
    # Registration resolves stable symbols in each process; caches contain no addresses.
    function_type = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer(), ir.IntType(32).as_pointer()])
    function = cgutils.get_or_insert_function(builder.module, function_type,
                                              "nsb_mpi_comm_" + attribute)
    holder = cgutils.create_struct_proxy(typ)(context, builder, value=value)
    output = cgutils.alloca_once_value(builder, ir.Constant(ir.IntType(32), 0))
    status = builder.call(function, [holder.pointer, output])
    with builder.if_then(builder.icmp_signed("!=", status, ir.Constant(status.type, 0)), likely=False):
        context.call_conv.return_user_exc(builder, RuntimeError, ("MPI communicator query failed: inactive MPI, invalid communicator, thread restriction or MPI error",))
    return builder.load(output)


@lower_getattr(CommunicatorType, "rank")
def lower_rank(context, builder, typ, value):
    return _query(context, builder, typ, value, "rank")


@lower_getattr(CommunicatorType, "size")
def lower_size(context, builder, typ, value):
    return _query(context, builder, typ, value, "size")


def is_communicator_annotation(annotation):
    if _provider is None:
        return False
    from mpi4py import MPI
    return annotation is MPI.Comm


def communicator_state_bridge():
    from .director_state import DeclaredObjectBridge

    class MPIStateBridge(DeclaredObjectBridge):
        numba_state_type = comm_type

    from mpi4py import MPI
    return MPIStateBridge("MPI_Comm", "Comm", MPI, "")


def communicator_state_property(slot):
    from .director_state import DirectorStateProperty
    from numba.core.runtime import _nrt_python

    class CommunicatorStateProperty(DirectorStateProperty):
        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            state = instance._require_nsb_director_state()
            address = int(state.ObjectAddress(self.slot))
            if not address:
                raise ReferenceError(f"communicator state field {self.name!r} is empty")
            return _provider.holder_owner(address)

        def __set__(self, instance, value):
            state = instance._require_nsb_director_state()
            if value is None:
                state.SetObjectBorrowed(self.slot, 0)
                return
            capsule = _provider.make_holder(value)
            address = _provider.holder_address(capsule)
            # The existing ABI consumes one NRT reference. The capsule retains
            # the Python communicator and owns only the C++ holder wrapper.
            token = _nrt_python.meminfo_new(address, capsule)
            try:
                state.SetObjectFromNative(self.slot, address, token)
            except BaseException:
                _nrt_python._MemInfo(token)  # Release the untransferred reference.
                raise

    return CommunicatorStateProperty(slot=slot, bridge=communicator_state_bridge())


def communicator_state_at(backend):
    from numba.extending import intrinsic

    @intrinsic
    def read(typingctx, state, slot):
        if state != types.voidptr or not isinstance(slot, types.Integer):
            return None
        signature = comm_type(state, slot)

        def codegen(context, builder, sig, args):
            # Use the same backend symbols as existing ordinary object slots.
            slot_value = context.cast(builder, args[1], sig.args[1], types.intc)
            integer = ir.IntType(types.uintp.bitwidth)
            function_type = ir.FunctionType(integer, [args[0].type, ir.IntType(32)])
            address_fn = cgutils.get_or_insert_function(builder.module, function_type,
                                                        backend.symbol_name("object_address_at"))
            meminfo_fn = cgutils.get_or_insert_function(builder.module, function_type,
                                                        backend.symbol_name("object_meminfo_at"))
            address = builder.call(address_fn, [args[0], slot_value])
            with builder.if_then(builder.icmp_unsigned("==", address, ir.Constant(integer, 0))):
                context.call_conv.return_user_exc(builder, ReferenceError, ("communicator state field is empty or closed",))
            result = cgutils.create_struct_proxy(comm_type)(context, builder)
            result.pointer = builder.inttoptr(address, context.get_value_type(types.voidptr))
            token = builder.call(meminfo_fn, [args[0], slot_value])
            result.meminfo = builder.inttoptr(token, context.get_value_type(types.MemInfoPointer(types.voidptr)))
            context.nrt.incref(builder, types.MemInfoPointer(types.voidptr), result.meminfo)
            owner_fn = cgutils.get_or_insert_function(builder.module,
                ir.FunctionType(context.get_value_type(types.pyobject), [result.pointer.type]),
                "nsb_mpi_comm_owner")
            result.owner = builder.call(owner_fn, [result.pointer])
            return result._getvalue()

        return signature, codegen

    return read
