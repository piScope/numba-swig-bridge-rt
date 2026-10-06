"""Native array-pointer utilities for Numba-compiled bridge code."""

from __future__ import annotations

from numba import types
from numba.extending import intrinsic


@intrinsic
def as_cpointer(typingctx, array_type):
    """Return a typed pointer to a contiguous Numba array's first element.

    The pointer is valid only while the source array remains live.  It is
    intended for immediate calls to explicitly annotated native view adapters;
    it must not escape compiled code or be retained by C++.
    """
    if not isinstance(array_type, types.Array):
        raise TypeError("nsb.as_cpointer() requires a Numba array")
    if array_type.ndim < 1 or array_type.layout not in {"C", "F"}:
        raise TypeError("nsb.as_cpointer() requires a C- or Fortran-contiguous array")
    if not array_type.mutable:
        raise TypeError("nsb.as_cpointer() requires a writable array")

    pointer_type = types.CPointer(array_type.dtype)
    signature = pointer_type(array_type)

    def codegen(context, builder, sig, args):
        array = context.make_array(array_type)(context, builder, args[0])
        return array.data

    return signature, codegen
