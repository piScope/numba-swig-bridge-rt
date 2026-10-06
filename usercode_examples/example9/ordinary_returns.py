"""Object returns crossing a normal Numba-to-Python boundary."""

import gc

import numpy as np
from numba import njit

import nsb_rt as nsb
import nsb_sample9
import nsb_sample9.registration  # noqa: F401


@njit
def ordinary_child(parent):
    return parent.GetChild()


@nsb.borrowed_from(owner_arg="parent")
@njit
def retained_python_child(parent):
    return parent.GetChild()


@njit
def retained_native_child(value):
    parent = nsb_sample9.Parent(nsb.as_cpointer(value))
    return nsb.borrowed_from(parent.GetChild(), parent)


@njit
def owned_parent():
    return nsb.own_this(nsb_sample9.Parent(14))


@njit
def implicit_owned_parent():
    return nsb_sample9.Parent(15)


parent = nsb_sample9.Parent(11)
child = ordinary_child(parent)
assert child.thisown is False
assert child.Value() == 11

parent = nsb_sample9.Parent(12)
child = retained_python_child(parent)
assert child.thisown is False
del parent
gc.collect()
assert child.Value() == 12
del child
gc.collect()

try:
    implicit_owned_parent()
except TypeError as error:
    assert "nsb.own_this" in str(error)
else:
    raise AssertionError("an unmarked frame-owned result escaped to Python")

before = nsb_sample9.DestroyedParentCount()
parent = owned_parent()
assert parent.thisown is True
assert parent.GetChild().Value() == 14
assert nsb_sample9.DestroyedParentCount() == before
del parent
gc.collect()
assert nsb_sample9.DestroyedParentCount() == before + 1

before = nsb_sample9.DestroyedParentCount()
value = np.array([13], dtype=np.int32)
child = retained_native_child(value)
assert child.thisown is False
assert child.Value() == 13
assert nsb_sample9.DestroyedParentCount() == before
del child
gc.collect()
assert nsb_sample9.DestroyedParentCount() == before + 1

print("ordinary object returns: Normal End")
