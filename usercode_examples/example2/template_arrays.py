"""Read and modify two concrete C++ template instantiations from Numba."""

from numba import njit
from nsb_sample2 import IntArray, DoubleArray
import nsb_sample2.registration  # Install the generated bridge.


@njit
def fill_and_sum(integers, doubles):
    integers[1] = 3
    doubles[1] = 2.5
    for i in range(integers.Size()):
        integers[i] = integers[i] + i
        doubles[i] = doubles[i] + 0.5 * i
    return integers[1], doubles[1]


ints = IntArray(3)
doubles = DoubleArray(3)
assert fill_and_sum(ints, doubles) == (4, 3.0)
assert ints[1] == 4  # Mutation is visible through the original SWIG proxy.
assert doubles[1] == 3.0


@njit
def copy_through_reference(source, target):
    source.CopyTo(target)
    return target[1]


integer_copy = IntArray(3)
double_copy = DoubleArray(3)
assert copy_through_reference(ints, integer_copy) == 4
assert copy_through_reference(doubles, double_copy) == 3.0
assert integer_copy[1] == 4
assert double_copy[1] == 3.0


@njit
def read_at(array, index):
    return array[index]


@njit
def write_at(array, index, value):
    array[index] = value


for array in (ints, doubles):
    for index in (-1, array.Size()):
        for action in (
            lambda: read_at(array, index),
            lambda: write_at(array, index, 1 if isinstance(array, IntArray) else 1.0),
        ):
            try:
                action()
            except RuntimeError:
                pass
            else:
                raise AssertionError("out-of-range indexing should fail")

print("C++ template arrays: indexed reads, writes, and bounds failures verified")
