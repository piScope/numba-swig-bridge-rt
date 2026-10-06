"""Keep a writable NumPy array in native director state."""

import numpy as np
from numba import types
import nsb_rt as nsb
import nsb_sample7
import nsb_sample7.registration  # noqa: F401


@nsb.director(state={"values": types.float64[::1]}, fallback="silent")
class ArrayAccumulator(nsb_sample7.Accumulator):
    def __init__(self, values):
        super().__init__()
        self.values = values

    @nsb.override("Accumulate", "double", stateaccess="values")
    def accumulate(self, value):
        pointer = nsb.as_cpointer(self.values)
        pointer[0] += value
        return pointer[0]


@nsb.director(state={"values": types.float64[::1, :]}, fallback="silent")
class MatrixAccumulator(nsb_sample7.Accumulator):
    def __init__(self, values):
        super().__init__()
        self.values = values

    @nsb.override("Accumulate", "double", stateaccess="values")
    def accumulate(self, value):
        self.values[1, 2] += value
        return self.values[1, 2]


@nsb.director(state={"values": types.Array(types.float64, 3, "A")}, fallback="silent")
class TensorAccumulator(nsb_sample7.Accumulator):
    def __init__(self, values):
        super().__init__()
        self.values = values

    @nsb.override("Accumulate", "double", stateaccess="values")
    def accumulate(self, value):
        self.values[1, 1, 1] += value
        return self.values[1, 1, 1]


@nsb.director(state={"values": types.float64[::1]}, fallback="silent")
class LazyArrayAccumulator(nsb_sample7.Accumulator):
    def __init__(self):
        super().__init__()

    @nsb.override("Accumulate", "double", stateaccess="values")
    def accumulate(self, value):
        if not nsb.is_slot_occupied(self, "values"):
            self.values = np.zeros(1, dtype=np.float64)
        self.values[0] += value
        return self.values[0]


values = np.array([2.0, 0.0], dtype=np.float64)
accumulator = ArrayAccumulator(values)
try:
    assert nsb_sample7.invoke_accumulate(accumulator, 3.0) == 5.0
    assert values.tolist() == [5.0, 0.0]
    print("array state director: Normal End")
finally:
    accumulator.close()

matrix_values = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
matrix = MatrixAccumulator(matrix_values)
try:
    assert nsb_sample7.invoke_accumulate(matrix, 4.0) == 4.0
    assert matrix_values[1, 2] == 4.0
finally:
    matrix.close()

tensor_values = np.zeros((3, 2, 2), dtype=np.float64)[::2]
tensor = TensorAccumulator(tensor_values)
try:
    assert nsb_sample7.invoke_accumulate(tensor, 6.0) == 6.0
    assert tensor_values[1, 1, 1] == 6.0
finally:
    tensor.close()


lazy = LazyArrayAccumulator()
try:
    assert nsb_sample7.invoke_accumulate(lazy, 3.0) == 3.0
    assert nsb_sample7.invoke_accumulate(lazy, 4.0) == 7.0
finally:
    lazy.close()
