"""Target spelling for frame-owned C++ views over NumPy storage."""

import numpy as np
from numba import njit
import nsb_rt as nsb
import nsb_sample8
import nsb_sample8_storage
import nsb_sample8.registration  # noqa: F401


@njit
def update(values, matrix_data):
    vector = nsb_sample8.Vector(nsb.as_cpointer(values), values.size)
    matrix = nsb_sample8.DenseMatrix(nsb.as_cpointer(matrix_data), 2, 3)
    vector.Add(2.0)
    matrix.Set(1, 2, 9.0)
    return vector.Sum() + matrix.Sum()


values = np.array([1.0, 2.0, 3.0])
matrix_data = np.asfortranarray(np.arange(6.0).reshape((2, 3), order="F"))
destroyed_before = nsb_sample8_storage.DestroyedViewCount()
assert update(values, matrix_data) == 31.0
assert values.tolist() == [3.0, 4.0, 5.0]
assert matrix_data[1, 2] == 9.0
assert nsb_sample8_storage.DestroyedViewCount() == destroyed_before + 2
print("owned array views: Normal End")
