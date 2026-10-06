"""Use typed 1-D, Fortran-layout 2-D, and strided 3-D data views from Numba."""
from numba import njit
from nsb_sample5 import registration  # noqa: F401
from nsb_sample5_storage import (
    Vector, IntVector, FloatVector, BoolVector, DenseMatrix, DenseTensor,
)

@njit
def fill(vector, int_vector, float_vector, bool_vector, matrix, tensor):
    v = vector.GetDataArray()
    iv = int_vector.GetDataArray()
    fv = float_vector.GetDataArray()
    bv = bool_vector.GetDataArray()
    a = matrix.GetDataArray()
    for i in range(v.size):
        v[i] = i + 1.0
        iv[i] = 2 * i
        fv[i] = i + 0.5
        bv[i] = i % 2 == 0
    for j in range(a.shape[1]):
        for i in range(a.shape[0]):
            a[i, j] = 10.0 * j + i
    q = tensor.GetDataArray()
    q[2, 1, 2] = 99.0
    return v.sum() + iv.sum() + fv.sum() + bv.sum() + a[1, 2] + q[2, 1, 2]

v, iv = Vector(3), IntVector(3)
fv, bv = FloatVector(3), BoolVector(3)
a, q = DenseMatrix(2, 3), DenseTensor(2, 3, 4)
assert fill(v, iv, fv, bv, a, q) == 6.0 + 6 + 4.5 + 2 + 21.0 + 99.0
assert v.Get(2) == 3.0
assert iv.Get(2) == 4
assert fv.Get(2) == 2.5
assert bv.Get(2)
print("Typed 1-D, Fortran 2-D, and strided 3-D GetDataArray views passed")
