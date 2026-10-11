"""Float32 ordinary ABI, callbacks, state, constructor and array inputs."""

import ctypes
import numpy as np
import nsb_rt as nsb
from numba import njit, types, carray
from numba.core.errors import TypingError
from nsb_sample4 import FloatEvaluator, FloatOps, invoke_float, invoke_float_values
import nsb_sample4.registration as bridge


@nsb.director(
    binding=bridge.director_binding("nsb_sample4::FloatEvaluator"),
    state={"gain": types.float32, "offset": types.float32},
    fallback="silent",
)
class AffineFloat(FloatEvaluator):
    def __init__(self, gain, offset):
        super().__init__(gain)
        self.gain = gain
        self.offset = offset

    @nsb.override("Apply", "nsb_sample4::scalar_t")
    def apply(self, value):
        return self.gain * value + self.offset

    @nsb.override("SumValues", "const float*", "std::int64_t")
    def sum_values(self, values, count):
        array = carray(values, count)
        total = np.float32(0)
        for i in range(count):
            total += array[i]
        return total + self.offset


@njit
def round_trip(ops, value):
    return ops.RoundTrip(value)


@njit
def double_round_trip(ops, value):
    return ops.DoubleRoundTrip(value)


@njit
def precision(ops, value):
    return ops.Precision(value)


@njit
def array_sum(ops, values):
    return ops.SumValues(values)


@njit
def apply_director(evaluator, value):
    return evaluator.Apply(value)


def main():
    ops = FloatOps()
    value = np.float32(1.25)
    assert double_round_trip(ops, np.float64(1.25)) == 1.25
    assert (
        double_round_trip.overloads[
            double_round_trip.signatures[0]
        ].signature.return_type
        == types.float64
    )
    assert round_trip(ops, value) == value
    assert (
        round_trip.overloads[round_trip.signatures[0]].signature.return_type
        == types.float32
    )
    assert round_trip(ops, np.float64(16777217)) == 16777216.0
    assert np.isnan(round_trip(ops, np.float32(np.nan)))
    assert round_trip(ops, np.float32(np.inf)) == np.inf
    assert round_trip(ops, np.float32(-np.inf)) == -np.inf
    assert precision(ops, np.float32(1)) == 32
    assert precision(ops, np.float64(1)) == 64
    values = np.array([1.25, 2.5, 3.75], dtype=np.float32)
    assert array_sum(ops, values) == 7.5
    assert array_sum(ops, np.array([], dtype=np.float32)) == 0.0
    assert (
        array_sum.overloads[array_sum.signatures[0]].signature.return_type
        == types.float32
    )
    for invalid in (values.astype(np.float64), values[::2], values.reshape(1, 3)):
        try:
            array_sum(ops, invalid)
        except TypingError:
            pass
        else:
            raise AssertionError("invalid float32 array was accepted")
    evaluator = AffineFloat(1.5, 0.25)
    try:
        assert evaluator.Gain() == 1.5
        assert invoke_float(evaluator, float(value)) == 2.125
        assert apply_director(evaluator, value) == 2.125
        assert (
            apply_director.overloads[apply_director.signatures[0]].signature.return_type
            == types.float32
        )
        assert evaluator.Scale(float(value)) == 1.875  # C++ const fallback
        assert invoke_float_values(evaluator) == 7.75
        # Native float state rounds on assignment, independently of callbacks.
        evaluator.offset = 16777217.0
        assert evaluator.offset == 16777216.0
        evaluator.offset = 0.5
        assert invoke_float(evaluator, float(value)) == 2.375
        for callback in evaluator._numba_callbacks.values():
            # Directors use status-plus-output callbacks: C++ receives the
            # float result through the final pointer argument.
            assert callback.ctypes.restype is ctypes.c_int
    finally:
        evaluator.close()
    try:
        _ = evaluator.offset
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed float state was readable")
    print("Float32 scalar ABI, overloads, callbacks, state and array inputs passed")


if __name__ == "__main__":
    main()
