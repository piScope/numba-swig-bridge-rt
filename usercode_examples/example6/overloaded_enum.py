"""Use both overloaded constructors of a native Numba director."""

from numba import types
import nsb_rt as nsb
import nsb_sample6
import nsb_sample6.registration  # noqa: F401


@nsb.director(state={"factor": types.float64}, fallback="silent")
class MultiplyEvaluator(nsb_sample6.Evaluator):
    def __init__(self, equations, dimension, factor):
        super().__init__(equations, dimension, nsb_sample6.Evaluator.MULTIPLY)
        self.factor = factor

    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return self.factor * value

    @nsb.override("Record", "double")
    def record(self, value):
        # A void C++ virtual maps to a Numba callback with no return value.
        pass


@nsb.director(state={"offset": types.float64}, fallback="silent")
class OffsetEvaluator(nsb_sample6.Evaluator):
    def __init__(self, scale, offset):
        super().__init__(scale, nsb_sample6.Evaluator.OFFSET)
        self.offset = offset

    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return value + self.offset

    @nsb.override("Record", "double")
    def record(self, value):
        pass


multiply = MultiplyEvaluator(3, 2, 2.5)
offset = OffsetEvaluator(2.0, 1.5)
try:
    assert nsb_sample6.invoke_evaluate(multiply, 4.0) == 10.0
    assert nsb_sample6.invoke_evaluate(offset, 4.0) == 5.5
    nsb_sample6.invoke_record(multiply, 3.0)
    nsb_sample6.invoke_record(offset, 4.0)
    print("overloaded enum director: Normal End")
finally:
    multiply.close()
    offset.close()
