"""Transfer a generated director to an owning C++ container."""

from numba import types

import nsb_sample1_directors as directors
import nsb_sample1_registration
import nsb_sample1_director_bindings as native
import nsb_rt as nsb


@nsb.director(state={"offset": types.float64}, fallback="silent")
class OffsetEvaluator(directors.Evaluator):
    def __init__(self, offset):
        self.offset = offset
        super().__init__()

    @nsb.override("Bias", "double")
    def bias(self, value):
        return value + self.offset

    @nsb.override(
        "Evaluate",
        "nsb_sample1::Buffer",
        "nsb_sample1::Buffer",
        "nsb_sample1::Counter",
        "double",
        "int",
        "std::int64_t",
        "bool",
    )
    def evaluate(self, source, destination, counter, scale, bias, count, reverse):
        return self.offset

    @nsb.override("EvaluateValues", "const double*", "std::int64_t")
    def evaluate_values(self, values, count):
        return self.offset


before = native.nsb_director_destruction_count()
director = OffsetEvaluator(3.5)
owner = directors.ExternalEvaluatorOwner()
owner.Take(director, False)
director.thisown = False
director.close()

assert owner.InvokeBias(2.0) == 5.5
owner.DeleteEvaluator()
assert native.nsb_director_destruction_count() == before + 1
print("director ownership handoff: Normal End")
