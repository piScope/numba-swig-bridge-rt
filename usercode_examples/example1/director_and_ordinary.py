"""A regular target-library user's Numba director workflow."""

from numba import carray, njit, types

import nsb_sample1_objects as objects
import nsb_sample1_directors as directors
import nsb_sample1_registration as sample_bridge
import nsb_rt as nsb

binding = sample_bridge.director_binding("nsb_sample1::Evaluator")


@nsb.director(
    base=directors.Evaluator,
    binding=binding,
    state={"offset": types.float64},
)
class OffsetEvaluator:
    def __init__(self, offset):
        self.offset = offset

    @nsb.override("Bias", "double", stateaccess="offset")
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
        stateaccess="offset",
    )
    def evaluate(self, source, destination, counter, scale, bias, count, reverse):
        total = 0.0
        for index in range(count):
            position = count - 1 - index if reverse else index
            value = source.Get(position) * scale + bias + self.offset
            destination.Set(position, value)
            total += value
        counter.Add(count)
        return total

    @nsb.override(
        "EvaluateValues", "const double*", "std::int64_t", stateaccess="offset"
    )
    def evaluate_values(self, values, count):
        view = carray(values, count)
        total = 0.0
        for index in range(count):
            total += view[index] + self.offset
        return total

    # Two C++ overloads share Probe.  The canonical selectors keep both
    # callback slots available without relying on SWIG's Python dispatcher.
    @nsb.override("Probe", "double", stateaccess="offset")
    def probe_scalar(self, value):
        return value + self.offset

    @nsb.override("Probe", "const double*", "std::int64_t", stateaccess="offset")
    def probe_values(self, values, count):
        view = carray(values, count)
        total = self.offset
        for index in range(count):
            total += view[index]
        return total


def buffer(*values):
    result = objects.Buffer(len(values))
    for index, value in enumerate(values):
        result.Set(index, value)
    return result


@njit
def transform(operator, source, destination, counter):
    return operator.Transform(source, destination, counter, 2.0, 1, 3, False)


source = buffer(1.0, 2.0, 4.0)
destination = objects.Buffer(3)
counter = objects.Counter()
evaluator = OffsetEvaluator(0.5)
try:
    base = evaluator.as_base_proxy()
    callback_total = directors.invoke_evaluate(
        base, source, destination, counter, 2.0, 1, 3, False
    )
    pointer_total = directors.invoke_evaluate_values(base, source)
    ordinary_total = transform(directors.Arithmetic(), source, destination, counter)
    assert (callback_total, pointer_total, ordinary_total, counter.Value()) == (
        18.5,
        8.5,
        17.0,
        6,
    )
    print("Numba director and ordinary bridge completed")
finally:
    evaluator.close()
