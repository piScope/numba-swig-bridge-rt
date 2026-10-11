"""Use ordinary state and two different Numba director bases in one callback."""

from numba import types

import nsb_sample1_objects as objects
import nsb_sample1_directors as directors
import nsb_sample1_registration as sample_bridge
import nsb_rt as nsb

inner_binding = sample_bridge.director_binding("nsb_sample1::Inner")
auxiliary_binding = sample_bridge.director_binding("nsb_sample1::Auxiliary")
outer_binding = sample_bridge.director_binding("nsb_sample1::Outer")


@nsb.director(
    base=directors.Inner,
    binding=inner_binding,
    state={"factor": types.float64},
    fallback="silent",
)
class ScaledInner:
    def __init__(self, factor):
        self.factor = factor

    @nsb.override("Apply", "double")
    def apply(self, value):
        return self.factor * value


@nsb.director(
    base=directors.Auxiliary,
    binding=auxiliary_binding,
    state={"offset": types.intc},
    fallback="silent",
)
class OffsetAuxiliary:
    def __init__(self, offset):
        self.offset = offset

    @nsb.override("Adjust", "double")
    def adjust(self, value):
        return value + self.offset


@nsb.director(
    base=directors.Outer,
    binding=outer_binding,
    state={
        "bias": types.float64,
        "increment": types.intc,
        "count": types.int64,
        "enabled": types.boolean,
        "buffer": objects.Buffer,
        "inner": inner_binding,
        "auxiliary": auxiliary_binding,
    },
    fallback="silent",
)
class CombinedOuter:
    def __init__(self, bias, increment, count, enabled, buffer, inner, auxiliary):
        self.bias = bias
        self.increment = increment
        self.count = count
        self.enabled = enabled
        self.buffer = buffer
        self.inner = inner
        self.auxiliary = auxiliary

    @nsb.override("Combine", "double")
    def combine(self, value):
        total = self.inner.Apply(value) + self.auxiliary.Adjust(value)
        if self.enabled:
            total += self.buffer.Get(0)
        return total + self.bias + self.increment + self.count


def buffer(value):
    result = objects.Buffer(1)
    result.Set(0, value)
    return result


inner = ScaledInner(3.0)
auxiliary = OffsetAuxiliary(4)
outer = CombinedOuter(0.5, 1, 2, True, buffer(5.0), inner, auxiliary)
try:
    assert directors.invoke_outer(outer.as_base_proxy(), 2.0) == 20.5

    # The outer native lifetime control retains both dependencies. Directly
    # closing their Python facades does not invalidate the C++ callback graph.
    inner.close()
    auxiliary.close()
    assert directors.invoke_outer(outer.as_base_proxy(), 2.0) == 20.5
    print("Heterogeneous director state completed")
finally:
    outer.close()
