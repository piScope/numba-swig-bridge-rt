"""Call ordinary methods across two SWIG extensions inside @njit."""

from numba import njit

import nsb_sample1_objects as objects
import nsb_sample1_directors as operators
import nsb_sample1_registration  # Install the sample library's bridge.


@njit
def transform(operator, source, destination, counter):
    # The operator belongs to one extension; its arguments belong to another.
    return operator.Transform(
        source, destination, counter, 2.0, 1, source.Size(), False
    )


@njit
def maybe_sum(buffer, present):
    # A selected C++ T* return is Optional[T] in Numba.  The returned object
    # is borrowed from ``buffer`` and remains valid only while its owner does.
    borrowed = buffer.MaybeSelf(present)
    if borrowed is None:
        return -1.0
    return borrowed.Sum()


def main():
    source = objects.Buffer(3)
    for index, value in enumerate((1.0, 2.0, 4.0)):
        source.Set(index, value)
    destination = objects.Buffer(3)
    counter = objects.Counter()
    total = transform(operators.Arithmetic(), source, destination, counter)
    assert total == 17.0
    assert [destination.Get(index) for index in range(3)] == [3.0, 5.0, 9.0]
    assert counter.Value() == 3
    assert maybe_sum(source, True) == 7.0
    assert maybe_sum(source, False) == -1.0
    print("Ordinary methods across SWIG extensions completed")


if __name__ == "__main__":
    main()
