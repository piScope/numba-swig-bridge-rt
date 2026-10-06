"""Inherited methods, explicit base views, and checked downcasts inside @njit."""
from numba import njit
from nsb_sample3 import Base, Derived
import nsb_rt as nsb
from nsb_sample3 import GrandDerived, Multiple, VirtualDerived, Consumer
import nsb_sample3.registration


@njit
def call_base_method(obj):
    return obj.BaseMethod()


@njit
def call_derived_method(obj):
    return obj.DerivedMethod()


base = Base(7)
derived = Derived(7)
assert derived.BaseMethod() == 7
print('Python: Derived.BaseMethod() passed', flush=True)
assert call_base_method(base) == 7
print('Numba: Base.BaseMethod() passed', flush=True)
assert call_derived_method(derived) == 14
print('Numba: Derived.DerivedMethod() passed', flush=True)

# The actual question: compile the same base method on the derived object.
assert call_base_method(derived) == 7
print('Numba: Derived.BaseMethod() passed', flush=True)

assert call_base_method(GrandDerived(8)) == 8


@njit
def read_argument(consumer, obj):
    # Automatic conversion also applies to Base& method parameters.
    return consumer.Read(obj)


assert read_argument(Consumer(), derived) == 7


@njit
def read_explicit_base(obj):
    base_view = nsb.upcast(obj, Base)
    return base_view.BaseMethod()


multiple = Multiple(9)
assert multiple.BaseIsOffset()  # This case needs a real C++ pointer adjustment.
assert read_explicit_base(multiple) == 9
assert read_explicit_base(VirtualDerived(10)) == 10


@njit
def checked_round_trip(obj):
    base_view = nsb.upcast(obj, Base)
    derived_view = nsb.dynamic_cast(base_view, Derived)
    return derived_view.DerivedMethod()


assert checked_round_trip(derived) == 14
assert checked_round_trip(multiple) == 18


@njit
def checked_downcast(base):
    return nsb.dynamic_cast(base, Derived).DerivedMethod()


try:
    checked_downcast(Base(7))
except TypeError:
    print('Checked downcast failure raised TypeError', flush=True)
else:
    raise AssertionError('a Base instance is not a Derived instance')

print('Automatic inheritance, explicit upcasts, and checked downcasts passed', flush=True)
