# Inherited methods and C++ casts

[`inherited_method.py`](inherited_method.py) uses
[`sample 3`](../../../../sample_libraries/3_inheritance/README.md). All demonstrated
calls now work inside `@njit`:

```python
import nsb_rt as nsb
from nsb_sample3 import Base, Derived

@njit
def use_derived(obj):
    value = obj.BaseMethod()             # Automatic inherited-method lookup.
    base_view = nsb.upcast(obj, Base)        # Explicit borrowed base view.
    derived_view = nsb.dynamic_cast(base_view, Derived)
    return value + derived_view.DerivedMethod()
```

Automatic lookup and base-argument conversion follow a single public base
chain with non-virtual conversions. Methods declared on a derived class hide
base overloads. The inventory records direct public bases without copying
base methods into derived classes; Numba resolves each call during typing.

For multiple or virtual inheritance, use `nsb.upcast(obj, Base)` explicitly. The
C++ compiler checks accessibility and ambiguity and performs the actual pointer
adjustment. Ambiguous or inaccessible conversions raise a Numba `TypingError`.
An ambiguous diamond can be disambiguated with two explicit upcasts through a
chosen intermediate base.

`nsb.dynamic_cast(base_view, Derived)` supports checked downcasts along the selected
public hierarchy. It requires a polymorphic source class and raises `TypeError`
if the actual object is not the requested derived type. Unsupported casts are
rejected during compilation. Sibling crosscasts and unchecked static downcasts
are not provided.

Casts create borrowed views, with no allocation or ownership transfer. Keep the
original Python owner alive throughout their use. These views can be used within
compiled code; returning one to Python still requires the object-boxing policy
that is deferred. Both source and destination must be selected classes in the
same bridge.

After installation, run from the repository root:

```bash
python package/numba-swig-bridge-rt/usercode_examples/example3/inherited_method.py
```
