# Sample 6: overloaded director constructors and enum arguments

`Evaluator` has two constructor overloads:

```cpp
Evaluator(int equations, int dimension, Mode mode);
Evaluator(double scale, Mode mode);
```

`Mode` is a nested C++ enum. The generated native director exposes both
constructor forms through one SWIG factory, so an `nsb.director` subclass can
use the normal Python spelling:

```python
super().__init__(3, 2, nsb_sample6.Evaluator.MULTIPLY)
super().__init__(2.0, nsb_sample6.Evaluator.OFFSET)
```

It also includes a `void Record(double) const` virtual method. The example
invokes it through the native C++ base without a callback return value.

Run the example after installation:

```bash
python usercode_examples/example6/overloaded_enum.py
```

`ProtectedEvaluator` and `ProtectedConcrete` cover protected constructors for
abstract and non-abstract bases. The generated subclass forwards supported
scalar/enum arguments and default forms; its user-supplied Numba override is
invoked by a real C++ caller. The unsupported pointer overload is diagnosed,
and private constructors are excluded. Run
`usercode_examples/example6/protected_constructors.py` after installing sample 6.
