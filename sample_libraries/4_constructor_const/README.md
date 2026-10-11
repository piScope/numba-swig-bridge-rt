# Sample 4: Constructor forwarding and const overrides

`Evaluator` requires two integer dimensions and declares a const pure virtual
`Evaluate(double)` and a const non-pure `Bias(double)`.

The decorated Python class inherits `Evaluator` and forwards constructor
arguments with `super().__init__(equations, dimension)`. The generated native
director owns the C++ object; the Python instance exposes its borrowed SWIG
pointer. Dimensions remain in C++ storage; mutable callback state is separate.
The matching [user example](../../usercode_examples/example4/constructor_const.py)
checks C++ dispatch through a const reference, inherited fallback, ordinary
Numba calls, mutable callback state, and rejected constructor arguments.

Build and install with the framework wheel available:

```bash
python -m pip install . --find-links /tmp/numba-swig-bridge-wheels
python ../../usercode_examples/example4/constructor_const.py
```

The initial constructor support requires exactly one public constructor with
supported scalar arguments. Overloaded constructors and object arguments are
not supported. Constructor failures currently produce a generic factory error.

## Float32 support

Run `package/numba-swig-bridge-rt/usercode_examples/example4/float32.py` from the repository root after
installing the sample. `FloatEvaluator` demonstrates float constructor arguments,
const float callbacks, float32 state, and a borrowed float32 input-array callback.
`FloatOps` demonstrates ordinary scalar calls, exact float/double overload
selection, and checked float32 array inputs. The framework and sample helpers
must be rebuilt together for state ABI version 2.
