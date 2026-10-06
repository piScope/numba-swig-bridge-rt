# Constructor arguments and const virtual methods

Install sample library `4_constructor_const`, then run:

```bash
python package/numba-swig-bridge-rt/usercode_examples/example4/constructor_const.py
```

The decorated class inherits the SWIG `Evaluator` base and calls
`super().__init__(equations, dimension)` to construct the generated native
director. `ScaledEvaluator(3, 2, 2.5)` initializes the C++ dimensions and the
separate mutable `scale` state. Both `super()` and explicit
`super(ExplicitEvaluator, self)` work.

Instances support inherited methods and direct arguments to SWIG functions
and Numba functions. `as_base_proxy()` remains available and returns the same
instance for subclass directors. Normal SWIG classes are unchanged.

The example checks const virtual dispatch, inherited fallback, preserved
Python helper methods, constructor errors, missing/repeated base initialization,
ownership protection, and rejection of closed instances.

A subclass with no Python initializer forwards its arguments to the C++ base
automatically. The initial interface supports direct single inheritance from
the selected SWIG base; additional Python inheritance chains are deferred.

## Float32 support

Run `package/numba-swig-bridge-rt/usercode_examples/example4/float32.py` from the repository root after
installing the sample. `FloatEvaluator` demonstrates float constructor arguments,
const float callbacks, float32 state, and a borrowed float32 input-array callback.
`FloatOps` demonstrates ordinary scalar calls, exact float/double overload
selection, and checked float32 array inputs. The framework and sample helpers
must be rebuilt together for state ABI version 2.

The scalar example also uses `scalar_t -> real_t -> float` and a separate double
alias. Its override selector names `nsb_sample4::scalar_t`, matching the type
reported by SWIG, while callback compilation uses float32.
