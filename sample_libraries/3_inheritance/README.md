# Sample 3: inheritance and casts

`Base` declares `BaseMethod`; `Derived` inherits it without redefining it.
`GrandDerived` extends the inheritance chain. All selected classes retain their
own declarations in the inventory; inherited methods are resolved by Numba
when compiling a call.

This sample uses custom setuptools commands: CMake builds the C++ library,
then bridge generation and SWIG produce `nsb_sample3_inheritance` and
`nsb_sample3.registration`.

From the repository root, build the updated framework and install the sample:

```bash
python -m pip wheel . --no-deps --wheel-dir /tmp/numba-swig-bridge-wheels
python -m pip install ./sample_libraries/3_inheritance \
  --find-links /tmp/numba-swig-bridge-wheels
python usercode_examples/example3/inherited_method.py
```

[The usage example](../../usercode_examples/example3/README.md) demonstrates
inherited calls, automatic base-argument conversion, explicit upcasts, and
checked downcasts. `Base` has a virtual destructor to permit `dynamic_cast`.
`Multiple` has its `Base` subobject at an adjusted address; `VirtualDerived`
requires a virtual-base conversion. Both use explicit `upcast`.

Additional classes exercise method hiding, ambiguous bases, private
inheritance, and rejection of downcasts from non-polymorphic bases in the
integration tests. A C++17 compiler with RTTI is required for checked downcasts.
