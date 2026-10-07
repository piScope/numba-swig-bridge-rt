# Sample 2: C++ templates

`array.hpp` defines a simple owning `Array<T>`. `arrays.i` explicitly
instantiates it as `IntArray` and `DoubleArray`, selects both for numba-swig-bridge,
and maps scalar accessors to Python indexing. No directors are needed.

The setuptools custom build invokes CMake, bridge generation, and SWIG, then
compiles one owning extension, `nsb_sample2_arrays`. Generated registration and
metadata live inside `nsb_sample2`, keeping them separate from sample 1.

From the repository root, build the updated framework wheel and install:

```bash
python -m pip wheel . --no-deps --wheel-dir /tmp/numba-swig-bridge-wheels
python -m pip install ./sample_libraries/2_templates \
  --find-links /tmp/numba-swig-bridge-wheels
python usercode_examples/example2/template_arrays.py
```

See [the usage example](../../usercode_examples/example2/README.md) for indexing
semantics and current limits. The C++ library requires a C++17 compiler.
