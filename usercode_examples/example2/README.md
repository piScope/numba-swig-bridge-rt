# C++ template arrays

[`2_templates`](../../sample_libraries/2_templates/) exposes one C++ template
as `IntArray` (`Array<int>`) and `DoubleArray` (`Array<double>`). Run
[`template_arrays.py`](template_arrays.py) after installing the sample:

```bash
python usercode_examples/example2/template_arrays.py
```

The example creates the arrays in Python, then reads and writes them in
`@njit` code, including `integers[1] = 3`. Changes are visible through the
original Python proxies. Each specialization has its own Numba bridge type.

SWIG renames value-based `Get`/`Set` accessors to `__getitem__`/`__setitem__`;
Numba indexing calls their generated C ABI adapters. The C++ `operator[]`
returns references, but those references are not exported through the bridge.
Integer arrays accept integer values; double arrays currently require floating
values, such as `3.0`. Slices and construction inside `@njit` are not supported.
Indices are zero-based; negative indices are rejected rather than interpreted
relative to the end. Out-of-range accesses raise the bridge's `RuntimeError`
in compiled code.
