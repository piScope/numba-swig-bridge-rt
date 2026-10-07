# User-code examples

Each directory contains Python usage examples for the corresponding numbered
C++ library in the repository's [`sample_libraries/`](../sample_libraries/)
companion distributions.

The samples are optional and are not installed with the runtime. From the
repository root, install them before running examples:

```bash
python -m pip install ./sample_libraries/*/
```

Run the examples after installing the runtime and the corresponding sample
package. The runtime release check uses
`scripts/validate_runtime_consumer.py` to execute sample-1 examples in a clean
consumer environment.

- [`example1/`](example1/README.md) uses `nsb_sample1`: ordinary C++ methods,
  compiled director callbacks, and persistent nested director state across two
  SWIG extensions.
- [`example2/`](example2/README.md) uses `nsb_sample2`: C++ template arrays with
  indexed reads and writes inside `@njit`.
- [`example3/`](example3/README.md) uses `nsb_sample3`: inherited methods, explicit
  upcasts, and checked downcasts inside `@njit`.
- [`example4/`](example4/README.md) uses `nsb_sample4`: C++ constructor arguments,
  const virtual dispatch, and inherited fallback.
- [`example5/`](example5/get_data_array.py) uses `nsb_sample5`: writable 1-D,
  2-D, and 3-D `GetDataArray()` views with their declared native layouts.
- [`example6/`](example6/overloaded_enum.py) uses `nsb_sample6`: overloaded
  director constructors, enum arguments, and a `void` virtual callback.
- [`example7/`](example7/array_in_state.py) uses `nsb_sample7`: writable
  NumPy director state with C-contiguous, Fortran-contiguous, and strided
  layouts, including lazy Numba-owned allocation.
- [`example8/`](example8/owned_array_views.py) uses `nsb_sample8`: raw pointer
  returns and frame-owned C++ array views over NumPy storage.
- [`example9/`](example9/object_returns.py) uses `nsb_sample9`: ordinary and
  native-director borrowed C++ object returns, retained parents, `own_this`
  transfer, and lazy native-owned director state.
