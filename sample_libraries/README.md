# Sample libraries

These optional packages provide the native libraries used by the examples in
the public runtime repository. They are separate distributions so installing
the runtime alone does not build native code.

From the repository root, install the samples when you want to run the
examples:

```bash
python -m pip install ./sample_libraries/*/
```

The sample packages require a C++17 compiler, CMake, and SWIG. Every sample
contains committed generated bridge artifacts under `nsb_artifacts/`, so a
sample build does not import the bridge generator. Sample 1 is also the clean
consumer fixture used by the runtime release checks.

- [`1_multiple_extensions/`](1_multiple_extensions/) demonstrates ordinary
  objects and directors across two SWIG extensions.
- [`2_templates/`](2_templates/README.md) demonstrates `Array<int>` and
  `Array<double>` with indexed reads and writes in Numba.
- [`3_inheritance/`](3_inheritance/README.md) demonstrates inherited methods,
  explicit upcasts, and checked downcasts in compiled code.

- [`4_constructor_const/`](4_constructor_const/README.md) demonstrates scalar
  constructor forwarding and const virtual overrides.
- [`5_GetDataArray/`](5_GetDataArray/) demonstrates writable 1-D, Fortran-layout
  2-D, and strided 3-D `GetDataArray()` views in Numba.
- [`6_overloaded_enum/`](6_overloaded_enum/README.md) demonstrates overloaded
  native director constructors and enum constructor arguments.
- [`7_array_in_state/`](7_array_in_state/README.md) is the integration fixture
  for Python-assigned and Numba-owned NumPy array state in a native director callback.
- [`8_owned_array_views/`](8_owned_array_views/README.md) is a compact standalone
  fixture for raw pointer returns and owned C++ array views.
- [`9_object_returns/`](9_object_returns/README.md) demonstrates ordinary and
  native-director borrowed object pointer/reference returns, owner retention,
  `own_this` transfer, and Numba-owned director state.
- [`10_data_members/`](10_data_members/README.md) demonstrates native
  data-member access.
