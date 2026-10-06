# numba-swig-bridge use cases

These examples use the C++ library in
[`sample_libraries/1_multiple_extensions/`](../../../../sample_libraries/1_multiple_extensions/).
Its ordinary objects and director bases live in separately compiled SWIG
extensions and share one bridge registration.

| Use case | Example | Behavior demonstrated |
| --- | --- | --- |
| Ordinary C++ methods in `@njit` | [`ordinary_methods.py`](ordinary_methods.py) | An operator from one extension reads and modifies objects from another. |
| Compiled virtual overrides | [`director_and_ordinary.py`](director_and_ordinary.py) | A Numba callback calls ordinary methods and uses pointer/count recipes and overload selectors. |
| Persistent nested director state | [`heterogeneous_directors.py`](heterogeneous_directors.py) | An outer director retains ordinary objects and two different nested directors after their direct Python handles close. |
| Native ownership handoff | [`director_ownership.py`](director_ownership.py) | A C++ owner keeps callbacks and state alive after the Python director facade closes, then deletes the director through its virtual base. |

Run all examples from the repository root after building:

```bash
./build.sh
./package/numba-swig-bridge-rt/usercode_examples/example1/run.sh
```

To run one example directly:

```bash
PYTHONPATH=build/python python package/numba-swig-bridge-rt/usercode_examples/example1/ordinary_methods.py
```
