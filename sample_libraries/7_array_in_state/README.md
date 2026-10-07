# Sample 7: array state in a native director

`Accumulator` has one virtual method. The companion example declares a
writable one-, two-, or three-dimensional `float64` NumPy array as director
state. The example covers C-contiguous, Fortran-contiguous, and strided
storage. Its compiled overrides declare `stateaccess="values"`, so natural
repeated `self.values` reads construct one callback-local array descriptor.
The one-dimensional case also uses `nsb.as_cpointer` to access the same
storage through a native typed pointer. The lazy case verifies that assignment
remains live and write-through.

This sample is the integration fixture for array-state support: the state owner
must keep the NumPy array alive while C++ calls the director callback.

Run after installation:

```bash
python usercode_examples/example7/array_in_state.py
```
