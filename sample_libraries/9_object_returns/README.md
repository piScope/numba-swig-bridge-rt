# Sample 9: object returns and ownership transfer

This MFEM-independent fixture demonstrates the native director ABI for
borrowed C++ object results. `Provider::GetChild()` returns `Child&`; its
Numba callback writes a non-null `void*` output. `Provider::MaybeChild(bool)`
returns `Child*`; its callback may successfully write a null address.

`object_returns.py` returns children held by a `Parent` retained in director
state, so no object ownership crosses the director boundary.

`native_state.py` leaves `parent` empty at Python construction time and creates
`Parent(41)` in the first Numba callback, then replaces it with `Parent(42)`.
The director-state slot takes the constructor result's NRT token; the example
checks that replacement releases the first object and `close()` releases the
second.

`ordinary_returns.py` covers the Python boundary: an ordinary non-owning child
return, `@nsb.borrowed_from(owner_arg="parent")` for a Python/SWIG parent,
`nsb.borrowed_from(child, parent)` for an NRT-owned parent, and `nsb.own_this`
for a C++ object whose ownership transfers to Python.

Run after installation:

```bash
python usercode_examples/example9/object_returns.py
python usercode_examples/example9/native_state.py
python usercode_examples/example9/ordinary_returns.py
```
