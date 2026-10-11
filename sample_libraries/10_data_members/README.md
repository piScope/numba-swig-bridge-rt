# Automatic C++ data-member access

Public scalar/enum fields need no opt-in. This standalone library tests native
const fields, SWIG immutable/renamed/ignored fields, optional readonly/exclusion,
borrowed object pointers, nullable values, and distinct pointer/pointee
constness. Const callback arguments and ordinary return values reject writes.
The example validates native-owned and borrowed storage, C++ dispatch, explicit
owner retention, and invalid assignments. `Opaque` verifies const pointer
access even when its class has no exposed members.

Install NSB and this sample, then run:

```sh
python -I package/numba-swig-bridge-rt/usercode_examples/example10/data_members.py
```

Pointer slots are never assigned in compiled code. The containing object only
serves as the retained owner in this example because it owns its stored child.
