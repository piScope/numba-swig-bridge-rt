# numba-swig-bridge-rt

`numba-swig-bridge-rt` is the public runtime used by bridge-enabled downstream
SWIG packages.  It provides the Numba-facing API used by an application's
compiled code; it is not a replacement for the downstream package's normal
Python API.

Applications import it as:

```python
import nsb_rt as nsb
```

Import the downstream package's generated registration module before compiling
code that passes its SWIG objects to Numba.  The registration module installs
the bridge types and its downstream-owned director-state backend.

```python
import nsb_rt as nsb
from numba import njit
from mylibrary import Buffer
import mylibrary.nsb_registration  # installs generated bridge support

@njit
def increment(buffer):
    buffer.Add(1.0)
    return buffer.Value()

buffer = Buffer()
assert increment(buffer) == 1.0
```

## Compiled virtual overrides

For the usual case, subclass the generated SWIG director base directly.
After importing `mylibrary.nsb_registration`, `nsb.director` finds that base's
unique generated binding automatically. Director state is declared in the
decorator and is available to `@nsb.override` methods as normal attributes:

```python
from numba import types
import mylibrary
import mylibrary.nsb_registration  # installs the Evaluator binding

@nsb.director(state={"offset": types.float64})
class OffsetEvaluator(mylibrary.Evaluator):
    @nsb.override("Evaluate", "double")
    def evaluate(self, value):
        return value + self.offset

evaluator = OffsetEvaluator()
evaluator.offset = 0.5
try:
    # Pass evaluator.as_base_proxy() to the downstream library as required.
    # mylibrary.use_evaluator(evaluator.as_base_proxy())
finally:
    evaluator.close()
```

Use `nsb.as_cpointer(array)` only inside `@njit`, and only for immediate calls
to a downstream API that does not retain the pointer.  Use `nsb.own_this`,
`nsb.borrowed_from`, and `nsb.is_slot_occupied` where supported by the
downstream generated bridge.

### Python subclasses of a compiled director

An ordinary Python subclass can customize initialization and add Python methods
to an already decorated director:

```python
class FixedOffsetEvaluator(OffsetEvaluator):
    def __init__(self):
        super().__init__()
        self.offset = 2.0
```

Each subclass instance receives its own StateBlock with the parent's declared
fields and ownership policies. It uses the parent's native director and compiled
callbacks; instances do not share their stored values or retained owners.

This does not extend the compiled director schema:

- Additional Python attributes are ordinary Python state, unavailable to
  compiled overrides unless declared in the parent's `state` mapping.
- A new or redefined `@nsb.override` method on the undecorated subclass does
  not replace a native callback. The parent's callbacks were selected and
  compiled when its `@nsb.director` decorator ran.
- Decorating the subclass again to add state or replace compiled overrides
  is unsupported. `@nsb.director` currently requires direct inheritance from
  the registered SWIG base; it does not merge decorated parent schemas or
  inherited override definitions.

To define another compiled director with different fields or overrides, declare
it directly from the registered SWIG base and supply its own `state` mapping
and override methods. Extending compiled directors through Python inheritance
requires additional support.

## Examples

The [`usercode_examples/`](usercode_examples/README.md) directory contains
end-user examples using the optional sample downstream packages.  They all
import the public runtime as `nsb_rt`; none needs the internal generator.

The sample packages are companion distributions in the repository source
bundle.  They are deliberately not runtime dependencies, so installing
`numba-swig-bridge-rt` alone does not build native samples.  From the
repository root, install them when you want to run the examples:

```bash
python -m pip install ./sample_libraries/*/
```

The same command works after installing a runtime wheel, provided the
repository's `sample_libraries/` directory is available.

## Downstream packages

The runtime is consumed by downstream packages that ship generated SWIG bridge
artifacts. End users install that downstream package and this runtime; they do
not need SWIG or the bridge generator. The generator is maintained separately
from this public runtime repository and is only needed when developing a new
bridge-enabled package.

Downstream state extensions compile the runtime-supplied native header and
sources. From 0.14.2, the private StateBlock/StateHandle classes use hidden
visibility on supported non-Windows GCC/Clang toolchains so internal state reads
can bind locally and inline. Release 0.15.0 advances the state ABI to 7 with
a borrowed-address setter. Rebuild downstream state extensions and regenerate
their artifacts with compatibility minimum 0.15.0. Windows, unsupported
compilers, and SWIG parsing use ordinary
declarations. No downstream compiler flag is required; rebuild the downstream
state extension with the updated runtime sources to acquire the optimization.

### Inferred borrowed director fields

NSB analyzes assignments in every selected override before compiling callbacks.
Assigning a borrowed callback object infers a borrowed field; assigning a
constructed object infers an owned field. Mixed ownership fails compilation.
The declared SWIG class must match the assigned type; use an explicit cast.
Always-inlined state-mutating helpers participate in analysis. Separate compiled
helpers that mutate state are rejected because their constraints are unavailable.

Const and mutable borrows together infer const, following C++ qualification
conversion. Const fields permit const methods and reject mutation. Python reads
of const borrowed fields cannot create a writable proxy and raise TypeError.
Borrowed fields reject Python object assignments, including initialization;
Python `None` clears them without changing their policy. Guard empty live reads
with `nsb.is_slot_occupied(self, "field")`; an unguarded read fails the callback.
Eager `stateaccess` caching requires occupied fields on callback entry.
Nullable assignments require `nsb.require_not_none` before storage.

Borrowed state neither deletes nor retains its pointee. Retain the true owner
separately and replace references after owner-controlled rebuilds before reuse.
NSB cannot detect external destruction or stale gradient addresses.

### Local block-view helpers

Both ordinary `@njit` and `@njit(inline="always")` helpers can return constructed
external-data wrappers to compiled callers. Check bounds before using
`Vector(nsb.as_cpointer(vector.GetDataArray()[start:end]), end-start)`:
require `0 <= start <= end <= vector.Size()`. Empty blocks are supported.
The wrapper owns its small C++ object and borrows the backing storage. Keep the
backing owner alive and views local to the call; these views must not enter
persistent state. Output writes reach the original storage without copying.
An API exposing mutable data through const objects requires callers to preserve
logical input constness.

Numba does not unwind owned locals when exceptions escape their allocating
frame. Validate every block before creating any local view, or catch failures
in each allocating frame and return normally to release its locals. Compiled
callers may receive owned helper results; Python escape still requires
`nsb.own_this` and a separately retained backing owner.
