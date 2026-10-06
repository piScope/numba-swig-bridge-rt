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

## Examples

The [`usercode_examples/`](usercode_examples/README.md) directory contains
end-user examples using the sample downstream packages.  They all import the
public runtime as `nsb_rt`; none needs the internal generator.

## Wrapper authors

SWIG annotations, generation, and frozen-artifact maintenance are wrapper
development tasks, not runtime-user tasks.  Follow the internal
`numba-swig-bridge` generator documentation for that workflow.  A released
downstream wheel contains its generated artifacts; an end user needs only that
wheel and this runtime, and does not need SWIG or the generator.
