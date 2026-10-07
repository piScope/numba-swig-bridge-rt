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
