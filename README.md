# numba-swig-bridge runtime

Numba-Swig-Bridge (NSB) uses Numba to compile Python code that calls C++ methods
on SWIG objects. It also provides bridged director classes that implement C++
virtual methods as Numba callbacks. This guide describes how end users can
annotate their Python code to use NSB's bridge features.

The runtime is a support library intended to be installed as a dependency of
libraries that provide NSB bindings. Your C++ library must ship NSB bridge
registration and native adapters; installing the runtime alone does not add
bridges to an arbitrary SWIG library.

## Installation

```sh
python -m pip install numba-swig-bridge-rt
```

Applications use `import nsb_rt as nsb`. Import your library's bridge
registration before compiling functions that use its objects. Ordinary library
operations continue to use its normal Python API. Python 3.11 or newer and
Numba 0.68 or newer are required.

The runtime wheel is pure Python and includes SWIG annotations, native support
sources, optional sample sources, frozen bridge artifacts and Python examples.
Native samples are separate distributions and are built only when requested.
Using a downstream wheel requires neither SWIG nor the NSB generator.

## Contents

- [Simplest use case](#simplest-use-case)
- [Constructed objects and helpers](#passing-constructed-c-objects-between-compiled-functions)
- [Returning objects to Python](#returning-c-objects-to-python)
- [Bridged directors](#bridged-director)
- [Director state](#native-owned-director-state)
- [Virtual overrides](#compiled-virtual-overrides)
- [Inheritance and casts](#inheritance-and-casts)
- [C++ data members](#c-data-members)
- [Build and run the optional samples](#build-and-run-the-optional-samples)

## Simplest use case

Import a library's bridge registration, then pass a normal SWIG object to an
`@njit` function and call its C++ methods:

```python
from numba import njit
import nsb_sample1_objects as objects
import nsb_sample1_registration  # Register the library's types and methods.

@njit
def total(buffer):
    return buffer.Sum()

buffer = objects.Buffer(1)
buffer.Set(0, 3.0)
print(total(buffer))  # 3.0
```

Numba compiles `total` on its first call and calls the selected C++ `Sum`
method directly. The Python SWIG proxy continues to own `buffer`; the compiled
function borrows it for the call. This requires a library with generated NSB
bindings, such as sample 1 above.

## Passing constructed C++ objects between compiled functions

A C++ object constructed inside `@njit` can be passed to another compiled
function. That helper can also return a constructed object to its compiled
caller. For example, sample 8's Vector constructor creates a small C++ wrapper
that borrows an array's storage:

```python
import numpy as np
from numba import njit
import nsb_rt as nsb
import nsb_sample8 as sample
import nsb_sample8.registration

@njit
def block_view(vector, start, end):
    if start < 0 or end < start or end > vector.Size():
        raise ValueError("invalid block bounds")
    data = vector.GetDataArray()
    return sample.Vector(nsb.as_cpointer(data[start:end]), end - start)

@njit
def run(data):
    vector = sample.Vector(nsb.as_cpointer(data), data.size)
    view = block_view(vector, 1, 3)
    view.Add(10.0)
    return view.Sum()

data = np.arange(5.0)
print(run(data))  # 23.0
print(data)       # [0. 11. 12. 3. 4.]
```

Numba's NRT owns both C++ wrappers and releases them when `run` returns.
The NumPy array owns the backing storage; writes through the view reach that
array without copying. Keep its owner alive while using the view.

Plain `@njit` is sufficient. `inline="always"` optionally inserts the helper's
body into its caller; `inline="never"` explicitly keeps an ordinary helper
call. Neither setting is required for this ownership behavior. Passing an owned
wrapper here happens entirely between compiled functions. Passing a regular
SWIG proxy from Python uses a separate borrowed-argument specialization.

Check all requested block bounds before creating local views. If an exception
escapes a compiled frame that already holds owned wrappers, Numba may leave
those wrappers unreleased. Alternatively, catch the error in each allocating
frame and return normally so its wrappers are released.

## Returning C++ objects to Python

An ordinary selected C++ `T&` or `T*` return can cross an `@njit` boundary as
the normal non-owning SWIG proxy (`thisown=False`). Its C++ owner must remain
alive, exactly as for an ordinary SWIG borrowed return:

```python
@njit
def child(parent):
    return parent.GetChild()
```

When the returned proxy itself must retain a Python/SWIG parent, declare that
parent on the outer decorator:

```python
@nsb.borrowed_from(owner_arg="parent")
@njit
def retained_child(parent):
    return parent.GetChild()
```

The result remains non-owning, but NSB retains `parent` privately until the
child proxy is released. For a parent constructed inside compiled code, use the
expression form instead:

```python
@njit
def retained_native_child(values):
    parent = library.Parent(nsb.as_cpointer(values))
    return nsb.borrowed_from(parent.GetChild(), parent)
```

`nsb.own_this(obj)` transfers an NSB frame-owned C++ object into an owning SWIG
proxy (`thisown=True`). It clears the NRT destructor payload, so the proxy is
the sole C++ owner after the return. It transfers the C++ wrapper object only:
when the constructor borrows external NumPy storage, the caller must keep that
storage alive for the returned proxy's lifetime.

For the Vector example above, explicitly transfer the wrapper when returning
it to Python:

```python
@njit
def python_block(data, start, end):
    if start < 0 or end < start or end > data.size:
        raise ValueError("invalid block bounds")
    view = sample.Vector(nsb.as_cpointer(data[start:end]), end - start)
    return nsb.own_this(view)

data = np.arange(5.0)
view = python_block(data, 1, 3)
print(view.thisown)  # True: the Python proxy owns the C++ wrapper
view.Add(10.0)
print(data)          # [0. 11. 12. 3. 4.]
del view            # release the wrapper before its backing array
```

Returning `view` directly from this function is rejected because it would
implicitly transfer a frame-owned object to Python. `nsb.own_this` makes that
transfer explicit; it does not retain or take ownership of `data`. No explicit
`inline` setting is needed for this function either.

## Bridged-Director

A C++ director lets a C++ virtual call reach an implementation defined in
Python. For example, a C++ algorithm accepting an `Evaluator&` can call its
virtual `Evaluate(double)` method, with a Python subclass providing the
implementation. A usual SWIG director enters Python for that call.

An NSB bridged director instead dispatches selected virtual calls to
Numba-compiled callbacks. NSB generates a native C++ subclass, connects its
virtual methods to those callbacks, and provides a Python facade for setup
and lifetime management. C++ callers keep using the original base-class API.

For a library exposing a bridged `Evaluator` base, a typical definition is:

```python
import nsb_rt as nsb
from numba import types
import library
import library.nsb_registration

@nsb.director(state={"offset": types.float64})
class OffsetEvaluator(library.Evaluator):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    @nsb.override
    def Evaluate(self, value):
        return value + self.offset

evaluator = OffsetEvaluator(0.5)
try:
    # Pass evaluator to the library's ordinary SWIG API as an Evaluator.
    ...
finally:
    evaluator.close()
```

The names above are illustrative; the library must ship generated bridge
registration for its base class. `@nsb.director` compiles the marked overrides,
so they do not need an additional `@njit`. Initialization and other Python
setup methods remain ordinary Python code. An override must accept every
parameter of the actual C++ virtual declaration, including inherited methods.
Python default arguments in overrides are rejected; native callers supply
the C++ defaults before callback dispatch.

### Accessible data members

Inside a compiled override, bridged C++ objects passed as arguments or stored
in director state support attribute access to their exposed **public native
instance members**. Supported member types include scalars, enums, and pointers
to bridged classes. For example, a bridged integration point can expose
`ip.x`, while a bridged transformation can expose `transformation.ElementNo`.

Mutable members of a mutable object can be written. Const objects, const
members, and members marked read-only can only be read. Protected and private
members are not exposed, even though a native C++ subclass could access its
protected members. Static members, bit-fields, and excluded or unsupported
members are also unavailable. A public bridged getter or setter can provide
access where the C++ API offers one.

The compiled override's `self` is a typed view of its declared director state,
not unrestricted access to the native subclass or Python facade. Thus
`self.offset` above is a state field; native base-class members and arbitrary
Python attributes or `@property` getters do not automatically become accessible
through `self`. The following sections describe how that state is stored and
accessed.

### Python subclasses of a compiled director

An ordinary Python subclass can customize initialization and add Python methods
to an already decorated director:

```python
class FixedOffsetEvaluator(OffsetEvaluator):
    def __init__(self):
        super().__init__(2.0)
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

## Native-owned director state

A director object or array field may start empty and be initialized by a Numba
callback. Test it without reading the field, then assign an NSB-selected
constructor result or a Numba-created array:

```python
@nsb.override("GetChild")
def get_child(self):
    if not nsb.is_slot_occupied(self, "parent"):
        self.parent = library.Parent(41)
    return self.parent.GetChild()
```

The `StateBlock` takes the assigned direct expression's NRT ownership.
Replacing the field releases its prior native value, and closing the director releases any
value that remains. Borrowed values instead use an inferred borrowed field
policy, described in [inferred borrowed fields](#inferred-borrowed-director-fields). References or pointers returned from a state-held value become
invalid if that field is later replaced, so replacement must occur only after
C++ has stopped using earlier results.

## Callback-local director state

An override can declare the state fields it uses so NSB resolves repeated
reads once at callback entry:

```python
@nsb.override("Accumulate", "double", stateaccess="values, scale")
def accumulate(self, value):
    self.values[0] += self.scale * value
    return self.values[0] + self.values[0]
```

The declaration is checked while `@nsb.director` compiles the class. Reading,
assigning, or passing an unlisted field to `nsb.is_slot_occupied` is a typing
error. `stateaccess="*"` authorizes all declared state but caches only fields
used by that override. `stateaccess=""` authorizes no state. Omitting the
keyword preserves unrestricted live access.

Cached values are refreshed for every callback. Scalars are snapshots for that
invocation; cached objects and arrays still refer to their shared native object
or storage. A field assigned by the override stays on the live write-through
path, so lazy initialization and a later read in the same callback see the new
value.

## Ordinary methods and template arrays

Create objects through their normal SWIG Python API, then pass them into
compiled functions. Importing the generated registration module installs the
library's bridge types and method adapters:

```python
from numba import njit
from nsb_sample2 import IntArray
import nsb_sample2.registration

@njit
def update(array):
    array[1] = 3
    return array[1] + array.Size()

array = IntArray(4)
assert update(array) == 7
assert array[1] == 3
```

The sample maps value-based C++ `Get`/`Set` methods to SWIG's `__getitem__` and
`__setitem__`. It does not expose the reference returned by C++ `operator[]`
through the C ABI. Integer and double template instantiations remain distinct
bridge types. Indexing currently supports scalar indices, with negative and
out-of-range accesses rejected; slices are unsupported.

C++ `float` uses Numba `types.float32`; `double` uses `types.float64`.
Python floats normally select double precision. Use `numpy.float32` when
selecting a float overload explicitly. See the
[float32 example](usercode_examples/example4/float32.py).

## Native array pointers

`nsb.as_cpointer(array)` is available only inside `@njit` code. It returns a
typed native pointer to the first element of a writable contiguous NumPy
array. Use it only for an immediate call to a native API that documents that it
does not retain the pointer:

```python
import nsb_rt as nsb

@njit
def first_value(values):
    pointer = nsb.as_cpointer(values)
    return pointer[0]
```

The pointer is invalid once its source array is released and cannot be returned
to Python.

## Compiled virtual overrides

Subclass the library's registered SWIG base and mark the methods to compile
with `@nsb.override`. The registration import lets `@nsb.director` find the
binding automatically:

```python
import nsb_rt as nsb
from numba import types
import nsb_sample1_directors as callbacks
import nsb_sample1_registration

@nsb.director(state={"offset": types.float64})
class OffsetInner(callbacks.Inner):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    @nsb.override("Apply", "double")
    def apply(self, value):
        return value + self.offset

instance = OffsetInner(0.5)
try:
    assert callbacks.invoke_inner(instance, 2.0) == 2.5
finally:
    instance.close()
```

Use the C++ method name in the selector. For overloaded methods, supply the
parameter types to select the signature, as above. The Python implementation
must accept every C++ parameter and must not declare default arguments. For
example, C++ `AddMult(x, y, a = 1.0)` requires an override accepting `x, y, a`;
ordinary bridged calls can still use `obj.AddMult(x, y)`.

### Methods you do not override

Selected pure virtual methods require an implementation. For selected
non-pure methods, `fallback` controls use of the inherited C++ implementation:

- `"warn"` (default): use it and emit `NumbaSwigBridgeFallbackWarning`.
- `"silent"`: use it without a warning.
- `"error"`: reject the director class if an implementation is missing.

Methods outside the library's selected bridge retain their C++ behavior.
If a compiled callback fails, the outer library call reports a callback error.
The original Numba exception message and traceback are not currently preserved.

### When C++ takes ownership

A new director owns its C++ object (`thisown=True`). If a library API takes
ownership and will delete that object, release Python ownership **after the
ownership-taking call succeeds**:

```python
owner.Take(instance)
instance.thisown = False
instance.close()  # Detach Python; the native owner can still call the overrides.
```

This requires a C++ base with a virtual destructor. Ownership cannot be taken
back after release. Follow the library's lifetime contract, and ensure native
owners stop calling overrides before Python interpreter shutdown.
See the [ownership example](usercode_examples/example1/director_ownership.py).

## Inheritance and casts

Artifact generation recursively checks public base declarations. Every
intermediate class on a path connecting two selected bridge classes must also
be selected. For example, selecting `Derived` and `Base` in
`Derived -> Middle -> Base` requires selecting `Middle`; generation reports
the incomplete path and missing classes. Selection is checked across all
configured interfaces, including imported base declarations. The exposed
hierarchy may stop before unselected ancestors, and unselected descendants do
not require additional bridge classes. Abstract bases expose supported methods
and conversions without ordinary constructors.

Inherited methods and base-typed arguments work automatically along a public,
non-virtual single-inheritance chain. The inventory records direct bases;
Numba finds the inherited method during compilation without duplicating base
methods into derived class records.

Use explicit casts for supported conversions outside that automatic scope:

```python
import nsb_rt as nsb
from numba import njit
from nsb_sample3 import Base, Derived
import nsb_sample3.registration

@njit
def use_derived(obj):
    inherited = obj.BaseMethod()
    base_view = nsb.upcast(obj, Base)
    derived_view = nsb.dynamic_cast(base_view, Derived)
    return inherited + derived_view.DerivedMethod()
```

C++ helpers perform the pointer adjustments, including multiple and virtual
inheritance. Ambiguous or inaccessible upcasts are rejected during compilation.
Checked downcasts require a polymorphic source class and raise `TypeError` if
the object has the wrong dynamic type. Both classes must belong to the same
bridge. The views are borrowed: the original owner must remain alive.

A derived method declaration hides the base overload set. Combining overloads
through `using Base::Method` is not supported yet. Sibling crosscasts, unchecked
static downcasts, and returning cast views to Python are also deferred.

## C++ data members

Selected bridge classes automatically expose supported public instance members.
Scalars (`float`, `double`, `int`, `std::int64_t`, `bool`) and enums can be read
and assigned in compiled code when C++ and SWIG allow it. Const members and
`%immutable` are read-only; `%rename` and `%ignore` are respected.

Public pointers to other bridged classes are automatically readable as borrowed
views or `None`. `const T*` yields a const view; `T* const` still yields a mutable
pointee. Pointer-slot assignment is unsupported. Reading does not transfer
ownership or infer that the containing object owns the pointee; retain its real
owner, using `nsb.borrowed_from` where an explicit lifetime anchor is needed.

Optional `NUMBA_SWIG_BRIDGE_MEMBER(Class::field, "readonly")` and `"exclude"`
annotations restrict exposure before SWIG parses the declaration. Unsupported
members are recorded in `skipped_members`. Inherited member lookup currently
requires an explicit upcast to the bridged declaring base. Const member-enabled
objects and const pointee views reject non-const methods, mutable arguments,
member assignments, and writable Python proxy escape. Legacy classes without
exposed members retain their prior qualification behavior unless reached through
a const object-pointer member.

Const callback arguments and inferred borrowed director fields always preserve
qualification, including classes without selected data members. See the
[runtime guide](#inferred-borrowed-director-fields)
for ownership inference, ABI-7 rebuilding, and validated local block-view helpers.

See [sample 10](sample_libraries/10_data_members/README.md) and its
[compiled example](usercode_examples/example10/data_members.py).

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

## Build and run the optional samples

Sources and frozen artifacts ship in both runtime wheels and source
distributions. Sample builds use those artifacts without importing the
NSB generator. They still require a C++17 compiler, CMake, SWIG 4.5 or newer,
and Python build tools. Install these tools explicitly:

```sh
python -m pip install 'setuptools>=75' wheel cmake 'swig>=4.5'
python -m nsb_rt.samples list
python -m nsb_rt.samples install 1
python -m nsb_rt.samples run 1
```

Omitting sample numbers installs or runs serial samples 1–10. Select multiple
samples with `install 1 2 8` or `run 1 2 8`. The command uses the current Python
interpreter, builds in a temporary writable copy, and streams build output.
It does not modify the distributed source tree. Build isolation is disabled
so the sample build uses your installed runtime and build tools.

`list` prints the resource location. Installed sources and examples live under
`share/numba-swig-bridge-rt/` in the installation prefix; a source checkout has
`sample_libraries/` and `usercode_examples/` beside this README. From a runtime
source distribution, first install the runtime with `python -m pip install .`,
then use the same sample commands.

Sample 11 requires MPI, mpi4py and MPI compiler wrappers matched to the same
MPI implementation. It is excluded from the default selection:

```sh
python -m pip install mpi4py
CC=mpicc CXX=mpicxx python -m nsb_rt.samples install 11
mpiexec -n 2 python -m nsb_rt.samples run 11
```

See the [example index](usercode_examples/README.md) and
[sample library sources](sample_libraries/README.md) for individual examples.
MPI communication stays in Python or the native library. NSB supports
communicator inputs, compiled `.rank`/`.size`, native communicator results
passed between compiled helpers, and communicator director state. Boxing
native-origin communicators back to Python and virtual callback signatures
carrying MPI_Comm are not supported. Keep borrowed MPI resources valid until
native use ends, and close directors before MPI finalization.

## Downstream build compatibility

Downstream state extensions compile the headers and sources supplied by the
runtime. Release 0.15.0 uses state ABI 7; MPI support requires runtime 0.15.1.
Runtime and generated artifacts must satisfy the compatibility minimum and ABI
recorded in each artifact manifest. Regenerate and rebuild downstream state
extensions when their ABI changes.

Private StateBlock/StateHandle classes use hidden visibility on supported
non-Windows GCC/Clang toolchains so internal state reads can bind locally and
inline. Windows, unsupported compilers and SWIG parsing use ordinary
declarations. No downstream compiler flag is required.

