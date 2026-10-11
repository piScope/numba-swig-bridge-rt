# Sample 11: MPI communicator foundation

This sample demonstrates the locally completed NSBP_21 scope. It requires MPI and mpi4py only
when built or used. The shared runtime remains MPI-free on serial import.

The user-facing demonstration is
[example11/mpi_communicator.py](../../usercode_examples/example11/mpi_communicator.py).
This directory contains the C++ fixture, annotations, generated artifacts and build.

MPI-enabled generator configurations specify `mpi_module = "package._mpi"`.
Their manifest contains `mpi.sources` and `mpi.headers`; build the provider as
that Python extension with MPI compiler/link flags and mpi4py's include directory.
The generated registration module registers the provider automatically.

The initial runtime accepts mpi4py communicator arguments and lowers read-only
`comm.rank` and `comm.size` to native queries. The native wrapper owns a copy of
the handle and retains the Python communicator; it never frees the MPI handle.
Explicit communicator Free and MPI finalization invalidate borrowed handles.

Native communicator returns and communicator-taking ordinary adapters are now
generated. Native-origin returns to Python raise TypeError; Python-origin returns
preserve the original mpi4py object. Constructors copy the handle without an
automatic lifetime dependency. The application keeps the MPI resource valid.

Build from this directory after generation:

```sh
CC=mpicc CXX=mpicxx python setup.py build
```

Dedicated director state accepts `state={"comm": MPI.Comm}`. Python assignment
retains its communicator through an NRT-owned holder. Compiled replacement
retains the new holder and releases the previous one; closing state releases
the final holder. Python reads preserve Python-origin communicators and reject
native-origin ones. `None` clears a field from Python.

The ordinary, MPI provider, state provider and director extensions have compiled
locally with Open MPI. The example and installed state worker passed on two
ranks. Virtual callback signatures carrying MPI_Comm are deferred; communicator
state and ordinary methods/constructors are supported.
Library-version checks detect reported mismatches but do not prove MPI ABI
compatibility. A second MPI implementation remains deferred to NSBP_13 CI.
