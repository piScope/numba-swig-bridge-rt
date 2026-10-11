# MPI communicator example

`mpi_communicator.py` demonstrates mpi4py communicator arguments, compiled
`.rank`/`.size`, communicator-taking C++ constructors, native `GetComm()` returns
passed to another compiled helper, and `state={"comm": MPI.Comm}` in a director.

Install the runtime and optional sample 11 with the same MPI implementation
used by mpi4py. From the repository root, after generating the sample artifacts:

```sh
CC=mpicc CXX=mpicxx python -m pip install ./sample_libraries/11_mpi_communicator
mpiexec -n 2 python package/numba-swig-bridge-rt/usercode_examples/example11/mpi_communicator.py
```

When using the example installed with the runtime, locate it under
`share/numba-swig-bridge-rt/usercode_examples/example11/` in the installation prefix.
Running the example needs the public runtime, mpi4py and the installed sample;
it does not need SWIG or the generator.

The application keeps the underlying MPI communicator valid during native use.
NSB owns holder wrappers and retains Python-origin communicators; it does not
free or duplicate MPI communicators. Native-origin returns to Python are rejected,
but passing them between compiled helpers is supported. Close directors before
MPI finalization. MPI communication operations remain outside this example.

The example passed locally on two Open MPI ranks, including a fresh installed
consumer without SWIG or the generator. Virtual callback signatures carrying
MPI_Comm are deferred; communicator director state is supported.
