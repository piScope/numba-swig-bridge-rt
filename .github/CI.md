# Runtime continuous integration

## Pull requests and branch pushes

`test.yml` validates the installed runtime and its bundled frozen sample sources:

| Jobs | Python | Sample coverage |
| --- | --- | --- |
| Serial | 3.11–3.14 | Samples 1–10, all 17 example scripts |
| Open MPI | 3.13 | Samples 1–11; sample 11 on two ranks |
| MPICH | 3.13 | Samples 1–11; sample 11 on two ranks |

The MPI jobs use separate Ubuntu 24.04 runners. Each installs its implementation
and selects named compiler/launcher wrappers. mpi4py is built from source with
that MPI compiler, and the loaded library version is checked before examples
run. Serial jobs do not install MPI or mpi4py.

Every job builds native sample wheels from the sources and frozen artifacts
installed with the runtime wheel. It installs those wheels together in a fresh
consumer without SWIG or the generator. Build tools are confined to the build
environment. Each example has a 180-second timeout; MPI execution uses an
absolute matching launcher. Logs and JSON reports are uploaded on failure as
well as success.

## Releases

`build-release.yml` checks all serial samples from the runtime wheel and from
a wheel rebuilt from its source distribution. Separate Open MPI and MPICH jobs
then check all eleven samples against the release wheel. Publication requires
all three jobs (including both MPI matrix entries) to succeed. Existing release
and TestPyPI triggers and secrets are unchanged.

## Local validation

From the runtime repository, build a wheel and validate all serial samples:

```sh
python -m build --wheel --outdir dist .
python scripts/validate_runtime_consumer.py \
  --runtime-wheel dist/numba_swig_bridge_rt-*.whl \
  --work-dir /tmp/nsb-serial-consumer
```

For Open MPI, select its compilers and include sample 11:

```sh
CC=mpicc.openmpi CXX=mpicxx.openmpi MPI4PY_BUILD_MPICC=mpicc.openmpi \
python scripts/validate_runtime_consumer.py \
  --runtime-wheel dist/numba_swig_bridge_rt-*.whl \
  --work-dir /tmp/nsb-openmpi-consumer \
  --samples 1 2 3 4 5 6 7 8 9 10 11 \
  --mpi-implementation openmpi --mpiexec mpiexec.openmpi
```

Use the `.mpich` wrappers and `--mpi-implementation mpich` for MPICH. Work
directories must be empty and outside the repository. Reports record the
sample wheels, every example run, and the loaded MPI implementation/version.
