#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/mpich-probe.XXXXXX")
trap 'rm -rf "${probe_dir}"' EXIT
compiler=$(command -v mpicc.mpich)
launcher=$(command -v mpiexec.mpich)
printf 'Compiler: %s\nLauncher: %s\n' "${compiler}" "${launcher}"
readlink -f "${compiler}"
readlink -f "${launcher}"
"${compiler}" -show
ldd "${launcher}"
cat > "${probe_dir}/probe.c" <<'C'
#include <mpi.h>
#include <stdio.h>
#include <string.h>
extern char **environ;
int main(int argc, char **argv) {
    int rank, size, length;
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_library_version(version, &length);
    printf("C MPI probe: rank=%d size=%d\n%.*s\n", rank, size, length, version);
    for (char **entry = environ; *entry; ++entry) {
        if (!strncmp(*entry, "PMI", 3))
            printf("C rank=%d %s\n", rank, *entry);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
C
"${compiler}" "${probe_dir}/probe.c" -o "${probe_dir}/probe"
ldd "${probe_dir}/probe"
mkdir "${probe_dir}/mpi-tools"
for name in ssh rsh orted prted hydra_pmi_proxy; do
    if tool=$(command -v "${name}"); then
        ln -s "${tool}" "${probe_dir}/mpi-tools/${name}"
    fi
done
# Preserve the MPI environment while varying only the executable search path.
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
status=0
printf '\nNormal runner PATH\n'
timeout 60 "${launcher}" -n 2 "${probe_dir}/probe" || status=$?
printf 'Normal launch exit code: %s\n' "${status}"
status=0
printf '\nRestricted consumer PATH\n'
timeout 60 env PATH="${probe_dir}/mpi-tools" "${launcher}" -n 2 "${probe_dir}/probe" || status=$?
printf 'Restricted launch exit code: %s\n' "${status}"
