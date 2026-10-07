#!/usr/bin/env bash
set -euo pipefail
repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
"${NSB_PYTHON:-python}" "${repo_root}/package/numba-swig-bridge-rt/usercode_examples/example1/ordinary_methods.py"
"${NSB_PYTHON:-python}" "${repo_root}/package/numba-swig-bridge-rt/usercode_examples/example1/director_and_ordinary.py"
"${NSB_PYTHON:-python}" "${repo_root}/package/numba-swig-bridge-rt/usercode_examples/example1/heterogeneous_directors.py"
"${NSB_PYTHON:-python}" "${repo_root}/package/numba-swig-bridge-rt/usercode_examples/example1/director_ownership.py"
