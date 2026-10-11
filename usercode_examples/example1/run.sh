#!/usr/bin/env bash
set -euo pipefail
example_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for example in ordinary_methods director_and_ordinary heterogeneous_directors director_ownership; do
    "${NSB_PYTHON:-python}" "${example_dir}/${example}.py"
done
