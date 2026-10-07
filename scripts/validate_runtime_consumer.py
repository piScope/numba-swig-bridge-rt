#!/usr/bin/env python3
"""Validate a runtime wheel with the frozen sample-1 consumer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_libraries" / "1_multiple_extensions"
EXAMPLES = (
    "ordinary_methods.py",
    "director_and_ordinary.py",
    "heterogeneous_directors.py",
    "director_ownership.py",
)
EXAMPLE_ROOT = ROOT / "package" / "numba-swig-bridge-rt" / "usercode_examples"
if not EXAMPLE_ROOT.is_dir():
    EXAMPLE_ROOT = ROOT / "usercode_examples"


def run(command, *, cwd, env, log):
    command = [str(value) for value in command]
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def python_for(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime_wheel = args.runtime_wheel.resolve()
    work = args.work_dir.resolve()
    if not runtime_wheel.is_file() or not runtime_wheel.name.startswith("numba_swig_bridge_rt-"):
        parser.error("--runtime-wheel must name a numba-swig-bridge-rt wheel")
    if work == ROOT or ROOT in work.parents:
        parser.error("--work-dir must be outside the repository")
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        parser.error("--work-dir must be empty")
    log = work / "validation.log"
    base_env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        base_env.pop(name, None)
    base_env["PYTHONNOUSERSITE"] = "1"

    build_env_dir = work / "build-env"
    consumer_env_dir = work / "consumer-env"
    venv.EnvBuilder(with_pip=True, clear=True).create(build_env_dir)
    venv.EnvBuilder(with_pip=True, clear=True).create(consumer_env_dir)
    build_python = python_for(build_env_dir)
    consumer_python = python_for(consumer_env_dir)
    wheels = work / "wheels"
    wheels.mkdir()
    run([build_python, "-m", "pip", "install", "--upgrade", "setuptools>=75", "wheel", "cmake", "swig"],
        cwd=work, env=base_env, log=log)
    run([build_python, "-m", "pip", "install", runtime_wheel],
        cwd=work, env=base_env, log=log)
    run([build_python, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps",
         SAMPLE, "--wheel-dir", wheels], cwd=work, env=base_env, log=log)
    sample_wheel = next(wheels.glob("numba_swig_bridge_sample1-*.whl"))

    run([consumer_python, "-m", "pip", "install", runtime_wheel, sample_wheel],
        cwd=work, env=base_env, log=log)
    run([consumer_python, "-m", "pip", "check"], cwd=work, env=base_env, log=log)
    execution = work / "examples"
    execution.mkdir()
    for name in EXAMPLES:
        shutil.copy2(EXAMPLE_ROOT / "example1" / name,
                     execution / name)

    isolated_env = dict(base_env)
    isolated_env["PATH"] = str(consumer_python.parent)
    check = execution / "consumer_check.py"
    check.write_text(
        "import importlib.util\n"
        "import shutil\n"
        "assert importlib.util.find_spec('numba_swig_bridge') is None\n"
        "assert shutil.which('swig') is None\n"
        "import nsb_sample1_state_bindings as state\n"
        "import nsb_sample1_registration\n"
        "from nsb_rt import state_backend\n"
        "assert state_backend('nsb_sample1').module is state\n"
        "print('clean runtime consumer checks passed')\n"
    )
    run([consumer_python, "-I", check], cwd=execution, env=isolated_env, log=log)
    for name in EXAMPLES:
        run([consumer_python, "-I", execution / name], cwd=execution, env=isolated_env, log=log)
    (work / "report.json").write_text(json.dumps({
        "status": "passed",
        "runtime_wheel": runtime_wheel.name,
        "sample_wheel": sample_wheel.name,
        "checks": ["clean consumer", "generator unavailable", "SWIG unavailable", "sample-1 examples"],
        "log": str(log),
    }, indent=2) + "\n")
    print(f"Runtime consumer validation passed: {work}")


if __name__ == "__main__":
    main()
