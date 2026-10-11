#!/usr/bin/env python3
"""Validate distributed sample wheels in a clean installed runtime consumer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import venv

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env, log, timeout=1200):
    """Stream subprocess diagnostics to both the job output and its saved log."""
    command = [str(value) for value in command]
    with log.open("a") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        ) as process:

            def copy_output():
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)

            reader = threading.Thread(target=copy_output)
            reader.start()
            try:
                status = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            finally:
                reader.join()
            if status:
                raise subprocess.CalledProcessError(status, command)


def python_for(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_resources(python, work, env):
    return Path(
        subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "from nsb_rt.samples import resource_root; print(resource_root())",
            ],
            cwd=work,
            env=env,
            text=True,
        ).strip()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        help="sample numbers; default: serial samples 1–10",
    )
    parser.add_argument("--mpi-implementation", choices=("openmpi", "mpich"))
    parser.add_argument("--mpiexec", help="explicit MPI launcher matching the compiler")
    args = parser.parse_args()
    selected = sorted(set(args.samples or range(1, 11)))
    if any(number not in range(1, 12) for number in selected):
        parser.error("samples must be numbered 1–11")
    mpi_enabled = 11 in selected
    if mpi_enabled != bool(args.mpi_implementation) or mpi_enabled != bool(
        args.mpiexec
    ):
        parser.error("sample 11 requires --mpi-implementation and --mpiexec")
    launcher = shutil.which(args.mpiexec) if mpi_enabled else None
    if mpi_enabled and not launcher:
        parser.error("MPI launcher is unavailable")
    runtime_wheel = args.runtime_wheel.resolve()
    work = args.work_dir.resolve()
    if not runtime_wheel.is_file() or not runtime_wheel.name.startswith(
        "numba_swig_bridge_rt-"
    ):
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
    base_env.update(PYTHONNOUSERSITE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    build_env_dir, consumer_env_dir = work / "build-env", work / "consumer-env"
    for directory in (build_env_dir, consumer_env_dir):
        venv.EnvBuilder(with_pip=True, clear=True).create(directory)
    build_python, consumer_python = python_for(build_env_dir), python_for(
        consumer_env_dir
    )
    wheels = work / "wheels"
    wheels.mkdir()
    run(
        [
            build_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "setuptools>=75",
            "wheel",
            "cmake",
            "swig",
        ],
        cwd=work,
        env=base_env,
        log=log,
    )
    run(
        [build_python, "-m", "pip", "install", runtime_wheel],
        cwd=work,
        env=base_env,
        log=log,
    )
    mpi_wheels = []
    if mpi_enabled:
        compiler = base_env.get("MPI4PY_BUILD_MPICC") or base_env.get("MPICC")
        if not compiler or not shutil.which(compiler):
            parser.error("set MPI4PY_BUILD_MPICC to the selected MPI compiler")
        run(
            [
                build_python,
                "-m",
                "pip",
                "wheel",
                "mpi4py>=4",
                "--no-deps",
                "--no-binary=mpi4py",
                "--no-cache-dir",
                "--wheel-dir",
                wheels,
            ],
            cwd=work,
            env=base_env,
            log=log,
        )
        mpi_wheels = list(wheels.glob("mpi4py-*.whl"))
        assert len(mpi_wheels) == 1
        run(
            [build_python, "-m", "pip", "install", *mpi_wheels],
            cwd=work,
            env=base_env,
            log=log,
        )
    distributed = installed_resources(build_python, work, base_env)
    sample_sources = []
    for number in selected:
        source = next((distributed / "sample_libraries").glob(f"{number}_*"))
        destination = work / "frozen-samples" / source.name
        shutil.copytree(source, destination)
        sample_sources.append(destination)
    run(
        [
            build_python,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            *sample_sources,
            "--wheel-dir",
            wheels,
        ],
        cwd=work,
        env=base_env,
        log=log,
    )
    sample_wheels = sorted(
        path for path in wheels.glob("*.whl") if path not in mpi_wheels
    )
    assert len(sample_wheels) == len(selected)
    run(
        [
            consumer_python,
            "-m",
            "pip",
            "install",
            runtime_wheel,
            *sample_wheels,
            *mpi_wheels,
        ],
        cwd=work,
        env=base_env,
        log=log,
    )
    run([consumer_python, "-m", "pip", "check"], cwd=work, env=base_env, log=log)
    resources = installed_resources(consumer_python, work, base_env)
    execution = work / "examples"
    execution.mkdir()
    isolated_env = dict(base_env)
    isolated_env["PATH"] = str(consumer_python.parent)
    if mpi_enabled:
        # MPI launchers need their runtime helpers even for local jobs. Expose
        # only those tools, keeping SWIG and native build tools unavailable.
        mpi_tools = execution / "mpi-tools"
        mpi_tools.mkdir()
        for name in ("ssh", "rsh", "orted", "prted", "hydra_pmi_proxy"):
            tool = shutil.which(name, path=base_env.get("PATH"))
            if tool:
                (mpi_tools / name).symlink_to(tool)
        isolated_env["PATH"] += os.pathsep + str(mpi_tools)
    check = execution / "consumer_check.py"
    check.write_text(
        "import importlib.util, shutil, sys\n"
        "assert importlib.util.find_spec('numba_swig_bridge') is None\n"
        "assert shutil.which('swig') is None\n"
        "import nsb_rt\n"
        "assert 'mpi4py.MPI' not in sys.modules\n"
        + (
            "assert importlib.util.find_spec('mpi4py') is None\n"
            if not mpi_enabled
            else ""
        )
        + "print('clean runtime consumer checks passed')\n"
    )
    run(
        [consumer_python, "-I", check],
        cwd=execution,
        env=isolated_env,
        log=log,
        timeout=180,
    )
    mpi_version = None
    if mpi_enabled:
        mpi_check = execution / "mpi_check.py"
        mpi_command = [launcher]
        if args.mpi_implementation == "openmpi":
            mpi_command.append("--oversubscribe")
        mpi_command.extend(["-n", "2"])
        if args.mpi_implementation == "mpich":
            native_source = execution / "mpi_probe.c"
            native_probe = execution / "mpi_probe"
            native_source.write_text(
                '#include <mpi.h>\n#include <stdio.h>\n#include <string.h>\n'
                'extern char **environ;\n'
                'int main(int argc, char **argv) {\n'
                '  int rank, size, length;\n'
                '  char version[MPI_MAX_LIBRARY_VERSION_STRING];\n'
                '  MPI_Init(&argc, &argv);\n'
                '  MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n'
                '  MPI_Comm_size(MPI_COMM_WORLD, &size);\n'
                '  MPI_Get_library_version(version, &length);\n'
                '  printf("C MPI probe: rank=%d size=%d\\n%.*s\\n",\n'
                '         rank, size, length, version);\n'
                '  for (char **entry = environ; *entry; ++entry) {\n'
                '    if (!strncmp(*entry, "PMI", 3))\n'
                '      printf("C rank=%d %s\\n", rank, *entry);\n'
                '  }\n'
                '  fflush(stdout);\n'
                '  MPI_Finalize();\n'
                '  return 0;\n}\n'
            )
            run(
                [compiler, native_source, "-o", native_probe],
                cwd=execution,
                env=base_env,
                log=log,
            )
            run(
                [*mpi_command, native_probe],
                cwd=execution,
                env=isolated_env,
                log=log,
                timeout=180,
            )
        mpi_check.write_text(
            "import json, os\nfrom mpi4py import MPI\n"
            "version = MPI.Get_library_version()\n"
            "print(json.dumps({'probe': 'mpi4py', 'rank': MPI.COMM_WORLD.rank,\n"
            "    'size': MPI.COMM_WORLD.size, 'version': version,\n"
            "    'extension': MPI.__file__,\n"
            "    'pmi_environment': {k: v for k, v in os.environ.items()\n"
            "        if k.startswith('PMI')}}), flush=True)\n"
            f"assert {args.mpi_implementation!r} in version.lower().replace(' ', ''), version\n"
            "assert MPI.COMM_WORLD.size == 2\n"
            "if MPI.COMM_WORLD.rank == 0:\n"
            "    print(version)\n"
            "    with open('mpi-version.json', 'w') as stream:\n"
            "        json.dump({'version': version}, stream)\n"
        )
        run(
            [*mpi_command, consumer_python, "-I", mpi_check],
            cwd=execution,
            env=isolated_env,
            log=log,
            timeout=180,
        )
        mpi_version = json.loads((execution / "mpi-version.json").read_text())[
            "version"
        ]
    examples_run = []
    for number in selected:
        examples = sorted(
            (resources / "usercode_examples" / f"example{number}").glob("*.py")
        )
        assert examples, f"sample {number} has no distributed examples"
        for example in examples:
            target = execution / f"sample{number}_{example.name}"
            shutil.copy2(example, target)
            command = [consumer_python, "-I", target]
            if number == 11:
                command = [*mpi_command, *command]
            run(command, cwd=execution, env=isolated_env, log=log, timeout=180)
            examples_run.append(f"example{number}/{example.name}")
    (work / "report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "runtime_wheel": runtime_wheel.name,
                "sample_wheels": [path.name for path in sample_wheels],
                "samples": selected,
                "examples": examples_run,
                "mpi_implementation": args.mpi_implementation,
                "mpi_version": mpi_version,
                "mpi4py_wheels": [path.name for path in mpi_wheels],
                "checks": [
                    "clean consumer",
                    "generator unavailable",
                    "SWIG unavailable",
                ],
                "log": str(log),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Runtime consumer validation passed: {work}")


if __name__ == "__main__":
    main()
