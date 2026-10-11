"""Build optional frozen samples and run their bundled examples.

Usage: python -m nsb_rt.samples list|install|run [sample numbers]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def resource_root() -> Path:
    """Locate resources in a source checkout or an installed runtime."""
    source = Path(__file__).resolve().parents[2]
    installed = Path(sysconfig.get_path("data")) / "share/numba-swig-bridge-rt"
    for root in (source, installed):
        if (root / "sample_libraries").is_dir() and (
            root / "usercode_examples"
        ).is_dir():
            return root
    raise FileNotFoundError(
        "runtime sample resources are missing; reinstall the runtime"
    )


def sample_paths(root: Path) -> dict[int, Path]:
    return {
        int(path.name.split("_", 1)[0]): path
        for path in (root / "sample_libraries").iterdir()
        if path.is_dir() and (path / "pyproject.toml").is_file()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "install", "run"))
    parser.add_argument(
        "samples",
        nargs="*",
        type=int,
        help="sample numbers; default is serial samples 1–10",
    )
    args = parser.parse_args(argv)
    root = resource_root()
    available = sample_paths(root)
    selected = args.samples or sorted(number for number in available if number != 11)
    unknown = set(selected) - available.keys()
    if unknown:
        parser.error(f"unknown samples: {sorted(unknown)}")
    if args.action == "list":
        for number, path in sorted(available.items()):
            print(
                f"{number:2}: {path.name}" + (" (MPI required)" if number == 11 else "")
            )
        print(f"Resources: {root}")
        return 0
    if args.action == "install":
        # The installed share directory may be read-only. Never build in it.
        with tempfile.TemporaryDirectory(prefix="nsb-samples-") as directory:
            projects = []
            for number in selected:
                destination = Path(directory) / available[number].name
                shutil.copytree(
                    available[number],
                    destination,
                    ignore=shutil.ignore_patterns(
                        "build", "dist", "*.egg-info", "__pycache__", "*.so"
                    ),
                )
                projects.append(str(destination))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    *projects,
                ],
                check=True,
            )
    else:
        for number in selected:
            examples = sorted(
                (root / "usercode_examples" / f"example{number}").glob("*.py")
            )
            if not examples:
                parser.error(f"sample {number} has no runnable examples")
            for example in examples:
                print(f"Running {example.name}", flush=True)
                subprocess.run([sys.executable, "-I", str(example)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
