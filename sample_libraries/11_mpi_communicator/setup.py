"""Build with CXX=mpicxx CC=mpicc and installed NSB runtime/mpi4py."""

import json
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
import mpi4py
from nsb_rt.resources import include_dir, cpp_dir

root = Path(".")
artifacts = root / "nsb_artifacts"
manifest = json.loads((artifacts / "manifest.json").read_text())
mpi = manifest["mpi"]
directors = manifest["director"]
native_includes = [
    ".",
    str(cpp_dir()),
    str(artifacts / "director/cpp"),
    str(artifacts / "mpi/cpp"),
]
swig_options = [
    "-c++",
    "-I" + str(include_dir()),
    "-I" + str(cpp_dir()),
    "-I.",
    "-outdir",
    str(artifacts / "python"),
]


class BuildPy(build_py):
    def run(self):
        self.run_command("build_ext")
        super().run()
        for relative in manifest["python_files"]:
            source = artifacts / relative
            target = Path(self.build_lib) / Path(relative).relative_to("python")
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(source), str(target))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / Path(relative).relative_to("python"))
            for relative in manifest["python_files"]
        ]


class BuildExt(build_ext):
    def run(self):
        extension = next(e for e in self.extensions if e.name == "_nsb_sample11_state")
        original = list(extension.sources)
        try:
            extension.sources.append(str(cpp_dir() / "nsb_director_state.cpp"))
            super().run()
        finally:
            extension.sources = original


setup(
    name="nsb-sample11",
    version="0.1.0",
    packages=["nsb_sample11"],
    py_modules=[
        "nsb_sample11_api",
        "nsb_sample11_state",
        "nsb_sample11_director_bindings",
    ],
    package_dir={"": str(artifacts / "python"), "nsb_sample11": "nsb_sample11"},
    cmdclass={"build_py": BuildPy, "build_ext": BuildExt},
    ext_modules=[
        Extension(
            "_nsb_sample11_api",
            ["sample.i"],
            language="c++",
            include_dirs=[".", mpi4py.get_include(), str(artifacts / "mpi/cpp")],
            swig_opts=[
                "-c++",
                "-DNUMBA_SWIG_BRIDGE_GENERATED",
                "-I" + str(include_dir()),
                "-I" + str(artifacts),
                "-outdir",
                str(artifacts / "python"),
            ],
        ),
        Extension(
            mpi["python_module"],
            [str(artifacts / p) for p in mpi["sources"]],
            language="c++",
            include_dirs=[mpi4py.get_include(), str(artifacts / "mpi/cpp")],
        ),
        Extension(
            "_nsb_sample11_state",
            ["state.i"],
            language="c++",
            include_dirs=native_includes,
            swig_opts=swig_options,
        ),
        Extension(
            "_nsb_sample11_director_bindings",
            [
                str(artifacts / directors["swig_interface"]),
                *[str(artifacts / p) for p in directors["sources"]],
            ],
            language="c++",
            include_dirs=native_includes,
            swig_opts=swig_options,
        ),
    ],
)
