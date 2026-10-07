"""
Sample 3: C++ class inheritance.
Custom setuptools hooks for the sample library and its SWIG extension.
"""


def _version_tuple(value):
    return tuple(int(part) for part in value.split("."))

import os
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def artifact_path(root, relative):
    return root / "nsb_artifacts" / relative


def load_manifest(root, runtime_info):
    import json
    artifacts = root / "nsb_artifacts"
    manifest = json.loads((artifacts / "manifest.json").read_text())
    if _version_tuple(runtime_info["version"]) < _version_tuple(manifest.get("required_runtime_version")):
        raise RuntimeError(
            f"{root.name} artifacts require runtime "
            f"{manifest.get('required_runtime_version')!r}, "
            f"but build uses {runtime_info['version']!r}"
        )
    if manifest.get("state_abi") != runtime_info["abi_version"]:
        raise RuntimeError(f"{root.name} artifacts require a different state ABI")
    for relative in manifest.get("generated_files", []):
        if not (artifacts / relative).is_file():
            raise RuntimeError(f"{root.name} generated artifact is missing: {relative}")
    return manifest


class EggInfo(egg_info):
    def find_sources(self):
        # Rebuild the generated manifest so earlier builds cannot contribute
        # obsolete absolute paths (including removed pip isolation directories).
        manifest = Path(self.egg_info) / "SOURCES.txt"
        manifest.unlink(missing_ok=True)
        super().find_sources()


class BuildExt(build_ext):
    def build_extensions(self):
        original_sources = {extension.name: list(extension.sources)
                            for extension in self.extensions}
        # Locate the installed framework's SWIG include files and generator.
        from nsb_rt.build_info import get_build_info

        info = get_build_info()
        temporary = Path(self.build_temp).resolve()
        cmake_build = temporary / "library"
        cmake = os.environ.get("CMAKE", "cmake")
        swig = os.environ.get("SWIG", "swig")
        # Adapt these CMake options and the archive name to your C++ library.
        subprocess.run([
            cmake, "-S", str(ROOT), "-B", str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=" + str(cmake_build / "lib"),
        ], check=True)
        subprocess.run([cmake, "--build", str(cmake_build), "--config", "Release",
                        "--parallel", str(self.parallel or 2)], check=True)
        archives = list((cmake_build / "lib").rglob("*nsb_sample3_core*"))
        archive = next(path for path in archives if path.suffix in {".a", ".lib"})
        # Set your interfaces, module names, and SWIG options in bridge.toml.
        manifest = load_manifest(ROOT, info)
        self.manifest = manifest
        proxies = temporary / "python"
        proxies.mkdir(parents=True, exist_ok=True)
        cpp_flags = (["/std:c++17"] if self.compiler.compiler_type == "msvc"
                     else ["-std=c++17"])
        by_name = {extension.name: extension for extension in self.extensions}
        for module, specification in manifest["extensions"].items():
            include_dirs = [str(ROOT / value) for value in specification["include_dirs"]]
            include_dirs += [info["include_dir"], info["cpp_dir"], str(ROOT / "nsb_artifacts")]
            wrapper = temporary / (module + "_wrap.cxx")
            subprocess.run([
                swig, "-c++", "-python", *specification["swig_options"],
                *["-D" + value for value in specification["wrapper_definitions"]],
                *["-I" + value for value in include_dirs],
                "-outdir", str(proxies), "-o", str(wrapper),
                str(ROOT / specification["interface"]),
            ], check=True)
            extension = by_name["_" + module]
            extension.sources = [str(wrapper)]
            # Add your library's compiler include paths and link dependencies here.
            extension.include_dirs = [str(ROOT)]
            extension.extra_objects = [str(archive)]
            extension.extra_compile_args = cpp_flags
        # Generated/external compilation inputs must not enter source manifests.
        # Metadata describes the original distribution, not this build's sources.
        try:
            super().build_extensions()
        finally:
            for extension in self.extensions:
                extension.sources = original_sources[extension.name]
        self.python_outputs = [(path, Path(path.name)) for path in proxies.glob("*.py")]
        self.python_outputs += [(artifact_path(ROOT, value), Path(value).relative_to("python"))
                                for value in manifest["python_files"]]

    def run(self):
        super().run()
        destination = ROOT if self.inplace else Path(self.build_lib)
        destination.mkdir(parents=True, exist_ok=True)
        for path, relative in self.python_outputs:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(path), str(target))


class BuildPy(build_py):
    def run(self):
        # Custom setuptools hooks coordinate native work before
        # the ordinary Python-file build, without a second setup.py.
        self.run_command("build_ext")
        super().run()

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        command = self.get_finalized_command("build_ext")
        for path, relative in getattr(command, "python_outputs", []):
            outputs.append(str(Path(self.build_lib) / relative))
        return outputs


setup(
    # Match the owning SWIG module configured in bridge.toml.
    # inheritance.cpp is a placeholder source; BuildExt replaces it with generated sources.
    ext_modules=[Extension("_" + module, sources=["inheritance.cpp"], language="c++")
                 for module in ["nsb_sample3_inheritance"]],
    cmdclass={"build_py": BuildPy, "build_ext": BuildExt, "egg_info": EggInfo},
)
