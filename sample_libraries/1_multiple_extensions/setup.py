"""
Sample 1: Multiple Extensions.
Custom setuptools hooks for the sample library  and SWIG extensions.
"""


def _version_tuple(value):
    return tuple(int(part) for part in value.split("."))


import os
import json
from pathlib import Path
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "nsb_artifacts"


def generated_manifest(runtime_info: dict) -> dict:
    """Read reviewed bridge artifacts without importing or running the generator."""
    path = ARTIFACTS / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(
            "sample 1 has no generated NSB artifacts; run the bridge generator "
            "during development and commit sample_libraries/1_multiple_extensions/nsb_artifacts"
        ) from error
    if (
        manifest.get("schema_version") != 2
        or manifest.get("artifact_schema_version") != 1
    ):
        raise RuntimeError(
            "sample 1 generated manifest has an unsupported artifact schema"
        )
    if manifest.get("bridge_id") != "nsb_sample1":
        raise RuntimeError("sample 1 generated manifest has the wrong bridge identity")
    if manifest.get("state_backend") != "nsb_sample1":
        raise RuntimeError("sample 1 generated manifest has the wrong state backend")
    if manifest.get("state_module") != "nsb_sample1_state_bindings":
        raise RuntimeError("sample 1 generated manifest has the wrong state module")
    if _version_tuple(runtime_info["version"]) < _version_tuple(
        manifest.get("required_runtime_version")
    ):
        raise RuntimeError(
            "sample 1 generated artifacts require runtime "
            f"{manifest.get('required_runtime_version')!r}, but build uses {runtime_info['version']!r}"
        )
    if manifest.get("state_abi") != runtime_info["abi_version"]:
        raise RuntimeError("sample 1 generated artifacts require a different state ABI")
    return manifest


def artifact(manifest: dict, relative: str) -> Path:
    path = ARTIFACTS / relative
    if not path.is_file():
        raise RuntimeError(f"sample 1 generated artifact is missing: {relative}")
    return path


class EggInfo(egg_info):
    def find_sources(self):
        # Rebuild the generated manifest so earlier builds cannot contribute
        # obsolete absolute paths (including removed pip isolation directories).
        manifest = Path(self.egg_info) / "SOURCES.txt"
        manifest.unlink(missing_ok=True)
        super().find_sources()


class BuildExt(build_ext):
    def build_extensions(self):
        original_sources = {
            extension.name: list(extension.sources) for extension in self.extensions
        }
        # Normal wheel builds consume the public runtime build resources and
        # reviewed generated artifacts. They never import the NSB generator.
        from nsb_rt.build_api import get_build_info

        info = get_build_info()
        manifest = generated_manifest(info)
        temporary = Path(self.build_temp).resolve()
        cmake_build = temporary / "library"
        cmake = os.environ.get("CMAKE", "cmake")
        swig = os.environ.get("SWIG", "swig")
        # Adapt these CMake options and the archive name to your C++ library.
        subprocess.run(
            [
                cmake,
                "-S",
                str(ROOT),
                "-B",
                str(cmake_build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=" + str(cmake_build / "lib"),
            ],
            check=True,
        )
        subprocess.run(
            [
                cmake,
                "--build",
                str(cmake_build),
                "--config",
                "Release",
                "--parallel",
                str(self.parallel or 2),
            ],
            check=True,
        )
        archives = list((cmake_build / "lib").rglob("*nsb_sample1_core*"))
        archive = next(path for path in archives if path.suffix in {".a", ".lib"})
        proxies = temporary / "python"
        proxies.mkdir(parents=True, exist_ok=True)
        cpp_flags = (
            ["/std:c++17"] if self.compiler.compiler_type == "msvc" else ["-std=c++17"]
        )
        by_name = {extension.name: extension for extension in self.extensions}
        for module, specification in manifest["extensions"].items():
            include_dirs = [
                str(ROOT / value) for value in specification["include_dirs"]
            ]
            include_dirs += [info["include_dir"], info["cpp_dir"], str(ARTIFACTS)]
            wrapper = temporary / (module + "_wrap.cxx")
            subprocess.run(
                [
                    swig,
                    "-c++",
                    "-python",
                    *specification["swig_options"],
                    *["-D" + value for value in specification["wrapper_definitions"]],
                    *["-I" + value for value in include_dirs],
                    "-outdir",
                    str(proxies),
                    "-o",
                    str(wrapper),
                    str(ROOT / specification["interface"]),
                ],
                check=True,
            )
            extension = by_name["_" + module]
            extension.sources = [str(wrapper)]
            # Add your library's compiler include paths and link dependencies here.
            extension.include_dirs = [str(ROOT)]
            extension.extra_objects = [str(archive)]
            extension.extra_compile_args = cpp_flags
        # The downstream wheel owns this one StateBlock implementation.  Its
        # director helper sees only the opaque lease C ABI.
        state_wrapper = temporary / "nsb_sample1_state_bindings_wrap.cxx"
        state_include_dirs = [info["include_dir"], info["cpp_dir"]]
        subprocess.run(
            [
                swig,
                "-c++",
                "-python",
                "-module",
                "nsb_sample1_state_bindings",
                *["-I" + value for value in state_include_dirs],
                "-outdir",
                str(proxies),
                "-o",
                str(state_wrapper),
                info["state_interface"],
            ],
            check=True,
        )
        state_extension = by_name["_nsb_sample1_state_bindings"]
        state_extension.sources = [str(state_wrapper), *info["sources"]]
        state_extension.include_dirs = state_include_dirs
        state_extension.extra_compile_args = cpp_flags

        director = manifest["director"]
        # Keep this helper name consistent with director_helper_module in bridge.toml.
        wrapper = temporary / "nsb_sample1_director_bindings_wrap.cxx"
        native_include = artifact(manifest, director["headers"][0]).parent
        # The generated helper consumes the registered opaque state-lease C
        # ABI.  It deliberately neither includes nor compiles StateBlock.
        include_dirs = [str(native_include), str(ROOT), info["include_dir"]]
        subprocess.run(
            [
                swig,
                "-c++",
                "-python",
                *["-I" + value for value in include_dirs],
                "-outdir",
                str(proxies),
                "-o",
                str(wrapper),
                str(artifact(manifest, director["swig_interface"])),
            ],
            check=True,
        )
        extension = by_name["_" + director["python_module"]]
        extension.sources = [str(wrapper)] + [
            str(artifact(manifest, value)) for value in director["sources"]
        ]
        extension.include_dirs = include_dirs
        extension.extra_objects = [str(archive)]
        extension.extra_compile_args = cpp_flags
        # Generated/external compilation inputs must not enter source manifests.
        # Metadata describes the original distribution, not this build's sources.
        try:
            super().build_extensions()
        finally:
            for extension in self.extensions:
                extension.sources = original_sources[extension.name]
        self.python_outputs = [path for path in proxies.glob("*.py")] + [
            artifact(manifest, value) for value in manifest["python_files"]
        ]

    def run(self):
        super().run()
        destination = ROOT if self.inplace else Path(self.build_lib)
        destination.mkdir(parents=True, exist_ok=True)
        for path in self.python_outputs:
            self.copy_file(str(path), str(destination / path.name))


class BuildPy(build_py):
    def run(self):
        # Custom setuptools hooks coordinate native work before
        # the ordinary Python-file build, without a second setup.py.
        self.run_command("build_ext")
        super().run()

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        command = self.get_finalized_command("build_ext")
        for path in getattr(command, "python_outputs", []):
            outputs.append(str(Path(self.build_lib) / path.name))
        return outputs


setup(
    # Match the owning SWIG modules and director helper configured in bridge.toml.
    # sample1.cpp is a placeholder source; BuildExt replaces it with generated sources.
    ext_modules=[
        Extension("_" + module, sources=["sample1.cpp"], language="c++")
        for module in [
            "nsb_sample1_objects",
            "nsb_sample1_directors",
            "nsb_sample1_director_bindings",
            "nsb_sample1_state_bindings",
        ]
    ],
    cmdclass={"build_py": BuildPy, "build_ext": BuildExt, "egg_info": EggInfo},
)
