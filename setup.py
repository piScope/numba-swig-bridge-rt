"""Ship optional sample sources without building native extensions."""

import json
from pathlib import Path

from setuptools import setup
from setuptools.command.egg_info import egg_info

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"build", "dist", "__pycache__", ".pytest_cache", ".git"}
SUFFIXES = {".py", ".md", ".sh", ".toml", ".json", ".i", ".cpp", ".hpp", ".txt", ".in"}
data_files = [
    ("share/numba-swig-bridge-rt", ["README.md", "LICENSE"]),
    ("share/numba-swig-bridge-rt/.github", [".github/README.md"]),
]
for tree in ("sample_libraries", "usercode_examples"):
    directories = {}
    for path in sorted((ROOT / tree).rglob("*")):
        relative = path.relative_to(ROOT)
        if any(
            part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if not path.is_file():
            continue
        if "_wrap." in path.name:
            continue
        if "nsb_artifacts" in relative.parts:
            artifact_root = ROOT.joinpath(
                *relative.parts[: relative.parts.index("nsb_artifacts") + 1]
            )
            manifest = json.loads((artifact_root / "manifest.json").read_text())
            if (
                path.relative_to(artifact_root).as_posix()
                not in manifest["generated_files"]
            ):
                continue
        if not path.is_file() or (
            path.suffix not in SUFFIXES and path.name != "LICENSE"
        ):
            continue
        destination = "share/numba-swig-bridge-rt/" + str(relative.parent)
        directories.setdefault(destination, []).append(str(relative))
    data_files.extend(directories.items())
ALLOWED_RESOURCES = {name for _, names in data_files for name in names}


class SourceManifest(egg_info):
    def find_sources(self):
        super().find_sources()
        self.filelist.files = [
            name
            for name in self.filelist.files
            if not name.startswith(("sample_libraries/", "usercode_examples/"))
            or name in ALLOWED_RESOURCES
        ]
        self.write_file(
            "filtered source manifest",
            str(Path(self.egg_info) / "SOURCES.txt"),
            "\n".join(self.filelist.files) + "\n",
        )


setup(data_files=data_files, cmdclass={"egg_info": SourceManifest})
