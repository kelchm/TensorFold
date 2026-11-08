"""Export the committed native harness source identity without importing GPU libraries."""
import argparse
import hashlib
import subprocess
from pathlib import Path

from .common import canonical_bytes, require
from .fixture import NATIVE_BASE


def manifest(repo):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    paths = ["zig", "build.zig", "build.zig.zon", ".zig-version", "tools/perf/glm_replay"]
    require(not git("ls-files", "--others", "--exclude-standard", "-z", "--", *paths),
            "untracked native harness sources must be committed before exporting identities")
    require(subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=repo).returncode == 0,
            "commit the native harness sources before exporting identities")
    git("merge-base", "--is-ancestor", NATIVE_BASE, "HEAD")
    names = sorted(git("ls-files", "-z", "--", *paths).decode().split("\0"))
    files = []
    for name in names:
        if not name:
            continue
        path = repo / name
        require(path.is_file() and not path.is_symlink(), "source identity requires regular committed files")
        raw = path.read_bytes()
        files.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"schema": "glm-replay-native-source-v1", "base_commit": NATIVE_BASE,
            "harness_commit": git("rev-parse", "HEAD").decode().strip(), "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = manifest(args.repo)
    with args.output.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
