"""Host-only identity checks and a closed import boundary for the frozen Python engine."""
import importlib.abc
import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from .common import digest, fields, file_digest, hex_id, integer, read_json, require
from .fixture import DRAFTER_REVISION, PYTHON_BASE, TARGET_REVISION, load_fixture

KIT = "bf5f94523a8251029cedae86bf61f8ce37fac218"
DEVICE = "sha256:b925352f3515caddc1dde8abf33a621afad8133d2964a091184ad4842bfff7b6"
CORE_GRAPHS = {1, 4, 8, 16, 32}
SOURCE_MANIFEST = Path(__file__).with_name("python_sources.json")
REQUIRED_ENV = {"TF_GLM_MULTI_LONE": "0", "TF_GLM_MULTI_VERIFY": "batched",
                "TF_GLM_MULTI_GRAPHS": "1", "TF_GLM_MULTI_DRAFT_GRAPHS": "1",
                "TF_GLM_MULTI_SAMPLER": "streams", "TF_GLM_MULTI_WINDOW": "32",
                "TF_GLM_KV_SHARD": "0", "TF_GLM_LATENT": "1", "TF_GLM_KV": "fp8",
                "TF_GLM_PROFILE": "0", "TF_GLM_MULTI_PROFILE": "0"}
HASHES = "fixture_sha256 config_sha256 schedule_sha256 tokens_sha256"


def relative_file(root, name):
    root = root.resolve()
    require(isinstance(name, str) and name, "unsafe artifact path")
    path = PurePosixPath(name)
    require(isinstance(name, str) and str(path) == name and not path.is_absolute()
            and ".." not in path.parts and "\\" not in name, "unsafe artifact path")
    candidate = root / name
    require(candidate.is_file() and not any(p.is_symlink() for p in (candidate, *candidate.parents)),
            f"missing file or symlink: {name}")
    require(candidate.resolve().is_relative_to(root.resolve()), "artifact escaped root")
    return candidate


def verify_files(root, assets, exact=False):
    require(isinstance(assets, list) and assets, "empty file manifest")
    names = set()
    for asset in assets:
        require(set(asset) in ({"path", "bytes", "sha256"}, {"path", "bytes", "sha256", "qualification"}),
                "invalid file identity fields")
        name = asset["path"]
        require(name not in names, "duplicate file identity")
        names.add(name)
        path = relative_file(root, name)
        integer(asset["bytes"], "file bytes")
        hex_id(asset["sha256"], "file hash")
        require(path.stat().st_size == asset["bytes"] and file_digest(path) == asset["sha256"],
                f"source/artifact identity mismatch: {name}")
    if exact:
        require(not any(p.is_symlink() for p in root.rglob("*")), "source/asset tree contains a symlink")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
        require(actual == names, "source tree contains missing or unpinned files")
    return digest(assets)


def verify_sources(root):
    manifest = read_json(SOURCE_MANIFEST)
    require(manifest["engine_commit"] == PYTHON_BASE and manifest["kit_commit"] == KIT
            and manifest["device_image"] == DEVICE, "source manifest identity mismatch")
    verify_files(root, manifest["files"], exact=True)
    return manifest


def graph_rows(fixture):
    return sorted(CORE_GRAPHS | {sum(p["offered"] for p in f["parts"])
                                 for phase in ("preparation", "forwards") for f in fixture[phase]})


def environment(env=None):
    env = os.environ if env is None else env
    for key, expected in REQUIRED_ENV.items():
        require(env.get(key) == expected, f"{key} must explicitly equal {expected}")
    require(env.get("TF_GLM_COMM", "nccl") in ("nccl", "roce"), "unsupported production transport")
    require(not env.get("TF_GLM_DRAFT_DUMP"), "draft dump must be disabled")
    return {k: v for k, v in sorted(env.items())
            if k.startswith(("TF_", "TENSORFOLD_", "CUDA_", "NCCL_", "TORCH_", "OMP_", "TRITON_"))}


def metadata_check(meta, fixture):
    fields(meta, "schema status arm bindings source source_tree_sha256 artifacts checkpoints "
           "launch ids validation_gate external_gate server_gate telemetry", "Python metadata")
    require(meta["schema"] == "glm-replay-python-metadata-v2" and meta["status"] == "ready",
            "metadata is unrun/template or unsupported")
    require(meta["arm"] == "P", "Python adapter supports contextual arm P only")
    fields(meta["bindings"], HASHES, "fixture bindings")
    for key in HASHES.split():
        require(meta["bindings"][key] == fixture[key], f"metadata {key} mismatch")
    require(meta["server_gate"] is None, "P server gate must be null: unmodified frozen engine")
    from .telemetry import reference_name

    reference_name(meta["telemetry"])
    source = meta["source"]
    fields(source, "engine_commit harness_commit binary_sha256 fatbin_sha256 compiler_sha256 "
           "kit_commit image_digest device_image_digest", "source")
    require(source["engine_commit"] == PYTHON_BASE and source["kit_commit"] == KIT
            and source["device_image_digest"] == DEVICE, "pinned source identity mismatch")
    hex_id(source["harness_commit"], "harness commit", 40)
    for key in ("binary_sha256", "fatbin_sha256", "compiler_sha256"):
        hex_id(source[key], key)
    require(isinstance(source["image_digest"], str) and source["image_digest"].startswith("sha256:"),
            "image digest required")
    hex_id(source["image_digest"][7:], "image digest")
    launch = meta["launch"]
    fields(launch, "prefill_rows grid draft_ring environment_sha256", "launch")
    integer(launch["prefill_rows"], "prefill rows", 64, 16384)
    require(launch["prefill_rows"] % 64 == 0, "prefill rows must tile 64")
    integer(launch["grid"], "prompt grid")
    require(not launch["grid"] or launch["prefill_rows"] % launch["grid"] == 0, "prefill/grid mismatch")
    require(launch["draft_ring"] is True, "production drafter rings required")
    require(isinstance(launch["environment_sha256"], list) and len(launch["environment_sha256"]) == 2,
            "one environment digest per rank required")
    for value in launch["environment_sha256"]:
        hex_id(value, "environment digest")
    fields(meta["ids"], "run_id pair_id start_id start_order block_id cell_id", "run identifiers")
    for key, value in meta["ids"].items():
        if key == "start_order":
            integer(value, key, 0, 1)
        else:
            require(isinstance(value, str) and value.strip(), f"invalid {key}")
    fields(meta["artifacts"], "interpreter fatbin compiler", "artifacts")
    fields(meta["checkpoints"], "target drafter", "checkpoints")


def preflight(fixture_path, source_root, metadata_path=None, roots=None, rank=0, allow_synthetic=False):
    integer(rank, "rank", 0, 1)
    fixture = load_fixture(fixture_path, allow_synthetic=allow_synthetic)
    sources = verify_sources(Path(source_root))
    pool = fixture["config"]["pool"]
    require((2 * pool["h"] * pool["page_tokens"]) % 2048 == 0, "Python pool must tile 2048-token extents")
    from .capacity import reservations

    extents = reservations(fixture, 512, 1)["stream_extents"]
    require(sum((n + 2047) // 2048 * 2048 for n in extents.values()) <= 2 * pool["h"] * 512,
            "Python aligned all-slot reservations exceed pool")
    report = {"schema": "glm-replay-python-preflight-v1", "status": "host-checked",
              "bindings": {k: fixture[k] for k in HASHES.split()}, "source_tree_sha256": digest(sources),
              "needed_graph_rows": graph_rows(fixture), "graph_rows": list(range(1, 33)), "hardware_checks": "unrun",
              "qualification": "seven retained installed matches; all other entries local-source-only"}
    if metadata_path is None:
        return fixture, None, report
    meta = read_json(metadata_path)
    metadata_check(meta, fixture)
    verify_harness(meta["source"]["harness_commit"])
    require(meta["source_tree_sha256"] == digest(sources), "metadata source tree mismatch")
    require(meta["launch"]["environment_sha256"][rank] == digest(environment()), "runtime environment mismatch")
    require(roots is not None, "artifact/checkpoint roots required")
    for role, key in (("interpreter", "binary_sha256"), ("fatbin", "fatbin_sha256"),
                      ("compiler", "compiler_sha256")):
        fields(meta["artifacts"][role], "path bytes sha256", role)
        verify_files(Path(roots["artifacts"]), [meta["artifacts"][role]])
        require(meta["artifacts"][role]["sha256"] == meta["source"][key], f"{role} identity mismatch")
    require(file_digest(sys.executable) == meta["source"]["binary_sha256"], "wrong Python executable")
    bundle = read_json(relative_file(Path(roots["artifacts"]), meta["artifacts"]["fatbin"]["path"]))
    fields(bundle, "schema files", "compiled kernel bundle")
    require(bundle["schema"] == "glm-replay-python-kernels-v1", "compiled kernel bundle required")
    verify_files(Path(roots["artifacts"]), bundle["files"])
    for role, revision in (("target", TARGET_REVISION), ("drafter", DRAFTER_REVISION)):
        checkpoint = meta["checkpoints"][role]
        fields(checkpoint, "revision ranks", role)
        require(checkpoint["revision"] == revision and len(checkpoint["ranks"]) == 2,
                f"{role} checkpoint revision/rank mismatch")
        verify_files(Path(roots[role]), checkpoint["ranks"][rank], exact=True)
    checkpoint_config(roots)
    report["metadata_sha256"] = digest(meta)
    return fixture, meta, report


def verify_harness(commit):
    repo = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    require(git("rev-parse", "HEAD").decode().strip() == commit, "harness commit differs from checkout HEAD")
    folder = Path(__file__).parent
    paths = set(folder.rglob("*.py")) | set(folder.glob("python_*.json"))
    for path in sorted(paths):
        if path.is_file():
            require(git("show", f"HEAD:{path.relative_to(repo).as_posix()}") == path.read_bytes(),
                    f"uncommitted harness identity: {path.name}")


def checkpoint_config(roots):
    raw = read_json(Path(roots["target"]) / "config.json")
    text = raw.get("text_config") or raw
    quant = raw.get("quantization") or raw.get("quantization_config") or {}
    require(quant.get("bits") == 4 and quant.get("group_size") == 64
            and quant.get("quant_method", "mlx") == "mlx", "target must use pinned affine q4/group64")
    require(raw.get("tensorfold_activation_dtype") in (None, "bfloat16"), "CUDA activation dtype mismatch")
    require(text.get("vocab_size") == 154880 and text.get("kv_lora_rank") == 512
            and text.get("num_attention_heads") == 64
            and sum(k != "linear_attention" for k in text.get("layer_types", [])) == 11,
            "target GLM geometry mismatch")
    draft = read_json(Path(roots["drafter"]) / "config.json")
    require("dflash_config" in draft and draft["dflash_config"].get("target_layer_ids"),
            "drafter normal target taps missing")


class FrozenLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(self.get_data(self.path), self.path, "exec", dont_inherit=True)


class FrozenImports(importlib.abc.MetaPathFinder):
    """Resolve tensorfold imports exclusively through already verified frozen files."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pins = {a["path"]: a for a in read_json(SOURCE_MANIFEST)["files"]}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "tensorfold" and not fullname.startswith("tensorfold."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(self.root.parent)] if path is None else path)
        require(spec is not None and spec.origin is not None, f"unresolved frozen module: {fullname}")
        origin = Path(spec.origin).resolve()
        require(origin.is_relative_to(self.root), "import attempted outside frozen source tree")
        name = origin.relative_to(self.root).as_posix()
        require(name in self.pins and file_digest(origin) == self.pins[name]["sha256"],
                f"import identity mismatch: {fullname}")
        spec.loader = FrozenLoader(fullname, str(origin))
        return spec


def install_import_boundary(root):
    require(not any(k == "torch" or k == "tensorfold" or k.startswith(("torch.", "tensorfold."))
                    for k in sys.modules), "GPU/engine module imported before validated preflight")
    sys.dont_write_bytecode = True
    guard = FrozenImports(root)
    sys.meta_path.insert(0, guard)
    return guard


def gate_binding(fixture, meta, runtime_sha256):
    return {**{k: fixture[k] for k in HASHES.split()}, "arm": "P", "source_sha256": digest(meta["source"]),
            "source_tree_sha256": meta["source_tree_sha256"], "runtime_sha256": runtime_sha256,
            "checkpoints_sha256": digest(meta["checkpoints"])}


def require_gate(gate, binding, checks, schema):
    require(gate.get("schema") == schema and gate.get("status") == "pass", "gate unrun/failed or wrong schema")
    require(gate.get("bindings") == binding, "gate binding mismatch")
    require(gate.get("completed_ranks") == [0, 1], "gate lacks both-rank completion")
    for key in checks:
        require(gate.get("checks", {}).get(key) is True, f"missing hardware gate: {key}")
    require(gate.get("execution") == "device", "host doubles cannot qualify hardware")
