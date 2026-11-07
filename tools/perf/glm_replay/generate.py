"""Local-only one-time tokenization; synthetic byte fixtures are explicitly host-only."""
import argparse
from pathlib import Path

from .capacity import minimum_h, reservations
from .common import canonical_bytes, fields, file_digest, hex_id, integer, read_json, require, text_digest, write_json
from .fixture import DRAFTER_REVISION, SCHEMA, TARGET_REVISION, seal_fixture, validate_fixture
from .histories import source_material

SYNTHETIC_TEMPLATE = "synthetic-byte-v1: canonical JSON of messages/tools, followed by newline assistant newline"


class SyntheticTokenizer:
    kind = "synthetic-byte"
    revision = "synthetic-byte-v1"
    vocab_size = 256
    files = []
    template = SYNTHETIC_TEMPLATE

    def render(self, messages, tools):
        return canonical_bytes({"messages": messages, "tools": tools}).decode("ascii") + "\nassistant\n"

    def encode(self, text):
        return list(text.encode("utf-8"))


class PinnedTokenizer:
    kind = "huggingface-local"

    def __init__(self, directory, manifest):
        root = Path(directory).resolve(strict=True)
        receipt = read_json(manifest)
        fields(receipt, "schema revision files", "tokenizer manifest")
        require(receipt["schema"] == "glm-replay-tokenizer-assets-v1", "invalid tokenizer manifest schema")
        hex_id(receipt["revision"], "tokenizer revision", 40)
        require(receipt["revision"] == TARGET_REVISION, "tokenizer must be pinned to target checkpoint revision")
        require(isinstance(receipt["files"], list) and receipt["files"], "empty tokenizer manifest")
        names = set()
        for item in receipt["files"]:
            fields(item, "path sha256", "tokenizer asset")
            relative = Path(item["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe tokenizer asset path")
            path = (root / relative).resolve(strict=True)
            require(path.is_relative_to(root) and path.is_file(), "tokenizer asset escapes directory")
            require(path.suffix in (".json", ".jinja", ".model", ".txt", ".tiktoken"), "unsupported tokenizer asset")
            require(item["path"] not in names, "duplicate tokenizer asset")
            names.add(item["path"])
            hex_id(item["sha256"], "tokenizer asset hash")
            require(file_digest(path) == item["sha256"], f"tokenizer asset hash mismatch: {item['path']}")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        require(actual == names, "all local tokenizer directory files must be hashed in manifest")
        require({"tokenizer.json", "tokenizer_config.json"} <= names, "missing tokenizer assets")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ValueError("real tokenization requires an already installed transformers package and local assets") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        self.template = self.tokenizer.get_chat_template()
        require(isinstance(self.template, str) and self.template, "missing pinned chat template")
        self.vocab_size = len(self.tokenizer)
        self.revision = receipt["revision"]
        self.files = sorted(receipt["files"], key=lambda item: item["path"])

    def render(self, messages, tools):
        options = {"tools": tools} if tools else {}
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def _schedule(streams, active, widths, keeps, preparation_rounds, rounds, permute):
    offsets = [0] * 4
    schedules = []
    for phase, count in enumerate((preparation_rounds, rounds)):
        participating = list(range(4)) if phase == 0 else active
        offered_counts = [1] * 4 if phase == 0 else widths
        kept_counts = [1] * 4 if phase == 0 else keeps
        forwards = []
        for fid in range(count):
            order = participating if not permute or fid % 2 == 0 else list(reversed(participating))
            parts = [{"stream_id": sid, "position": len(streams[sid]["prompt_tokens"]) + offsets[sid],
                      "tape_offset": offsets[sid], "offered": offered_counts[sid], "kept": kept_counts[sid]} for sid in order]
            forwards.append({"round_id": fid, "forward_id": fid, "parts": parts})
            for part in parts:
                offsets[part["stream_id"]] += part["kept"]
        schedules.append(forwards)
    return schedules


def build_fixture(family="prose", tokenizer=None, context_tokens=128, rounds=3, preparation_rounds=1,
                  active_streams=(0, 1, 2, 3), widths=(4, 4, 4, 4), keeps=None,
                  placement="balanced", permute=False, repetitions=None, pool_h=None, configured_capacity=None):
    require(family in ("prose", "code", "pi"), "unknown family")
    integer(context_tokens, "context tokens", 1)
    integer(rounds, "rounds", 1)
    integer(preparation_rounds, "preparation rounds")
    require(len(widths) == 4 and (keeps is None or len(keeps) == 4), "exactly four width/keep counts required")
    keeps = list(widths) if keeps is None else list(keeps)
    for offered, kept in zip(widths, keeps, strict=True):
        integer(offered, "offered", 1, 16)
        integer(kept, "kept", 1, offered)
    active = list(active_streams)
    require(len(active) in (1, 4) and len(set(active)) == len(active), "solo or four distinct active streams required")
    for sid in active:
        integer(sid, "active stream", 0, 3)
    require(sum(widths[s] for s in active) <= 32, "shared forward exceeds 32 offered rows")
    require(placement in ("balanced", "reversed"), "invalid placement")
    tokenizer = tokenizer or SyntheticTokenizer()
    require(isinstance(tokenizer, (SyntheticTokenizer, PinnedTokenizer)), "unrecognized tokenizer provenance")
    repetitions = repetitions if repetitions is not None else max(2, context_tokens // 64 + 2, (rounds + preparation_rounds) // 4 + 1)
    integer(repetitions, "source repetitions", 1)
    streams, sources = [], []
    owners = [0, 0, 1, 1] if placement == "balanced" else [1, 1, 0, 0]
    for sid in range(4):
        material = source_material(family, sid, repetitions)
        rendered = tokenizer.render(material["messages"], material["tools"])
        prompt = tokenizer.encode(rendered)
        tape = tokenizer.encode(material["tape"])
        require(len(prompt) >= context_tokens, "rendered source too short; increase source repetitions and regenerate")
        source_id = f"authored-{family}-{sid}-v1"
        sources.append({"source_id": source_id, "attribution": "TensorFold public replay fixture authors, 2026",
                        "license": "Apache-2.0", "text": material["text"], "sha256": text_digest(material["text"])})
        streams.append({"stream_id": sid, "owner": owners[sid], "source_id": source_id,
                        "rendered_prompt": rendered, "rendered_tape": material["tape"],
                        "rendered_prompt_sha256": text_digest(rendered),
                        "rendered_tape_sha256": text_digest(material["tape"]),
                        "prompt_encoding": {"token_ids": prompt, "used_tokens": context_tokens},
                        "prompt_tokens": prompt[:context_tokens], "tape_tokens": tape})
    preparation, forwards = _schedule(streams, active, widths, keeps, preparation_rounds, rounds, permute)
    fixture = {"schema": SCHEMA, "fixture_id": f"{family}-{context_tokens}-{'-'.join(map(str, widths))}-{placement}",
               "kind": "synthetic" if tokenizer.kind == "synthetic-byte" else "campaign", "family": family,
               "provenance": {"target_revision": TARGET_REVISION, "drafter_revision": DRAFTER_REVISION,
                              "tokenizer": {"kind": tokenizer.kind, "revision": tokenizer.revision,
                                            "vocab_size": tokenizer.vocab_size, "files": tokenizer.files,
                                            "template": tokenizer.template, "template_sha256": text_digest(tokenizer.template)},
                              "sources": sources},
               "config": {"speed": "standard", "tp": 2, "slots": 4, "dense_bits": 8, "head_bits": 8, "kv": "fp8",
                          "reset": "fresh_prefill", "selection": "greedy", "proposer": False,
                          "active_streams": active, "placement": placement, "pool": {}},
               "streams": streams, "preparation": preparation, "forwards": forwards}
    largest = max(reservations(fixture, 512, 1)["stream_extents"].values())
    capacity = configured_capacity if configured_capacity is not None else max(131072, largest)
    h = pool_h if pool_h is not None else minimum_h(fixture, 512, capacity)
    fixture["config"]["pool"] = {"h": h, "page_tokens": 512, "configured_capacity": capacity, "prompt_twins": 1}
    return validate_fixture(seal_fixture(fixture), allow_synthetic=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--family", choices=("prose", "code", "pi"), default="prose")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="explicit host-only byte encoding")
    mode.add_argument("--tokenizer-dir", type=Path, help="local assets only; never downloaded")
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--preparation-rounds", type=int, default=1)
    parser.add_argument("--active-streams", default="0,1,2,3")
    parser.add_argument("--widths", default="4,4,4,4")
    parser.add_argument("--keeps")
    parser.add_argument("--placement", choices=("balanced", "reversed"), default="balanced")
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--pool-h", type=int)
    parser.add_argument("--configured-capacity", type=int)
    args = parser.parse_args(argv)
    require(args.synthetic or args.tokenizer_manifest is not None, "real generator requires tokenizer manifest")
    tokenizer = SyntheticTokenizer() if args.synthetic else PinnedTokenizer(args.tokenizer_dir, args.tokenizer_manifest)
    counts = lambda value: tuple(int(part) for part in value.split(","))
    fixture = build_fixture(args.family, tokenizer, args.context_tokens, args.rounds, args.preparation_rounds,
                            counts(args.active_streams), counts(args.widths), counts(args.keeps) if args.keeps else None,
                            args.placement, args.permute, args.repetitions, args.pool_h, args.configured_capacity)
    require(not args.output.exists(), "refusing to overwrite immutable fixture")
    write_json(args.output, fixture)
    print(f"{fixture['kind']} {fixture['fixture_sha256']} {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
