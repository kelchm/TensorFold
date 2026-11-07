"""Validate retained server comparisons offline; never contact an endpoint or run a server."""
import argparse
from collections import defaultdict
from pathlib import Path

from .common import digest, fields, file_digest, hex_id, integer, read_json, require, string, write_json

SCHEMA = "glm-replay-server-neutrality-v1"
IDENTITY = "engine harness_commit fatbin_sha256 original_server_sha256 instrumented_server_sha256 completed_ranks"
GATE_FIELDS = "schema status execution " + IDENTITY + " token_evidence performance_evidence"


def artifact(value, roots=None):
    fields(value, "path sha256", "server artifact")
    path = Path(string(value["path"], "artifact path")).resolve(strict=True)
    require(path.is_file(), "server artifact must be a file")
    if roots is not None:
        require(any(path.is_relative_to(Path(root).resolve(strict=True)) for root in roots),
                "server artifact escapes permitted roots")
    hex_id(value["sha256"], "artifact sha256")
    require(file_digest(path) == value["sha256"], "server artifact hash mismatch")
    return path


def identity(value, source, engine):
    require(value["engine"] == engine and engine in ("native", "python"), "server engine mismatch")
    for key, length in (("harness_commit", 40), ("fatbin_sha256", 64),
                        ("original_server_sha256", 64), ("instrumented_server_sha256", 64)):
        hex_id(value[key], key, length)
    for key in ("harness_commit", "fatbin_sha256"):
        require(value[key] == source[key], f"server {key} mismatch")
    require(value["execution"] == "device", "server device evidence is unrun")
    require(value["completed_ranks"] == [0, 1] and all(type(r) is int for r in value["completed_ranks"]),
            "server comparison lacks both ranks")


def evidence_identity(evidence, gate, source, engine):
    identity(evidence, source, engine)
    for key in IDENTITY.split():
        require(type(evidence[key]) is type(gate[key]) and evidence[key] == gate[key],
                f"server evidence identity mismatch: {key}")


def check_tokens(evidence, gate, source, engine):
    fields(evidence, "schema execution " + IDENTITY + " cases", "server token evidence")
    require(evidence["schema"] == "glm-replay-server-tokens-v1", "server token schema mismatch")
    evidence_identity(evidence, gate, source, engine)
    require(isinstance(evidence["cases"], list) and evidence["cases"], "empty server token comparisons")
    cases = {}
    for case in evidence["cases"]:
        fields(case, "case_id request_sha256 config_sha256 original_tokens instrumented_tokens", "server token case")
        key = string(case["case_id"], "case_id")
        require(key not in cases, "duplicate server token case")
        for name in ("request_sha256", "config_sha256"):
            hex_id(case[name], name)
        for name in ("original_tokens", "instrumented_tokens"):
            require(isinstance(case[name], list) and case[name], "empty server tokens")
            for token in case[name]:
                integer(token, "server token", 0, 154879)
        require(case["original_tokens"] == case["instrumented_tokens"], "server tokens differ")
        cases[key] = case
    return cases


def check_performance(evidence, gate, source, engine, cases):
    fields(evidence, "schema execution " + IDENTITY + " policy samples", "server performance evidence")
    require(evidence["schema"] == "glm-replay-server-performance-v1", "server performance schema mismatch")
    evidence_identity(evidence, gate, source, engine)
    policy = evidence["policy"]
    fields(policy, "schema predeclared_policy_sha256 minimum_pairs max_slowdown_ppm", "server comparison policy")
    require(policy["schema"] == "glm-replay-server-policy-v1", "server policy schema mismatch")
    hex_id(policy["predeclared_policy_sha256"], "predeclared_policy_sha256")
    minimum = integer(policy["minimum_pairs"], "minimum_pairs", 3)
    limit = integer(policy["max_slowdown_ppm"], "max_slowdown_ppm", 0, 1_000_000)
    require(policy["predeclared_policy_sha256"] == digest(
        {k: v for k, v in policy.items() if k != "predeclared_policy_sha256"}), "server policy digest mismatch")
    require(isinstance(evidence["samples"], list) and evidence["samples"], "empty server performance comparisons")
    grouped, seen = defaultdict(list), set()
    for sample in evidence["samples"]:
        fields(sample, "case_id pair_id start_order request_sha256 config_sha256 original_elapsed_ns "
               "instrumented_elapsed_ns", "server performance sample")
        key = string(sample["case_id"], "case_id")
        pair = string(sample["pair_id"], "pair_id")
        require(key in cases and (key, pair) not in seen, "unknown or duplicate server pair")
        seen.add((key, pair))
        integer(sample["start_order"], "start_order", 0, 1)
        for name in ("request_sha256", "config_sha256"):
            require(sample[name] == cases[key][name], "server performance work differs from token comparison")
        for name in ("original_elapsed_ns", "instrumented_elapsed_ns"):
            integer(sample[name], name, 1)
        grouped[key].append(sample)
    require(set(grouped) == set(cases), "server performance cases incomplete")
    for samples in grouped.values():
        require(len(samples) >= minimum, "insufficient independent server pairs")
        orders = [s["start_order"] for s in samples]
        require(abs(orders.count(0) - orders.count(1)) <= 1, "server start order is not counterbalanced")
        original = sum(s["original_elapsed_ns"] for s in samples)
        candidate = sum(s["instrumented_elapsed_ns"] for s in samples)
        require(candidate * 1_000_000 <= original * (1_000_000 + limit),
                "server performance exceeds predeclared slowdown limit")


def validate_server_gate(value, source, engine, roots=None):
    gate = read_json(artifact(value, roots))
    fields(gate, GATE_FIELDS, "server neutrality receipt")
    require(gate["schema"] == SCHEMA and gate["status"] == "pass", "server neutrality is unrun or failed")
    identity(gate, source, engine)
    tokens = read_json(artifact(gate["token_evidence"], roots))
    performance = read_json(artifact(gate["performance_evidence"], roots))
    cases = check_tokens(tokens, gate, source, engine)
    check_performance(performance, gate, source, engine, cases)
    return gate


def produce(output, token_evidence, performance_evidence, original_server, instrumented_server, source, engine):
    output = Path(output)
    require(not output.exists(), "refusing to overwrite server neutrality receipt")
    gate = {"schema": SCHEMA, "status": "pass", "execution": "device", "engine": engine,
            "harness_commit": source["harness_commit"], "fatbin_sha256": source["fatbin_sha256"],
            "original_server_sha256": file_digest(original_server),
            "instrumented_server_sha256": file_digest(instrumented_server), "completed_ranks": [0, 1],
            "token_evidence": token_evidence, "performance_evidence": performance_evidence}
    identity(gate, source, engine)
    cases = check_tokens(read_json(artifact(token_evidence)), gate, source, engine)
    check_performance(read_json(artifact(performance_evidence)), gate, source, engine, cases)
    write_json(output, gate)
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--engine", choices=("native", "python"), required=True)
    parser.add_argument("--original-server", type=Path, required=True)
    parser.add_argument("--instrumented-server", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    args = parser.parse_args(argv)
    ref = lambda p: {"path": str(p.resolve(strict=True)), "sha256": file_digest(p)}
    produce(args.output, ref(args.tokens), ref(args.performance), args.original_server,
            args.instrumented_server, read_json(args.source), args.engine)
    print("PASS: retained server evidence validated; no endpoint or device work performed by this command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
