"""Validate retained server comparisons offline; never contact an endpoint or run a server."""
import argparse
import time
from collections import defaultdict
from pathlib import Path

from .common import digest, fields, file_digest, hex_id, integer, read_json, require, string, write_json

SCHEMA = "glm-replay-server-neutrality-v2"
IDENTITY = "engine harness_commit fatbin_sha256 original_server_sha256 instrumented_server_sha256 completed_ranks"
GATE_FIELDS = "schema status execution contract_sha256 " + IDENTITY + " token_evidence performance_evidence"


def artifact(value, roots=None):
    fields(value, "path sha256", "server artifact")
    path = Path(string(value["path"], "artifact path"))
    if not path.is_absolute() and roots is not None:
        require(len(roots) == 1, "relative server paths need one receipts root")
        path = Path(roots[0]) / path
    path = path.resolve(strict=True)
    require(path.is_file(), "server artifact must be a file")
    if roots is not None:
        require(any(path.is_relative_to(Path(root).resolve(strict=True)) for root in roots),
                "server artifact escapes permitted roots")
    hex_id(value["sha256"], "artifact sha256")
    require(file_digest(path) == value["sha256"], "server artifact hash mismatch")
    return path


def read_contract(value, source, engine, roots=None):
    require(engine == "native", "unmodified Python has no instrumented server comparison")
    contract = read_json(artifact(value, roots))
    fields(contract, "schema engine engine_commit image_digest harness_commit fatbin_sha256 "
           "original_server_sha256 instrumented_server_sha256 policy_sha256 declared_unix_ns", "server contract")
    require(contract["schema"] == "glm-replay-server-contract-v1" and contract["engine"] == engine,
            "server contract schema/engine mismatch")
    from .fixture import NATIVE_BASE

    require(contract["engine_commit"] == NATIVE_BASE and contract["image_digest"] == source["image_digest"],
            "server original revision/image mismatch")
    for key in ("harness_commit", "fatbin_sha256"):
        require(contract[key] == source[key], f"server contract {key} mismatch")
    for key in ("original_server_sha256", "instrumented_server_sha256", "policy_sha256"):
        hex_id(contract[key], key)
    require(contract["original_server_sha256"] != contract["instrumented_server_sha256"],
            "original and instrumented server must differ")
    integer(contract["declared_unix_ns"], "contract declaration time", 1)
    return contract


def identity(value, source, engine, contract):
    require(value["engine"] == engine and engine == "native", "server engine mismatch")
    for key, length in (("harness_commit", 40), ("fatbin_sha256", 64),
                        ("original_server_sha256", 64), ("instrumented_server_sha256", 64)):
        hex_id(value[key], key, length)
    for key in ("harness_commit", "fatbin_sha256"):
        require(value[key] == source[key], f"server {key} mismatch")
    for key in ("original_server_sha256", "instrumented_server_sha256"):
        require(value[key] == contract[key], f"server contract identity mismatch: {key}")
    require(value["execution"] == "device", "server device evidence is unrun")
    require(value["completed_ranks"] == [0, 1] and all(type(r) is int for r in value["completed_ranks"]),
            "server comparison lacks both ranks")


def evidence_identity(evidence, gate, source, engine, contract):
    identity(evidence, source, engine, contract)
    integer(evidence["started_unix_ns"], "server evidence start", contract["declared_unix_ns"] + 1)
    integer(evidence["ended_unix_ns"], "server evidence end", evidence["started_unix_ns"])
    for key in IDENTITY.split():
        require(type(evidence[key]) is type(gate[key]) and evidence[key] == gate[key],
                f"server evidence identity mismatch: {key}")


def check_tokens(evidence, gate, source, engine, contract):
    fields(evidence, "schema execution " + IDENTITY + " started_unix_ns ended_unix_ns cases", "server token evidence")
    require(evidence["schema"] == "glm-replay-server-tokens-v2", "server token schema mismatch")
    evidence_identity(evidence, gate, source, engine, contract)
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


def check_performance(evidence, gate, source, engine, cases, contract):
    fields(evidence, "schema execution " + IDENTITY + " started_unix_ns ended_unix_ns policy samples", "server performance evidence")
    require(evidence["schema"] == "glm-replay-server-performance-v2", "server performance schema mismatch")
    evidence_identity(evidence, gate, source, engine, contract)
    policy = evidence["policy"]
    fields(policy, "schema predeclared_policy_sha256 minimum_pairs max_slowdown_ppm", "server comparison policy")
    require(policy["schema"] == "glm-replay-server-policy-v1", "server policy schema mismatch")
    hex_id(policy["predeclared_policy_sha256"], "predeclared_policy_sha256")
    minimum = integer(policy["minimum_pairs"], "minimum_pairs", 3)
    limit = integer(policy["max_slowdown_ppm"], "max_slowdown_ppm", 0, 999_999)
    require(policy["predeclared_policy_sha256"] == digest(
        {k: v for k, v in policy.items() if k != "predeclared_policy_sha256"}), "server policy digest mismatch")
    require(policy["predeclared_policy_sha256"] == contract["policy_sha256"],
            "server policy differs from frozen contract")
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


def validate_server_gate(value, source, engine, roots=None, contract=None):
    require(contract is not None, "predeclared server contract required")
    pins = read_contract(contract, source, engine, roots)
    gate = read_json(artifact(value, roots))
    fields(gate, GATE_FIELDS, "server neutrality receipt")
    require(gate["schema"] == SCHEMA and gate["status"] == "pass", "server neutrality is unrun or failed")
    require(gate["contract_sha256"] == contract["sha256"], "server contract hash mismatch")
    identity(gate, source, engine, pins)
    tokens = read_json(artifact(gate["token_evidence"], roots))
    performance = read_json(artifact(gate["performance_evidence"], roots))
    cases = check_tokens(tokens, gate, source, engine, pins)
    check_performance(performance, gate, source, engine, cases, pins)
    return gate


def produce(output, token_evidence, performance_evidence, original_server, instrumented_server, source, engine, contract, roots=None):
    output = Path(output)
    require(not output.exists(), "refusing to overwrite server neutrality receipt")
    pins = read_contract(contract, source, engine, roots)
    gate = {"contract_sha256": contract["sha256"], "schema": SCHEMA, "status": "pass", "execution": "device", "engine": engine,
            "harness_commit": source["harness_commit"], "fatbin_sha256": source["fatbin_sha256"],
            "original_server_sha256": file_digest(original_server),
            "instrumented_server_sha256": file_digest(instrumented_server), "completed_ranks": [0, 1],
            "token_evidence": token_evidence, "performance_evidence": performance_evidence}
    identity(gate, source, engine, pins)
    cases = check_tokens(read_json(artifact(token_evidence, roots)), gate, source, engine, pins)
    check_performance(read_json(artifact(performance_evidence, roots)), gate, source, engine, cases, pins)
    write_json(output, gate)
    return gate


def declare(output, original_server, instrumented_server, source, policy):
    from .fixture import NATIVE_BASE

    require(not Path(output).exists(), "refusing to overwrite server contract")
    require(source["engine_commit"] == NATIVE_BASE, "server original revision mismatch")
    fields(policy, "schema predeclared_policy_sha256 minimum_pairs max_slowdown_ppm", "server comparison policy")
    require(policy["schema"] == "glm-replay-server-policy-v1", "server policy schema mismatch")
    integer(policy["minimum_pairs"], "minimum_pairs", 3)
    integer(policy["max_slowdown_ppm"], "max_slowdown_ppm", 0, 999_999)
    require(policy["predeclared_policy_sha256"] == digest(
        {k: v for k, v in policy.items() if k != "predeclared_policy_sha256"}), "server policy digest mismatch")
    contract = {"schema": "glm-replay-server-contract-v1", "engine": "native", "engine_commit": NATIVE_BASE,
                "image_digest": source["image_digest"], "harness_commit": source["harness_commit"],
                "fatbin_sha256": source["fatbin_sha256"], "original_server_sha256": file_digest(original_server),
                "instrumented_server_sha256": file_digest(instrumented_server),
                "policy_sha256": policy["predeclared_policy_sha256"], "declared_unix_ns": time.time_ns()}
    require(contract["original_server_sha256"] != contract["instrumented_server_sha256"],
            "original and instrumented server must differ")
    write_json(output, contract)
    return contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "produce"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--engine", choices=("native",), required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--original-server", type=Path, required=True)
    parser.add_argument("--instrumented-server", type=Path, required=True)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--receipts-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.receipts_root.resolve(strict=True)
    def ref(path):
        path = path.resolve(strict=True)
        require(path.is_relative_to(root), "server evidence escapes receipts root")
        return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    if args.command == "declare":
        require(args.policy is not None, "declaration needs a frozen policy")
        declare(args.output, args.original_server, args.instrumented_server, read_json(args.source), read_json(args.policy))
    else:
        require(args.contract and args.tokens and args.performance, "production needs contract, tokens and performance")
        produce(args.output, ref(args.tokens), ref(args.performance), args.original_server,
                args.instrumented_server, read_json(args.source), args.engine, ref(args.contract), [root])
    print("PASS: retained server evidence validated; no endpoint or device work performed by this command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
