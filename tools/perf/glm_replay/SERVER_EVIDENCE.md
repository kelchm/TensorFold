# Default-server comparison evidence

Native compares the pinned original server with the distinct serving binary built from the
replay patch. These are different artifacts from `tf-cuda-test`. This device comparison is
**UNRUN** during host implementation. The external Python adapter leaves its frozen engine
unchanged and records `default_server_neutral="not_applicable_unmodified_engine"`; it must instead
prove the installed-source equivalence required by its qualification gate.

Before capturing any server comparison, freeze a policy and create a contract:

```sh
python3 -m tools.perf.glm_replay.server_neutrality declare CONTRACT.json \
  --source SOURCE.json --engine native --policy POLICY.json \
  --original-server ORIGINAL_SERVER --instrumented-server INSTRUMENTED_SERVER
```

Put the exact contract `{path,sha256}` in native metadata's `default_server_contract` before
capture. Contract schema `glm-replay-server-contract-v1` has `schema`, `engine="native"`,
`engine_commit` (historical e52), `image_digest` (original pinned image), `harness_commit`,
`fatbin_sha256`, `original_server_sha256`, `instrumented_server_sha256`, `policy_sha256` and
`declared_unix_ns`. The producer hashes actual server files and records declaration time.
Original and instrumented hashes must differ. Retain provenance mapping the original file
to its pinned deployment image; naming an arbitrary file is not deployment qualification.

Policy fields are `schema="glm-replay-server-policy-v1"`, `minimum_pairs` (at least three),
`max_slowdown_ppm` and `predeclared_policy_sha256` (canonical hash of the other fields).
There is no automatically chosen acceptable slowdown. The coordinator must review and freeze
the bound before capture; technical validation proves the declared bound, not its suitability.
The contract separately pins that policy hash so recomputing a policy checksum after seeing
results cannot change an already bound comparison. A 100% slowdown limit is refused.

After the coordinator retains real evidence, produce the comparison receipt offline:

```sh
python3 -m tools.perf.glm_replay.server_neutrality produce NEW_RECEIPT.json \
  --source SOURCE.json --engine native --contract CONTRACT.json \
  --original-server ORIGINAL_SERVER --instrumented-server INSTRUMENTED_SERVER \
  --tokens TOKENS.json --performance PERFORMANCE.json --receipts-root RECEIPTS_DIRECTORY
```

Evidence and contract references emitted by this CLI are relative to the receipts root. Run
native preflight from that directory, or retain equivalent resolvable process-relative paths
on both ranks. Neither command starts a server, contacts an endpoint or supplies measurements.
Contracts and passing receipts are distinct; no passing device gate is supplied by host work.

Receipt schema `glm-replay-server-neutrality-v2` contains `schema`, `status`, `execution`,
`contract_sha256`, `engine`, `harness_commit`, `fatbin_sha256`, `original_server_sha256`,
`instrumented_server_sha256`, `completed_ranks`, `token_evidence` and `performance_evidence`.
Only passed device evidence with both ranks admits measurement. Every reference is exactly
`{path,sha256}`. Source, distinct server identities and policy must match the frozen contract.

Token/performance evidence carry the same server/source identity fields, `execution="device"`,
`completed_ranks=[0,1]`, `started_unix_ns` and `ended_unix_ns`. Capture must start after the
contract declaration and end afterwards. These retained timestamps and the metadata pin are
auditable ordering evidence, not a trusted timestamp service or protection against forged files.

`glm-replay-server-tokens-v2` adds nonempty `cases` with `case_id`, `request_sha256`,
`config_sha256`, `original_tokens` and `instrumented_tokens`. Tokens must match exactly and
fit the model vocabulary. `glm-replay-server-performance-v2` adds the pinned `policy` and
`samples` with `case_id`, `pair_id`, `start_order`, `request_sha256`, `config_sha256`,
`original_elapsed_ns` and `instrumented_elapsed_ns`. Each token case needs at least the declared
number of distinct counterbalanced paired starts. All durations are positive and work hashes
match. The ratio of summed candidate times to summed original times must meet the frozen bound.

Native independently checks the exact contract and evidence file hashes, identities, ordering,
tokens, work and policy. Old unbound v1 receipts are refused. No replay runtime or replay-binary
identity is substituted for a server identity.
