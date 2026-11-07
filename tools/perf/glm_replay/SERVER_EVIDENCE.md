# Default-server neutrality evidence

Server neutrality compares the original server with the serving binary built from the replay
patch. Its identities differ from the replay executable and its rank-session runtime digest.
The receipt binds the harness commit, kernel bundle and actual server artifacts. The hardware
comparison is **UNRUN** during host implementation.

After the coordinator captures real device evidence, validate it offline:

```sh
python3 -m tools.perf.glm_replay.server_neutrality NEW_RECEIPT.json \
  --source SOURCE.json --engine native \
  --original-server ORIGINAL_SERVER --instrumented-server INSTRUMENTED_SERVER \
  --tokens TOKENS.json --performance PERFORMANCE.json
```

For Python, supply retained server artifact manifests and `--engine python`. The command hashes
those files and validates retained evidence. It never launches a server, contacts an endpoint,
performs inference or supplies missing measurements.

Receipt schema `glm-replay-server-neutrality-v1` has exactly `schema`, `status`, `execution`,
`engine`, `harness_commit`, `fatbin_sha256`, `original_server_sha256`, `instrumented_server_sha256`,
`completed_ranks`, `token_evidence`, `performance_evidence`. Only passed device evidence with
both ranks admits measurement. Artifact references have exactly `path` and `sha256`; evidence
paths must remain available during runner preflight.

Both token/performance evidence carry identity fields `execution`, `engine`, `harness_commit`,
`fatbin_sha256`, `original_server_sha256`, `instrumented_server_sha256`, `completed_ranks`.
They must match the server receipt and source metadata.

Token evidence schema `glm-replay-server-tokens-v1` adds `cases`, a nonempty list with exactly
`case_id`, `request_sha256`, `config_sha256`, `original_tokens`, `instrumented_tokens` per case.
Tokens must match exactly, and IDs must fit the model vocabulary. Request/configuration hashes
bind the same work in token and performance comparisons.

Performance schema `glm-replay-server-performance-v1` adds `policy` and `samples`. Every sample
has `case_id`, `pair_id`, `start_order` (0 or 1), `request_sha256`, `config_sha256`,
`original_elapsed_ns`, `instrumented_elapsed_ns`. Durations are positive integers. Every token
case needs at least three distinct paired starts with counterbalanced order.

Policy fields are `schema=glm-replay-server-policy-v1`, `minimum_pairs` (at least three),
`max_slowdown_ppm` (an explicitly predeclared bound), and `predeclared_policy_sha256`. That hash
is the canonical JSON hash of the other three policy fields. The coordinator freezes and retains
this policy before the comparison. No universal slowdown threshold is invented. The sum of
candidate times divided by the sum of original times must stay within the declared bound;
all samples remain in the evidence.

Native independently checks receipt/evidence hashes, source/server identities, tokens, matched
work and the performance policy. Python calls the shared offline validator. Neither binds this
receipt to replay runtime or treats the replay executable as a server. A bare handwritten
`status=pass` without the comparisons is insufficient.
