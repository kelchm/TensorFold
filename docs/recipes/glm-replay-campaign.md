# Local fixed-work GLM replay campaigns

The developer command runs a frozen ordered block list in one process per rank and one
production TP2 `Session`. It keeps the existing `validate` and single-block `measure`
commands. This driver prescribes work; it adds no serving path or inference scheduler.
Hardware correctness, transport, original/default-server neutrality and performance remain
separate gates. Host tests and a compiled command do not establish those gates.

## Commands

From the repository root, with Zig 0.17.0:

```sh
zig build test-glm-replay-host compile-glm-replay glm-replay-host --summary all
zig build compile-cuda-tests -Dtarget=aarch64-linux-gnu --summary all
```

The first command runs driver-free negative controls and compiles the entire campaign body.
The second compiles and links the real developer dispatcher without executing it.
The existing per-fixture commands are:

```text
tf-cuda-test glm-replay-check FIXTURE
tf-cuda-test glm-replay-compare OFF.logical.json ON.logical.json [GATE_PREFIX]
tf-cuda-test glm-replay validate FIXTURE MODEL DRAFTER OUTPUT_PREFIX METADATA
tf-cuda-test glm-replay measure FIXTURE MODEL DRAFTER OUTPUT.jsonl METADATA
```

The new command has exactly five arguments after `glm-replay`:

```text
tf-cuda-test glm-replay measure-many CAMPAIGN.json MODEL DRAFTER FAILURE_PREFIX
```

On each rank, in a separately authorized hardware window with its ordinary production
rank/transport environment already configured, use the same arm and ordered manifest:

```sh
TF_KV_PAGES=on zig-out/bin/tf-cuda-test glm-replay measure-many campaign.json MODEL DRAFTER attempts/start-a
```

`MODEL` and `DRAFTER` name existing local frozen checkpoints. TP2, standard speed,
dense/head 8/8, four slots, paged FP8 KV and DFlash2 q4 are required. The command reads
the ordinary GLM settings; intrusive profiling is refused. Set `TF_GLM_KV_SHARD` to the
arm under test while keeping the other frozen execution and transport settings identical.
This bounded mode requires explicit `TF_KV_PAGES=on` before opening CUDA; `auto` does not
enable pages for FP8 in the production startup resolver.
The driver never starts another process, creates a start ID, gathers telemetry or changes a
service. A new independent start requires a genuinely new rank process invocation and a
new externally assigned `start_id`; three paired starts require six native invocations.

## Ordered descriptor schema

`CAMPAIGN.json` has exactly these fields; unknown and duplicate JSON fields are refused:

```json
{
  "schema": "glm-replay-native-campaign-v1",
  "cases": [
    {"fixture": "fixtures/solo.json", "metadata": "metadata/solo-0.json", "output": "results/solo-0.jsonl"},
    {"fixture": "fixtures/wide.json", "metadata": "metadata/wide-0.json", "output": "results/wide-0.jsonl"},
    {"fixture": "fixtures/solo.json", "metadata": "metadata/solo-1.json", "output": "results/solo-1.jsonl"}
  ]
}
```

Paths are relative to the process working directory, including paths in metadata. Output
parent directories must already exist. Outputs must be absent, including dangling symlinks;
resolved parent paths detect aliases. Creation remains exclusive after the preflight so a
file created concurrently is preserved. The list contains 1–4096 cases and the manifest
is limited to 4 MiB. Each descriptor is one measured block, in its listed position.
Use repeated descriptors with distinct metadata/output paths for five or more blocks per
cell. Deterministic rotations and within-start anchor repeats are ordinary list entries.
There is no automatic expansion, permutation, retry, process launch or dataset-completeness
claim. The common offline validator enforces replication requirements across starts.

All cases must have the same run, pair, start, start order, complete build/source identity,
H and configured capacity. Each `(cell_id, block_id)` and output must be unique within the
start; the same `block_id` can appear in different cells. A cell always names the same
fixture hash. Distinct cells may change fixture family, context, active streams, widths,
placement and schedule, subject to the existing fixture validator and fixed pool budget.
All four slots are freshly prepared even for solo cases.

## Existing metadata and gates

Every descriptor requires a complete `glm-replay-native-metadata-v1` file. Its exact fields
are `schema`, `run_id`, `pair_id`, `start_id`, `start_order` (integer 0 or 1), `block_id`
(nonempty string), `cell_id`, `fixture_sha256`, `config_sha256`, `source`, `fatbin`,
`compiler`, `telemetry`, `correctness`, `sharding_equality`, `timing_neutrality` and
`default_server_neutrality`. IDs are nonempty strings. Fixture/config hashes must match
the fully validated fixture. No fields are inferred from filenames or case indices.

`source` has exactly `engine_commit`, `harness_commit`, `binary_sha256`, `fatbin_sha256`,
`compiler_sha256`, `kit_commit`, `image_digest` and `device_image_digest`.
Commits are 40 lowercase hex digits, SHA-256 values are 64 lowercase hex digits, and image
digests use the `sha256:` prefix. The historical engine commit is
`e52eb42b52b1e50f3f23a5eae8115bbe80c5e7d9`; kit/image identities retain the existing frozen
metadata checks. The binary hash is verified against the running executable.

Every artifact is exactly `{"path":"LOCAL_FILE","sha256":"64 lowercase hex digits"}`.
`fatbin` and `compiler` must agree with their source hashes. All four gate artifacts are
mandatory for `measure-many`; `null` is refused. Telemetry must already be a frozen external
artifact and is checked before loading, then checked again after each block.
This mode cannot emit validation gates or manufacture a passing hardware receipt.

Each gate file has exactly these fields:

```text
schema="glm-replay-native-gate-v1", gate, status="pass", fixture_sha256,
config_sha256, binary_sha256, completed_ranks=[0,1], comparisons>0,
kv_shard, runtime_sha256
```

`gate` equals the corresponding artifact name. Hashes, arm and actual runtime must match.
Existing per-fixture validation/compare commands produce their existing gates;
default-server neutrality remains an external device check. Rebuilds require newly bound
gates because `binary_sha256` changes.

## Execution and failures

Before `Driver.open`, both ranks independently validate every fixture, token/source hash,
tokenizer asset, model config, metadata, gate and artifact; compile every unique fixture's
preparation and block buffers; and reject mixed identities and output collisions. The
ordered content/start/source digest plus settings hash is passed through existing Session
agreement before engine allocation. Rank-local paths are excluded from that digest.
Receipt/event capacity is the maximum forward count over every preparation and block.
After load, every gate is matched to the actual combined `Session.runtime_sha256` on each
rank, followed by a rank agreement barrier before any preparation or inference.

For each case the existing measurement helper runs fresh four-slot prefill plus prescribed
preparation, a disposable warmup block, then another fresh four-slot prefill and identical
preparation, then the reported block. Both warmup and reported completion are checked.
The reported preparation duration excludes warmup. Verification, selection, keep and grouped
ingest use existing production paths. No extra allocation, JSON/telemetry IO or diagnostic
state readback is added inside the timed block. Existing all-rank completion and post-block
timing readbacks precede the exclusive output write. Each output contains both rank JSONL
records and uses string block IDs accepted by the common validator.

A caught failure attempts an exclusive
`FAILURE_PREFIX.rank-R.failed.json` with schema
`glm-replay-native-campaign-attempt-v1`, `status="failed"`, `failure`,
`measurement_complete=false`, `rank`, `session_open`, `completed_output_cases`,
`active_case_index` and `ordered_sha256`. The digest is null until preflight succeeds.
`completed_output_cases` is known only to rank 0; it counts fully written preceding outputs,
not independent starts. The active index is zero based. Files already written stay intact;
the current output may be partial on a write failure and must not be used as completion
evidence. A killed process or unavailable filesystem cannot promise a failure receipt.
Follower shutdown remains owned by the production follower/leader path; caught session
failures notify the watchdog and use the existing failed-session exit behavior.
