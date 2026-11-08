# Local fixed-work GLM replay campaigns

The developer command runs a frozen ordered block list in one process per rank and one
production TP2 `Session`. It keeps the existing `validate` and single-block `measure`
commands. This driver prescribes work; it adds no serving path or inference scheduler.
Hardware correctness, transport, original/default-server neutrality and performance remain
separate gates. Host tests and a compiled command do not establish those gates.
Before booking this full campaign, complete the production workload-shape check and the cheap
served q8 sharding screen in [DECISION.md](../../tools/perf/glm_replay/DECISION.md), report both
with a recommendation, and wait for the user's campaign decision. Preserve the host implementation.

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
Every new Session also mints a cryptographically random nonce outside the replay blocks.
Both ranks gather it and use rank 0's nonce in every JSONL record and phase anchor.
`campaign_sha256` is the actual ordered preflight/settings digest agreed by that Session,
constant across all its cells. Neither value substitutes for independent process starts.
Single-block `measure` is a pilot command: copying or relabelling its output cannot supply
the common validator's five blocks per cell or independent-start requirements.

## Actual checkpoint identity and optional pins

Replay Sessions gather both ranks' actual target `Engine.checkpoint_digest` and a CPU
identity of the local drafter directory, using production `ranks.artifact/fullIdentity`.
The `tensorfold-full-identity-v1` identity covers configuration, shard metadata and full
shard bytes with the existing identity/cache rules; it does not use device pointers or readbacks.
Both identities enter each rank's stable loaded runtime digest and appear in logical
evidence, gates, JSONL and phase anchors as `checkpoints`:

```json
{"scheme":"tensorfold-full-identity-v1","target":["RANK0_SHA256","RANK1_SHA256"],"drafter":["RANK0_SHA256","RANK1_SHA256"]}
```

Each digest must be 64 lowercase hex digits. Metadata may supply `expected_checkpoints`
with this exact shape. Its schema is checked before CUDA opens, it is included in the
preflight/settings agreement digest, and every specified pin must equal the gathered
actual identity before either rank enters inference. Gates and logical references require
checkpoint receipts even when optional metadata pins are absent; old receipts are refused.
The Session nonce is excluded from runtime and correctness-gate identity so fresh validation
and measurement processes can agree on the same runtime while having different nonces.
Ordinary serving gains no drafter hashing; only opening a replay Session adds it.

To produce local pins later without a driver, service or GPU, build `glm-replay-host` and run:

```text
zig-out/bin/tf-glm-replay-identity MODEL DRAFTER
```

Run this CPU-only producer against each rank's existing frozen local stores. It prints one
local target and drafter digest, the identity scheme, and `revision_qualification="unverified"`.
Place those two ranks' digests into `expected_checkpoints` only after the coordinator verifies
the stores' provenance separately against the fixture's frozen target/drafter revisions.
The command never infers a Hugging Face revision from a directory name, and its output is
not revision qualification. No real checkpoint inputs have been qualified by host controls.

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
`default_server_neutrality`, `default_server_contract` and `logical_evidence`. IDs are nonempty strings. Fixture/config hashes must match
the fully validated fixture. No fields are inferred from filenames or case indices.
`expected_checkpoints` is optional and defaults to null; when present all four pins are required.

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

The correctness, timing and sharding gate files have these fields:

```text
schema="glm-replay-native-gate-v1", gate, status="pass", fixture_sha256,
config_sha256, binary_sha256, completed_ranks=[0,1], comparisons>0,
kv_shard, runtime_sha256, checkpoints, evidence_sha256,
coverage="adversarial-one-row-preparation-and-whole-block-v2"
```

`gate` equals the corresponding artifact name. Hashes, arm and actual runtime must match.
The evidence digest must match the authenticated logical artifact with the same schedule,
owners and output extents. Old coverage or unbound gates fail before CUDA opens.
Per-fixture validation/compare commands produce these gates. Default-server neutrality uses
the separate server schema and offline producer documented in
[SERVER_EVIDENCE.md](../../tools/perf/glm_replay/SERVER_EVIDENCE.md).
It binds the actual original/instrumented server hashes, harness commit and fatbins, with
retained token and counterbalanced performance evidence. Rebuilds require newly bound
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

For each case the measurement helper runs three independently prepared four-slot schedules:
an observer-off warmup, an observer-on phase anchor, and the observer-off reported block.
Every block's completion, kept outputs and final both-rank logical state are checked against
the authenticated reference after completion. The reported preparation duration excludes warmup
and the phase anchor. Verification, selection, keep and grouped
ingest use existing production paths. No extra allocation, JSON/telemetry IO or diagnostic
state readback is added inside the timed block. Existing all-rank completion and post-block
timing readbacks precede the exclusive output write. Each output contains both rank JSONL
records and uses string block IDs accepted by the common validator. The separate
`OUTPUT.anchor.json` retains the anchor and headline completion records, including actual
block layout-check deltas. Its file digest appears in both JSONL timing-source objects.
Phase intervals come from the anchor and must not be summed into headline wall time.
`wall_ns` is that rank's actual completion wall, including both streams' joins and the
all-rank timed completion boundary. A subsequent untimed receipt exchange gathers these
local clocks. The aggregate uses their maximum. `preparation_ns` on both records explicitly
uses rank 0's leader clock around the headline's fresh four-slot preparation; it is not a
rank 1 preparation measurement. Each JSONL `layout_checks` is the saved headline receipt's
rank-local block delta, while the anchor retains both headline and observed receipt counts.
Saved completion receipts also expose `realtime_start_ns` and `realtime_end_ns` as `u64`
nanoseconds from `.real`, sampled before monotonic start and after the all-rank monotonic
stop respectively. The same untimed receipt exchange gathers each rank's own window before
record/timing readbacks or correctness comparisons. Host fakes may leave these fields zero;
a telemetry window producer must require positive ordered real windows. No telemetry scrape,
file, schema or controller is added by these clock seams.
Replay Session setup also gathers `Session.device_uuids: [2][40]u8` from each actual
`Context.device`, querying `cuDeviceGetUuid_v2` with explicit symbol fallback to `cuDeviceGetUuid`.
The 16-byte ABI is checked and formatted as `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` outside
timing. The global driver API and ordinary serving remain unchanged. These fields support
the coordinator's later post-block telemetry binding; they are not hardware qualification.

Before validation opens CUDA, all three output paths (`.logical.json`, `.correctness.json`
and `.timing.json`) must be absent, including dangling symlinks and paths through parent aliases.
Writes remain exclusive, and failures preserve existing outputs and attempted failure ledgers.
Validation coverage is `adversarial-one-row-preparation-and-whole-block-v2`: the one-row oracle
freshly prefills all four slots and replays preparation in fixture round/part/kept-row order,
using the same solo verify, keep and context ingest before the next row. Only needed replay
row/endpoints are captured; full exactness comparisons and poisoned/whole-block checks remain.
The later hardware pilot must measure validation cost separately before freezing the campaign.

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
