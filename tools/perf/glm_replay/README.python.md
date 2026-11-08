# External Python replay adapter

This adapter imports the frozen Python engine; it does not edit or copy its forward,
commit, scheduler, arithmetic, kernels or transport. It supplies prescribed fixture rows
and keeps to the existing production boundaries. Python arm P is q4 deployment context.
Native A/B remains the primary sharding comparison. A native raw verify call includes
selection/readback; a Python raw forward call does not. Their raw timings are not equivalent.

**Status: host implementation and host doubles only. All device, TP transport, correctness,
timing neutrality, installed-source equivalence, physical-fit and performance gates are unrun.**
No campaign fixtures, model assets or hardware gate passes ship here. Host tests cannot
qualify device behavior. Arm P has no instrumented server: its server comparison is explicitly
not applicable, conditional on frozen imports and full both-rank installed-source equivalence.
Common authenticated-anchor and process-start provenance integration is required before device mode.
Finish host work, then the workload/served screens in [DECISION.md](DECISION.md), before deciding
on a full hardware campaign. No frozen engine patch is required.

## Host commands

Run from the repository root with Python 3.11 or later:

```sh
python3 -m unittest discover -s tools/perf/glm_replay/tests -v
python3 -m tools.perf.glm_replay.python_adapter host-check \
  tools/perf/glm_replay/fixtures/host-pi.json --allow-synthetic
python3 -m tools.perf.glm_replay.python_adapter preflight \
  tools/perf/glm_replay/fixtures/host-prose.json \
  --source-root FROZEN_SRC/tensorfold --allow-synthetic
```

`host-check` validates fixture/config/token/schedule hashes before any engine access.
`preflight` additionally hashes every frozen file, without importing torch or tensorfold.
Its `status=host-checked` and `hardware_checks=unrun` do not authorize measurement.
`python_preflight.example.json` is an actual host-only inspection of the shipped synthetic
prose fixture, with device status explicitly unrun. Synthetic fixtures are refused in device mode.

`python_sources.json` pins all 439 files in the inspected frozen `src/tensorfold` tree.
Seven entries retain the earlier installed-manifest match. `segments.py`, `pool.py`, `l2pf.py`
and every other newly pinned dependency remain **local-source-only**, requiring retained complete
installed-image listings from both ranks; the adapter does not invent earlier installed hashes. Paths are relative.
Missing, changed, added, escaped or symlinked files fail closed. Imports are restricted to
the verified tree, rehashed at import and compiled from source without reading cached bytecode
or writing bytecode into the frozen tree. An already-imported torch/engine is refused.

## Metadata and provenance

`python_metadata.example.json`, `python_validation.example.json` and
`python_gate.example.json`, `python_activation.example.json` and `python_qualification.example.json`
are **unrun templates** with null unknown identities. They cannot pass preflight or authorize device measurement. Populate a new
metadata file from independently verified local evidence; never relabel an example as pass.

Ready metadata binds the four common fixture hashes, pinned engine/kit/device identities,
the actual committed harness HEAD, Python executable, compiler and compiled-kernel bundle.
It also binds exact checkpoint files for each rank to the pinned target and DFlash2 revisions.
Checkpoint roots must be dedicated read-only asset trees: no downloads, retokenization or
remote code are used. Every file manifest entry has relative `path`, exact `bytes` and
lowercase `sha256`. Rank manifests may differ, but both ranks use the same metadata object.
`source.binary_sha256` means the actual Python executable in this contextual arm.

The `fatbin` artifact is a hashed JSON file of this shape, with every file resolved beneath
the artifact root. It describes the qualified compiled extension/cubin set, not source text:

```json
{"schema":"glm-replay-python-kernels-v1","files":[{"path":"compiled/qualified.cubin","bytes":123,"sha256":"<real 64 hex digest>"}]}
```

The adapter hashes every bundle member before GPU import and checks that actually loaded
tensorfold native extensions belong to the bundle. Qualification that this is the actual
complete CUDA/Triton kernel set remains a mandatory external device gate. A source hash or
unused arbitrary cubin does not qualify a compiled-kernel identity.

Set the required environment explicitly on both ranks; preflight never changes it:

```sh
export TF_GLM_MULTI_LONE=0 TF_GLM_MULTI_VERIFY=batched TF_GLM_MULTI_GRAPHS=1
export TF_GLM_MULTI_DRAFT_GRAPHS=1 TF_GLM_MULTI_SAMPLER=streams TF_GLM_MULTI_WINDOW=32
export TF_GLM_KV_SHARD=0 TF_GLM_LATENT=1 TF_GLM_KV=fp8
export TF_GLM_PROFILE=0 TF_GLM_MULTI_PROFILE=0
```

Preserve the authorized production transport and numerical settings. No adapter command
changes host clocks, pinning, drivers or a serving process. `launch.environment_sha256`
contains two rank-ordered hashes of `python_provenance.environment()`; it includes all
TF/TENSORFOLD/CUDA/NCCL/TORCH/OMP/TRITON-prefixed environment values, including new knobs.
The launch also pins prefill rows, grid and production drafter rings. Four slots and q4
affine/group64 weights with a real q4 head are required, including solo cases.

The effective `runtime_sha256` combines both rank fingerprints in rank order, after
disposable warmup of every case has loaded its real prefill/forward extensions. Each binds
the source tree, checkpoints, artifact identities, loaded native Python modules, loaded
module flags, actual transport class/limit, split settings, Torch/CUDA/cuDNN, numerical
flags, loaded weight config, graph keys, and real production pool layout. It contains no
sharding treatment in this off-only contextual arm. It is the same combined hash on both
ranks and must match all validation/qualification receipts and common results. A fresh random
`session_nonce` is agreed through the bootstrap store per real Runtime, outside this stable hash.
Results and phase anchors also bind the actual ordered `campaign_sha256` (single-run: `digest(meta)`)
and `checkpoints={scheme:"file-manifest-v1",target:[rank0,rank1],drafter:[rank0,rank1]}`. Each
checkpoint digest hashes that role/rank's already verified file-manifest array. `layout_checks=0` for P.

## Device entry point, not executed by host checks

The following describes a separately authorized hardware run. It is not an instruction to
stop or replace an existing service. Launch one process on each allocated host using the
existing production environment and transport; rank 1 can start first. No SSH, campaign
or service manager is built into this adapter.

```sh
python3 -m tools.perf.glm_replay.python_adapter gpu-run CAMPAIGN.json \
  --source-root FROZEN_SRC/tensorfold --metadata READY_METADATA.json \
  --artifacts-root ARTIFACTS --target-root TARGET_RANK_FILES --drafter-root DRAFTER_FILES \
  --rank RANK --master EXISTING_BOOTSTRAP_ADDRESS --port ALLOCATED_PORT \
  --receipts-root RECEIPTS --mode validate --output NEW_VALIDATION.json
```

Every local fixture/config/source/checkpoint/artifact check precedes GPU import. The two
ranks then agree fixture, source and metadata identities through the production NCCL
bootstrap store before weights or inference. Both execute the same prescribed schedule
symmetrically with production NCCL/RoCE, split connection warmup and loaders. The NCCL barrier
precedes RoCE construction, matching production. Startup derives prefill rows from the actual
`tensorfold.cuda.geometry.PREFILL_ROWS`/`TF_GLM_PREFILL_ROWS` and calls the frozen
`engine.prompt_grid(rows)`, refusing metadata disagreement. Nonzero `display_kv.credit()` is
refused, rather than silently discarded. There is no fake communicator or alternate follower implementation.

The path is `Segment -> BatchedVerify.forward -> multi.sample_streams(..., None) ->
BatchedVerify.commit -> MultiDrafter.commit([(context, taps[:keep]), ...])`. Selection includes
its declared greedy collective/readback, and predictions never select subsequent input.
All target commits precede one context group per forward. The full context update includes
the production tap staging, FC/norm/KV computation and live ring writes. A failed transaction
cannot be reused. Guards refuse proposal entry points; stale positions or lost/incomplete
context prevent a result rather than being counted as successful work.

Capture calls the real `BatchedVerify.capture()` without a subset argument, preserving the full
production bank **1 through 32** and its graph pool. All needed solo/shared/preparation/unequal
widths must be a subset of that actual bank, which is bound in the runtime fingerprint. Every measured forward
requires one actual graph replay and zero eager fallbacks. Production drafter allocation
and both block/tap graph captures remain present, although the proposer is never called.

Warmup uses disposable logical state. Each block clears every real cache plane, index ring,
slot and drafter ring at their graph-bound addresses, creates fresh `State` views over four
distinct aligned production `Pool` extents, then freshly prefills **all four** contexts and
replays the prescribed all-slot preparation. `State.reset` alone is never treated as a cache
reset. This preserves captured addresses while removing prior cache contents. No cache
snapshot restore, carries, prompt twins or cache eviction occurs in this unsharded path.

## Validation, gates and timing

Validate mode rebuilds fresh state for event-off, repeated event-off, event-on, and the true
production `SerialVerify` one-row teacher-forced oracle. The oracle offers/keeps one tape
row at a time, omitting discarded lookahead. Validation replaces discarded tokens with
deterministically different in-vocabulary IDs; kept tokens and measurement inputs stay
unchanged. At every scheduled committed boundary it
compares each kept logit/tap row, selected IDs, active recurrent bank, conv, committed FP8
codes/scales, complete pooled keys, the live incomplete index pool, host/device positions
and relevant last 2048 drafter K/V rows. Intermediate oracle captures read only taps/logits
and selection; full state is read at original part ends. Physical addresses, inactive banks, discarded and
future rows do not enter comparisons. Both ranks complete each checked forward before
untimed capture. Repeated resets and timing-on/off must preserve outputs, state and dispatch.
The whole poisoned and natural schedules also run without inter-forward completion barriers.
Their kept outputs and final state are checked after completion against oracle-validated
captures. Actual inline logical evidence is canonically SHA-bound and rehashed by the
measurement gate; passing booleans without this evidence cannot authorize measurement.

A generated `glm-replay-python-validation-v2` receipt marks only its fixture correctness,
fresh-prefill and timing-neutrality after actual device execution. It records the actual number
of discarded rows poisoned in each run: zero on full-keep fixtures, positive exactly when partial
coverage exists. Retained evidence includes typed initial states, one-row oracle captures,
kept/final output hashes, dispatch and repeated/timed/unsynchronized run digests. The offline
validator rechecks these hashes, positions, per-row types/counts, oracle agreement and graph/commit
counts; old opaque or boolean-only receipts are refused.

`default_server_neutral` is exactly **`not_applicable_unmodified_engine`**, never `true`, for P.
Metadata v2 requires `server_gate=null`. No Python call uses `server_neutrality.py`. Measurement
requires all 439 pinned source files to match both installed-image listings in the qualification
below. Native still requires its actual server-neutrality gate. These gates do not claim q8/q4 equality.

Production startup deviations are explicit and included in the runtime fingerprint. Fixed equal
capacity and all-slot reservations replace production `admit`/automatic cache budgeting; physical
fit is UNRUN. Full launch/environment store agreement and actual both-rank fingerprints replace
server-specific settings/spill gathers. `GlmEngine._calibrate()` is omitted because it invokes
`Drafter.propose`, forbidden here; no scheduler or draft-choice cost table is instantiated. Disposable
fixed-work warmup remains. This is not a claim to reproduce all normal server startup work.

Use `--mode measure` with the same metadata bindings, populated validation/external gate
artifact references and a separate new output path. It performs disposable warmup followed
by another fresh block. `Runtime.run()` can repeat fresh blocks within one process start;
the `gpu-run` pilot emits one block. The campaign loop supplies distinct block IDs and
external telemetry per block, preserving process-start identity across those calls.

Future telemetry references contain **only** `{"path":"blocks/cell-block.json"}`. Paths are
canonical POSIX paths relative to `--receipts-root`; absolute, escaped, symlinked, reused and
pre-existing measurement destinations are refused. No future telemetry SHA is predeclared.
Both rank realtime start/end timestamps are read outside the timed headline; after completion
the ranks exchange their windows. Rank 0 writes an exclusive `<path>.window.json` request for
a future external controller. Only then does each rank wait (30 seconds, at most 60 via API),
read actual telemetry, compute its file-byte SHA and agree that SHA across ranks.

`telemetry.py` is a pure host parser with no device imports. Its API is:

- `block_window(bindings, ids, ranks, session_nonce, campaign_sha256, checkpoints)` validates a request.
- `reference(spec, root)` validates a path-only future reference; `write_request(spec, root, window)` writes exclusively.
- `validate_telemetry(value, window)` rechecks the typed payload and coverage.
- `wait_telemetry(spec, root, window, timeout_s=30, expected_sha256=None)` returns `(value, computed_file_sha256)`;
  an optional known SHA is checked only for later offline authentication.

`glm-replay-block-window-v1` has exactly `schema,bindings,ids,completed_ranks,ranks,session_nonce,
campaign_sha256,checkpoints`. Bindings are the exact `gate_binding` object: fixture/config/schedule/
tokens SHA256, arm, source SHA256, source-tree SHA256, runtime SHA256 and checkpoints SHA256.
IDs are run/pair/start/order/cell/block. Ranks are ordered `[0,1]`, each exactly
`{rank,realtime_start_ns,realtime_end_ns}` with positive ordered integer nanoseconds.
Checkpoint schemes match the common result contract: `file-manifest-v1` for P and
`tensorfold-full-identity-v1` for native A/B. Both carry rank-ordered target and drafter SHA256 arrays.

`glm-replay-telemetry-v1` has exactly `schema,status,window_sha256,bindings,ids,completed_ranks,
ranks,session_nonce,campaign_sha256,checkpoints`. Status is `measured`; `window_sha256` is the
canonical `digest(window)`, not the request-file SHA. All identities must match. Each ordered rank
contains exactly `rank,device_uuid,realtime_start_ns,realtime_end_ns,max_sample_gap_ns,samples`.
Device UUIDs must match activation evidence. Each sample has exactly `realtime_ns,sm_clock_mhz,
memory_clock_mhz,temperature_millicelsius,power_milliwatts,used_memory_bytes`, all integers with
positive clocks/power/memory and temperature 0–150000. Samples must increase, bracket that rank's
window, include an in-window sample and have gaps no greater than the declared maximum, itself
at most five seconds. Empty prewritten stubs cannot qualify. Missing/invalid telemetry leaves
the block incomplete; no telemetry collector or controller is implemented or run here.

The primary metric is both-rank-completed replay-block wall time, never generation rate.
The headline runs with the observer **off**. A separate fresh identical-work observer-on
anchor supplies phases. Both runs' kept outputs and final state must match the passed
logical evidence after completion. `timing_source` labels the wall as `observer_off` and
phases as `separate_observer_on_anchor`; `anchor_sha256` hashes the actual combined-rank
JSON artifact beneath `OUTPUT.anchors/`, including its newline. Phases are not presented
as observations of the headline block.
Preallocated CUDA events in the anchor observe staging and the real graph's target/normal taps,
tap concatenation/bookkeeping, complete selection, target commit and full context commit.
The graph includes the normal `l2pf.join`; target end follows graph replay, and tap
bookkeeping ends when the existing forward returns. No forward implementation is replaced.
Events include stream waits, host gaps and submission starvation; they are not kernel-busy
time. Do not sum rank intervals or subtract their sum from block wall time.

No rich state reads, correctness comparisons, telemetry or file output occur in measurement
blocks. Ordinary production allocations/transfers remain accounted for; the adapter does
not rewrite them into a faster substitute. Event resolution and rank receipt exchange occur
after the wall end. Actual graph counters and completed context rows are recorded. Python
does not expose native `dr.lost`; its lost audit is zero only when every context end and
target position agrees at each ingest, backed by the separate logical device validation.

Two `glm-replay-result-v1` rank records are validated together by the common validator and
written exclusively by rank 0 after complete peer output. Missing ranks, absent timings,
unpassed gates and schema/config mismatches produce no complete result. Known arena bytes
are measured from actual tensor geometry (Python FP8 rows include 16 bytes of padding);
unmeasured tables/scratch/other/physical peak are null. Existing files are never overwritten.

The module/API split is `python_provenance.preflight`, `python_runtime.Runtime`,
`python_control.Controller`, `python_validation.validate_device`, and
`python_receipts.{measurement_gates,build_record,publish_local}`. `collect_records(runtime,record,anchor)`
and `publish_local(runtime,output,record,anchor)` pass the actual canonical anchor bytes to common
`validate_results(...,anchors={sha:canonical_bytes(anchor)+b"\n"})` before publication. All are external tools;
no native source or common schema ownership moves to this adapter.

## Offline suite qualification

`python_qualification.build_qualification(entries,activation,installed_sources,fixture,meta,runtime_sha256,root)`
reopens and authenticates every reference. `validate_qualification(value,fixture,meta,runtime_sha256,root)`
rebuilds the same result. Default APIs refuse synthetic fixtures; the explicit host-test opt-in does
not change device entry points. No deployed installed-image listing or activation record is retained
or claimed here, so real qualification remains **UNRUN**.

Each suite entry is exactly `{"fixture":{"path":...,"sha256":...},"validation":{"path":...,"sha256":...}}`.
Fixture files are immutable, byte-hashed and revalidated by the common parser. Each actual v2
validation receipt is rechecked with that fixture and common source/tree/runtime/checkpoints.
Both rank runtime digests must agree across the suite. The measured fixture must appear. Coverage
is derived across the suite: solo/shared participation, full/partial forwards, multiple ordered
permutations of the same participating stream set, committed 512-token page crossings and actual
repeated-reset evidence. `owner_swap` is inapplicable to unsharded P and is absent. No individual
solo fixture must attest shared or partial work it did not execute.

The output `glm-replay-python-qualification-v2` has exactly `schema,status,bindings,completed_ranks,
suite,coverage,activation,installed_sources`. Bindings are the common arm/source/source-tree/runtime/
checkpoints subset of `gate_binding`. It has no coverage or hardware booleans supplied by the caller.
The emitted coverage is recomputed on every validation. All retained evidence paths resolve
beneath the receipts root, including the kernel bundle and its members; retain immutable copies
at the metadata's same relative artifact paths when the artifact root differs.

`installed_sources` is exactly two rank-ordered path/SHA references. Each
`glm-replay-python-installed-source-v1` contains exactly `schema,status,rank,source,source_tree_sha256,files`.
Status is `listed`; the entire source object (including engine, kit, interpreter/compiler/kernel and
image/device-image identities) must equal measured metadata. The ordered file listing must match
**all 439** pinned paths, byte counts and SHA256s, including source-only dependencies. Seven-file
listings, missing entries or a mismatch on either rank fail. This driver-free command hashes an
already extracted installed-image tree. The coordinator must establish the extraction/image
identity; pointing it at a source checkout does not establish installed-image provenance.

```sh
python3 -m tools.perf.glm_replay.python_qualification installed-list \
  --installed-root EXTRACTED_IMAGE/src/tensorfold --metadata READY_METADATA.json --rank RANK \
  --image-digest sha256:ACTUAL_IMAGE_SHA256 --device-image-digest sha256:ACTUAL_DEVICE_IMAGE_SHA256 \
  --output RECEIPTS/installed-rank-RANK.json
```

`activation` is a retained `glm-replay-python-activation-v1` observation, with exactly
`schema,status,execution,bindings,completed_ranks,ranks`. It requires `status=observed,execution=device`.
Each rank binds `rank,runtime_rank_sha256,source,environment,device,transport,kernels`. Environment
must pass the adapter's production constraints and hash to that rank's launch environment.
Device has UUID, PCI bus ID, compute capability and driver version; rank UUIDs must differ.
Transport has the actual `kind`, production implementation/bootstrap class, TP world/peer, positive
observed all-gather/barrier counts and send/receive bytes, and observed route-device names. The peer
and transport mode must match that rank and environment. Kernels bind the exact retained bundle
and each compiled artifact's path/bytes/SHA with nonempty launched symbols and positive observed
launch counts; the observed set must equal the complete bundle. These are typed external
observations, not an arbitrary JSON attachment or `production_transport=true`.

The coordinator must collect actual loaded-kernel/transport observations in a later authorized
device window. This host adapter neither observes those activations nor turns source inspection
into a pass. `python_activation.example.json` remains UNRUN; nulls, empty launches, generic flags,
host execution and stale runtime/source/image identities are rejected. The following offline
producer is reproducible after actual evidence exists; running it with the shipped templates
refuses qualification and cannot unblock measurement:

```sh
python3 -m tools.perf.glm_replay.python_qualification build \
  --request QUALIFICATION_REQUEST.json --fixture MEASURED_FIXTURE.json --metadata READY_METADATA.json \
  --runtime-sha256 ACTUAL_COMBINED_RUNTIME_SHA256 --receipts-root RECEIPTS --output RECEIPTS/hardware.json
```

The request is exactly `suite,activation,installed_sources`, as illustrated by the refusing template
`python_qualification.example.json`. Outputs are exclusive. Authenticating hashes protects retained
bytes and bindings; it does not independently prove the truth of an external controller's device
observations. The later controller/activation collection remains a required, unimplemented part
of hardware qualification. The workload/served screens and user decision still precede a full campaign.

## One-session campaigns

Use `campaign` for the full cell matrix and five or more blocks per cell in each real
process start. `gpu-run` remains a single-block pilot. The campaign is a prevalidated
ordered developer case/block list; it does not choose tokens, acceptance or scheduling policy.
`python_campaign.example.json` is an unrun template with relative fixture/metadata references.
Each reference hashes the exact file bytes, while each fixture also carries its canonical
common hashes. Each block has an explicit string `block_id` and path-only future telemetry reference.
A cell may recur in multiple entries for interleaved drift anchors. The cell-to-fixture map is
stable and `(cell_id,block_id)` is unique; different cells may reuse a block string. Metadata carries the cell ID and the same genuine run/pair/start/order
as the top-level campaign; those identifiers cannot change between cases.

```sh
python3 -m tools.perf.glm_replay.python_adapter campaign CAMPAIGN_LIST.json \
  --campaign-root CASE_INPUTS --source-root FROZEN_SRC/tensorfold \
  --artifacts-root ARTIFACTS --target-root TARGET_RANK_FILES --drafter-root DRAFTER_FILES \
  --receipts-root RECEIPTS --rank RANK --mode preflight
```

For an independently authorized device window, add the existing bootstrap `--master` and
`--port`, choose `--mode validate` or `--mode measure`, and supply a new `--output` path.
Validate writes each successful fixture receipt exclusively and immediately into a new directory.
A later failure preserves earlier successes. Duplicate fixture entries in the same run reuse the
retained success after revalidation and byte comparison, without a second run or overwrite.
Measure writes common rank JSONL. Both ranks must consume the same ordered campaign file.

All case files, fixture schedules, metadata, shared identities and measurement gates are
checked before torch import. Shared source/checkpoint/artifact files are hashed once after
all cases prove they refer to the same identity. Measurement requires at least five blocks
per **cell/start in total across ordered entries**, and all case gates must name the same combined runtime hash. Every case
uses fixed H, configured capacity, checkpoint/load/transport configuration and the full production
graph bank. The production pool reserves the maximum aligned extent required by each slot
across the entire campaign; that joint reservation must fit before startup.

One `Runtime` loads the production engine and bootstraps the rank pair once. It warms all
cases on disposable state, seals the actual runtime identities, then executes the exact
case/block order. Every block clears/rebuilds and freshly prefills all four slots and rings,
including inactive solo slots, and runs the fixture's prescribed preparation. There is no
checkpoint reload, synthetic new start ID, hidden block repetition or per-case graph recapture.
Each declared block explicitly runs a phase anchor and observer-off headline; only the
headline gets a common block receipt. Unique exchange keys prevent stale rank receipts.

Only completed, gated two-rank blocks are appended. Earlier valid blocks remain if a later
case fails; the failed/incomplete block cannot acquire a complete receipt. A successful
process start supplies every declared case/block; the external campaign controller must
retain failed-attempt logs and predeclared exclusions. Repeated blocks in this session
remain within-start repeats, not independent starts. The Python context arm still needs
separately coordinated independent process starts for its prescribed replication.
