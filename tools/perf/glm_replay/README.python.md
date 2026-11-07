# External Python replay adapter

This adapter imports the frozen Python engine; it does not edit or copy its forward,
commit, scheduler, arithmetic, kernels or transport. It supplies prescribed fixture rows
and keeps to the existing production boundaries. Python arm P is q4 deployment context.
Native A/B remains the primary sharding comparison. A native raw verify call includes
selection/readback; a Python raw forward call does not. Their raw timings are not equivalent.

**Status: host implementation and host doubles only. All device, TP transport, correctness,
timing neutrality, default-server neutrality, physical-fit and performance gates are unrun.**
No campaign fixtures, model assets or hardware gate passes ship here. Host tests cannot
qualify device behavior. The parent must integrate `runtime_sha256`, `timing_source` and
the shared server gate validator before device mode starts. No frozen engine patch is required.

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
and every other newly pinned dependency remain **local-source-only**, requiring hardware
qualification; the adapter does not invent earlier installed hashes. Paths are relative.
Missing, changed, added, escaped or symlinked files fail closed. Imports are restricted to
the verified tree, rehashed at import and compiled from source without reading cached bytecode
or writing bytecode into the frozen tree. An already-imported torch/engine is refused.

## Metadata and provenance

`python_metadata.example.json`, `python_validation.example.json` and
`python_gate.example.json` and `python_server_gate.example.json` are **unrun templates** with null unknown
identities. They cannot pass preflight or authorize device measurement. Populate a new
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
ranks and must match all validation/default-server/hardware receipts and common results.

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
symmetrically with production NCCL/RoCE, normal split connection warmup, loaders and engine
initialization. There is no fake communicator or alternate follower implementation.

The path is `Segment -> BatchedVerify.forward -> multi.sample_streams(..., None) ->
BatchedVerify.commit -> MultiDrafter.commit([(context, taps[:keep]), ...])`. Selection includes
its declared greedy collective/readback, and predictions never select subsequent input.
All target commits precede one context group per forward. The full context update includes
the production tap staging, FC/norm/KV computation and live ring writes. A failed transaction
cannot be reused. Guards refuse proposal entry points; stale positions or lost/incomplete
context prevent a result rather than being counted as successful work.

Capture keys are total offered rows: solo 1/4/8, shared 4/16/32, plus all preparation/unequal
widths. Four-row preparation and the one-row reference are included. Every measured forward
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

A generated validation receipt can mark only its fixture correctness, fresh-prefill and
timing-neutrality checks as passed after actual device execution. It leaves default-server
and general hardware qualification **unrun**. Measurement requires that receipt plus an
independently retained hardware gate for production transport/kernel bundle, full/partial,
solo/shared, permutations, page edges, repeated resets and owner-swap coverage. Correctness
and hardware gates bind fixture/config/source, checkpoint, runtime and **arm P**.
The separate server-neutrality artifact binds actual original/instrumented server binaries,
harness commit, fatbin, token evidence and a predeclared performance policy. It does not
equate server identities with the replay interpreter or runtime. Its deferred shared API is
`server_neutrality.validate_server_gate(artifact, source, "python", roots=...)`.
These gates are not cross-engine q8/q4 equality claims.

Use `--mode measure` with the same metadata bindings, populated validation/external gate
artifact references and a separate new output path. It performs disposable warmup followed
by another fresh block. `Runtime.run()` can repeat fresh blocks within one process start;
the `gpu-run` pilot emits one block. The campaign loop supplies distinct block IDs and
external telemetry per block, preserving process-start identity across those calls.

An external telemetry artifact is read/hashed only after both ranks finish the block. It
must bind this runtime/fixture/arm and block ID and identify both ranks. The campaign caller
must arrange for this receipt to exist before finalization; the adapter does not scrape
telemetry or invent zero temperatures/footprints. Missing evidence fails without a complete
measurement output. External gates/evidence live beneath `--receipts-root` and are hashed.

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
`python_receipts.{measurement_gates,build_record,publish_local}`. All are external tools;
no native source or common schema ownership moves to this adapter.

## One-session campaigns

Use `campaign` for the full cell matrix and five or more blocks per cell in each real
process start. `gpu-run` remains a single-block pilot. The campaign is a prevalidated
ordered developer case/block list; it does not choose tokens, acceptance or scheduling policy.
`python_campaign.example.json` is an unrun template with relative fixture/metadata references.
Each reference hashes the exact file bytes, while each fixture also carries its canonical
common hashes. Each block has its explicit unique string `block_id` and external telemetry
artifact reference. Metadata carries the cell ID and the same genuine run/pair/start/order
as the top-level campaign; those identifiers cannot change between cases.

```sh
python3 -m tools.perf.glm_replay.python_adapter campaign CAMPAIGN_LIST.json \
  --campaign-root CASE_INPUTS --source-root FROZEN_SRC/tensorfold \
  --artifacts-root ARTIFACTS --target-root TARGET_RANK_FILES --drafter-root DRAFTER_FILES \
  --receipts-root RECEIPTS --rank RANK --mode preflight
```

For an independently authorized device window, add the existing bootstrap `--master` and
`--port`, choose `--mode validate` or `--mode measure`, and supply a new `--output` path.
Validate writes a new directory containing one bound validation JSON artifact per fixture.
Measure writes common rank JSONL. Both ranks must consume the same ordered campaign file.

All case files, fixture schedules, metadata, shared identities and measurement gates are
checked before torch import. Shared source/checkpoint/artifact files are hashed once after
all cases prove they refer to the same identity. Measurement requires at least five blocks
in **each** case, and all case gates must name the same combined runtime hash. Every case
uses fixed H, configured capacity, checkpoint/load/transport configuration and one union
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
