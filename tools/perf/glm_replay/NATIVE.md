# Native GLM replay

The local experiment is based on historical `e52eb42b52b1e50f3f23a5eae8115bbe80c5e7d9`.
It retains dense/head 8/8, FP8 paged KV, TP2, four configured slots and DFlash2 q4 allocations.
The same patched binary provides both sharding arms. No proposer runs in replay.
The full hardware campaign is conditional on the workload-shape and served-sharding screens
and the user's decision in [DECISION.md](DECISION.md). Host readiness does not book a window.

## Host commands

Use the exact Zig version from `.zig-version`:

```sh
zig build compile-glm-replay test-glm-replay-host glm-replay-host
zig build compile-cuda-tests -Dtarget=aarch64-linux-gnu
zig-out/bin/tf-glm-replay-fixture FIXTURE.json
```

The standalone preflight never initializes CUDA. `tf-cuda-test glm-replay-check FIXTURE.json`
also dispatches before `Driver.open`; it accepts explicitly labelled synthetic host controls.
The cross-compile uses absent/inert GPU images when no qualified fatbins are supplied. It is
a host compile check, not a runnable model binary or a device qualification receipt.

## Coordinated device commands, unrun in host implementation

Each rank uses its ordinary production rank/transport environment. Launch and service ownership
remain with the hardware coordinator. The commands below do not authorize a hardware window.

```sh
tf-cuda-test glm-replay validate FIXTURE.json MODEL DRAFTER OUTPUT_PREFIX METADATA.json
tf-cuda-test glm-replay-compare OFF.logical.json ON.logical.json GATE_PREFIX
tf-cuda-test glm-replay measure FIXTURE.json MODEL DRAFTER OUTPUT.jsonl METADATA.json
```

Validation uses a true one-row teacher-forced reference, commits context before another verify,
and compares both ranks' logical state, kept logits/taps and live drafter ring. It replaces every
discarded lookahead ID with a different deterministic token and checks whole schedules without
per-forward synchronization as well as timing disabled/enabled. Compare checks native off/on equality at actual
owners. The one-block measure command is useful for a pilot; a campaign must preserve genuine
process starts and repeated blocks, rather than assigning one start ID to unrelated processes.
The [campaign entry point](../../../docs/recipes/glm-replay-campaign.md) runs an ordered case list
within one actual rank session.

## Metadata and gates

Metadata schema `glm-replay-native-metadata-v1` requires:

- `run_id`, `pair_id`, `start_id`, `start_order` (0 or 1), `block_id` (nonempty string), `cell_id`.
- `fixture_sha256`, `config_sha256` and the common `source` object from [SCHEMA.md](SCHEMA.md).
- `fatbin`, `compiler`, `telemetry`: local `{path,sha256}` artifact references.
- `correctness`, `timing_neutrality`, `sharding_equality`, `default_server_neutrality`: null for
  validation, or local `{path,sha256}` gate artifacts for measurement.
- `logical_evidence`: null for validation, or the exact retained logical artifact for measurement.

The complete required metadata object also has `schema`. Source binary hashes are checked
against the running executable; compiler/fatbin artifacts and tokenizer assets are hashed before
CUDA opens. Gate runtime bindings are compared with the actual combined rank/transport digest
after production startup and before replay preparation. Frozen external telemetry is authenticated
before startup and rechecked after completion.

Gate schema `glm-replay-native-gate-v1` requires `schema`, `gate`, `status`, `fixture_sha256`,
`config_sha256`, `binary_sha256`, `completed_ranks`, `comparisons`, `kv_shard`, `runtime_sha256`,
`evidence_sha256`, `coverage="adversarial-and-whole-block-v1"`.
Only `status=pass`, both completed ranks and positive comparisons admit measurement. Validation
retains logical evidence before emitting gates that hash those exact bytes; compare produces
sharding equality bound to the corresponding arm's logical artifact. Old unbound gates fail.
Default-server token/performance neutrality uses the separate server-identity schema and offline
producer in [SERVER_EVIDENCE.md](SERVER_EVIDENCE.md), rather than the replay gate schema/runtime.
The host implementation supplies no passing device gates.

## Measurement boundary

All four slots start from fresh prefill and the declared untimed preparation, even in solo cases.
Each round calls production solo `Leader.verify` or multi `Leader.verifyShared`, normal selection,
all target keeps and exactly one grouped DFlash2 ingest. Both ranks finish before headline wall
timing ends. Events are preallocated and resolved afterwards; JSON, rich correctness readbacks
and telemetry stay outside the block. Phase spans include normal waits and submission gaps.
Each case runs fresh warmup, a separate observed phase anchor and a fresh observer-off headline.
Every completed block's kept outputs and final logical state must match the loaded reference.
`OUTPUT.jsonl.anchor.json` retains both ranks' phase and headline completion records, including
block counter deltas and layout checks. Its file hash binds the JSONL timing source. Phase
intervals describe the anchor and cannot be summed into or subtracted from headline wall time.

Off uses `2H` pages; on uses `2H+4`, retaining equal `2H` logical pages and the production part
floor. Actual owner placement is asserted, prompt twins are returned, and carries/twins must
be zero. Rank-local owner sends are checked per unique forward, rather than per request.

Device correctness, live TP2 transport, owner activation, timing/default-server neutrality,
physical fit and performance remain **UNRUN** until the hardware coordinator executes them.
