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
zig-out/bin/tf-glm-replay-identity MODEL DRAFTER
```

The standalone preflight never initializes CUDA. `tf-cuda-test glm-replay-check FIXTURE.json`
also dispatches before `Driver.open`; it accepts explicitly labelled synthetic host controls.
The cross-compile uses absent/inert GPU images when no qualified fatbins are supplied. It is
a host compile check, not a runnable model binary or a device qualification receipt.
The CPU identity producer uses the production full checkpoint identity/cache algorithm and
prints local target/drafter digests with revision qualification explicitly unverified.
Directory names do not establish revisions; obtain both ranks' pins from frozen local stores
and qualify their source revisions separately before any device run.
The host replay suite also invokes `python3` to validate actual native formatter output and
retained anchors with the common validator. That test uses explicitly synthetic fixtures,
invented counters and no driver; it is a format check, not passing device evidence.

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
Oracle preparation also starts with four fresh prefills and replays every kept preparation
row through solo verify/keep/ingest in the declared tape/position order before the next row.

## Metadata and gates

Metadata schema `glm-replay-native-metadata-v2` requires:

- `run_id`, `pair_id`, `start_id`, `start_order` (0 or 1), `block_id` (nonempty string), `cell_id`.
- `fixture_sha256`, `config_sha256` and the common `source` object from [SCHEMA.md](SCHEMA.md).
- `fatbin`, `compiler`: local `{path,sha256}` artifact references.
- `telemetry`: a future `{path}` only, relative to the process working directory.
- `source_manifest`: null for validation, or the authenticated committed source export for measurement.
- `correctness`, `timing_neutrality`, `sharding_equality`, `default_server_neutrality`: null for
  validation, or local `{path,sha256}` gate artifacts for measurement.
- `logical_evidence`: null for validation, or the exact retained logical artifact for measurement.
- `default_server_contract`: null for validation, or the pre-capture server/policy contract artifact.
- Optional `expected_checkpoints`: `{scheme:"tensorfold-full-identity-v1",target:[rank0,rank1],drafter:[rank0,rank1]}`.

The complete required metadata object also has `schema`. Source binary hashes are checked
against the running executable; compiler/fatbin artifacts and tokenizer assets are hashed before
CUDA opens. Gate runtime bindings are compared with the actual combined rank/transport digest
after production startup and before replay preparation. Telemetry destinations must be absent
before startup; a predeclared telemetry hash is refused. Old metadata v1 is refused.
Explicit checkpoint pins require all four 64-lowercase-hex digests. They are schema-checked
and bound in preflight before CUDA opens, then compared with actual gathered identities
on both ranks before inference. Replay alone adds CPU drafter-directory hashing; serving is unchanged.
On first use this reads the full drafter contents and may add entries to the production digest
cache in the serving user's home directory on both hosts. Include that untimed IO and those cache
writes in the coordinator's startup plan. It does not modify the checkpoint or sealed rerun evidence.

Gate schema `glm-replay-native-gate-v1` requires `schema`, `gate`, `status`, `fixture_sha256`,
`config_sha256`, `binary_sha256`, `completed_ranks`, `comparisons`, `kv_shard`, `runtime_sha256`,
`checkpoints`, `evidence_sha256`, `coverage="adversarial-one-row-preparation-and-whole-block-v2"`.
Only `status=pass`, both completed ranks and positive comparisons admit measurement. Validation
retains logical evidence before emitting gates that hash those exact bytes; compare produces
sharding equality bound to the corresponding arm's logical artifact. Old unbound gates fail.
Default-server token/performance neutrality uses the separate server-identity schema and offline
producer in [SERVER_EVIDENCE.md](SERVER_EVIDENCE.md), rather than the replay gate schema/runtime.
The host implementation supplies no passing device gates.
Actual target and drafter checkpoint identities bind the stable loaded runtime and logical
evidence as well as gates. Missing checkpoint receipts and changed weights on either rank fail.

## Post-block telemetry and source export

After committing the harness, export the exact committed source inventory on the build host:

```sh
python3 -m tools.perf.glm_replay.native_source_manifest --repo . --output native-source.json
```

The exporter refuses dirty or untracked source files and checks historical-base ancestry.
Bind that file's SHA-256 through `source_manifest` and use its commit as `source.harness_commit`.
The inventory, executable, compiler and fatbin identities are separate; host cross-compilation
with inert images supplies no qualified GPU binary or fatbin identity.

Start a separately coordinated telemetry collector before replay. After the observer-off block
completes and correctness readbacks finish, the runner exclusively writes
`telemetry.path + ".window.json"`. A complete temporary file is published atomically without
replacing an existing request. It then waits at most 30 seconds for the controller's receipt.
The window follows `glm-replay-block-window-v1` in [telemetry.py](telemetry.py): fixture/config/
schedule/token hashes, arm, canonical source hash, source-inventory file hash, runtime,
canonical native checkpoint-object hash, IDs, nonce, ordered campaign hash and each actual
rank's positive realtime start/end. These realtime windows enclose the monotonic headline wall;
they are not used to calculate throughput. Telemetry names also enter startup agreement.

The controller joins already collected samples to those windows and atomically publishes
`glm-replay-telemetry-v1`, `status="measured"` at the future path. Each rank needs its actual
GPU UUID, at least two ordered clock/temperature/power/memory samples bracketing its window,
an in-window sample, and a declared maximum sample gap no greater than five seconds.
The runner checks actual gathered UUIDs, both rank windows and every binding, then hashes
the exact bytes it parsed into both JSONL records. Missing samples, stubs, stale windows,
old sessions and reused or colliding paths fail. Paths must be canonical relative POSIX names,
with no symlink components. Result and request destinations must be distinct across cases.
No telemetry scrape or file operation runs inside the timed block. The collector and real
telemetry remain unrun; host negative controls use explicitly invented samples.
Retain the source inventory, window request and telemetry alongside JSONL and anchors: the
source-inventory hash is linked through the window/telemetry and ordered campaign hash rather
than copied into the JSONL. Offline rank agreement requires the same telemetry hash on both lines;
the common result validator still does not reopen telemetry files. Native-only logical snapshots
retain internal byte-digest representations; a future non-native consumer must normalize them first.

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
Every JSONL/anchor includes a fresh random `session_nonce`, the Session's ordered
preflight/settings `campaign_sha256`, and actual both-rank `checkpoints`. The nonce stays
outside stable runtime and correctness bindings. Each JSONL uses its own rank's actual
`wall_ns` and headline `layout_checks`; elapsed clocks are exchanged after the timed completion
boundary. `preparation_ns` on both records is explicitly rank 0's leader-clock preparation duration.
Validation preflights all three absent output paths through parent aliases before opening CUDA;
exclusive writes preserve previous attempts, including files or dangling symlinks created meanwhile.

Off uses `2H` pages; on uses `2H+4`, retaining equal `2H` logical pages and the production part
floor. Actual owner placement is asserted, prompt twins are returned, and carries/twins must
be zero. Rank-local owner sends are checked per unique forward, rather than per request.

Device correctness, live TP2 transport, owner activation, timing/default-server neutrality,
physical fit and performance remain **UNRUN** until the hardware coordinator executes them.
