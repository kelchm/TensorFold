# Fixed-work GLM replay host tools

These tools prepare immutable token fixtures and validate offline replay receipts. They do
not load an engine, call a model, start a service, or establish device correctness. Native
and external Python adapters consume the exact same arrays and ordered forward schedule.
All work uses standard speed. Native arithmetic remains dense/head 8/8, TP2 and FP8 paged KV;
the recorded Python q4 profile is a contextual comparison.

The schemas and exact hash rules are in [SCHEMA.md](SCHEMA.md). Python 3.11+ is sufficient
for validation, generation of synthetic host fixtures, tests and analysis. Real tokenization
also requires a locally installed `transformers` package and the pinned checkpoint's assets.
Nothing downloads assets or executes remote tokenizer code.
Real rendering parses serialized tool-call arguments into objects for the pinned template,
while retaining the authored source unchanged. Tokenizer entries may be fewer than the model's
padded 154880-row ID capacity; every tokenizer ID must fit that capacity.

## Host checks

From the repository root:

```sh
python3 -m unittest discover -s tools/perf/glm_replay/tests -v
python3 -m tools.perf.glm_replay validate-fixture \
  --fixture tools/perf/glm_replay/fixtures/host-prose.json \
  --fixture tools/perf/glm_replay/fixtures/host-code.json \
  --fixture tools/perf/glm_replay/fixtures/host-pi.json --allow-synthetic
```

The three shipped examples use synthetic UTF-8 byte IDs with vocabulary 256. They are small
host controls, explicitly marked `kind=synthetic`; the default validator refuses them for
campaign/device use. They contain no claimed GLM model tokenization or performance evidence.
The Pi example has unequal widths, partial commits and alternating part order. Its source
is a newly authored tool conversation, not a private Pi transcript or a copied Pi template.

The corpus in `histories.py` is newly authored public prose, Python examples, and Pi-style
read/bash conversation histories. Each family has four distinct topics, rendered prompts,
continuation tapes and source identities. Their retained source text, attribution and
Apache-2.0 license travel inside every fixture. JSON string `text` encodes the exact UTF-8
source bytes (with no newline normalization); its SHA-256 hashes those bytes directly.
The example file checksums are in `fixtures/manifest.json`.

## Real fixture generation

Real campaign fixtures are intentionally absent until pinned local assets are provided.
Use a dedicated tokenizer directory containing only files named in an independently pinned
manifest. Every file is hashed before `AutoTokenizer` is imported or constructed. Both
`tokenizer.json` and `tokenizer_config.json` are required; include the template, added-token,
special-token or config assets if present. Paths in the manifest are relative to that directory.
Its `revision` must be `078455ffe6472f9a52fbc1139f58b9db2881b25c`:

```json
{"schema":"glm-replay-tokenizer-assets-v1","revision":"078455ffe6472f9a52fbc1139f58b9db2881b25c","files":[{"path":"tokenizer.json","sha256":"<64 lowercase hex>"},{"path":"tokenizer_config.json","sha256":"<64 lowercase hex>"}]}
```

```sh
python3 -m tools.perf.glm_replay.generate campaign-prose-8k.json \
  --family prose --tokenizer-dir TOKENIZER_DIR --tokenizer-manifest ASSETS.json \
  --context-tokens 8192 --rounds FROZEN_ROUNDS --preparation-rounds 2 \
  --widths 4,4,4,4 --pool-h FROZEN_H --configured-capacity FROZEN_CAPACITY
python3 -m tools.perf.glm_replay validate-fixture --fixture campaign-prose-8k.json
```

Use the same frozen H and configured capacity across cells sharing a process start. Derive
H from the largest campaign fixture and both owners' reservations, including temporary prompt
twins, continuation and lookahead. The host helper `capacity.minimum_h` calculates the bound;
`capacity.validate_pool` returns the two page arguments and owner reservations. H must also
satisfy the production minimum of 131072 tokens per part and capacity/4. Synthetic examples
preserve that gate; their large declared pool is a reservation calculation, not an allocation.
Actual owner admission, physical fit, twin return and zero carries require device checks.

The generator renders with the retained pinned template and encodes each prompt and tape
exactly once. To select an exact context length without repeated tokenization it retains the
complete `prompt_encoding.token_ids`, then sets `prompt_tokens` to the explicit prefix of
`used_tokens` IDs. `rendered_prompt` describes that full reservoir; a prefix may end within a
message. It is a token-history replay fixture, not a request-body round-trip claim. Increase
`--repetitions` if a rendered reservoir is too short; the generator fails instead of padding
with fabricated tokens. Adapters never tokenize, decode/re-encode, or append predicted IDs.

`--active-streams 0` or `2` selects a solo owner control, with four configured slots preserved.
Untimed preparation always covers all four streams with width=keep=1; timed forwards contain
only the selected active streams. `--placement reversed` swaps both owners. Full-keep widths
1/4/8 cover the screening matrix; shared rows must total at most 32. Explicit `--keeps` and
`--permute` generate partial/unequal validation schedules. Discarded lookahead is reoffered
from the committed tape offset. Every part has a following forced anchor, including the last.

The generator refuses to overwrite existing fixtures. Preserve frozen fixtures and manifests
with the campaign. Pilot block lengths, repetitions and exclusions must be frozen before
interpreting paired native results.

For the complete screening/follow-up matrix and partial/owner controls, use the bundle generator.
It tokenizes each family's prompt/tape once per context, then reuses those arrays across shapes.
The manifest is written only after every fixture passes validation; a failed directory is retained
without a completion manifest. Use a new output directory for a corrected attempt.

```sh
python3 -m tools.perf.glm_replay.bundle NEW_BUNDLE \
  --tokenizer-dir TOKENIZER_DIR --tokenizer-manifest ASSETS.json \
  --asset-provenance VERIFIED_ASSET_RECEIPT.json \
  --pool-h 1280 --configured-capacity 262144 --rounds FROZEN_ROUNDS
```

This produces 51 fixtures: the 42 screening/follow-up cells plus nine partial, reversed-owner
and solo-owner controls. The template comes from the hashed tokenizer directory. To reproduce
the retained deployment, explicitly stage its serving override as `chat_template.jinja` and
include that file's hash in ASSETS.json; do not silently substitute the different HF template.
The chosen pool satisfies host reservation checks for short diagnostic schedules. Revalidate
headroom after the pilot fixes longer blocks; host admission calculations do not establish
physical fit. Native command and gate boundaries are described in [NATIVE.md](NATIVE.md).

## Offline receipts and paired-start analysis

```sh
python3 -m tools.perf.glm_replay validate-results --fixture campaign-prose-8k.json \
  --results rank0.jsonl --results rank1.jsonl
python3 -m tools.perf.glm_replay summarize --fixture campaign-prose-8k.json \
  --results rank0.jsonl --results rank1.jsonl --output report.json
```

Both ranks supply complete post-block receipts with exact fixture/schedule/config/token
hashes, build and compiler identities, all-rank completion, timings, graph paths and per-forward
commit/context/owner counters. Native paired arms must use the same binary and all other
settings, with off=2H and on=2H+4 page arguments. Owner byte counts are rank-local sends:
queries for peer-owned rows, outputs for locally owned rows. Shared forwards count once.

The validator rejects incomplete schedules, duplicate ranks/blocks, unmatched native arms,
changed work/builds/settings, unexpected graph fallback, wrong owner geometry and unbalanced
start order. Excluded attempts require a reason and both-rank receipts; they stay in the
report and remove the corresponding block from both arms. Incomplete rank data are a failed
measurement, not an automatically excluded success. Keep failed raw attempts separately.

The report uses max(rank wall intervals), never their sum. It reports milliseconds per unique forward and replay throughput, and averages matched block times
within each start before calculating B/A and B-A, exposes every paired-start effect and
uses Student-t intervals across independent starts (log scale for ratios). At least three
native pairs and five valid blocks per cell/start are required. A thousand rounds or extra
blocks in one boot cannot replace independent starts. An interval overlapping one is called
inconclusive. With only three pairs the interval relies strongly on assumptions; every pair
remains visible. Python context starts are separate and require at least two starts.

Phase durations remain separate rank-local intervals. They can overlap and include waits or
submission starvation; no phase sum is presented as busy time or subtracted to invent CPU
cost. Preparation is outside replay wall time. Missing physical peak/tables/scratch/other
measurements use `null`, and exact row-page allocation bytes remain separate from logical
capacity. Telemetry hashes identify external post-block receipts; this host code does not
validate their contents or scrape telemetry.

## Adapter imports and remaining gates

```python
from tools.perf.glm_replay import load_fixture, offered_tokens, next_anchor
fixture = load_fixture("campaign-prose-8k.json", expected={"vocab_size": 154880})
for forward in fixture["forwards"]:
    for part in forward["parts"]:
        token_ids = offered_tokens(fixture, part)
        anchor_id = next_anchor(fixture, part)
```

The import surface has no GPU, Torch, model or tokenizer dependency. Call `load_fixture`
with expected hashes/model/tokenizer/template/vocabulary bindings before engine loading or
collectives. Result validation also binds effective configuration and schedule. Common
hashing/validation utilities may be imported by the external Python adapter; that adapter
is maintained separately from these files and never changes the frozen Python engine.

Pinned-model tokenization, GPU/model correctness, production TP transport, timing neutrality,
server neutrality, live owner counters, graph activation, physical memory and replay
performance are **UNRUN**. Passing host tests or accepting externally supplied gate booleans
does not establish those facts. Hardware runs remain separately coordinated.
