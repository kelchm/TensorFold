# GLM replay fixture and result schemas

## Canonical hashes

Canonical JSON = UTF-8 of json.dumps(value, ensure_ascii=True, sort_keys=True,
separators=(",", ":"), allow_nan=False), no trailing newline. Keys are ASCII; strings
may be Unicode (escaped by the serializer). JSON numbers in fixtures are integers;
booleans are distinct from integers. SHA-256 digests are lowercase 64 hex characters.
Reject duplicate JSON keys, nonfinite numbers, unknown fields and booleans as counts.
fixture_sha256 = SHA256(canonical fixture with ONLY fixture_sha256 omitted).
schedule_sha256 = SHA256(canonical {"preparation":preparation,"forwards":forwards}).
tokens_sha256 = SHA256(canonical [{"stream_id":id,"prompt_tokens":array,"tape_tokens":array},...])
in stream_id order. config_sha256 = SHA256(canonical config).
source/rendered string hashes hash raw UTF-8 bytes, not JSON strings.

## Fixture object (glm-replay-fixture-v1)

Required top-level keys: schema, fixture_id, kind (synthetic|campaign), family
(prose|code|pi), provenance, config, streams, preparation, forwards,
tokens_sha256, schedule_sha256, config_sha256, fixture_sha256.
provenance = {target_revision, drafter_revision, tokenizer, sources}.
tokenizer = {kind:synthetic-byte|huggingface-local, revision (campaign must equal target_revision), vocab_size,
files:[{path,sha256}], template, template_sha256}. File paths are relative, not local paths.
sources = [{source_id, attribution, license, text, sha256}]. Authored sources Apache-2.0;
rendered Pi-style conversations are authored public synthetic histories, not private logs.
config = {speed:"standard", tp:2, slots:4, dense_bits:8, head_bits:8, kv:"fp8",
reset:"fresh_prefill", selection:"greedy", proposer:false, active_streams:[ids],
placement:"balanced"|"reversed", pool:{h,page_tokens,configured_capacity,prompt_twins}}.
No sharding flag in fixture: identical fixture across native off/on. Python q4
is a contextual arm, not a change to the common native baseline config.

streams: exactly four distinct objects, ordered by stream_id 0..3:
{stream_id, owner:0|1, source_id, rendered_prompt, rendered_tape,
rendered_prompt_sha256, rendered_tape_sha256, prompt_encoding, prompt_tokens, tape_tokens}.
prompt_encoding = {token_ids:[full one-time encoding of rendered_prompt], used_tokens:N}.
prompt_tokens equals prompt_encoding.token_ids[:N]; N may be shorter than the rendered
reservoir for exact 8K/32K/128K contexts. Retain full encoding to make slicing explicit.
tape_tokens is the one-time encoding of rendered_tape (no token slicing/re-tokenization).
All IDs are in [0,vocab_size); campaign vocab_size must equal pinned GLM 154880. Synthetic-byte fixtures never claim pinned model tokenization.
Owners balanced=[0,0,1,1], reversed=[1,1,0,0]; actual admission must equal requested owner.

preparation and forwards: arrays of {round_id, forward_id, parts:[part,...]}.
Each array starts its contiguous round_id and forward_id at 0 (separate untimed/timed spaces).
part = {stream_id, position, tape_offset, offered, kept}.
1<=offered<=16, 1<=kept<=offered, sum(offered)<=32, no repeated stream per forward.
Timed forwards contain all active streams once in prescribed order. Preparation forwards
contain ALL FOUR configured streams once, including solo cases. Generator preparation uses
width=keep=1 for each stream; preparation remains an explicit independent schedule. Solo is one part;
shared is multiple parts. Native MUST use solo verify for one part.
Offsets start at 0 and advance only by kept across preparation then timed forwards.
position = len(prompt_tokens)+tape_offset. Inputs = tape_tokens[offset:offset+offered].
Next forced anchor = tape_tokens[offset+kept]; require this extra token even on final full keep.
Discarded lookahead is offered again from the next committed offset, never appended twice.
Counter denominator is unique forward count, never stream participation count.

Pool controls: off page argument=2H, on=2H+4, equal logical pages=2H.
Production per-part minimum=max(131072,configured_capacity//4) tokens, no synthetic bypass.
Reserve ceil((prompt length+maximum offered/anchor extent)/page_tokens) pages per stream,
plus prompt_twins times ceil(prompt length/page_tokens), separately per actual owner.
H must cover both owners' reservations and production minimum. Twins returned before timing.

## Results (glm-replay-result-v1 JSONL)

One self-contained rank block receipt per line, emitted AFTER both-rank block completion.
Required keys: schema, fixture_sha256, schedule_sha256, tokens_sha256, config_sha256,
run_id, pair_id, start_id, start_order (0|1), block_id, cell_id, arm (A|B|P), rank (0|1),
status (complete|excluded), exclusion_reason (null|string), source, effective_config,
effective_config_sha256, preparation_ns, wall_ns, completed_ranks:[0,1], forwards,
checks, memory, telemetry_sha256.
source={engine_commit,harness_commit,binary_sha256,fatbin_sha256,compiler_sha256,
kit_commit,image_digest,device_image_digest}; all commits full 40 hex, all digests full.
effective_config={engine:"native"|"python",speed:"standard",tp:2,slots:4,
dense_bits,head_bits,kv:"fp8",kv_shard:bool,multi_lone:0,
pool_h,page_argument,page_tokens,configured_capacity,dsa_layers,
local_heads,latent_width,row_dtype_bytes,query_bytes_per_row,output_bytes_per_row,
graph_rows:[positive total-row keys]}.
A native q8 off; B native q8 on; P Python q4 off. Same source identities for native A/B.
checks={correctness:true,timing_neutral:true,default_server_neutral:true,
fresh_prefill:true,proposer_calls:0,lost:0,carries:0,twins:0}.
memory={logical_pages,key_page_bytes,latent_page_bytes,row_page_bytes,
tables_bytes,scratch_bytes,other_bytes,peak_physical_bytes}. logical_pages/key_page_bytes/latent_page_bytes/row_page_bytes
are nonnegative integers; tables_bytes/scratch_bytes/other_bytes/peak_physical_bytes may be
null when unmeasured, never fabricated zero. Physical peak, if provided, covers known allocations.
telemetry_sha256 identifies an external post-block telemetry receipt (never timed scraping).
forwards = [{forward_id,round_id,parts,phases_ns,owner,graph_replays,eager_fallbacks,
target_commits,context_commits,ingest_calls,proposer_calls,lost}].
parts EXACTLY equals fixture timed parts, no reorder. target_commits=context_commits=sum(kept),
ingest_calls=1 (one group, even shared), proposer_calls=lost=0.
phases_ns = {staging,target,tap_bookkeeping,selection,target_commit,context_commit};
each numeric finite nonnegative duration. Rank-local intervals may overlap; never sum ranks.
owner={layer_visits,rows,query_send_bytes,output_send_bytes}. B expects D,D*R visits/rows.
Rank-local query sends = D * offered rows whose owner != rank * query_bytes_per_row;
output sends = D * offered rows whose owner == rank * output_bytes_per_row.
Both row byte sizes = local_heads * latent_width * row_dtype_bytes (BF16=2).
A/P all owner counters zero. Do not multiply all rows by both send widths on each rank.
P graph_replays=1 per forward and eager_fallbacks=0, requires graph_rows containing R.
Native graph_replays=0 and eager_fallbacks=0 (no Python graph claims).

Offline validation rejects malformed/hash/config/vocab/owner/schedule mismatches, incomplete
rank sets, duplicate blocks/ranks, missing/unmatched A/B start/block/cell pairs, changed
native A/B build identities, unbalanced start order and unequal work. Excluded attempts
are retained but not analyzed; an excluded pair needs both ranks and a reason.
Aggregate blocks within each start/cell/arm first (mean synchronized block wall time),
then report paired-start B/A ratios and B-A ns, all per-pair values, and a Student-t interval
across independent starts (three pairs is only three independent units). Do not pool rounds,
streams or blocks as independent replicates. P context has separate start summaries.

## Import API

from tools.perf.glm_replay import load_fixture, validate_fixture, canonical_bytes, digest
from tools.perf.glm_replay import validate_results, summarize_paired
load_fixture(path, allow_synthetic=False, expected=None) validates before returning dict;
expected optional mapping binds fixture/config/token/schedule hashes and tokenizer/model IDs.
validate_fixture(fixture, allow_synthetic=False, expected=None) returns fixture.
validate_results(records, fixtures) -> validated list; fixtures keyed by fixture_sha256.
summarize_paired(records, fixtures, minimum_pairs=3, minimum_blocks=5) -> dict.
The generator and command examples are documented in README.md. Result APIs additionally
accept allow_synthetic=True explicitly for host tests only. Expected bindings also support
tokenizer_files_sha256 = digest(tokenizer.files).
