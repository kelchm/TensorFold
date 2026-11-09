# GLM CUDA prompt MLA experiment

This branch adapts MiaAI-Lab's engine patches 0005, 0011, 0041 and 0068 from
`GLM-5.3-Flash-DGX-Sparks-1104@13f2464` to `e6f522d05`.
The target is GLM-5.3-Flash on two GB10s (sm_121), TP=2, with FP8 KV.
There is no local CUDA device or nvcc; device correctness, compiler resources,
model fidelity and speed remain unqualified.

The FP8 chunk reader fetches eight E4M3 codes at once and overlaps tile + 2's reads
with the current tile's math. Conversion, FP32 row scaling and BF16 rounding stay
unchanged. This replacement is default-on whenever FP8 KV is selected.
Both chunk widths require exactly 256 threads and trap on another block width.
The BF16 attention arithmetic also stays unchanged.

A sparse call whose chunk partials would not fit its buffers (`latent.fuses`) runs `tf_glm_chunks_{16,32}{,_fp8}_fused` instead of the chunk kernel and `tf_glm_merge`. Each block (row, head block) computes the row's chunks in order with the chunk kernel's own code (`chunk_softmax`), then folds each into its running o, m and l with `tf_glm_merge`'s operations in its order, so the output bytes are unchanged and no FP32 partials are written. Count-0 rows stay untouched, as with the merge. Prompt buffers now size their partials for dense calls only, so a sparse prompt call wider than they hold merges in the kernel; with 32 heads a rank and a context of at least 2,560, that frees 867.375 MiB of partials at 4,096 rows and 1,895.375 MiB at 8,192. Dense passes and decode windows keep the chunk kernels and the merge. This is default-on and needs no setting.

`TF_GLM_ABSORB_MMA=1` selects the source patch's head-batched BF16 tensor-core
GEMMs for a prompt's aligned absorb_q and expand_v calls at 64 or more rows.
Unset, blank or `0` retains the upstream SIMT dispatch; other values are refused.
Decode windows, shorter prompt tails and unaligned pointers retain SIMT.
The MMA kernels change FP32 accumulation order, so their arithmetic is opt-in.
The profile registry labels this arithmetic-changing, and fingerprint layout 20
requires every rank to agree before weights load. Ordinary, paired and batched
prompt passes carry the setting.

The adapted GEMMs retain 128x128 tiles, K32 steps, three cp.async stages and the
stored weight layouts. They use upstream's common cp.async and MMA helpers.
Source-patch speed and exactness receipts were on an RTX PRO 6000, not GB10;
those receipts do not qualify this port or establish a GB10 speedup.

## Contract coverage

The CUDA contract runner adds FP8 dense and sparse replacements at both head
widths. It compares FP32 partials and merged BF16 outputs with the frozen
`e6f522d05` byte reader, including key and row tails, empty sparse rows and
exceptional inputs. Code-pair swap controls must fail the shapes gate.
The softmax fixture does not prove accumulator precision; gate 4 states that
limitation explicitly.

The four fused kernels register as byte replacements of the sparse chunk kernel then `tf_glm_merge`, with BF16 and E4M3 rows at both head widths, an empty row and planted NaN and infinity inputs. Gate 4 replays their sums against the FP64 attention reference. Controls that leave out a row's last chunk must fail the shapes gate. `tf-cuda-test glm-ref` also compares fused and unfused calls byte for byte at 1, 16, 70 and 300 rows with counts from 0 to 2,051.

Absorb and expand register separately against independent FP64 head projections,
with SIMT entries on the same fixtures. The fixture distinguishes FP16/BF16 sums
in all three reduction orders. Next-head weight controls must fail gate 3.
Launch geometry is registered for production kernels and controls. All new or
changed compile-resource receipts are pending; previous register counts are not
reused as evidence for the changed source.

## Hardware checks still required

1. Compile `fatbins kernel-contract-fatbins` with the qualified nvcc for
   `-Dsm=80,86,89,120,121`, then rederive records with
   `tf-cuda-test kernel-contract-limits zig-out/fatbin`. Record the nvcc release,
   register usage, spills and shared-memory limits, including sm_121.
2. Run the complete CUDA kernel contract and `tf-cuda-test glm-ref`, retaining its SUMMARY and planted-control
   results. Check the frozen-reader and fused-merge byte replacements and both FP64 projection
   classes. Compare MMA and SIMT RMS and worst per-column FP64 errors separately;
   require no worse native error before qualifying the arithmetic opt-in.
3. At TP=2 with FP8 KV, compare unset and `=0` against the base for chunk partials,
   latent outputs, KV, final logits and token hashes. Check prompt/decode seams,
   dense and sparse attention, ordinary/paired/batched prompts, short tails,
   draft/plain equality, resumed/fresh and concurrent/solo equality.
4. Repeat with `=1`, retaining teacher-forced top-token changes and margins and
   independent projection errors. Do not describe the opt-in as bit-identical.
   Check that ranks with different settings refuse startup.
5. Alternate base/candidate cold 2k/8k/32k/64k prompt runs with fixed token arrays,
   checkpoint/runtime/settings and fresh KV (`--prompt-cache-gib 0`). Record MLA
   time per row, absorb/expand timings, total TTFT and decode cost. Compare the
   candidate opt-in with the older patched deployment's opt-in as well.
