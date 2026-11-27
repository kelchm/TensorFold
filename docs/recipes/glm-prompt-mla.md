# GLM CUDA prompt MLA experiment

This branch adapts MiaAI-Lab's engine patches 0005, 0011 and 0041 from
`GLM-5.3-Flash-DGX-Sparks-1104@13f2464` onto TensorFold `v1.3.6` (with GLM PDL
`glm_pdl::enter()` preserved on every kernel).
The target is GLM-5.3-Flash on two GB10s (sm_121), TP=2, with FP8 KV.
Device correctness, model fidelity and speed on live GB10 remain to be verified.

The FP8 chunk reader fetches eight E4M3 codes at once and overlaps tile + 2's reads
with the current tile's math. Conversion, FP32 row scaling and BF16 rounding stay
unchanged. This replacement is default-on whenever FP8 KV is selected.
Both chunk widths require exactly 256 threads and trap on another block width.
The BF16 attention arithmetic also stays unchanged.

`TF_GLM_ABSORB_MMA=1` selects the source patch's head-batched BF16 tensor-core
GEMMs (`tf_glm_latent_mma_kn`, `tf_glm_latent_mma_nk`) for a prompt's aligned
`absorb_q` and `expand_v` calls at 64 or more rows (non-roped GLM-5.3-Flash).
Unset, blank or `0` retains the upstream SIMT dispatch; other values are refused.
Decode windows, shorter prompt tails, unaligned pointers and full roped GLM-5.3
(`rd != 0`) retain their existing paths.
The MMA kernels change FP32 accumulation order, so their arithmetic is opt-in.
The profile registry labels this arithmetic-changing, and fingerprint layout 35
requires every rank to agree before weights load. Ordinary, paired and batched
prompt passes carry `ranks.Numerics` (covering both `TF_GLM_ABSORB_MMA` and
`TF_GLM_EXL3_BF16_SLOTS`).

The adapted GEMMs retain 128x128 tiles, K32 steps, three cp.async stages and the
stored weight layouts. They use upstream's common cp.async and MMA helpers and
open with `glm_pdl::enter()` without `__restrict__` inputs so dependent launches
never hoist non-coherent loads above `ACQBULK`.
Source-patch speed and exactness receipts were on an RTX PRO 6000, not GB10;
those receipts do not qualify this port or establish a GB10 speedup.

## Contract coverage

The CUDA contract runner covers FP8 dense and sparse replacements at both head
widths (`contract_glm_fp8_attn.zig`) and fused sparse chunk merge (`contract_glm_fused_attn.zig`).
It compares FP32 partials and merged BF16 outputs with the frozen byte reader,
including key and row tails, empty sparse rows and exceptional inputs.
Code-pair swap controls must fail the shapes gate.

Absorb and expand register separately against independent FP64 head projections
(`contract_glm_mma.zig`, `glm_mla_ref.zig`), with SIMT entries on the same fixtures.
The fixture distinguishes FP16/BF16 sums in all three reduction orders. Next-head
weight controls (`tf_ctl_glm_mma_kn`, `tf_ctl_glm_mma_nk`) must fail gate 3.
Launch geometry is registered in `glm_mla_records.zig` for production kernels and
controls.

## Hardware checks still required

1. Compile `fatbins kernel-contract-fatbins` with the qualified nvcc for
   `-Dsm=80,86,89,120,121`, record exact 5-SM compiler resources and verify 0 PDL
   early reads under `tools/pdl_early_loads.py`.
2. Run the complete CUDA kernel contract on live GB10, retaining its SUMMARY and
   planted-control results. Compare MMA and SIMT RMS and worst per-column FP64
   errors separately.
3. At TP=2 with FP8 KV, compare unset and `=0` against the base for chunk partials,
   latent outputs, KV, final logits and token hashes across ordinary, paired,
   batched and solo prompt passes.
4. Repeat with `TF_GLM_ABSORB_MMA=1` and `TF_GLM_EXL3_BF16_SLOTS=1`, retaining
   teacher-forced top-token changes and margins. Do not describe either opt-in as
   bit-identical. Verify that ranks with mismatched settings refuse startup.
5. Alternate base/candidate cold prompt runs with fixed token arrays, checkpoint,
   runtime settings and fresh KV (`--prompt-cache-gib 0`) to measure MLA and MoE
   speedups and total TTFT on GB10.
