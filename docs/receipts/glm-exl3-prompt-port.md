# GLM EXL3 routed prompt kernels

This is an unqualified performance experiment based on `e6f522d05f9b95caab32c34541d9a64f4034e61d`.
Target: GLM-5.3-Flash CUDA, mcg 4-bit routed experts, 4,096 model channels, 1,024 expert channels per rank,
288 experts, top-8, TP=2 on two GB10s (sm_121). No checkpoint revision or device performance is qualified here.
The base already makes one common EXL3 call per prompt pass, capped by `cuda_state.solo_rows`.

## Ported behavior

- Kit patch 0007: down writes pair `p` at `(p / 8) * 9 + p % 8`, preserving the shared expert's ninth slot.
  GLM omits its temporary FP32 output and the output `cuMemcpy2D`; pick compaction and the grouped fallback retain their copies.
  At 4,096 rows this eliminates 512 MiB of temporary output per rank, by size calculation.
- Kit patch 0078: the GLM-only kernels specialize for 1–8 blocks of eight members, stage decoded fragments ahead,
  precompute async-copy addresses, extract the same MCG states, use exact-sign butterflies, and pack output stores.
  The eight-half-bit specialization is retained; the GLM loader and common prompt eligibility still admit only mcg 4-bit.
- Kit patches 0026/0027 and 0028 were already present in the base: grouped load-ahead on sm_120+, its replacement
  contracts and compiler records, and non-allocating read-only trellis loads. They were not replayed.
- Patches 0003/0004 were not dependencies: no 3-bit admission, k6 entry points or mixed-width prompt dispatch is added.

`tf_exl3_prompt_gu` and `tf_exl3_prompt_down`, their helper source and ABI, and all grouped kernels remain unchanged.
New symbols are `tf_exl3_prompt_down_slots`, `tf_exl3_prompt_gu_glm` and `tf_exl3_prompt_down_glm`.
Only GLM calls `runGlm`. Other EXL3 families retain the common kernels and compact output. Rebuild the EXL3 image
alongside the executable, since symbol resolution now includes the additional kernels.

## Exactness argument and checks

Each output retains one FP32 `mma.m16n8k16` chain in ascending K-tile order, starting from zero. Specializations remove
only blocks without members. Lookahead changes when bytes arrive and decode, preserving their addresses and states.
The MCG multiply, mask, XOR and half-pair sum are unchanged. Each Hadamard butterfly retains its order:
`fmaf(+1, v, o)` or `fmaf(-1, v, o)` has an exact product and one final rounding, as the original add/subtract does.
Activation rounding, clipping, scaling and output conversions retain their order. Vector stores pack the same values;
GLM refuses a destination not aligned to 16 bytes, or intermediate output not aligned to eight bytes.

`contract_exl3_prompt.zig` registers the new launches against retained common kernels. It tests all item counts 1–64,
multiple items, shuffled members, shared and separate input rotations, both activation modes, GLM projection widths,
nonfinite inputs, repeated bytes, input immutability, guard bytes and untouched shared slots. The baseline down output
is scattered on the host. Planted controls drop the final member or use an eight-slot output stride.
Independent FP64 constant-weight stress exercises the full projections and epilogues: FP16 accumulators overflow.
BF16 epilogue rounding can hide balanced reduced-precision sums, so the contract explicitly declines that stronger proof.
These are registered device checks, not executed device receipts. The host target compiles every fixture hook even when
empty CUDA images would prune the main runner. Recording-driver tests cover one-pass execution and the next chunk's
input, pick and direct-output offsets, reduced scratch, vector alignment refusal and the unchanged grouped fallback.

## Kit 0079 resources

The source kit's nvcc 13.3 measurements, in SM order 80 / 86 / 89 / 120 / 121, were:

| Original symbol | Registers per thread | Static shared bytes | Launch bound |
| --- | --- | --- | --- |
| `tf_exl3_prompt_gu` | 122 / 125 / 125 / 122 / 122 | 43,264 | 256 |
| `tf_exl3_prompt_down` | 123 / 125 / 125 / 122 / 122 | 43,264 | 256 |
| `tf_exl3_prompt_gu_k6` | 128 / 128 / 128 / 128 / 128 | 40,192 | 256 |
| `tf_exl3_prompt_down_k6` | 128 / 128 / 128 / 122 / 122 | 40,192 | 256 |

Patch 0078 reported sm_120 gate/up 128→122 registers and down 127→122, no spills, two blocks per SM, shared memory
unchanged. Patch 0007 reported unchanged slot-output usage on sm_120 (4-bit down 127 registers / 43,264 shared bytes).
Those are historical measurements of different entry points and a different build, not measurements of this port.
Kit 0079's live resource overrides are deliberately not copied. The new symbols register pending usage under every
compiler release, and a host test verifies they cannot borrow old measurements to pass qualification.
The retained common kernels keep their existing records. Re-derive new records with the actual five-SM fatbin.

## Local validation

Zig 0.17.0 on macOS ARM64; no nvcc, CUDA device, or remote hardware used. Commands below use that Zig binary.

```sh
zig build compile-cuda-tests -Dtarget=aarch64-linux-gnu --summary failures -j8
zig build test-cuda-host test-exl3-cuda-host test-exl3-prompt-host test-exl3-prompt-contract-host test-glm-prompts-host test-qwen35-exl3-admission-host test-qwen35-exl3-owner-host --summary all -j8
zig fmt --check <changed Zig files>
python3 tools/zig/lean_check.py
python3 tools/zig/lean_new.py BASE.txt HEAD.txt
git diff --check
```

The cross-build passes. The focused prompt host and contract targets pass 11/11 tests. The full host command passes
361/361 steps, with 3,776/3,805 tests passing and 29 skipped (cached results affect the displayed test counts).
Formatting and diff checks pass. Lean exits 1 with 399 inherited findings;
comparison against the untouched base reports 399 before, 399 after, zero new findings and exits 0.
Every changed source module is at most 600 lines.
A plain `zig build --summary failures -j8` fails because `xcrun` cannot find the Metal compiler (exit 72).
Initial new-fixture assertions that overstated BF16 stress discrimination were corrected before the passing checks.
Expected diagnostic output from the unchanged DeepSeek chat controls also reproduces in a pristine base archive;
that focused baseline target exits 0.

## Required hardware run

Build both images and runner with the same nvcc, flags and checkpoint revision, then run the complete CUDA contract,
including every planted control. Derive limits with `kernel-contract-limits` for sm_80/86/89/120/121; record registers,
static shared bytes, launch bounds, spills and occupancy. On sm_121, verify shared-memory accounting (including the
reserved block bytes), four-stage shared-input and three-stage separate-input async-copy lifetimes and barriers.
Compare retained and new kernels byte for byte for Xd and Y, all item tails, uneven/skewed routing and real GLM widths.
Use fresh/cold 2k/4k/8k/32k/64k prompts as admitted, TP=2, matching baseline/candidate checkpoint and settings, and
compare stage timings, whole-prefill times, memory, logits and token hashes. Check chunk seams, greedy/drafted output,
resume and concurrent/solo equality. Run other EXL3 families, including Qwen3.5, to confirm their existing dispatch,
bytes and performance; grouped decode and other architectures retain their previous kernels.
