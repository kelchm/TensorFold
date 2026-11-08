# CUDA hardware ceilings

Build the CUDA test runner and its benchmark images in the Linux CUDA 13 image:

```sh
zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
./zig-out/bin/tf-cuda-test roofline 1
```

The optional argument is positive seconds of GPU event time per measurement (default 1, maximum 3600).
Device 0 is the CUDA-visible ordinal. Run without a model process or competing GPU workload.
The 4 GiB memory case needs two 4 GiB device allocations, plus small probe outputs.
Keep the binary with its image build; a prebuilt `-Dfatbins=DIR` set must also contain
`roofline.fatbin` and `roofline_low.fatbin`.

Every measurement prints `ROOFLINE name best unit median=...` and timing and workload details.
Three warm-up launches and an event calibration precede graph construction.
Each batch targets approximately 10 ms of work (at most 1024 launches).
One graph warm-up precedes at least seven samples and the requested total GPU event time.
Best and median are per-launch rates; `repeats` counts graph samples and `batch` counts launches per sample.
Allocation, input generation, module loading, heuristic selection and validation are outside the measured intervals.
Kernel/GEMM graph replay avoids host submission gaps; GPU launch overhead remains part of each measured operation.
The `cuMemcpyDtoD` reference uses direct default-stream batches and includes any host submission gaps.

- Memory: 32, 64, 128, 256, 512, 1024, 2048 and 4096 MiB per buffer, plus a power of two at most L2/4.
  Reads use explicit 16-byte `.cg` loads, four integer sums and a warp reduction to an observable output.
  Writes use explicit 16-byte `.wb` stores of an index-dependent pattern; copies use both instructions.
  Read and write count one buffer's bytes; copy and `cuMemcpyDtoD` count both read and written bytes.
  The L2 case's two buffers fit within half the reported L2 capacity; repeated launches warm that working set.
  Its custom kernels make 32 passes per launch to amortize launch cost; other memory cases make one pass.
  Cases at most four times L2 are labelled `cached-*`; only larger cases contribute to `summary-memory-*`.
  GB/s means decimal billions of logical bytes per second, not hardware memory-controller counter traffic.
- Tensor: dense `mma.sync`, FP16/BF16 to FP32 at m16n8k16, E4M3 to FP32 and INT8 to INT32 at m16n8k32.
  Four independent accumulator fragments issue eight times each per round, for 4096 rounds.
  Inputs stay in registers; each thread stores one final sum. FMA counts as two operations.
  Blocks have eight warps; the grid has two waves of the driver-reported maximum resident blocks over all SMs.
  Loaded functions must report zero local bytes; registers and local bytes accompany each measurement.
  The printed launch and occupancy identify the geometry; this is an observed ceiling, not a saturation proof.
- FP4/FP6: separate `sm_120a`/`sm_121a` image with m16n8k32 `kind::f8f6f4`, E2M1 and E3M2 respectively,
  FP32 accumulation, and values in padded byte lanes (FP4 in bits 2..5, FP6 in bits 0..5).
  These are unscaled operations, not block-scaled NVFP4 peaks.
  Other architectures, absent eligible images and unsupported driver images print a skip.
- CUDA cores: eight independent FP32 `fma.rn` chains, eight issues per chain per round, 4096 rounds.
  Only a final sum is stored; both tensor and FMA sums are checked against their exact synthetic results.
- GEMM: BF16 and, when cuBLASLt admits it, tensorwide E4M3, FP32 accumulation and BF16 output, scales one.
  Shapes (M,N,K) are (4096,4096,4096), (8192,8192,8192) and (2048,{4096,8192,16384},4096).
  Up to 32 heuristic candidates compete using three warm event samples, within 64 MiB workspace.
  Nonzero signed synthetic inputs have exactly representable FP32 sums; nine output locations are checked
  against FP64 and independent BF16 rounding. Unsupported types/shapes and absent libraries print a skip.

Summary peaks are the best observed rate per category; each associated median belongs to that winning test.
They are dense operation counts, with no sparsity multiplier. Score engine stages against the appropriate
memory, register-compute or achievable-GEMM ceiling, using the same thermal and power conditions.
Device facts include SM count, L2 capacity, driver clock attribute and optional NVML current/max SM and memory clocks.
NVML selects by CUDA UUID; unsupported queries are explicit. A driver clock attribute is not a live clock sample.

These probes have no served shapes and are linked only into `tf-cuda-test`, like `probe.cu`'s launch-cost kernels.
They do not register model operators or qualify engine arithmetic through the kernel contract.
The engine kernel catalog, image requirements and inference paths are unchanged.

The instruction forms and architecture requirements were checked against the
[CUDA 13.0 PTX ISA](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
and the FP8 TN layout/default tensorwide scales against the
[CUDA 13.0 cuBLASLt guide](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltmatmul).
Local host builds do not assemble CUDA or establish GB10 throughput; nvcc compilation and GPU execution remain required.
