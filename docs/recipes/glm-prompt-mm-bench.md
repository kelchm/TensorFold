# GLM packed prompt projection benchmark

`tf-cuda-test glm-prompt-mm-bench [bits] [seconds]` runs on device 0 without a checkpoint or a second rank.
Bits default to 4; 6 and 8 are also accepted. Seconds default to 1 per projection, row count and path.
Every projection runs at M = 2048, 4096 and 8192. On a CUDA build host, using Zig 0.17.0:

```sh
zig build -Dtarget=aarch64-linux-gnu -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j8
./zig-out/bin/tf-cuda-test glm-prompt-mm-bench 4 1
./zig-out/bin/tf-cuda-test glm-prompt-mm-bench 8 1
```

Existing matching fatbins can replace nvcc with `-Dfatbins=DIR`. The compile-only check
`zig build compile-cuda-tests -Dtarget=aarch64-linux-gnu --summary failures -j8` needs neither nvcc nor a GPU;
it builds a runner with absent kernel images and does not install it into `zig-out/bin`.

## Shapes

The shape list uses `Config{}` and the CUDA loader's `Share.of` for TP=2. Indexer heads, q_a and kv_a are replicated.
Shapes follow `cuda_tile.layerLinears`, `cuda_kda_block`, `cuda_dsa_block`, `cuda_mlp` and `cuda_moe`.
Each listed call occurs once in every layer of its family. The two gate projections are separate calls.

| Projection | N × K per rank | Layers | Output |
| --- | --- | ---: | --- |
| KDA q/k/v/f_a/g_a/b stack | 12576 × 4096 | 34 | BF16 |
| KDA f_b, g_b (each) | 4096 × 128 | 34 | BF16 |
| KDA output | 4096 × 4096 | 34 | FP32 |
| MLA q_a/kv_a stack | 2048 × 4096 | 11 | BF16 |
| MLA q_b | 8192 × 1536 | 11 | BF16 |
| MLA indexer key/head weights | 160 × 4096 | 11 | BF16 |
| MLA indexer q_b | 4096 × 1536 | 11 | BF16 |
| MLA output | 4096 × 8192 | 11 | FP32 |
| Shared gate/up | 2048 × 4096 | 42 | BF16 |
| Shared down | 4096 × 1024 | 42 | FP32 |
| Dense gate/up | 12288 × 4096 | 3 | BF16 |
| Dense down | 4096 × 6144 | 3 | FP32 |

KDA f_b and g_b retain the served stride of 12576 BF16 elements and offsets 12288 and 12416.
The packed path calls `b16.matmul` with the forward's packed registry and prompt flag, selecting `glm_mm{bits}_128s2`.
Its activation input is already BF16; this path performs no activation conversion. cuBLASLt receives identical
activations, strides, BF16 dequantized weights and the same output type, with FP32 compute.
Packing, tiling, allocations, heuristic selection and validation are outside the timed regions.

## Results and checks

The benchmark reuses roofline's CUDA-event graph batches, three warmups, at least seven samples and 10 ms target
batches. cuBLASLt selects the fastest measured supported candidate from up to 32 heuristics with 64 MiB workspace.
`best_ms` and `median_ms` are per-call GPU times. TFLOPS counts `2*M*N*K`; peak percentages divide by the supplied
GB10 BF16 ceiling of 126.5 TFLOPS. `packed_over_cublaslt` divides packed latency by cuBLASLt latency; lower is faster.
`repeats`, `batch` and `gpu_ms` expose the sampling budget.

Family sums multiply each call's latency and operation count by its layer count. `per_layer` divides that family sum
by its count. The backbone sum covers all 45 layers and reports the average per backbone layer too. These are sums
of isolated warm-buffer measurements, including the sparse indexer query in every MLA layer, not an end-to-end pass;
summed bests and medians are estimates, not statistics from a captured full layer. Head/MTP, routed experts, routers,
MLA absorb/expand, nonlinearities and communication are excluded. The served prompt head only projects the last row.
FP8 is not measured by this command.

Deterministic signed fixtures use varied power-of-two scales and the loader's qpack quantizer. Dequantization must
reproduce every BF16 weight exactly, avoiding extra BF16 weight rounding in the reference. Every output is checked
before and after graph timing against cuBLASLt, with `abs(error) <= 1e-4 + rtol*abs(reference)` and rtol = 0.008 for
BF16 output or 0.0001 for FP32. All outputs must be finite. Nine reference outputs per case also check independent
FP64 dots. The report includes maximum absolute error and maximum fraction of the allowed bound; any mismatch fails.

Local cross-compilation does not establish device correctness or throughput. Run both bit widths on the target GPU
before interpreting the measurements.
