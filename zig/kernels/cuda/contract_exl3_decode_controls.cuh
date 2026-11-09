// Gate 3: GLM's EXL3 decode-window kernels' planted faults (contract_exl3_decode.zig), each red against grouped_1.
#pragma once

#include "exl3_grouped.cuh"

extern "C" __global__ void tf_ctl_exl3_decode_order(tf_exl3::GroupedArgs a) {  // k8r8 summing its warps last to first
    tf_exl3::grouped_body_one<1, 4, 4, tf_exl3::PLAIN, 8, 8, true>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_ctl_exl3_decode_shfl_src(tf_exl3::GroupedArgs a) {  // k8r8s: a word from the wrong lane
    tf_exl3::grouped_body_one<1, 4, 4, tf_exl3::SHFL_SRC, 8, 8>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_ctl_exl3_decode_shfl_a(tf_exl3::GroupedArgs a) {  // k8r8s: the last tile's stale A fragment
    tf_exl3::grouped_body_one<1, 4, 4, tf_exl3::SHFL_A, 8, 8>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}
