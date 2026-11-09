// Gate 3: GLM's EXL3 decode-window kernels' planted faults (contract_exl3_decode.zig), each red against grouped_1.
#pragma once

#include "exl3_grouped.cuh"

extern "C" __global__ void tf_ctl_exl3_decode_order(tf_exl3::GroupedArgs a) {  // k8r8 summing its warps last to first
    tf_exl3::grouped_body_one<1, 4, 4, tf_exl3::PLAIN, 8, 8, true>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}
