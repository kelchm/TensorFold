// CUDA 13 / PTX 8.8+: separate sm_120a / sm_121a image; FP4/FP6 occupy padded byte lanes, without block scales.
#include "roofline_mma.cuh"

#if __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
RF_MMA(tf_roof_f4, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32", float, "+f")
RF_MMA(tf_roof_f6, "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32", float, "+f")
#endif
