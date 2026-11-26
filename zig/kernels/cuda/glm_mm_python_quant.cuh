// CUDA load-time encoding of the GLM Python family's qmm.quantize4_mse_raw and make_f8.
// One warp owns a contiguous row group. Output is row-major for the tiler.
#include <cuda_fp8.h>

__device__ __forceinline__ float least_clipping_error(float x0, float x1, float s, float b) {
    float q0 = fminf(15.f, fmaxf(0.f, rintf(__fdiv_rn(__fsub_rn(x0, b), s))));
    float q1 = fminf(15.f, fmaxf(0.f, rintf(__fdiv_rn(__fsub_rn(x1, b), s))));
    float e0 = __fsub_rn(__fadd_rn(__fmul_rn(q0,s), b), x0);
    float e1 = __fsub_rn(__fadd_rn(__fmul_rn(q1,s), b), x1);
    float err = __fadd_rn(__fmul_rn(e0,e0), __fmul_rn(e1,e1));
#pragma unroll
    for (int d=16; d; d>>=1) err = __fadd_rn(err,__shfl_down_sync(0xffffffffu,err,d));
    return __shfl_sync(0xffffffffu,err,0);
}

extern "C" __global__ void glm_mm_quant20(const __nv_bfloat16* src, unsigned char* codes,
                                           uint32_t* meta, long long groups) {
    glm_pdl::enter_weights();
    const long long g=(long long)blockIdx.x*(blockDim.x/32)+(threadIdx.x/32);
    if (g>=groups) return;
    const int l=threadIdx.x&31;
    const float x0=__bfloat162float(src[g*64+l]), x1=__bfloat162float(src[g*64+l+32]);
    float low=fminf(x0,x1), high=fmaxf(x0,x1);
#pragma unroll
    for (int d=16; d; d>>=1) {
        low=fminf(low,__shfl_down_sync(0xffffffffu,low,d));
        high=fmaxf(high,__shfl_down_sync(0xffffffffu,high,d));
    }
    low=__shfl_sync(0xffffffffu,low,0);
    high=__shfl_sync(0xffffffffu,high,0);
    float best=INFINITY, bs=0.0f, bb=0.0f;
    // torch's Python scalar clipping sequence is computed in double and used in float.
#pragma unroll 1
    for(int i=0;i<16;++i){
        const float a=(float)(1.0-0.45*(double)i/15.0);
        const float lo=__fmul_rn(low,a), hi=__fmul_rn(high,a);
        const float s=__bfloat162float(__float2bfloat16_rn(fmaxf(1.0e-8f,__fmul_rn(__fsub_rn(hi,lo),1.0f/15.0f))));
        const float b=__bfloat162float(__float2bfloat16_rn(lo));
        const float err=least_clipping_error(x0,x1,s,b);
        if(err<best){best=err; bs=s; bb=b;}
    }
    const int q0=min(15,max(0,__float2int_rn(__fdiv_rn(__fsub_rn(x0,bb),bs))));
    const int q1=min(15,max(0,__float2int_rn(__fdiv_rn(__fsub_rn(x1,bb),bs))));
    const int hi0=__shfl_down_sync(0xffffffffu,q0,1);
    const int hi1=__shfl_down_sync(0xffffffffu,q1,1);
    if ((l&1)==0){
        codes[g*32+l/2]=q0|(hi0<<4);
        codes[g*32+16+l/2]=q1|(hi1<<4);
    }
    if(l==0) meta[g]=(uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(bs))|
                      ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(bb))<<16);
}

extern "C" __global__ void glm_mm_quant24(const __nv_bfloat16* src, unsigned char* codes,
                                           float* scales, long long groups) {
    glm_pdl::enter_weights();
    const long long g=(long long)blockIdx.x*(blockDim.x/32)+(threadIdx.x/32);
    if(g>=groups) return;
    const int l=threadIdx.x&31;
    float x[4], hi=0.0f;
#pragma unroll
    for(int j=0;j<4;++j){x[j]=__bfloat162float(src[g*128+j*32+l]); hi=fmaxf(hi,fabsf(x[j]));}
#pragma unroll
    for(int d=16;d;d>>=1) hi=fmaxf(hi,__shfl_down_sync(0xffffffffu,hi,d));
    const float scale=fmaxf(1.e-30f,__fmul_rn(__shfl_sync(0xffffffffu,hi,0),1.0f/448.0f));
#pragma unroll
    for(int j=0;j<4;++j) codes[g*128+j*32+l]=__nv_cvt_float_to_fp8(fminf(448.f,fmaxf(-448.f,__fdiv_rn(x[j],scale))),__NV_SATFINITE,__NV_E4M3);
    // A 128-column scale repeated for the two 64-column tiles.
    if(l==0){scales[g*2]=scale; scales[g*2+1]=scale;}
}
