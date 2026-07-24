#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace wllm_native::kernels {

void qwen3_vl_bf16_gemv_m1(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int N,
    int K,
    cudaStream_t stream);

}  // namespace wllm_native::kernels
