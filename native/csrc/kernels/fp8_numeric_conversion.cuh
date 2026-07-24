/*
 * Minimal FP8 conversion helpers used by wLLM elementwise kernels.
 *
 * Derived from SageAttention numeric_conversion.cuh:
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define WLLM_FP8_CAST_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define WLLM_RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define WLLM_RUNTIME_ASSERT(x) assert(0 && x)
#endif

__device__ __forceinline__ void floatx4_to_e4m3x4(
    uint32_t* dest, float* source0, float* source1) {
#ifdef WLLM_FP8_CAST_ENABLED
  asm volatile(
      "{\n"
      ".reg .b16 lo;\n"
      ".reg .b16 hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(dest[0])
      : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]),
        "f"(source1[1]));
#else
  WLLM_RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ int8_t float_to_int8_rn(float x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
}

#undef WLLM_RUNTIME_ASSERT
