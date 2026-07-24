# wLLM native C++ runtime path.
#
# This tree is intentionally separate from runtime/:
#   runtime/ = stable C ABI hand-off (wrt_runtime_export_v1)
#   cpp/     = native model runtime implementation layers
#
# Build:
#   cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build -j

cmake_minimum_required(VERSION 3.18)
project(wllm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(CTest)
include(GNUInstallDirs)

option(WLLM_CPP_WITH_EXEC "Build/link the exec layer for native replay" ON)
option(WLLM_CPP_WITH_CUDA_STAGING "Enable conservative CUDA H2D/D2H modality staging" ON)
option(WLLM_CPP_WITH_CUDA_KERNELS "Enable CUDA modality kernels" ON)
option(WLLM_ENABLE_NATIVE_CPP
       "Build opt-in native C++ operation and model extensions" OFF)
option(WLLM_CPP_WITH_PI05
       "Build the PI0.5 native C++ model frontend" OFF)
option(WLLM_CPP_WITH_SENTENCEPIECE
       "Enable native SentencePiece text tokenization" OFF)
option(WLLM_CPP_WITH_PI05_SM120_TARGET
       "Build the PI0.5 SM120 frontend target" OFF)
option(WLLM_CPP_WITH_PI05_SM110_TARGET
       "Build the PI0.5 SM110 frontend target" OFF)
if((WLLM_CPP_WITH_PI05 OR WLLM_CPP_WITH_SENTENCEPIECE) AND
   NOT WLLM_ENABLE_NATIVE_CPP)
  message(FATAL_ERROR
    "Native tokenizer and model frontends require "
    "WLLM_ENABLE_NATIVE_CPP=ON")
endif()
if((WLLM_CPP_WITH_PI05_SM120_TARGET OR
    WLLM_CPP_WITH_PI05_SM110_TARGET) AND
   NOT WLLM_CPP_WITH_PI05)
  message(FATAL_ERROR
    "PI0.5 hardware targets require WLLM_CPP_WITH_PI05=ON")
endif()
if(WLLM_ENABLE_NATIVE_CPP AND NOT WLLM_CPP_WITH_EXEC)
  message(FATAL_ERROR
    "WLLM_ENABLE_NATIVE_CPP requires WLLM_CPP_WITH_EXEC=ON")
endif()
if(WLLM_CPP_WITH_PI05_SM120_TARGET AND
   WLLM_CPP_WITH_PI05_SM110_TARGET)
  message(FATAL_ERROR
    "PI0.5 SM110 and SM120 targets use different CUDA compilation profiles; "
    "build one deployment target per build tree")
endif()
if(WLLM_CPP_WITH_PI05_SM110_TARGET)
  set(WLLM_CPP_PI05_CUDA_ARCH 110)
elseif(WLLM_CPP_WITH_PI05_SM120_TARGET)
  set(WLLM_CPP_PI05_CUDA_ARCH 120)
endif()
set(WLLM_CPP_FA2_LIBRARY "" CACHE FILEPATH
    "Path to the Python-free libwllm_fa2_raw shared library")
set(WLLM_CPP_CUTLASS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass" CACHE PATH
    "Path to CUTLASS for the PI0.5 SM110 frontend target")
if(WLLM_CPP_WITH_CUDA_STAGING)
  find_package(CUDAToolkit REQUIRED)
endif()
if(WLLM_CPP_WITH_CUDA_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(WLLM_CPP_WITH_PI05_SM120_TARGET)
  if(NOT WLLM_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM120_TARGET requires CUDA kernels")
  endif()
  if(NOT WLLM_CPP_WITH_CUDA_STAGING)
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM120_TARGET requires CUDA staging")
  endif()
  if(NOT WLLM_CPP_WITH_EXEC)
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM120_TARGET requires the exec layer")
  endif()
  if(TARGET wllm_fa2_raw)
    add_library(wllm_cpp_fa2_external ALIAS wllm_fa2_raw)
  else()
    if(NOT WLLM_CPP_FA2_LIBRARY)
      find_library(WLLM_CPP_FA2_LIBRARY
        NAMES wllm_fa2_raw
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../wllm_native
        NO_DEFAULT_PATH)
    endif()
    if(NOT WLLM_CPP_FA2_LIBRARY OR
       NOT EXISTS "${WLLM_CPP_FA2_LIBRARY}")
      message(FATAL_ERROR
        "The PI0.5 SM120 target requires the opt-in FA2 native C library; "
        "build wllm_fa2_raw or set WLLM_CPP_FA2_LIBRARY")
    endif()
    add_library(wllm_cpp_fa2_external SHARED IMPORTED GLOBAL)
    set_target_properties(wllm_cpp_fa2_external PROPERTIES
      IMPORTED_LOCATION ${WLLM_CPP_FA2_LIBRARY})
  endif()
endif()
if(WLLM_CPP_WITH_PI05_SM110_TARGET)
  if(NOT WLLM_CPP_WITH_CUDA_KERNELS)
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM110_TARGET requires CUDA kernels")
  endif()
  if(NOT WLLM_CPP_WITH_CUDA_STAGING)
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM110_TARGET requires CUDA staging")
  endif()
  if(NOT WLLM_CPP_WITH_EXEC)
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM110_TARGET requires the exec layer")
  endif()
  if(NOT EXISTS
     "${WLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "WLLM_CPP_WITH_PI05_SM110_TARGET requires CUTLASS; set "
      "WLLM_CPP_CUTLASS_DIR to a compatible checkout")
  endif()
endif()
if(WLLM_CPP_WITH_SENTENCEPIECE)
  find_path(WLLM_SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h)
  find_library(WLLM_SENTENCEPIECE_LIBRARY sentencepiece)
  if(WLLM_SENTENCEPIECE_INCLUDE_DIR AND WLLM_SENTENCEPIECE_LIBRARY)
    add_library(wllm_sentencepiece_external INTERFACE)
    target_include_directories(wllm_sentencepiece_external
      INTERFACE ${WLLM_SENTENCEPIECE_INCLUDE_DIR})
    target_link_libraries(wllm_sentencepiece_external
      INTERFACE ${WLLM_SENTENCEPIECE_LIBRARY})
    set(WLLM_CPP_SENTENCEPIECE_TARGET wllm_sentencepiece_external)
  else()
    include(FetchContent)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      sentencepiece
      GIT_REPOSITORY https://github.com/google/sentencepiece.git
      GIT_TAG v0.2.1)
    FetchContent_MakeAvailable(sentencepiece)
    set(WLLM_SENTENCEPIECE_INCLUDE_DIR ${sentencepiece_SOURCE_DIR}/src)
    set(WLLM_CPP_SENTENCEPIECE_TARGET sentencepiece-static)
  endif()
endif()
if(WLLM_CPP_WITH_EXEC AND NOT TARGET wllm_exec)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../exec
                   ${CMAKE_CURRENT_BINARY_DIR}/exec)
endif()
if(NOT TARGET wllm_runtime)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../runtime
                   ${CMAKE_CURRENT_BINARY_DIR}/runtime)
endif()

add_library(wllm_cpp INTERFACE)
target_include_directories(wllm_cpp
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include)
if(WLLM_CPP_WITH_EXEC)
  target_link_libraries(wllm_cpp INTERFACE wllm_exec)
endif()

set(WLLM_CPP_MODALITY_SRCS
  modalities/src/types.cpp
  modalities/src/vision_cpu.cpp
  modalities/src/action_cpu.cpp)
if(WLLM_ENABLE_NATIVE_CPP)
  list(APPEND WLLM_CPP_MODALITY_SRCS modalities/src/text_cpu.cpp)
  if(WLLM_CPP_WITH_SENTENCEPIECE)
    list(APPEND WLLM_CPP_MODALITY_SRCS
      modalities/src/sentencepiece_tokenizer.cpp)
  else()
    list(APPEND WLLM_CPP_MODALITY_SRCS
      modalities/src/tokenizer_unavailable.cpp)
  endif()
endif()
if(WLLM_CPP_WITH_CUDA_KERNELS)
  list(APPEND WLLM_CPP_MODALITY_SRCS modalities/src/vision_cuda.cu)
  if(WLLM_ENABLE_NATIVE_CPP)
    list(APPEND WLLM_CPP_MODALITY_SRCS modalities/src/text_cuda.cu)
  endif()
endif()

add_library(wllm_cpp_modalities STATIC ${WLLM_CPP_MODALITY_SRCS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_property(SOURCE modalities/src/vision_cpu.cpp APPEND PROPERTY
    COMPILE_OPTIONS -ffp-contract=off)
endif()
target_include_directories(wllm_cpp_modalities
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/modalities/include)
if(WLLM_CPP_WITH_CUDA_STAGING)
  target_compile_definitions(wllm_cpp_modalities
    PUBLIC WLLM_CPP_WITH_CUDA_STAGING=1)
  target_link_libraries(wllm_cpp_modalities PUBLIC CUDA::cudart)
endif()
if(WLLM_CPP_WITH_SENTENCEPIECE)
  target_compile_definitions(wllm_cpp_modalities
    PUBLIC WLLM_CPP_HAS_SENTENCEPIECE=1)
  target_include_directories(wllm_cpp_modalities
    PUBLIC ${WLLM_SENTENCEPIECE_INCLUDE_DIR})
  target_link_libraries(wllm_cpp_modalities
    PUBLIC ${WLLM_CPP_SENTENCEPIECE_TARGET})
endif()
if(WLLM_CPP_WITH_CUDA_KERNELS)
  target_compile_definitions(wllm_cpp_modalities
    PRIVATE WLLM_CPP_WITH_CUDA_KERNELS=1)
  target_link_libraries(wllm_cpp_modalities PUBLIC CUDA::cudart)
  set_target_properties(wllm_cpp_modalities PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  if(WLLM_CPP_PI05_CUDA_ARCH)
    set_target_properties(wllm_cpp_modalities PROPERTIES
      CUDA_ARCHITECTURES ${WLLM_CPP_PI05_CUDA_ARCH})
  endif()
endif()

add_library(wllm_cpp_vla INTERFACE)
target_include_directories(wllm_cpp_vla
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/families/vla/include)
target_link_libraries(wllm_cpp_vla
  INTERFACE wllm_cpp_modalities wllm_cpp)

if(WLLM_ENABLE_NATIVE_CPP)
  add_library(wllm_cpp_native STATIC
    native/src/config_object.cpp)
  target_include_directories(wllm_cpp_native
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native/include)

  if(WLLM_CPP_WITH_CUDA_KERNELS AND WLLM_CPP_WITH_EXEC)
    add_library(wllm_cpp_native_cuda STATIC
      native/src/cuda_graph_set.cpp)
    target_include_directories(wllm_cpp_native_cuda
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native/include)
    target_link_libraries(wllm_cpp_native_cuda
      PUBLIC wllm_cpp_modalities wllm_exec CUDA::cudart)
  endif()
endif()

if(WLLM_CPP_WITH_PI05_SM120_TARGET OR
   WLLM_CPP_WITH_PI05_SM110_TARGET)
  set(_wllm_csrc_dir "${CMAKE_CURRENT_SOURCE_DIR}/../csrc")

  function(_wllm_configure_csrc_target target)
    target_include_directories(${target} PUBLIC
      ${_wllm_csrc_dir}
      ${_wllm_csrc_dir}/attention
      ${_wllm_csrc_dir}/gemm
      ${_wllm_csrc_dir}/kernels)
    target_link_libraries(${target} PUBLIC
      CUDA::cublas CUDA::cublasLt CUDA::cudart)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr -O3
        --ftz=true --prec-div=false --prec-sqrt=false>)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    if(WLLM_CPP_PI05_CUDA_ARCH)
      set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES ${WLLM_CPP_PI05_CUDA_ARCH})
    endif()
  endfunction()

  add_library(wllm_native_cpp_cuda_support STATIC
    ${_wllm_csrc_dir}/native_cpp/operations.cu
    ${_wllm_csrc_dir}/native_cpp/fp8_quantize.cu
    ${_wllm_csrc_dir}/native_cpp/attention_seqused.cu)
  _wllm_configure_csrc_target(wllm_native_cpp_cuda_support)
  target_include_directories(wllm_native_cpp_cuda_support
    PUBLIC ${_wllm_csrc_dir}/native_cpp/include
    PRIVATE ${_wllm_csrc_dir}/native_cpp)
  set_target_properties(wllm_native_cpp_cuda_support PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

  add_library(wllm_cpp_csrc_core STATIC
    ${_wllm_csrc_dir}/gemm/gemm_runner.cu
    ${_wllm_csrc_dir}/kernels/activation.cu
    ${_wllm_csrc_dir}/kernels/elementwise.cu
    ${_wllm_csrc_dir}/kernels/norm.cu
    ${_wllm_csrc_dir}/kernels/patch_embed.cu
    ${_wllm_csrc_dir}/kernels/quantize.cu
    ${_wllm_csrc_dir}/kernels/rope.cu)
  _wllm_configure_csrc_target(wllm_cpp_csrc_core)

  if(WLLM_CPP_WITH_PI05_SM120_TARGET)
    add_library(wllm_cpp_csrc_sm120 STATIC
      ${_wllm_csrc_dir}/kernels/dit_bf16.cu
      ${_wllm_csrc_dir}/kernels/fusion.cu)
    _wllm_configure_csrc_target(wllm_cpp_csrc_sm120)
    target_link_libraries(wllm_cpp_csrc_sm120 PUBLIC
      wllm_cpp_csrc_core)
  endif()

  if(WLLM_CPP_WITH_PI05_SM110_TARGET)
    add_library(wllm_cpp_csrc_sm110_precise OBJECT
      ${_wllm_csrc_dir}/attention/fmha_fp16_strided.cu)
    target_include_directories(wllm_cpp_csrc_sm110_precise PRIVATE
      ${_wllm_csrc_dir}/attention
      ${WLLM_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${WLLM_CPP_CUTLASS_DIR}/include
      ${WLLM_CPP_CUTLASS_DIR}/tools/util/include)
    target_compile_options(wllm_cpp_csrc_sm110_precise PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr --expt-extended-lambda -O3>)
    set_target_properties(wllm_cpp_csrc_sm110_precise PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)

    add_library(wllm_cpp_csrc_sm110 STATIC
      $<TARGET_OBJECTS:wllm_cpp_csrc_sm110_precise>
      ${_wllm_csrc_dir}/gemm/cutlass_sm100.cu
      ${_wllm_csrc_dir}/kernels/attention_cublas.cu
      ${_wllm_csrc_dir}/kernels/decoder_fused.cu
      ${_wllm_csrc_dir}/kernels/softmax.cu)
    _wllm_configure_csrc_target(wllm_cpp_csrc_sm110)
    target_include_directories(wllm_cpp_csrc_sm110 PRIVATE
      ${WLLM_CPP_CUTLASS_DIR}/examples/77_blackwell_fmha
      ${WLLM_CPP_CUTLASS_DIR}/include
      ${WLLM_CPP_CUTLASS_DIR}/tools/util/include)
    target_compile_options(wllm_cpp_csrc_sm110 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda --use_fast_math>)
    set_target_properties(wllm_cpp_csrc_sm110 PROPERTIES
      CUDA_ARCHITECTURES 110a
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(wllm_cpp_csrc_sm110 PUBLIC
      wllm_cpp_csrc_core)
  endif()
endif()

if(WLLM_ENABLE_NATIVE_CPP)
  add_library(wllm_cpp_loader STATIC
    loader/src/safetensors.cpp
    loader/src/sha256.cpp)
  set_property(SOURCE loader/src/sha256.cpp APPEND PROPERTY COMPILE_OPTIONS -O2)
  target_include_directories(wllm_cpp_loader
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/loader/include)
  find_package(OpenSSL QUIET)
  if(OpenSSL_FOUND)
    target_compile_definitions(wllm_cpp_loader
      PRIVATE WLLM_CPP_HAS_OPENSSL=1)
    target_link_libraries(wllm_cpp_loader PRIVATE OpenSSL::Crypto)
  endif()
endif()

if(WLLM_CPP_WITH_PI05)
  set(WLLM_CPP_PI05_SUPPORT_SRCS
    models/pi05/src/support/native_rope_unavailable.cpp)
  if(WLLM_CPP_WITH_PI05_SM120_TARGET OR
     WLLM_CPP_WITH_PI05_SM110_TARGET)
    set(WLLM_CPP_PI05_SUPPORT_SRCS
      models/pi05/src/support/native_rope.cu)
  endif()

  add_library(wllm_cpp_pi05 STATIC
    models/pi05/src/spec.cpp
    models/pi05/src/io.cpp
    models/pi05/src/runtime.cpp
    models/pi05/src/model/dims.cpp
    models/pi05/src/model/execution_plan.cpp
    models/pi05/src/model/linear_weight_groups.cpp
    models/pi05/src/model/prompt_embed.cpp
    models/pi05/src/model/prompt_format.cpp
    models/pi05/src/model/resolved_resources.cpp
    models/pi05/src/model/semantic_pipeline.cpp
    models/pi05/src/support/native_calibration.cpp
    models/pi05/src/support/native_device_weights.cpp
    models/pi05/src/support/native_float16.cpp
    models/pi05/src/support/native_weight_materializer.cpp
    models/pi05/src/support/native_resource_resolver.cpp
    models/pi05/src/support/native_weight_ops.cpp
    models/pi05/src/support/native_weights.cpp
    models/pi05/src/support/native_workspace.cpp
    ${WLLM_CPP_PI05_SUPPORT_SRCS})
  target_include_directories(wllm_cpp_pi05
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(wllm_cpp_pi05
    PRIVATE wllm_cpp_loader
    PUBLIC wllm_cpp_modalities wllm_cpp_vla)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_property(SOURCE models/pi05/src/support/native_weight_ops.cpp
      APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
  endif()
  if(WLLM_CPP_WITH_CUDA_KERNELS)
    target_compile_definitions(wllm_cpp_pi05
      PRIVATE WLLM_CPP_WITH_CUDA_KERNELS=1)
    target_link_libraries(wllm_cpp_pi05 PRIVATE CUDA::cudart)
    if(TARGET wllm_native_cpp_cuda_support)
      target_link_libraries(wllm_cpp_pi05 PRIVATE
        wllm_native_cpp_cuda_support)
    endif()
    set_target_properties(wllm_cpp_pi05 PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    if(WLLM_CPP_PI05_CUDA_ARCH)
      set_target_properties(wllm_cpp_pi05 PROPERTIES
        CUDA_ARCHITECTURES ${WLLM_CPP_PI05_CUDA_ARCH})
    endif()
  endif()
  if(TARGET wllm_cpp_native_cuda)
    add_library(wllm_cpp_pi05_capture STATIC
      models/pi05/src/model/captured_program.cpp
      models/pi05/src/model/native_session.cpp)
    target_include_directories(wllm_cpp_pi05_capture
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
    target_link_libraries(wllm_cpp_pi05_capture
      PUBLIC wllm_cpp_pi05 wllm_cpp_native_cuda CUDA::cudart)
  endif()

  if(WLLM_CPP_WITH_PI05_SM120_TARGET)
  add_library(wllm_cpp_pi05_target_sm120 STATIC
    models/pi05/src/targets/sm120/attention.cpp
    models/pi05/src/targets/sm120/attention_driver.cpp
    models/pi05/src/targets/sm120/bf16_linear.cpp
    models/pi05/src/targets/sm120/fp8_autotune.cpp
    models/pi05/src/targets/sm120/bf16_scratch.cpp
    models/pi05/src/targets/sm120/fp8_linear.cpp
    models/pi05/src/targets/sm120/fp8_weight_packer.cpp
    models/pi05/src/targets/sm120/target.cpp)
  target_include_directories(wllm_cpp_pi05_target_sm120
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(wllm_cpp_pi05_target_sm120
    PUBLIC wllm_cpp_pi05
    PRIVATE wllm_cpp_loader wllm_cpp_csrc_core wllm_cpp_csrc_sm120
            wllm_native_cpp_cuda_support wllm_cpp_fa2_external
            CUDA::cudart)
  endif()

  if(WLLM_CPP_WITH_PI05_SM110_TARGET)
  add_library(wllm_cpp_pi05_target_sm110 STATIC
    models/pi05/src/targets/sm110/frontend_bindings.cpp
    models/pi05/src/targets/sm110/fp8_weight_packer.cpp
    models/pi05/src/targets/sm110/operation_driver.cu
    models/pi05/src/targets/sm110/physical_resources.cpp
    models/pi05/src/targets/sm110/target.cpp)
  target_include_directories(wllm_cpp_pi05_target_sm110
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(wllm_cpp_pi05_target_sm110
    PUBLIC wllm_cpp_pi05
    PRIVATE wllm_cpp_loader wllm_cpp_csrc_core
            wllm_cpp_csrc_sm110 wllm_native_cpp_cuda_support
            CUDA::cublas CUDA::cudart)
  set_target_properties(wllm_cpp_pi05_target_sm110 PROPERTIES
    CUDA_ARCHITECTURES 110a
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  endif()

  if(WLLM_CPP_WITH_PI05_SM120_TARGET OR
     WLLM_CPP_WITH_PI05_SM110_TARGET)
    add_library(wllm_cpp_pi05_c SHARED
      models/pi05/src/c_api.cpp
      models/pi05/src/model_runtime.cpp)
    target_link_libraries(wllm_cpp_pi05_c
      PUBLIC wllm_cpp_pi05 wllm_runtime)
    target_include_directories(wllm_cpp_pi05_c
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
    target_compile_definitions(wllm_cpp_pi05_c
      PRIVATE WLLM_PI05_C_BUILDING_LIBRARY=1)
    set_target_properties(wllm_cpp_pi05_c PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN YES)
    if(UNIX AND NOT APPLE)
      target_link_options(wllm_cpp_pi05_c PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/pi05_c_exports.map"
        "LINKER:--exclude-libs,ALL")
      add_custom_command(TARGET wllm_cpp_pi05_c POST_BUILD
        COMMAND ${CMAKE_COMMAND}
          -DNM_TOOL=${CMAKE_NM}
          -DINPUT_FILE=$<TARGET_FILE:wllm_cpp_pi05_c>
          -DALLOWED_SYMBOLS_FILE=${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/pi05_c_exports.txt
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_dynamic_exports.cmake
        VERBATIM)
    endif()
    if(TARGET wllm_cpp_pi05_capture)
      target_link_libraries(wllm_cpp_pi05_c PRIVATE
        wllm_cpp_pi05_capture)
    endif()
    if(TARGET wllm_cpp_pi05_capture AND
       (TARGET wllm_cpp_pi05_target_sm120 OR
        TARGET wllm_cpp_pi05_target_sm110))
      target_sources(wllm_cpp_pi05_c PRIVATE
        models/pi05/src/native_calibration_c_api.cpp
        models/pi05/src/native_model_runtime.cpp
        models/pi05/src/native_open.cpp
        models/pi05/src/native_calibration_session.cpp
        models/pi05/src/native_session_factory.cpp)
      target_link_libraries(wllm_cpp_pi05_c PRIVATE
        wllm_cpp_loader wllm_cpp_native)
    endif()
    if(TARGET wllm_cpp_pi05_target_sm120)
      target_compile_definitions(wllm_cpp_pi05_c
        PRIVATE WLLM_CPP_WITH_PI05_SM120_TARGET=1)
      target_link_libraries(wllm_cpp_pi05_c
        PRIVATE wllm_cpp_pi05_target_sm120)
    endif()
    if(TARGET wllm_cpp_pi05_target_sm110)
      target_compile_definitions(wllm_cpp_pi05_c
        PRIVATE WLLM_CPP_WITH_PI05_SM110_TARGET=1)
      target_link_libraries(wllm_cpp_pi05_c
        PRIVATE wllm_cpp_pi05_target_sm110)
    endif()
    set_target_properties(wllm_cpp_pi05_c PROPERTIES
      INSTALL_RPATH "$ORIGIN")
    install(TARGETS wllm_cpp_pi05_c wllm_runtime wllm_exec
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/../exec/include/wllm/exec.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/wllm/runtime.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/wllm/model_runtime.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/wllm)
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include/wllm/cpp/models/pi05/export.h
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include/wllm/cpp/models/pi05/c_api.h
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/include/wllm/cpp/models/pi05/model_runtime.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/wllm/cpp/models/pi05)
  endif()
endif()

if(BUILD_TESTING)
  if(TARGET wllm_native_cpp_cuda_support)
    add_executable(test_csrc_rope_table tests/test_csrc_rope_table.cu)
    target_link_libraries(test_csrc_rope_table PRIVATE
      wllm_native_cpp_cuda_support CUDA::cudart)
    set_target_properties(test_csrc_rope_table PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_rope_table COMMAND test_csrc_rope_table)

    if(TARGET wllm_cpp_csrc_sm110)
      add_executable(test_csrc_attention_seqused
        tests/test_csrc_attention_seqused.cu)
      target_link_libraries(test_csrc_attention_seqused PRIVATE
        wllm_native_cpp_cuda_support wllm_cpp_csrc_sm110
        CUDA::cublas CUDA::cudart)
      target_compile_options(test_csrc_attention_seqused PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
          --expt-relaxed-constexpr -O3
          --ftz=true --prec-div=false --prec-sqrt=false>)
      set_target_properties(test_csrc_attention_seqused PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
      add_test(NAME csrc_attention_seqused
        COMMAND test_csrc_attention_seqused)
    endif()
  endif()

  if(TARGET wllm_cpp_csrc_core)
    add_executable(test_csrc_bf16_operations
      tests/test_csrc_bf16_operations.cu)
    target_link_libraries(test_csrc_bf16_operations PRIVATE
      wllm_native_cpp_cuda_support CUDA::cudart)
    set_target_properties(test_csrc_bf16_operations PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME csrc_bf16_operations COMMAND test_csrc_bf16_operations)

    add_executable(test_csrc_fp8_weight_quantize
      tests/test_csrc_fp8_weight_quantize.cu)
    target_include_directories(test_csrc_fp8_weight_quantize PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/native_cpp)
    target_link_libraries(test_csrc_fp8_weight_quantize PRIVATE
      wllm_native_cpp_cuda_support wllm_cpp_csrc_core
      wllm_runtime CUDA::cudart)
    set_target_properties(test_csrc_fp8_weight_quantize PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    if(WLLM_CPP_PI05_CUDA_ARCH)
      set_target_properties(test_csrc_fp8_weight_quantize PROPERTIES
        CUDA_ARCHITECTURES ${WLLM_CPP_PI05_CUDA_ARCH})
    endif()
    add_test(NAME csrc_fp8_weight_quantize
      COMMAND test_csrc_fp8_weight_quantize)
  endif()

  if(WLLM_ENABLE_NATIVE_CPP)
    add_executable(test_safetensors_loader tests/test_safetensors_loader.cpp)
    target_link_libraries(test_safetensors_loader PRIVATE wllm_cpp_loader)
    add_test(NAME safetensors_loader COMMAND test_safetensors_loader)

    add_executable(test_sha256 tests/test_sha256.cpp)
    target_link_libraries(test_sha256 PRIVATE wllm_cpp_loader)
    add_test(NAME sha256 COMMAND test_sha256)

    add_executable(test_native_config tests/test_native_config.cpp)
    target_link_libraries(test_native_config PRIVATE wllm_cpp_native)
    add_test(NAME native_config COMMAND test_native_config)

    add_executable(test_text_modalities tests/test_text_modalities.cpp)
    target_link_libraries(test_text_modalities PRIVATE wllm_cpp_modalities)
    add_test(NAME text_modalities COMMAND test_text_modalities)

    add_executable(test_text_tokenizer tests/test_text_tokenizer.cpp)
    target_link_libraries(test_text_tokenizer PRIVATE wllm_cpp_modalities)
    add_test(NAME text_tokenizer COMMAND test_text_tokenizer)
  endif()

  if(WLLM_CPP_WITH_PI05)
    add_executable(test_cpp_modalities tests/test_modalities.cpp)
  target_link_libraries(test_cpp_modalities
    PRIVATE wllm_cpp_pi05 wllm_cpp_modalities)
  add_test(NAME cpp_modalities COMMAND test_cpp_modalities)

  add_executable(test_pi05_runtime tests/test_pi05_runtime.cpp)
  target_link_libraries(test_pi05_runtime
    PRIVATE wllm_cpp_pi05 wllm_cpp_modalities)
  add_test(NAME pi05_runtime COMMAND test_pi05_runtime)

  add_executable(test_pi05_prompt_format tests/test_pi05_prompt_format.cpp)
  target_include_directories(test_pi05_prompt_format PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_prompt_format PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_prompt_format COMMAND test_pi05_prompt_format)

  add_executable(test_pi05_prompt_embed tests/test_pi05_prompt_embed.cpp)
  target_include_directories(test_pi05_prompt_embed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  if(WLLM_CPP_WITH_CUDA_KERNELS)
    target_compile_definitions(test_pi05_prompt_embed PRIVATE
      WLLM_CPP_TEST_WITH_CUDA_KERNELS=1)
  endif()
  target_link_libraries(test_pi05_prompt_embed PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_prompt_embed COMMAND test_pi05_prompt_embed)

  add_executable(test_pi05_native_calibration
    tests/test_pi05_native_calibration.cpp)
  target_include_directories(test_pi05_native_calibration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_native_calibration PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_native_calibration COMMAND test_pi05_native_calibration)

  add_executable(test_pi05_semantic_pipeline
    tests/test_pi05_semantic_pipeline.cpp)
  target_include_directories(test_pi05_semantic_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_semantic_pipeline
    PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_semantic_pipeline COMMAND test_pi05_semantic_pipeline)

  add_executable(test_pi05_linear_weight_groups
    tests/test_pi05_linear_weight_groups.cpp)
  target_include_directories(test_pi05_linear_weight_groups PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_linear_weight_groups
    PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_linear_weight_groups
    COMMAND test_pi05_linear_weight_groups)

  add_executable(test_pi05_execution_plan
    tests/test_pi05_execution_plan.cpp)
  target_include_directories(test_pi05_execution_plan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_execution_plan
    PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_execution_plan COMMAND test_pi05_execution_plan)

  add_executable(test_pi05_resolved_resources
    tests/test_pi05_resolved_resources.cpp)
  target_include_directories(test_pi05_resolved_resources PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_resolved_resources
    PRIVATE wllm_cpp_pi05)
  add_test(NAME pi05_resolved_resources COMMAND test_pi05_resolved_resources)

  add_executable(test_pi05_native_resource_resolver
    tests/test_pi05_native_resource_resolver.cpp)
  target_include_directories(test_pi05_native_resource_resolver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
  target_link_libraries(test_pi05_native_resource_resolver PRIVATE
    wllm_cpp_pi05)
  add_test(NAME pi05_native_resource_resolver
    COMMAND test_pi05_native_resource_resolver)
  if(TARGET wllm_cpp_pi05_target_sm120)
    add_executable(test_pi05_sm120_fp8_linear
      tests/test_pi05_sm120_fp8_linear.cpp)
    target_include_directories(test_pi05_sm120_fp8_linear PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
    target_link_libraries(test_pi05_sm120_fp8_linear PRIVATE
      wllm_cpp_pi05_target_sm120 wllm_exec CUDA::cudart)
    add_test(NAME pi05_sm120_fp8_linear
      COMMAND test_pi05_sm120_fp8_linear)

    add_executable(test_pi05_sm120_target_bundle
      tests/test_pi05_sm120_target_bundle.cpp)
    target_link_libraries(test_pi05_sm120_target_bundle PRIVATE
      wllm_cpp_pi05_target_sm120 wllm_exec CUDA::cudart)
    add_test(NAME pi05_sm120_target_bundle
      COMMAND test_pi05_sm120_target_bundle)

  endif()

  if(TARGET wllm_cpp_native_cuda)
    add_executable(test_native_cuda_graph_set
      tests/test_native_cuda_graph_set.cpp)
    target_link_libraries(test_native_cuda_graph_set
      PRIVATE wllm_cpp_native_cuda)
    add_test(NAME native_cuda_graph_set COMMAND test_native_cuda_graph_set)

    add_executable(test_pi05_captured_program
      tests/test_pi05_captured_program.cpp)
    target_link_libraries(test_pi05_captured_program
      PRIVATE wllm_cpp_pi05_capture)
    add_test(NAME pi05_captured_program COMMAND test_pi05_captured_program)

    add_executable(test_pi05_native_session
      tests/test_pi05_native_session.cpp)
    target_link_libraries(test_pi05_native_session
      PRIVATE wllm_cpp_pi05_capture)
    add_test(NAME pi05_native_session COMMAND test_pi05_native_session)
  endif()

  if(WLLM_CPP_WITH_CUDA_STAGING)
    add_executable(test_device_staging tests/test_device_staging.cpp)
    if(WLLM_CPP_WITH_CUDA_KERNELS)
      target_compile_definitions(test_device_staging PRIVATE
        WLLM_CPP_TEST_WITH_CUDA_KERNELS=1)
    endif()
    target_link_libraries(test_device_staging
      PRIVATE wllm_cpp_pi05 wllm_cpp_modalities CUDA::cudart)
    add_test(NAME device_staging COMMAND test_device_staging)

    if(WLLM_CPP_WITH_PI05_SM120_TARGET OR
       WLLM_CPP_WITH_PI05_SM110_TARGET)
      add_executable(test_pi05_c_api tests/test_pi05_c_api.cpp)
      target_link_libraries(test_pi05_c_api
        PRIVATE wllm_cpp_pi05_c wllm_exec CUDA::cudart)
      add_test(NAME pi05_c_api COMMAND test_pi05_c_api)

      add_executable(test_pi05_model_runtime tests/test_pi05_model_runtime.cpp)
      target_link_libraries(test_pi05_model_runtime
        PRIVATE wllm_cpp_pi05_c wllm_exec wllm_runtime CUDA::cudart)
      add_test(NAME pi05_model_runtime COMMAND test_pi05_model_runtime)

      add_executable(test_pi05_native_open tests/test_pi05_native_open.cpp)
      target_include_directories(test_pi05_native_open PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/models/pi05/internal/include)
      target_link_libraries(test_pi05_native_open
        PRIVATE wllm_cpp_pi05_c wllm_runtime)
      if(WLLM_CPP_WITH_SENTENCEPIECE)
        target_compile_definitions(test_pi05_native_open
          PRIVATE WLLM_CPP_PI05_NATIVE_OPEN_ENABLED=1)
      endif()
      add_test(NAME pi05_native_open COMMAND test_pi05_native_open)
    endif()
  endif()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    get_property(_wllm_cpp_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _wllm_cpp_targets)
      if(_target MATCHES "^test_")
        target_compile_options(${_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>
          $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
      endif()
    endforeach()
  endif()
endif()
