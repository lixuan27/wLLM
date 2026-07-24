# wLLM execution-contract layer — standalone build (Phase A).
#
# Builds:
#   libwllm_exec.so   — the C ABI shared library (link target for C/Rust/robot hosts)
#   _wllm_exec*.so    — the pybind dev module (if pybind11 is found)
#
# Standalone on purpose: it does NOT touch the root CMake build. Build with:
#   cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build exec/build -j
# The pybind module imports as `_wllm_exec`; wllm_native/runtime/exec.py wraps it.

cmake_minimum_required(VERSION 3.18)
project(wllm_exec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# --- core C ABI library (CUDA runtime only; no kernels) ---
add_library(wllm_exec SHARED
  src/context.cpp
  src/buffer.cpp
  src/graph.cpp
  src/plan.cpp
  backend/cuda/cuda_backend.cpp)
target_include_directories(wllm_exec
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CMAKE_CURRENT_SOURCE_DIR}/backend)
target_link_libraries(wllm_exec PUBLIC CUDA::cudart)

# --- pybind dev module (optional) ---
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND AND Python3_Interpreter_FOUND)
  # Fall back to pybind11 shipped with the active interpreter.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pb_dir OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _pb_rc)
  if(_pb_rc EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${_pb_dir}")
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()

if(pybind11_FOUND)
  pybind11_add_module(_wllm_exec bindings/exec_pybind.cpp)
  target_include_directories(_wllm_exec
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/backend)
  target_link_libraries(_wllm_exec PRIVATE wllm_exec)
else()
  message(STATUS "pybind11 not found — building C ABI library only (no Python module).")
endif()
