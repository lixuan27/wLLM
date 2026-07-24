#ifndef WLLM_CPP_MODELS_PI05_TARGETS_SM120_DEVICE_BUFFER_H
#define WLLM_CPP_MODELS_PI05_TARGETS_SM120_DEVICE_BUFFER_H

#include "wllmrt/cpp/modalities/types.h"
#include "wllmrt/exec.h"

#include <cstddef>

namespace wllm {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

struct Sm120DeviceBuffer final {
    wrt_buffer buffer = nullptr;
    modalities::DType dtype = modalities::DType::kUInt8;
    modalities::Shape shape;

    void* device_data() const {
        return buffer ? wrt_buffer_dptr(buffer) : nullptr;
    }
    std::size_t bytes() const {
        return buffer ? wrt_buffer_bytes(buffer) : 0;
    }
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace wllm

#endif  // WLLM_CPP_MODELS_PI05_TARGETS_SM120_DEVICE_BUFFER_H
