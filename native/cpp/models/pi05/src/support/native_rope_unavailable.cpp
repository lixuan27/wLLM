#include "wllmrt/cpp/models/pi05/support/native_rope.h"

namespace wllm {
namespace models {
namespace pi05 {

modalities::Status generate_native_rope_f16(
    void*, int, int, std::uintptr_t) {
    return modalities::Status::error(
        modalities::StatusCode::kUnsupported,
        "native RoPE generation requires the CUDA kernels build");
}

}  // namespace pi05
}  // namespace models
}  // namespace wllm
