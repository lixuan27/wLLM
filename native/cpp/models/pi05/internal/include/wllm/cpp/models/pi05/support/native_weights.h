#ifndef WLLM_CPP_MODELS_PI05_NATIVE_WEIGHTS_H
#define WLLM_CPP_MODELS_PI05_NATIVE_WEIGHTS_H

#include <cstdint>
#include <string>
#include <vector>

namespace wllm {
namespace models {
namespace pi05 {

struct NativeTensorRequirement {
    std::string key;
    std::vector<std::uint64_t> shape;
};

const std::vector<NativeTensorRequirement>& native_tensor_requirements();

}  // namespace pi05
}  // namespace models
}  // namespace wllm

#endif  // WLLM_CPP_MODELS_PI05_NATIVE_WEIGHTS_H
