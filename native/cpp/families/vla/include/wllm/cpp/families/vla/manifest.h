#ifndef WLLM_CPP_FAMILIES_VLA_MANIFEST_H
#define WLLM_CPP_FAMILIES_VLA_MANIFEST_H

#include "wllmrt/cpp/modalities/action.h"
#include "wllmrt/cpp/modalities/vision.h"

#include <string>
#include <vector>

namespace wllm {
namespace families {
namespace vla {

struct GraphNames {
    std::string infer = "infer";
    std::string decode_only = "decode_only";
};

struct StateRegion {
    std::string name;
    std::string buffer;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct Manifest {
    modalities::VisionPreprocessSpec vision;
    modalities::ActionPostprocessSpec action;
    GraphNames graphs;
    std::vector<StateRegion> state_regions;
};

}  // namespace vla
}  // namespace families
}  // namespace wllm

#endif  // WLLM_CPP_FAMILIES_VLA_MANIFEST_H
