#ifndef WLLM_MODELS_PI05_CPP_RUNTIME_SPEC_H
#define WLLM_MODELS_PI05_CPP_RUNTIME_SPEC_H

#include "wllmrt/cpp/modalities/action.h"
#include "wllmrt/cpp/modalities/vision.h"

#include <vector>

namespace wllm {
namespace models {
namespace pi05 {

static constexpr int kImageSize = 224;
static constexpr int kDefaultChunk = 10;
static constexpr int kModelActionDim = 32;
static constexpr int kLiberoActionDim = 7;

modalities::VisionPreprocessSpec vision_preprocess_spec(int num_views);

modalities::ActionPostprocessSpec action_postprocess_spec(
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    int chunk = kDefaultChunk,
    int model_dim = kModelActionDim,
    int robot_dim = kLiberoActionDim);

}  // namespace pi05
}  // namespace models
}  // namespace wllm

#endif  // WLLM_MODELS_PI05_CPP_RUNTIME_SPEC_H
