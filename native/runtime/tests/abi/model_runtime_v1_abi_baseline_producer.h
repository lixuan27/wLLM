#ifndef WLLM_MODEL_RUNTIME_V1_ABI_BASELINE_PRODUCER_H
#define WLLM_MODEL_RUNTIME_V1_ABI_BASELINE_PRODUCER_H

#include "wllmrt/runtime.h"

namespace wllm::model_runtime_v1_abi {

void* create_baseline(const wrt_runtime_export_v1* exp, void* owner,
                      void (*retain_owner)(void*),
                      void (*release_owner)(void*));
void destroy_baseline(void* model);

}  // namespace wllm::model_runtime_v1_abi

#endif  // WLLM_MODEL_RUNTIME_V1_ABI_BASELINE_PRODUCER_H
