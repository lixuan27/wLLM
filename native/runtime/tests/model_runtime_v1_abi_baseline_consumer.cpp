#include "abi/model_runtime_v1_abi_baseline.h"

extern "C" int wllm_model_runtime_v1_prefix_consume(const void* object) {
    auto* model = static_cast<const wrt_model_runtime_v1*>(object);
    if (!model || model->abi_version != WRT_MODEL_RUNTIME_ABI_VERSION ||
        model->struct_size < sizeof(wrt_model_runtime_v1) || !model->exp ||
        !model->retain || !model->release) return -1;
    model->retain(model->owner);
    model->release(model->owner);
    return 0;
}

extern "C" int wllm_model_runtime_v1_exact_size_consume(
        const void* object) {
    auto* model = static_cast<const wrt_model_runtime_v1*>(object);
    if (!model || model->abi_version != WRT_MODEL_RUNTIME_ABI_VERSION ||
        model->struct_size != sizeof(wrt_model_runtime_v1)) return -1;
    return 0;
}
