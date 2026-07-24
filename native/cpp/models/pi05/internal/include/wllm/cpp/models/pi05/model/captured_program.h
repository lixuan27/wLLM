#ifndef WLLM_CPP_MODELS_PI05_MODEL_CAPTURED_PROGRAM_H
#define WLLM_CPP_MODELS_PI05_MODEL_CAPTURED_PROGRAM_H

#include "wllmrt/cpp/models/pi05/model/execution_plan.h"
#include "wllmrt/cpp/models/pi05/model/semantic_pipeline.h"
#include "wllmrt/cpp/native/cuda_graph_set.h"

#include <cstddef>

namespace wllm {
namespace models {
namespace pi05 {

class Pi05CapturedProgram final {
public:
    // The captured program takes ownership of context.
    explicit Pi05CapturedProgram(wrt_ctx context);

    Pi05CapturedProgram(const Pi05CapturedProgram&) = delete;
    Pi05CapturedProgram& operator=(const Pi05CapturedProgram&) = delete;

    modalities::Status warmup(const Pi05SemanticPipeline& pipeline,
                               Pi05OperationSink& operations);
    modalities::Status capture(
        const Pi05SemanticPipeline& pipeline,
        Pi05OperationSink& operations,
        const Pi05ResolvedGraphBindings& bindings);

    wrt_ctx context() const { return graphs_.context(); }
    wrt_graph graph(Pi05GraphId id) const;
    int stream_id() const { return graphs_.stream_id(); }
    void* native_stream() const { return graphs_.native_stream(); }
    int replay(Pi05GraphId id) const;
    modalities::Status synchronize() const;

private:
    struct RecordCall;

    static modalities::Status record_graph(
        void* owner, std::size_t slot, void* stream);
    static modalities::Status record_body(
        const Pi05SemanticPipeline& pipeline,
        Pi05OperationSink& operations,
        Pi05RecordBody body,
        Pi05Stream stream);

    native::CudaGraphSet graphs_;
    bool capture_attempted_ = false;
    bool captured_ = false;
};

}  // namespace pi05
}  // namespace models
}  // namespace wllm

#endif  // WLLM_CPP_MODELS_PI05_MODEL_CAPTURED_PROGRAM_H
