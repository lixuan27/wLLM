#pragma once

#include "wllmrt/cpp/modalities/types.h"
#include "wllmrt/exec.h"

#include <cstddef>
#include <vector>

namespace wllm {
namespace native {

struct CudaGraphBinding {
    const char* name = nullptr;
    wrt_buffer buffer = nullptr;
};

class CudaGraphSet {
public:
    using RecordFn = modalities::Status (*)(
        void* owner, std::size_t slot, void* stream);

    // The graph set takes ownership of context.
    CudaGraphSet(wrt_ctx context, std::size_t graph_count);
    ~CudaGraphSet();

    CudaGraphSet(const CudaGraphSet&) = delete;
    CudaGraphSet& operator=(const CudaGraphSet&) = delete;

    modalities::Status capture(
        std::size_t slot,
        const char* name,
        const std::vector<CudaGraphBinding>& bindings,
        RecordFn record,
        void* owner);
    modalities::Status create_replay_stream();

    wrt_ctx context() const { return context_; }
    wrt_graph graph(std::size_t slot) const;
    int stream_id() const { return stream_id_; }
    void* native_stream() const { return replay_stream_; }
    int replay(std::size_t slot) const;
    modalities::Status synchronize() const;

private:
    struct CaptureCall;
    static void record_graph(void* user, void* stream);

    wrt_ctx context_ = nullptr;
    std::vector<wrt_graph> graphs_;
    void* replay_stream_ = nullptr;
    int stream_id_ = -1;
};

}  // namespace native
}  // namespace wllm
