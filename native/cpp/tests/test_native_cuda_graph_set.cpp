#include "wllmrt/cpp/native/cuda_graph_set.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

struct RecordCall {
    void* destination = nullptr;
    std::size_t bytes = 0;
    std::size_t expected_slot = 0;
    bool inject_failure = false;
    std::size_t calls = 0;
};

wllm::modalities::Status record_fill(
    void* user, std::size_t slot, void* stream) {
    auto* call = static_cast<RecordCall*>(user);
    if (!call || slot != call->expected_slot || !stream ||
        !call->destination) {
        return wllm::modalities::Status::error(
            wllm::modalities::StatusCode::kInvalidArgument,
            "invalid graph test record request");
    }
    ++call->calls;
    if (call->inject_failure) {
        return wllm::modalities::Status::error(
            wllm::modalities::StatusCode::kBackend,
            "injected graph record failure");
    }
    const cudaError_t result = cudaMemsetAsync(
        call->destination, 0x5a, call->bytes,
        static_cast<cudaStream_t>(stream));
    return result == cudaSuccess
               ? wllm::modalities::Status::ok()
               : wllm::modalities::Status::error(
                     wllm::modalities::StatusCode::kBackend,
                     cudaGetErrorString(result));
}

}  // namespace

int main() {
    wrt_ctx context = wrt_ctx_create();
    CHECK(context != nullptr);
    wllm::native::CudaGraphSet graphs(context, 2);

    constexpr std::size_t kBytes = 64;
    wrt_buffer output = wrt_buffer_alloc(context, "output", kBytes);
    CHECK(output != nullptr);
    const std::vector<wllm::native::CudaGraphBinding> bindings = {
        {"output", output},
    };

    RecordCall rejected{wrt_buffer_dptr(output), kBytes, 1, true, 0};
    wllm::modalities::Status status =
        graphs.capture(1, "rejected", bindings, record_fill, &rejected);
    CHECK(!status.ok_status());
    CHECK(status.code == wllm::modalities::StatusCode::kBackend);
    CHECK(rejected.calls == 1);
    CHECK(graphs.graph(1) == nullptr);

    RecordCall accepted{wrt_buffer_dptr(output), kBytes, 0, false, 0};
    CHECK(graphs.capture(0, "fill", bindings, record_fill, &accepted)
              .ok_status());
    CHECK(accepted.calls == 1);
    CHECK(graphs.graph(0) != nullptr);
    CHECK(wrt_graph_variant_count(graphs.graph(0)) == 1);
    CHECK(!graphs.capture(0, "duplicate", bindings, record_fill, &accepted)
               .ok_status());
    CHECK(graphs.replay(0) == WRT_ERR_INVALID);
    CHECK(graphs.replay(2) == WRT_ERR_INVALID);
    CHECK(!graphs.synchronize().ok_status());

    CHECK(graphs.create_replay_stream().ok_status());
    CHECK(!graphs.create_replay_stream().ok_status());
    CHECK(graphs.replay(0) == WRT_OK);
    CHECK(graphs.synchronize().ok_status());

    std::array<unsigned char, kBytes> result{};
    CHECK(cudaMemcpy(result.data(), wrt_buffer_dptr(output), kBytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    for (unsigned char value : result) CHECK(value == 0x5a);

    std::cout << "PASS - native CUDA graph set\n";
    return 0;
}
