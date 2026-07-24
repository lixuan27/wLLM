#include "wllmrt/cpp/models/pi05/model/captured_program.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace pi05 = wllm::models::pi05;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                              \
    do {                                               \
        if (!(expression)) fail(#expression, __LINE__); \
    } while (false)

pi05::Pi05ResolvedShape canonical_shape() {
    pi05::Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 1;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    pi05::Pi05ResolvedShape shape;
    CHECK(pi05::resolve_pi05_shape(config, &shape).ok_status());
    return shape;
}

pi05::Pi05ResolvedGraphBindings allocate_bindings(
    wrt_ctx context,
    wrt_buffer* output,
    bool omit_last = false) {
    pi05::Pi05ResolvedGraphBindings bindings;
    const std::size_t count =
        static_cast<std::size_t>(pi05::Pi05GraphBindingId::kCount);
    for (std::size_t i = 0; i < count; ++i) {
        if (omit_last && i + 1 == count) break;
        const auto id = static_cast<pi05::Pi05GraphBindingId>(i);
        wrt_buffer buffer = wrt_buffer_alloc(
            context, pi05::pi05_graph_binding_name(id), 32);
        CHECK(buffer != nullptr);
        CHECK(bindings.bind(id, buffer).ok_status());
        if (id == pi05::Pi05GraphBindingId::kNoise && output) {
            *output = buffer;
        }
    }
    return bindings;
}

class CudaTraceSink final : public pi05::Pi05OperationSink {
public:
    explicit CudaTraceSink(
        void* output,
        std::size_t fail_at = std::numeric_limits<std::size_t>::max())
        : output_(output), fail_at_(fail_at) {}

    wllm::modalities::Status record(
        const pi05::Pi05OperationCall& call,
        const pi05::Pi05ResolvedShape& shape,
        pi05::Pi05Stream stream) override {
        wllm::modalities::Status status =
            pi05::validate_pi05_operation_call(call, shape);
        if (!status.ok_status()) return status;
        calls.push_back(call.id);
        if (calls.size() - 1 == fail_at_) {
            return wllm::modalities::Status::error(
                wllm::modalities::StatusCode::kBackend,
                "injected PI0.5 operation failure");
        }
        const cudaError_t result = cudaMemsetAsync(
            output_, static_cast<int>(call.id) + 1, 1,
            reinterpret_cast<cudaStream_t>(stream));
        return result == cudaSuccess
                   ? wllm::modalities::Status::ok()
                   : wllm::modalities::Status::error(
                         wllm::modalities::StatusCode::kBackend,
                         cudaGetErrorString(result));
    }

    std::vector<pi05::Pi05OperationId> calls;

private:
    void* output_ = nullptr;
    std::size_t fail_at_ = 0;
};

unsigned char read_byte(wrt_buffer buffer) {
    unsigned char result = 0;
    CHECK(cudaMemcpy(&result, wrt_buffer_dptr(buffer), 1,
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    return result;
}

void check_segment(
    const std::vector<pi05::Pi05OperationId>& calls,
    std::size_t offset,
    std::size_t count,
    pi05::Pi05OperationId first,
    pi05::Pi05OperationId last) {
    CHECK(offset + count <= calls.size());
    CHECK(calls[offset] == first);
    CHECK(calls[offset + count - 1] == last);
}

void test_complete_lifecycle() {
    wrt_ctx context = wrt_ctx_create();
    CHECK(context != nullptr);
    wrt_buffer output = nullptr;
    const pi05::Pi05ResolvedGraphBindings bindings =
        allocate_bindings(context, &output);
    CHECK(output != nullptr);

    const pi05::Pi05SemanticPipeline pipeline(canonical_shape());
    CudaTraceSink sink(wrt_buffer_dptr(output));
    pi05::Pi05CapturedProgram program(context);

    CHECK(program.graph(pi05::Pi05GraphId::kInfer) == nullptr);
    CHECK(program.replay(pi05::Pi05GraphId::kInfer) == WRT_ERR_INVALID);
    CHECK(!program.synchronize().ok_status());

    CHECK(program.warmup(pipeline, sink).ok_status());
    CHECK(sink.calls.size() == 482);
    check_segment(sink.calls, 0, 482,
                  pi05::Pi05OperationId::kComposePrompt,
                  pi05::Pi05OperationId::kDiffusionUpdate);
    CHECK(read_byte(output) ==
          static_cast<unsigned char>(
              pi05::Pi05OperationId::kDiffusionUpdate) + 1);

    sink.calls.clear();
    CHECK(cudaMemset(wrt_buffer_dptr(output), 0, 1) == cudaSuccess);
    CHECK(program.capture(pipeline, sink, bindings).ok_status());
    CHECK(sink.calls.size() == 482 + 390 + 92);
    check_segment(sink.calls, 0, 482,
                  pi05::Pi05OperationId::kComposePrompt,
                  pi05::Pi05OperationId::kDiffusionUpdate);
    check_segment(sink.calls, 482, 390,
                  pi05::Pi05OperationId::kDiffusionInputProject,
                  pi05::Pi05OperationId::kDiffusionUpdate);
    check_segment(sink.calls, 482 + 390, 92,
                  pi05::Pi05OperationId::kComposePrompt,
                  pi05::Pi05OperationId::kEncoderCacheFinalize);

    for (const pi05::Pi05GraphId id : {
             pi05::Pi05GraphId::kInfer,
             pi05::Pi05GraphId::kDecodeOnly,
             pi05::Pi05GraphId::kContext}) {
        CHECK(program.graph(id) != nullptr);
        CHECK(wrt_graph_variant_count(program.graph(id)) == 1);
    }
    CHECK(program.graph(pi05::Pi05GraphId::kCount) == nullptr);
    CHECK(program.stream_id() >= 0);
    CHECK(program.native_stream() != nullptr);
    CHECK(!program.warmup(pipeline, sink).ok_status());
    CHECK(!program.capture(pipeline, sink, bindings).ok_status());

    CHECK(program.replay(pi05::Pi05GraphId::kContext) == WRT_OK);
    CHECK(program.synchronize().ok_status());
    CHECK(read_byte(output) ==
          static_cast<unsigned char>(
              pi05::Pi05OperationId::kEncoderCacheFinalize) + 1);
    CHECK(program.replay(pi05::Pi05GraphId::kDecodeOnly) == WRT_OK);
    CHECK(program.synchronize().ok_status());
    CHECK(read_byte(output) ==
          static_cast<unsigned char>(
              pi05::Pi05OperationId::kDiffusionUpdate) + 1);
    CHECK(program.replay(pi05::Pi05GraphId::kInfer) == WRT_OK);
    CHECK(program.synchronize().ok_status());
    CHECK(read_byte(output) ==
          static_cast<unsigned char>(
              pi05::Pi05OperationId::kDiffusionUpdate) + 1);
    CHECK(program.replay(pi05::Pi05GraphId::kCount) == WRT_ERR_INVALID);
}

void test_missing_binding_is_atomic() {
    wrt_ctx context = wrt_ctx_create();
    CHECK(context != nullptr);
    wrt_buffer output = nullptr;
    const pi05::Pi05ResolvedGraphBindings bindings =
        allocate_bindings(context, &output, true);
    CHECK(output != nullptr);
    const pi05::Pi05SemanticPipeline pipeline(canonical_shape());
    CudaTraceSink sink(wrt_buffer_dptr(output));
    pi05::Pi05CapturedProgram program(context);

    const wllm::modalities::Status status =
        program.capture(pipeline, sink, bindings);
    CHECK(!status.ok_status());
    CHECK(status.code == wllm::modalities::StatusCode::kNotFound);
    CHECK(sink.calls.empty());
    CHECK(program.graph(pi05::Pi05GraphId::kInfer) == nullptr);
    CHECK(program.graph(pi05::Pi05GraphId::kDecodeOnly) == nullptr);
    CHECK(program.graph(pi05::Pi05GraphId::kContext) == nullptr);
}

void test_record_failure_is_not_publishable() {
    wrt_ctx context = wrt_ctx_create();
    CHECK(context != nullptr);
    wrt_buffer output = nullptr;
    const pi05::Pi05ResolvedGraphBindings bindings =
        allocate_bindings(context, &output);
    CHECK(output != nullptr);
    const pi05::Pi05SemanticPipeline pipeline(canonical_shape());
    CudaTraceSink sink(wrt_buffer_dptr(output), 482 + 11);
    pi05::Pi05CapturedProgram program(context);

    const wllm::modalities::Status status =
        program.capture(pipeline, sink, bindings);
    CHECK(!status.ok_status());
    CHECK(status.code == wllm::modalities::StatusCode::kBackend);
    CHECK(sink.calls.size() == 482 + 12);
    CHECK(program.graph(pi05::Pi05GraphId::kInfer) == nullptr);
    CHECK(program.graph(pi05::Pi05GraphId::kDecodeOnly) == nullptr);
    CHECK(program.graph(pi05::Pi05GraphId::kContext) == nullptr);
    CHECK(program.replay(pi05::Pi05GraphId::kInfer) == WRT_ERR_INVALID);
    CHECK(!program.capture(pipeline, sink, bindings).ok_status());
}

}  // namespace

int main() {
    test_complete_lifecycle();
    test_missing_binding_is_atomic();
    test_record_failure_is_not_publishable();
    std::cout << "PASS - PI0.5 captured program\n";
    return 0;
}
