#include "wllmrt/cpp/modalities/action.h"
#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
#include "wllmrt/cpp/modalities/text.h"
#endif
#include "wllmrt/cpp/modalities/vision.h"
#include "wllmrt/cpp/models/pi05/spec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using wllm::modalities::DType;
using wllm::modalities::ActionStaging;
#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
using wllm::modalities::EmbeddingGatherSpec;
#endif
using wllm::modalities::Layout;
using wllm::modalities::MemoryPlace;
using wllm::modalities::PixelFormat;
using wllm::modalities::Shape;
using wllm::modalities::TensorView;
#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
using wllm::modalities::TextEmbeddingStaging;
#endif
using wllm::modalities::VisionFrame;
using wllm::modalities::bfloat16_to_float;
using wllm::modalities::float_to_bfloat16;
#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
using wllm::modalities::gather_token_embeddings;
using wllm::modalities::gather_token_embeddings_cpu;
#endif
using wllm::modalities::postprocess_action;
using wllm::modalities::preprocess_vision_cpu;
using wllm::modalities::preprocess_vision;
using wllm::modalities::required_action_output_bytes;
using wllm::modalities::required_vision_output_bytes;

namespace {

bool has_cuda_device() {
    int n = 0;
    cudaError_t rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
}

void test_vision_h2d_staging() {
    const auto spec = wllm::models::pi05::vision_preprocess_spec(1);
    const std::uint64_t bytes = required_vision_output_bytes(spec);

    void* device = nullptr;
    assert(cudaMalloc(&device, bytes) == cudaSuccess);

    const std::uint8_t rgb[] = {
        0, 127, 255, 255, 127, 0,
        10, 20, 30, 40, 50, 60,
    };
    VisionFrame frame;
    frame.name = "image";
    frame.image = {const_cast<std::uint8_t*>(rgb), sizeof(rgb),
                   DType::kUInt8, MemoryPlace::kHost, Layout::kHWC,
                   Shape{2, 2, 3}};
    frame.format = PixelFormat::kRGB8;
    frame.width = 2;
    frame.height = 2;

    TensorView dst{device, bytes, DType::kBFloat16, MemoryPlace::kDevice,
                   Layout::kNHWC, Shape{1, 224, 224, 3}};
    auto st = preprocess_vision(spec, {frame}, dst);
    assert(st.ok_status());

    std::vector<std::uint16_t> got(bytes / 2);
    assert(cudaMemcpy(got.data(), device, bytes,
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<std::uint16_t> ref(bytes / 2);
    TensorView ref_dst{ref.data(), bytes, DType::kBFloat16, MemoryPlace::kHost,
                       Layout::kNHWC, Shape{1, 224, 224, 3}};
    st = preprocess_vision_cpu(spec, {frame}, ref_dst);
    assert(st.ok_status());

    for (std::size_t i = 0; i < got.size(); ++i) {
        assert(std::fabs(bfloat16_to_float(got[i]) -
                         bfloat16_to_float(ref[i])) < 0.01f);
    }
    assert(std::fabs(bfloat16_to_float(got[0]) - (-1.0f)) < 0.01f);
    assert(std::fabs(bfloat16_to_float(got[1]) -
                     (127.0f / 127.5f - 1.0f)) < 0.01f);
    assert(std::fabs(bfloat16_to_float(got[2]) - 1.0f) < 0.01f);

    /* the persistent staging pool (hot path: no per-frame allocation) must
     * produce the same bytes as the allocating dev path, tick after tick */
    wllm::modalities::VisionStaging pool;
    st = wllm::modalities::vision_staging_create(&pool, 1, sizeof(rgb));
    assert(st.ok_status() && pool.device && pool.host_pinned);
    std::vector<std::uint16_t> pooled(bytes / 2);
    for (int round = 0; round < 3; ++round) {
        assert(cudaMemset(device, 0, bytes) == cudaSuccess);
        st = preprocess_vision(spec, {frame}, dst, nullptr, &pool);
        assert(st.ok_status());
        assert(cudaMemcpy(pooled.data(), device, bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(pooled == got);
    }
    /* over-capacity frames are a hard error, never a fallback allocation */
    VisionFrame big = frame;
    big.width = 64; big.height = 64; big.stride_bytes = 64 * 3;
    big.image.bytes = 64ull * 64 * 3;
    std::vector<std::uint8_t> big_pixels(64ull * 64 * 3, 7);
    big.image.data = big_pixels.data();
    big.image.shape = Shape{64, 64, 3};
    st = preprocess_vision(spec, {big}, dst, nullptr, &pool);
    assert(!st.ok_status());
    assert(st.code == wllm::modalities::StatusCode::kInsufficientStorage);
    wllm::modalities::vision_staging_destroy(&pool);
    assert(pool.device == nullptr && pool.host_pinned == nullptr);

    cudaFree(device);
}

#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
void test_divide_vision_cpu_cuda_exact() {
    wllm::modalities::VisionPreprocessSpec spec;
    spec.view_order = {"image"};
    spec.target_width = 7;
    spec.target_height = 5;
    spec.output_dtype = DType::kBFloat16;
    spec.output_layout = Layout::kNHWC;
    spec.normalize.mode = wllm::modalities::NormalizeMode::kDivideShift;
    spec.normalize.divisor = 127.5f;
    spec.normalize.shift = -1.0f;

    constexpr int width = 3;
    constexpr int height = 2;
    constexpr int stride = width * 3 + 5;
    std::vector<std::uint8_t> pixels(stride * height, 0xa5);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                pixels[static_cast<std::size_t>(y * stride + x * 3 +
                                                channel)] =
                    static_cast<std::uint8_t>(x * 31 + y * 47 +
                                              channel * 73);
            }
        }
    }
    VisionFrame frame;
    frame.name = "image";
    frame.image = {pixels.data(), pixels.size(), DType::kUInt8,
                   MemoryPlace::kHost, Layout::kHWC,
                   Shape{height, width, 3}};
    frame.format = PixelFormat::kRGB8;
    frame.width = width;
    frame.height = height;
    frame.stride_bytes = stride;

    const std::uint64_t bytes = required_vision_output_bytes(spec);
    std::vector<std::uint16_t> expected(bytes / sizeof(std::uint16_t));
    TensorView host_output{expected.data(), bytes, DType::kBFloat16,
                           MemoryPlace::kHost, Layout::kNHWC,
                           Shape{1, 5, 7, 3}};
    auto status = preprocess_vision_cpu(spec, {frame}, host_output);
    assert(status.ok_status());

    void* device = nullptr;
    assert(cudaMalloc(&device, bytes) == cudaSuccess);
    TensorView device_output{device, bytes, DType::kBFloat16,
                             MemoryPlace::kDevice, Layout::kNHWC,
                             Shape{1, 5, 7, 3}};
    wllm::modalities::VisionStaging staging;
    status = wllm::modalities::vision_staging_create(
        &staging, 1, pixels.size());
    assert(status.ok_status());
    status = preprocess_vision(spec, {frame}, device_output, nullptr,
                               &staging);
    assert(status.ok_status());
    std::vector<std::uint16_t> actual(expected.size());
    assert(cudaMemcpy(actual.data(), device, bytes,
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(actual == expected);
    wllm::modalities::vision_staging_destroy(&staging);
    cudaFree(device);
}
#endif

void test_action_d2h_staging() {
    auto spec = wllm::models::pi05::action_postprocess_spec(
        {10.0f, 20.0f, 30.0f}, {2.0f, 3.0f, 4.0f},
        /*chunk=*/1, /*model_dim=*/4, /*robot_dim=*/3);
    std::vector<std::uint16_t> host(4);
    host[0] = float_to_bfloat16(1.0f);
    host[1] = float_to_bfloat16(-2.0f);
    host[2] = float_to_bfloat16(3.0f);
    host[3] = float_to_bfloat16(99.0f);

    const std::uint64_t bytes = required_action_output_bytes(spec, DType::kBFloat16);
    void* device = nullptr;
    assert(cudaMalloc(&device, bytes) == cudaSuccess);
    assert(cudaMemcpy(device, host.data(), bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess);

    TensorView src{device, bytes, DType::kBFloat16, MemoryPlace::kDevice,
                   Layout::kFlat, Shape{1, 4}};
    std::vector<float> actions;
    auto st = postprocess_action(spec, src, &actions);
    assert(st.ok_status());
    assert(actions.size() == 3);
    assert(std::fabs(actions[0] - 12.0f) < 0.01f);
    assert(std::fabs(actions[1] - 17.0f) < 0.01f);
    assert(std::fabs(actions[2] - 34.0f) < 0.01f);
    ActionStaging staging;
    st = wllm::modalities::action_staging_create(&staging, bytes);
    assert(st.ok_status());
    const std::size_t capacity = actions.capacity();
    for (int round = 0; round < 1000; ++round) {
        st = postprocess_action(spec, src, &actions, nullptr, &staging);
        assert(st.ok_status());
        assert(actions.capacity() == capacity);
    }
    wllm::modalities::action_staging_destroy(&staging);
    cudaFree(device);
}

#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
void test_text_embedding_device_gather() {
    const std::vector<float> table = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    const std::int32_t ids[] = {2, 0};
    std::vector<float> expected(8, 0.0f);
    TensorView host_table{const_cast<float*>(table.data()),
                          table.size() * sizeof(float), DType::kFloat32,
                          MemoryPlace::kHost, Layout::kFlat, Shape{3, 4}};
    TensorView host_output{expected.data(), expected.size() * sizeof(float),
                           DType::kFloat32, MemoryPlace::kHost,
                           Layout::kFlat, Shape{2, 4}};
    const EmbeddingGatherSpec spec{3, 4, 2.0f};
    auto status = gather_token_embeddings_cpu(spec, ids, 2, host_table,
                                               host_output);
    assert(status.ok_status());

    void* device_table = nullptr;
    void* device_output = nullptr;
    assert(cudaMalloc(&device_table, table.size() * sizeof(float)) ==
           cudaSuccess);
    assert(cudaMalloc(&device_output, expected.size() * sizeof(float)) ==
           cudaSuccess);
    assert(cudaMemcpy(device_table, table.data(), table.size() * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    TensorView table_view{device_table, table.size() * sizeof(float),
                          DType::kFloat32, MemoryPlace::kDevice,
                          Layout::kFlat, Shape{3, 4}};
    TensorView output_view{device_output, expected.size() * sizeof(float),
                           DType::kFloat32, MemoryPlace::kDevice,
                           Layout::kFlat, Shape{2, 4}};
    TextEmbeddingStaging staging;
    status = wllm::modalities::text_embedding_staging_create(&staging, 2);
    assert(status.ok_status());
    status = gather_token_embeddings(spec, ids, 2, table_view, output_view,
                                     nullptr, &staging);
    assert(status.ok_status());
    std::vector<float> actual(expected.size(), 0.0f);
    assert(cudaMemcpy(actual.data(), device_output,
                      actual.size() * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(actual == expected);
    wllm::modalities::text_embedding_staging_destroy(&staging);
    cudaFree(device_output);
    cudaFree(device_table);
}
#endif

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::cout << "SKIP - no CUDA device\n";
        return 0;
    }
    test_vision_h2d_staging();
#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
    test_divide_vision_cpu_cuda_exact();
#endif
    test_action_d2h_staging();
#ifdef WLLM_CPP_TEST_WITH_CUDA_KERNELS
    test_text_embedding_device_gather();
#endif
    std::cout << "PASS - CUDA modality kernels/staging\n";
    return 0;
}
