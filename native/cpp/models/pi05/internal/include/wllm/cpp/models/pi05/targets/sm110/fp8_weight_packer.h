#ifndef WLLM_CPP_MODELS_PI05_TARGETS_SM110_FP8_WEIGHT_PACKER_H
#define WLLM_CPP_MODELS_PI05_TARGETS_SM110_FP8_WEIGHT_PACKER_H

#include "wllmrt/cpp/models/pi05/model/linear_weight_groups.h"
#include "wllmrt/exec.h"

#include <cstddef>
#include <vector>

namespace wllm {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {

struct Sm110Fp8PackedWeight final {
    Pi05LinearWeightKey key;
    wrt_buffer values = nullptr;
    wrt_buffer scale = nullptr;
    Pi05ResolvedWeight view;
    modalities::Shape physical_shape;
    float host_scale = 0.0f;
};

class Sm110Fp8WeightPacker final : public Pi05LinearWeightGroupSink {
public:
    explicit Sm110Fp8WeightPacker(wrt_ctx context,
                                  Pi05Stream stream = 0);

    modalities::Status record(
        const Pi05LinearWeightGroup& group) override;
    modalities::Status finish();

    std::size_t size() const { return packed_.size(); }
    const Sm110Fp8PackedWeight* packed_weight(std::size_t index) const;
    const Sm110Fp8PackedWeight* packed_weight(
        const Pi05LinearWeightKey& key) const;
    bool finished() const { return finished_; }

private:
    modalities::Status pack(const Pi05LinearWeightGroup& group,
                            bool pair,
                            bool transpose);
    modalities::Status fail(modalities::Status status);

    wrt_ctx context_ = nullptr;
    Pi05Stream stream_ = 0;
    std::vector<Sm110Fp8PackedWeight> packed_;
    bool finished_ = false;
    bool failed_ = false;
};

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace wllm

#endif  // WLLM_CPP_MODELS_PI05_TARGETS_SM110_FP8_WEIGHT_PACKER_H
