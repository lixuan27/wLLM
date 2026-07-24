#ifndef WLLM_CPP_MODELS_PI05_NATIVE_CALIBRATION_SESSION_H
#define WLLM_CPP_MODELS_PI05_NATIVE_CALIBRATION_SESSION_H

#include "wllmrt/cpp/models/pi05/support/native_calibration.h"
#include "wllmrt/cpp/modalities/vision.h"

#include <memory>

namespace wllm {
namespace models {
namespace pi05 {

class NativeCalibrationSession final {
public:
    static std::unique_ptr<NativeCalibrationSession> create(
        const NativeCalibrationConfig& config,
        double percentile,
        modalities::Status* status);
    ~NativeCalibrationSession();

    NativeCalibrationSession(const NativeCalibrationSession&) = delete;
    NativeCalibrationSession& operator=(const NativeCalibrationSession&) =
        delete;

    modalities::Status observe(
        const std::string& prompt,
        const float* state,
        std::uint64_t n_state,
        const std::vector<modalities::VisionFrame>& frames,
        const float* noise,
        std::uint64_t n_noise,
        std::uint64_t noise_seed);
    modalities::Status finalize(const std::string& artifact_path) const;
    std::uint64_t sample_count() const;

private:
    struct Impl;
    explicit NativeCalibrationSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace pi05
}  // namespace models
}  // namespace wllm

#endif  // WLLM_CPP_MODELS_PI05_NATIVE_CALIBRATION_SESSION_H
