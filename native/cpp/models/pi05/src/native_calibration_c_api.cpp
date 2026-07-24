#include "wllmrt/cpp/models/pi05/c_api.h"

#include "config_map.h"
#include "wllmrt/cpp/models/pi05/spec.h"
#include "native_open_internal.h"

#if defined(WLLM_CPP_HAS_SENTENCEPIECE) && \
    (defined(WLLM_CPP_WITH_PI05_SM120_TARGET) || \
     defined(WLLM_CPP_WITH_PI05_SM110_TARGET))
#define WLLM_CPP_HAS_NATIVE_CALIBRATION 1
#include "wllmrt/cpp/models/pi05/support/native_calibration_session.h"
#endif

#include <cstddef>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

using wllm::modalities::DType;
using wllm::modalities::Layout;
using wllm::modalities::MemoryPlace;
using wllm::modalities::PixelFormat;
using wllm::modalities::Shape;
using wllm::modalities::VisionFrame;

thread_local std::string g_calibration_create_error;

struct wrt_pi05_calibration_session_s {
#if defined(WLLM_CPP_HAS_NATIVE_CALIBRATION)
    std::unique_ptr<wllm::models::pi05::NativeCalibrationSession> impl;
#endif
    std::string last_error;
    std::vector<std::string> view_names;
    std::vector<VisionFrame> frames;
};

namespace {

#if defined(WLLM_CPP_HAS_NATIVE_CALIBRATION)
int set_error(wrt_pi05_calibration_session* session,
              const wllm::modalities::Status& status) {
    if (session) session->last_error = status.message;
    return wllm::models::pi05::cface::status_code(status);
}

bool convert_frames(wrt_pi05_calibration_session* session,
                    const wrt_pi05_calibration_sample_v1& sample) {
    if (!session || session->view_names.empty() || !sample.frames ||
        sample.n_frames != session->view_names.size()) {
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(sample.n_frames);
    session->frames.assign(count, VisionFrame{});
    std::vector<bool> seen(count, false);
    for (std::size_t i = 0; i < count; ++i) {
        const wrt_pi05_vision_frame& source = sample.frames[i];
        if (source.struct_size < sizeof(wrt_pi05_vision_frame) ||
            !source.name || !source.data || source.width <= 0 ||
            source.height <= 0 || source.stride_bytes <= 0 ||
            source.pixel_format != WRT_PI05_PIXEL_RGB8) {
            return false;
        }
        std::size_t slot = count;
        for (std::size_t candidate = 0; candidate < count; ++candidate) {
            if (session->view_names[candidate] == source.name) {
                slot = candidate;
                break;
            }
        }
        if (slot == count || seen[slot]) return false;
        seen[slot] = true;
        VisionFrame& frame = session->frames[slot];
        frame.name = source.name;
        frame.image.data = const_cast<void*>(source.data);
        frame.image.bytes = source.bytes;
        frame.image.dtype = DType::kUInt8;
        frame.image.place = MemoryPlace::kHost;
        frame.image.layout = Layout::kHWC;
        frame.image.shape =
            Shape{static_cast<std::uint64_t>(source.height),
                  static_cast<std::uint64_t>(source.width), 3};
        frame.format = PixelFormat::kRGB8;
        frame.width = source.width;
        frame.height = source.height;
        frame.stride_bytes = source.stride_bytes;
        frame.timestamp_ns = source.timestamp_ns;
    }
    for (bool present : seen) {
        if (!present) return false;
    }
    return true;
}
#endif

}  // namespace

extern "C" int wrt_pi05_calibration_create_v1(
    const char* config_json,
    double percentile,
    wrt_pi05_calibration_session** out) try {
    g_calibration_create_error.clear();
    if (!out) {
        g_calibration_create_error = "calibration out is null";
        return -1;
    }
    *out = nullptr;
#if defined(WLLM_CPP_HAS_NATIVE_CALIBRATION)
    wllm::models::pi05::NativeOpenConfig open_config;
    int rc = wllm::models::pi05::parse_native_open_config(
        config_json, &open_config, &g_calibration_create_error);
    if (rc != 0) return rc;
    if (open_config.precision == "bf16") {
        g_calibration_create_error =
            "Pi0.5 calibration precision must resolve to fp8_e4m3fn";
        return -1;
    }
    wllm::models::pi05::NativeCalibrationConfig config;
    config.checkpoint_path = std::move(open_config.checkpoint_path);
    config.tokenizer_model_path = std::move(open_config.tokenizer_model_path);
    config.max_prompt_tokens = open_config.max_prompt_tokens;
    config.state_dim = open_config.state_dim;
    config.num_views = open_config.num_views;
    config.chunk_size = open_config.chunk;
    config.num_steps = open_config.num_steps;
    config.vision_pool_factor = open_config.vision_pool_factor;
    config.max_frame_width = open_config.max_frame_width;
    config.max_frame_height = open_config.max_frame_height;
    config.state_q01 = std::move(open_config.state_q01);
    config.state_q99 = std::move(open_config.state_q99);

    wllm::modalities::Status status;
    auto impl = wllm::models::pi05::NativeCalibrationSession::create(
        config, percentile, &status);
    if (!impl) {
        g_calibration_create_error = status.message;
        return wllm::models::pi05::cface::status_code(status);
    }
    std::unique_ptr<wrt_pi05_calibration_session> session(
        new (std::nothrow) wrt_pi05_calibration_session);
    if (!session) {
        g_calibration_create_error = "calibration handle allocation failed";
        return -6;
    }
    session->impl = std::move(impl);
    session->view_names =
        wllm::models::pi05::vision_preprocess_spec(config.num_views)
            .view_order;
    *out = session.release();
    return 0;
#else
    (void)config_json;
    (void)percentile;
    g_calibration_create_error =
        "Pi0.5 calibration requires a native FP8 target and SentencePiece";
    return -3;
#endif
} catch (const std::exception& error) {
    if (out) *out = nullptr;
    g_calibration_create_error = error.what();
    return -6;
} catch (...) {
    if (out) *out = nullptr;
    g_calibration_create_error = "calibration creation failed";
    return -6;
}

extern "C" int wrt_pi05_calibration_observe_v1(
    wrt_pi05_calibration_session* session,
    const wrt_pi05_calibration_sample_v1* sample) try {
#if defined(WLLM_CPP_HAS_NATIVE_CALIBRATION)
    if (!session || !session->impl || !sample ||
        sample->struct_size < sizeof(wrt_pi05_calibration_sample_v1) ||
        !sample->prompt || (!sample->state && sample->n_state) ||
        (!sample->noise && sample->n_noise) ||
        !convert_frames(session, *sample)) {
        if (session) session->last_error = "calibration sample is invalid";
        return -1;
    }
    const wllm::modalities::Status status = session->impl->observe(
        sample->prompt, sample->state, sample->n_state, session->frames,
        sample->noise, sample->n_noise, sample->noise_seed);
    if (!status.ok_status()) return set_error(session, status);
    session->last_error.clear();
    return 0;
#else
    (void)session;
    (void)sample;
    return -3;
#endif
} catch (const std::exception& error) {
    if (session) session->last_error = error.what();
    return -6;
} catch (...) {
    if (session) session->last_error = "calibration observation failed";
    return -6;
}

extern "C" int wrt_pi05_calibration_finalize_v1(
    wrt_pi05_calibration_session* session,
    const char* artifact_path) try {
#if defined(WLLM_CPP_HAS_NATIVE_CALIBRATION)
    if (!session || !session->impl || !artifact_path || !artifact_path[0]) {
        if (session) session->last_error = "calibration output path is invalid";
        return -1;
    }
    const wllm::modalities::Status status =
        session->impl->finalize(artifact_path);
    if (!status.ok_status()) return set_error(session, status);
    session->last_error.clear();
    return 0;
#else
    (void)session;
    (void)artifact_path;
    return -3;
#endif
} catch (const std::exception& error) {
    if (session) session->last_error = error.what();
    return -6;
} catch (...) {
    if (session) session->last_error = "calibration finalization failed";
    return -6;
}

extern "C" std::uint64_t wrt_pi05_calibration_sample_count_v1(
    const wrt_pi05_calibration_session* session) {
#if defined(WLLM_CPP_HAS_NATIVE_CALIBRATION)
    return session && session->impl ? session->impl->sample_count() : 0;
#else
    (void)session;
    return 0;
#endif
}

extern "C" const char* wrt_pi05_calibration_last_error_v1(
    const wrt_pi05_calibration_session* session) {
    return session ? session->last_error.c_str()
                   : g_calibration_create_error.c_str();
}

extern "C" const char* wrt_pi05_calibration_create_last_error_v1() {
    return g_calibration_create_error.c_str();
}

extern "C" void wrt_pi05_calibration_destroy_v1(
    wrt_pi05_calibration_session* session) {
    delete session;
}
