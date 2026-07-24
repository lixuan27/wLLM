#ifndef WLLM_CPP_MODELS_PI05_EXPORT_H
#define WLLM_CPP_MODELS_PI05_EXPORT_H

#if defined(_WIN32)
#if defined(WLLM_PI05_C_BUILDING_LIBRARY)
#define WLLM_PI05_C_API __declspec(dllexport)
#else
#define WLLM_PI05_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define WLLM_PI05_C_API __attribute__((visibility("default")))
#else
#define WLLM_PI05_C_API
#endif

#endif  // WLLM_CPP_MODELS_PI05_EXPORT_H
