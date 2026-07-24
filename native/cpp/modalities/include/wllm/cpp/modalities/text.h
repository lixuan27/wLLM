#ifndef WLLM_MODALITIES_TEXT_H
#define WLLM_MODALITIES_TEXT_H

#include "wllmrt/cpp/modalities/types.h"

#include <cstdint>

namespace wllm {
namespace modalities {

struct EmbeddingGatherSpec {
    std::uint64_t vocab_size = 0;
    std::uint64_t hidden_dim = 0;
    float scale = 1.0f;
};

struct TextEmbeddingStaging {
    void* device_token_ids = nullptr;
    void* device_status = nullptr;
    std::uint64_t max_tokens = 0;
};

Status gather_token_embeddings_cpu(const EmbeddingGatherSpec& spec,
                                   const std::int32_t* token_ids,
                                   std::uint64_t n_tokens,
                                   TensorView embedding_table,
                                   TensorView output);

Status text_embedding_staging_create(TextEmbeddingStaging* out,
                                     std::uint64_t max_tokens);
void text_embedding_staging_destroy(TextEmbeddingStaging*);

Status gather_token_embeddings(const EmbeddingGatherSpec& spec,
                               const std::int32_t* token_ids,
                               std::uint64_t n_tokens,
                               TensorView embedding_table,
                               TensorView output,
                               void* stream = nullptr,
                               TextEmbeddingStaging* staging = nullptr);

}  // namespace modalities
}  // namespace wllm

#endif  // WLLM_MODALITIES_TEXT_H
