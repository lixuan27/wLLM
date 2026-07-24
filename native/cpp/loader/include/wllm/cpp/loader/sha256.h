#ifndef WLLM_CPP_LOADER_SHA256_H
#define WLLM_CPP_LOADER_SHA256_H

#include <string>

namespace wllm {
namespace loader {

bool sha256_file(const std::string& path, std::string* hex,
                 std::string* error = nullptr);

/* Cache a verified digest next to the source file. A cache entry is reused
 * only while device, inode, size, mtime, and ctime still match. Read-only
 * checkpoint directories simply fall back to sha256_file(). */
bool sha256_file_cached(const std::string& path, std::string* hex,
                        bool* cache_hit = nullptr,
                        std::string* error = nullptr);

}  // namespace loader
}  // namespace wllm

#endif  // WLLM_CPP_LOADER_SHA256_H
