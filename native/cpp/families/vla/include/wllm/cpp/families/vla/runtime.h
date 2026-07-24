#ifndef WLLM_CPP_FAMILIES_VLA_RUNTIME_H
#define WLLM_CPP_FAMILIES_VLA_RUNTIME_H

#include "wllmrt/cpp/families/vla/manifest.h"
#include "wllmrt/cpp/runtime.h"

namespace wllm {
namespace families {
namespace vla {

/* Common VLA runtime shape. Concrete model frontends bind this family
 * contract to their own buffers, input path, and output schema. */
class Runtime : public wllm::runtime::ModelRuntime {
public:
    ~Runtime() override = default;
    virtual const Manifest& manifest() const = 0;
};

}  // namespace vla
}  // namespace families
}  // namespace wllm

#endif  // WLLM_CPP_FAMILIES_VLA_RUNTIME_H
