import torch
from torch.utils.cpp_extension import load_inline

cpp_src = r"""
#include <torch/extension.h>
#include <omp.h>
#include <cstdint>

// 固定 LUT
static inline int64_t lut16_to9(int idx) {
    static const int8_t LUT[16] = {
        0, 1, 2, -1,
        3, 5, 7, -1,
        4, 6, 8, -1,
        -1, -1, -1, -1
    };
    return (int64_t)LUT[idx];
}

torch::Tensor trans_one_hot_to_label9_cpu(torch::Tensor one_hot4) {
    TORCH_CHECK(one_hot4.device().is_cpu(), "must be CPU");
    TORCH_CHECK(one_hot4.dtype() == torch::kLong, "must be int64");
    TORCH_CHECK(one_hot4.dim() == 2 && one_hot4.size(1) == 4, "shape must be (N,4)");
    TORCH_CHECK(one_hot4.is_contiguous(), "must be contiguous");

    const int64_t N = one_hot4.size(0);
    auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));

    const int64_t* __restrict inp = one_hot4.data_ptr<int64_t>();
    int64_t* __restrict outp = out.data_ptr<int64_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const int64_t* row = inp + i * 4;

        // 非0即1
        const int idx =
            ((row[0] != 0) << 0) |
            ((row[1] != 0) << 1) |
            ((row[2] != 0) << 2) |
            ((row[3] != 0) << 3);

        outp[i] = lut16_to9(idx);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trans_one_hot_to_label9_cpu", &trans_one_hot_to_label9_cpu,
        "Fast trans_one_hot (int64, CPU, OMP)");
}
"""

trans_ext = load_inline(
    name="trans_lut_ext_fast",
    cpp_sources=cpp_src,
    with_cuda=False,
    extra_cflags=["-O3", "-march=native", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)


def trans_one_hot_to_label9_cpp(trans_one_hot: torch.Tensor):
    if not trans_one_hot.is_contiguous():
        trans_one_hot = trans_one_hot.contiguous()
    return trans_ext.trans_one_hot_to_label9_cpu(trans_one_hot)
