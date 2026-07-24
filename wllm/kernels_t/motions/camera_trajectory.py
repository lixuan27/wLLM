import torch
from torch.utils.cpp_extension import load_inline

cpp_src = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

torch::Tensor generate_camera_trajectory_local_cpu(
    torch::Tensor motion_codes,   // (N,) int64 CPU
    torch::Tensor T,              // (4,4) float32 CPU, will be modified in-place
    double step                   // scalar
) {
    TORCH_CHECK(motion_codes.device().is_cpu(), "motion_codes must be CPU");
    TORCH_CHECK(T.device().is_cpu(), "T must be CPU");
    TORCH_CHECK(motion_codes.dtype() == torch::kLong, "motion_codes must be int64");
    TORCH_CHECK(T.dtype() == torch::kFloat32, "T must be float32");
    TORCH_CHECK(motion_codes.dim() == 1, "motion_codes must be (N,)");
    TORCH_CHECK(T.sizes() == torch::IntArrayRef({4,4}), "T must be (4,4)");
    TORCH_CHECK(T.is_contiguous(), "T must be contiguous");
    TORCH_CHECK(motion_codes.is_contiguous(), "motion_codes must be contiguous");

    const auto N = motion_codes.size(0);
    auto poses = torch::empty({N, 4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    auto codes = motion_codes.data_ptr<int64_t>();
    float* Tptr = T.data_ptr<float>();
    float* Pptr = poses.data_ptr<float>();

    // Read constant rotation R = T[:3,:3]
    // T is row-major contiguous: T[r*4 + c]
    const float R00 = Tptr[0],  R01 = Tptr[1],  R02 = Tptr[2];
    const float R10 = Tptr[4],  R11 = Tptr[5],  R12 = Tptr[6];
    const float R20 = Tptr[8],  R21 = Tptr[9],  R22 = Tptr[10];

    // Current translation
    float tx = Tptr[3];
    float ty = Tptr[7];
    float tz = Tptr[11];

    const float s = static_cast<float>(step);

    // For each step: local delta in {(+/-x), (+/-z)} then world delta = R * local
    for (int64_t i = 0; i < N; ++i) {
        float lx = 0.f, ly = 0.f, lz = 0.f;
        const int64_t c = codes[i];
        // 1:w(+z), 2:s(-z), 3:d(+x), 4:a(-x)
        if (c == 1) lz =  1.f;
        else if (c == 2) lz = -1.f;
        else if (c == 3) lx =  1.f;
        else if (c == 4) lx = -1.f;

        lx *= s; ly *= s; lz *= s;

        // world_delta = R @ local
        const float dx = R00*lx + R01*ly + R02*lz;
        const float dy = R10*lx + R11*ly + R12*lz;
        const float dz = R20*lx + R21*ly + R22*lz;

        tx += dx; ty += dy; tz += dz;

        // write snapshot pose i (copy full T then overwrite translation)
        float* Pi = Pptr + i * 16;
        // copy T (16 floats)
        // (T is small; memcpy is fine)
        std::memcpy(Pi, Tptr, 16 * sizeof(float));

        Pi[3]  = tx;
        Pi[7]  = ty;
        Pi[11] = tz;
    }

    // in-place update T translation to final
    Tptr[3]  = tx;
    Tptr[7]  = ty;
    Tptr[11] = tz;

    return poses;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_camera_trajectory_local_cpu", &generate_camera_trajectory_local_cpu,
        "generate_camera_trajectory_local (CPU float32, in-place T)");
}
"""

ext = load_inline(
    name="traj_ext_cpu_f32",
    cpp_sources=cpp_src,
    functions=None,
    extra_cflags=["-O3"],
    with_cuda=False,
    verbose=False,
)

def generate_camera_trajectory_local(motions: torch.Tensor, T: torch.Tensor, step: float = 0.08):
    # 强制 CPU/float32（符合你的要求）
    assert T.device.type == "cpu"
    assert T.dtype == torch.float32
    poses = ext.generate_camera_trajectory_local_cpu(motions, T, float(step))
    return poses
