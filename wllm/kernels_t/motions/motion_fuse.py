import torch
from torch.utils.cpp_extension import load_inline

cpp_src = r"""
#include <torch/extension.h>
#include <cmath>
#include <cstring>

// 16 -> 9 LUT (illegal -> -1)
static const int LUT16_TO_9[16] = {
    0,1,2,-1,
    3,5,7,-1,
    4,6,8,-1,
    -1,-1,-1,-1
};

std::vector<torch::Tensor> motions_to_matrix_fused_latency_fastmath_cpu(
    torch::Tensor motions,   // (N,) int64 CPU: 0 noop, 1 w, 2 s, 3 d, 4 a
    torch::Tensor T,         // (4,4) float32 CPU, in-place updated
    torch::Tensor Cinv,      // (4,4) float32 CPU, in-place updated to last w2c
    bool first_chunk,
    float step
) {
    TORCH_CHECK(motions.device().is_cpu(), "motions must be CPU");
    TORCH_CHECK(T.device().is_cpu(), "T must be CPU");
    TORCH_CHECK(Cinv.device().is_cpu(), "Cinv must be CPU");

    TORCH_CHECK(motions.dtype() == torch::kLong, "motions must be int64");
    TORCH_CHECK(T.dtype() == torch::kFloat32, "T must be float32");
    TORCH_CHECK(Cinv.dtype() == torch::kFloat32, "Cinv must be float32");

    TORCH_CHECK(motions.dim() == 1, "motions must be (N,)");
    TORCH_CHECK(T.sizes() == torch::IntArrayRef({4,4}), "T must be (4,4)");
    TORCH_CHECK(Cinv.sizes() == torch::IntArrayRef({4,4}), "Cinv must be (4,4)");

    TORCH_CHECK(motions.is_contiguous(), "motions must be contiguous");
    TORCH_CHECK(T.is_contiguous(), "T must be contiguous");
    TORCH_CHECK(Cinv.is_contiguous(), "Cinv must be contiguous");

    const int64_t N = motions.size(0);

    auto viewmats = torch::empty({N,4,4}, T.options()); // float32
    auto action   = torch::empty({N},     T.options()); // float32

    const int64_t* __restrict mptr = motions.data_ptr<int64_t>();
    float* __restrict Tptr = T.data_ptr<float>();
    float* __restrict Cptr = Cinv.data_ptr<float>();
    float* __restrict Vptr = viewmats.data_ptr<float>();
    float* __restrict Aptr = action.data_ptr<float>();

    // ---- intrinsic base (3,3), Ks = expand view (N,3,3) ----
    auto Kbase = torch::empty({3,3}, T.options());
    float* Kb = Kbase.data_ptr<float>();
    Kb[0]=0.5051f; Kb[1]=0.f;    Kb[2]=0.5f;
    Kb[3]=0.f;    Kb[4]=0.8979f; Kb[5]=0.5f;
    Kb[6]=0.f;    Kb[7]=0.f;    Kb[8]=1.f;
    auto Ks = Kbase.unsqueeze(0).expand({N,3,3}); // view, zero-copy

    // ---- constant rotation from initial T (trajectory doesn't rotate) ----
    const float R00=Tptr[0],  R01=Tptr[1],  R02=Tptr[2];
    const float R10=Tptr[4],  R11=Tptr[5],  R12=Tptr[6];
    const float R20=Tptr[8],  R21=Tptr[9],  R22=Tptr[10];

    // current translation
    float tx=Tptr[3], ty=Tptr[7], tz=Tptr[11];

    // For relative at i==0 when !first_chunk, we need Cinv row0 and row2 only.
    const float C0_0=Cptr[0],  C0_1=Cptr[1],  C0_2=Cptr[2],  C0_3=Cptr[3];
    const float C2_0=Cptr[8],  C2_1=Cptr[9],  C2_2=Cptr[10], C2_3=Cptr[11];

    // prev_b0 / prev_b2 store previous w2c translation components for row0 and row2:
    // w2c row0: [R00 R10 R20 b0], b0 = -(R00*tx_prev + R10*ty_prev + R20*tz_prev)
    // w2c row2: [R02 R12 R22 b2], b2 = -(R02*tx_prev + R12*ty_prev + R22*tz_prev)
    float prev_b0 = C0_3; // only used if !first_chunk at i=0 (but we compute via full Cinv anyway)
    float prev_b2 = C2_3;

    for (int64_t i=0; i<N; ++i) {
        // ---- local delta lookup ----
        float lx=0.f, lz=0.f;
        const int64_t c = mptr[i];
        if (c==1)      lz =  1.f;
        else if (c==2) lz = -1.f;
        else if (c==3) lx =  1.f;
        else if (c==4) lx = -1.f;

        lx *= step; lz *= step;

        // ---- integrate translation ----
        tx += (R00*lx + R02*lz);
        ty += (R10*lx + R12*lz);
        tz += (R20*lx + R22*lz);

        // ---- compute dot products once ----
        const float dot0 = (R00*tx + R10*ty + R20*tz);
        const float dot2 = (R02*tx + R12*ty + R22*tz);

        // ---- current w2c translation components ----
        const float b0 = -dot0;
        const float b2 = -dot2;

        // ---- write w2c[i] ----
        float* Vi = Vptr + i*16;

        Vi[0]=R00; Vi[1]=R10; Vi[2]=R20; Vi[3]=b0;
        Vi[4]=R01; Vi[5]=R11; Vi[6]=R21; Vi[7]=-(R01*tx + R11*ty + R21*tz);
        Vi[8]=R02; Vi[9]=R12; Vi[10]=R22;Vi[11]=b2;
        Vi[12]=0.f; Vi[13]=0.f; Vi[14]=0.f; Vi[15]=1.f;

        // ---- relative translation (only x,z) ----
        float rel_x, rel_z;

        if (i == 0) {
            if (first_chunk) {
                // your behavior: action[0]=0
                Aptr[i] = 0.f;

                // update prev_b for next i
                prev_b0 = b0;
                prev_b2 = b2;
                continue;
            } else {
                // rel = Cinv @ pose0, only need rel_x (row0) and rel_z (row2)
                rel_x = C0_0*tx + C0_1*ty + C0_2*tz + C0_3;
                rel_z = C2_0*tx + C2_1*ty + C2_2*tz + C2_3;
            }
        } else {
            // rel_x = dot0 + prev_b0, rel_z = dot2 + prev_b2
            rel_x = dot0 + prev_b0;
            rel_z = dot2 + prev_b2;
        }

        // ---- trans action (no sqrt) ----
        const float norm2 = rel_x*rel_x + rel_z*rel_z;
        int idx = 0;

        if (norm2 > 1e-8f) {              // (1e-4)^2
            const float thr2 = 0.25f * norm2;

            const float z2 = rel_z*rel_z;
            if (z2 > thr2) idx |= (rel_z > 0.f) ? 1 : 2;

            const float x2 = rel_x*rel_x;
            if (x2 > thr2) idx |= (rel_x > 0.f) ? 4 : 8;
        }

        Aptr[i] = (float)LUT16_TO_9[idx];

        // update prev_b for next step
        prev_b0 = b0;
        prev_b2 = b2;
    }

    // ---- in-place update T and Cinv ----
    if (N > 0) {
        Tptr[3]=tx; Tptr[7]=ty; Tptr[11]=tz;
        std::memcpy(Cptr, Vptr + (N-1)*16, 16*sizeof(float));
    }

    return {viewmats, Ks, action};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("motions_to_matrix_fused_latency_fastmath_cpu",
          &motions_to_matrix_fused_latency_fastmath_cpu,
          "Fused motions_to_matrix (CPU float32, N~small latency optimized, -ffast-math)");
}
"""

motions_ext = load_inline(
    name="motions_fused_ext_latency_fastmath",
    cpp_sources=cpp_src,
    with_cuda=False,
    extra_cflags=["-O3", "-march=native", "-ffast-math"],
    verbose=True,
)

def motions_to_matrix_cpp(
    motions: torch.Tensor,
    T: torch.Tensor,
    Cinv: torch.Tensor,
    first_chunk: bool = False,
    step: float = 0.08,
):
    """
    motions: (N,) int64 CPU  [0 noop, 1 w, 2 s, 3 d, 4 a]
    T: (4,4) float32 CPU (in-place updated)
    Cinv: (4,4) float32 CPU (in-place updated to last w2c)

    returns float32:
      viewmats: (N,4,4)
      Ks:       (N,3,3) (expand view, zero-copy)
      action:   (N,)
    """
    assert motions.device.type == "cpu" and motions.dtype == torch.long and motions.is_contiguous()
    assert T.device.type == "cpu" and T.dtype == torch.float32 and T.is_contiguous()
    assert Cinv.device.type == "cpu" and Cinv.dtype == torch.float32 and Cinv.is_contiguous()

    return motions_ext.motions_to_matrix_fused_latency_fastmath_cpu(
        motions, T, Cinv, bool(first_chunk), float(step)
    )

if __name__ == "__main__":
    T = torch.eye(4, dtype=torch.float32)
    Cinv = torch.eye(4, dtype=torch.float32)
    motions = torch.tensor([1, 3, 2, 4], dtype=torch.long)  # N ~= 4

    viewmats, Ks, action = motions_to_matrix_cpp(motions, T, Cinv, first_chunk=True, step=0.08)
    print("viewmats:", viewmats.shape)
    print("Ks:", Ks.shape, "Ks is view:", not Ks.is_contiguous())
    print("action:", action)
    print("T updated:\n", T)
    print("Cinv updated:\n", Cinv)
