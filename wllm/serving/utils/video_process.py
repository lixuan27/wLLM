import torch





@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def latent2rgb(x: torch.Tensor):
    return ((x * 0.5 + 0.5)
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 4, 1))

