"""One Krea SP rank process. Bootstraps the shared distributed / model-parallel
groups (SP over the Krea GPUs, tp=1), then runs the SP orchestrator. Launched
by launch_mgpu.py under torchrun-style env (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*).

KREA_SP_VAE_SPLIT=1 selects the DiT-SP || VAE topology, which needs its own
group layout and orchestrator (see orchestrator_sp_vae_split).
"""

import os
import sys

import torch  # noqa: E402

from wllm.serving.distributed.parallel_state import (  # noqa: E402
    maybe_init_distributed_environment_and_model_parallel,
    get_world_rank, get_world_size, get_local_torch_device,
)
from wllm.serving.distributed.communication_op import warmup_sequence_parallel_communication  # noqa: E402
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_sp import KreaOrchestratorSP  # noqa: E402
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_stream_sp import KreaOrchestratorStreamSP  # noqa: E402
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_sp_vae_split import (  # noqa: E402
    KreaSPVaeSplit, init_sp_vae_split_parallel_state)


def main():
    cfg_path = sys.argv[1]
    sam_link_name = sys.argv[2]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    split = os.environ.get("KREA_SP_VAE_SPLIT", "0") == "1"
    # SP_SIZE < WORLD_SIZE ⇒ DiT is replicated (SP=1) but the VAE decoder still
    # width-tiles over the full world group (get_world_size()), isolating the VAE
    # decode-tiling lever. Default: SP = world (full frame-SP); under
    # --sp-vae-split the last rank runs the VAE stage instead of joining SP, so
    # SP defaults to world-1 there.
    sp_size = int(os.environ.get("SP_SIZE", str(world - 1 if split else world)))

    if split:
        # The stock bootstrap slices the world into uniform consecutive SP
        # groups, which leaves the extra VAE rank in no group at all.
        init_sp_vae_split_parallel_state(sp_size)
    else:
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp_size)
    warmup_sequence_parallel_communication()

    device = get_local_torch_device()
    torch.cuda.set_device(device)
    if split:
        KreaSPVaeSplit(cfg_path=cfg_path, sam_link_name=sam_link_name, device=device,
                       rank=get_world_rank(), sp_size=sp_size).loop()
        return
    if os.environ.get("KREA_PIPELINE", "0") == "1":
        from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_pipeline import KreaPipeline
        KreaPipeline(cfg_path=cfg_path, sam_link_name=sam_link_name,
                     device=device, rank=get_world_rank()).loop()
        return
    cls = KreaOrchestratorStreamSP if os.environ.get("KREA_STREAM", "0") == "1" else KreaOrchestratorSP
    orch = cls(cfg_path=cfg_path, sam_link_name=sam_link_name,
               device=device, rank=get_world_rank(), world=get_world_size())
    orch.loop()


if __name__ == "__main__":
    main()
