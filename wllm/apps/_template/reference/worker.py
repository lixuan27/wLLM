"""Sequential single-GPU reference worker for <app>.

    poll frontend inputs -> pipeline step -> write outputs to shared memory

The worker owns the shared-memory buffers (``create=True``); the adapter and
frontend attach to them. ``wllm/apps/worldplay/reference/worker.py`` is a
worked example of the same loop.
"""

from __future__ import annotations

import time

import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.logger import init_logger
from wllm.apps._template.reference.config import AppReferenceConfig
from wllm.apps._template.reference.pipeline import AppPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

logger = init_logger(__name__)
set_torch_options()


class AppWorker:  # TODO rename to <App>Worker
    def __init__(self, cfg_path: str):
        self.reference_cfg = AppReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.pipe = AppPipeline(cfg=self.cfg, device=self.device)
        self.pipe.start_instance()
        self.session_started = False

        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        # TODO create the input/output SharedTensorBuffers named in the config

        self.warmup()
        # Keep this exact wording pattern: the frontend docs and the example
        # scripts tell users to wait for the READY line.
        logger.info("<App> backend READY")

    def warmup(self):
        # TODO drive enough dummy work through the pipeline that kernel
        # selection / compilation happens before the first real input, then
        # reset the session state.
        pass

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        # TODO self.pipe.init_session(...)
        self.ctrl_buffer.commit()

    def reset(self):
        self.session_started = False
        self.pipe.reset()
        # TODO clear the input buffers
        self.ctrl_buffer.commit()

    def terminate(self):
        self.session_started = False
        self.pipe.terminate_instance()
        self.ctrl_buffer.unlink()
        # TODO unlink the input/output buffers

    def loop(self):
        while True:
            opcode = int(self.ctrl_buffer.recv())
            if opcode == 2 and self.session_started:
                self.terminate()
                break
            elif opcode == 1 and not self.session_started:
                self.start()
            elif opcode == 3 and self.session_started:
                self.reset()

            if self.session_started:
                # TODO read pending inputs; when a chunk's worth is ready,
                # run self.pipe.step(...) and write the outputs
                pass
            else:
                time.sleep(0.005)
