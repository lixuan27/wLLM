"""Async ASR sidecar process.

Placement of the black-box ASR engine on its own GPU + process so transcription
never blocks (GIL or GPU) the DiT generation loop. Reads the shared audio
buffer, runs VAD + ASR, and posts transcripts into a small shared uint8 buffer
(``<audio>_txt``) that rank 0 polls. Resets its VAD when it sees the audio
buffer cleared (session reset).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.rt_config import RTConfig
from wllm.apps.longlive.backend.cuda.vad import StreamingVADSegmenter

logger = init_logger(__name__)
_MAX = 200


def main():
    cfg = RTConfig.from_yaml(os.environ["CONFIG_PATH"], is_path=True)
    dev = os.environ.get("ASR_DEVICE", "cuda:0")
    audio = SharedTensorBuffer(cfg.audio_buffer_name,
                               frame_shape=(int(cfg.audio_frame_samples),),
                               dtype=np.float32, max_len=int(cfg.audio_max_chunks),
                               create=False, wait=True, timeout_s=1800.0)
    txt = SharedTensorBuffer(cfg.audio_buffer_name + "_txt", frame_shape=(512,),
                             dtype=np.uint8, max_len=256, create=False, wait=True,
                             timeout_s=1800.0)
    from qwen_asr import Qwen3ASRModel
    asr = Qwen3ASRModel.from_pretrained(
        cfg.asr_model_name, dtype=torch.bfloat16, device_map=dev,
        attn_implementation="sdpa", max_inference_batch_size=1,
        max_new_tokens=256)
    if os.getenv("WLLM_SKIP_ASR_WARMUP", "0") != "1":
        try:
            asr.transcribe(audio=(np.zeros(int(cfg.audio_sample_rate * 0.5), np.float32),
                                  int(cfg.audio_sample_rate)), language="English")
        except Exception as e:
            logger.warning("sidecar ASR warmup skipped: %s", e)
    seg = StreamingVADSegmenter()
    read = 0
    last_num = 0
    # rank 0 holds the READY marker until this flag is set
    ready_flag = SharedTensorBuffer(cfg.audio_buffer_name + "_asr_ready",
                                    frame_shape=(1,), dtype=np.int64, max_len=1,
                                    create=False, wait=True, timeout_s=1800.0)
    ready_flag.write(np.array([1], dtype=np.int64))
    logger.info("ASR sidecar up device=%s", dev)
    while True:
        num = audio.num
        if num < last_num:                     # audio buffer was cleared (reset)
            seg = StreamingVADSegmenter()
            read = 0
        last_num = num
        utt = None
        for _ in range(_MAX):
            read, ch = audio.read(read, 1)
            if ch is None:
                break
            ok, s = seg.process_chunk(ch[0])
            if ok and s is not None:
                utt = s
        if utt is None:
            time.sleep(0.004)
            continue
        try:
            res = asr.transcribe(audio=(np.asarray(utt, np.float32).reshape(-1),
                                        int(cfg.audio_sample_rate)), language="English")
            t = (res[0].text or "").strip()
        except Exception:
            logger.exception("sidecar ASR failed")
            t = ""
        if t:
            b = t.encode("utf-8")[:511]
            frame = np.zeros(512, dtype=np.uint8)
            frame[:len(b)] = np.frombuffer(b, dtype=np.uint8)
            txt.write(frame)
            logger.info("sidecar transcript: %s", t)


if __name__ == "__main__":
    main()
