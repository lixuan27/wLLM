import asyncio
import os
import time
import uuid
from enum import Enum
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import vllm_omni
from qwen_asr import Qwen3ASRModel
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.liveavatar.reference.config import LiveAvatarReferenceConfig
from wllm.apps.liveavatar.reference.pipeline import LiveAvatarPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

logger = init_logger(__name__)
set_torch_options()


class AudioState(Enum):
    IN_SILENCE = 0
    IN_SPEECH = 1


class StreamingVADSegmenter:
    def __init__(
        self,
        speech_threshold: float = 0.005,
        speech_start_frames: int = 10,
        speech_end_frames: int = 50,
        preroll_size: int = 5,
        min_utterance_frames: int = 25,
        max_utterance_frames: int = 2000,
        drop_trailing_silence: bool = True,
    ):
        self.speech_threshold = speech_threshold
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.preroll_size = preroll_size
        self.min_utterance_frames = min_utterance_frames
        self.max_utterance_frames = max_utterance_frames
        self.drop_trailing_silence = drop_trailing_silence

        self.state = AudioState.IN_SILENCE
        self.silence_buffer: List[bool] = []
        self.speech_buffer: List[bool] = []
        self.preroll_frames: List[np.ndarray] = []
        self.utterance_frames: List[np.ndarray] = []

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        is_speech = rms >= self.speech_threshold

        self.preroll_frames.append(audio_chunk.copy())
        if len(self.preroll_frames) > self.preroll_size:
            self.preroll_frames.pop(0)

        if self.state == AudioState.IN_SILENCE:
            self.speech_buffer.append(is_speech)
            if len(self.speech_buffer) > self.speech_start_frames:
                self.speech_buffer.pop(0)

            self.silence_buffer.clear()
            if len(self.speech_buffer) == self.speech_start_frames and all(self.speech_buffer):
                self.state = AudioState.IN_SPEECH
                self.utterance_frames = [f.copy() for f in self.preroll_frames]
                self.speech_buffer.clear()
                self.silence_buffer.clear()

            return False, None

        self.utterance_frames.append(audio_chunk.copy())
        if len(self.utterance_frames) >= self.max_utterance_frames:
            audio = np.concatenate(self.utterance_frames, axis=0)
            self._reset_to_silence()
            return True, audio

        self.silence_buffer.append(not is_speech)
        if len(self.silence_buffer) > self.speech_end_frames:
            self.silence_buffer.pop(0)

        self.speech_buffer.clear()
        if len(self.silence_buffer) == self.speech_end_frames and all(self.silence_buffer):
            frames = self.utterance_frames
            if self.drop_trailing_silence and len(frames) >= self.speech_end_frames:
                frames = frames[:-self.speech_end_frames]

            if len(frames) < self.min_utterance_frames:
                self._reset_to_silence()
                return False, None

            audio = np.concatenate(frames, axis=0) if frames else None
            self._reset_to_silence()
            return audio is not None, audio

        return False, None

    def _reset_to_silence(self):
        self.state = AudioState.IN_SILENCE
        self.silence_buffer.clear()
        self.speech_buffer.clear()
        self.utterance_frames.clear()


def truncate_to_last_sentence(text, end_tokens={".", "!", "?"}):
    text = text.replace("*", "")
    for i in range(len(text) - 1, -1, -1):
        if text[i] in end_tokens:
            return text[:i + 1]
    return text


def time_stretch_audio(samples: np.ndarray, stretch_ratio: float) -> np.ndarray:
    if stretch_ratio <= 0:
        raise ValueError("stretch_ratio must be > 0")

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(samples) < 2:
        return samples.copy()

    rate = 1.0 / stretch_ratio
    n = len(samples)
    n_fft = min(512, max(32, 2 ** int(np.floor(np.log2(n)))))
    hop_length = max(8, n_fft // 4)

    stretched = librosa.effects.time_stretch(
        samples,
        rate=rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return np.asarray(stretched, dtype=np.float32)


class LiveAvatarWorker:
    """Reference single-GPU worker.

    The worker keeps the application pipeline strictly sequential:
    ASR -> LLM -> full TTS -> full LiveAvatar generation.
    """

    def __init__(self, cfg_path: str):
        self.reference_cfg = LiveAvatarReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self._init_worker()
        self.warmup()
        logger.info("LiveAvatar backend READY")

    def _init_worker(self):
        if self.cfg.device != "cuda":
            raise ValueError(
                f"The reference worker expects cfg.device='cuda', got {self.cfg.device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The reference worker requires a visible CUDA GPU.")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.pipe = self._create_pipeline()
        self.pipe.start_instance()
        self.session_started = False

        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name,
            create=True,
        )
        self.audio_input_buffer = SharedTensorBuffer(
            self.cfg.audio_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=True,
        )
        self.audio_output_buffer = SharedTensorBuffer(
            self.cfg.audio_output_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=True,
        )

        if self.cfg.signal_buffer_name is not None:
            self.signal_buffer = SharedControlBuffer(
                self.cfg.signal_buffer_name,
                create=True,
            )
        else:
            self.signal_buffer = None

        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0
        self._async_runner = None

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.cfg.llm_gpu_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(self.cfg.llm_gpu_index))
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=self.cfg.llm_model_name,
                dtype="auto",
                gpu_memory_utilization=float(self.cfg.llm_gpu_memory_utilization),
                max_model_len=int(self.cfg.llm_max_model_len),
                trust_remote_code=True,
            )
        )

        self.llm_tokenizer = self.llm_engine.get_tokenizer()

        if self.cfg.tts_stage_configs_path is not None:
            tts_stage_config_path = self.cfg.tts_stage_configs_path
        else:
            tts_stage_config_path = os.path.join(
                os.path.dirname(vllm_omni.__file__),
                "model_executor/stage_configs/qwen3_tts.yaml",
            )

        if self.cfg.tts_gpu_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(self.cfg.tts_gpu_index))

        self.tts_engine = AsyncOmni(
            model=self.cfg.tts_model_name,
            stage_configs_path=tts_stage_config_path,
            trust_remote_code=True,
        )

        self.tts_tokenizer_for_prompt = AutoTokenizer.from_pretrained(
            self.cfg.tts_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tts_talker_config = getattr(
            self.tts_engine.model_config.hf_config, "talker_config", None,
        )

        if visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        self._async_runner = asyncio.Runner()

        self.asr_model = Qwen3ASRModel.from_pretrained(
            self.cfg.asr_model_name,
            dtype=torch.bfloat16,
            device_map=(
                f"cuda:{int(self.cfg.asr_gpu_index)}"
                if self.cfg.asr_gpu_index is not None
                else "cuda:0"
            ),
            attn_implementation="flash_attention_2",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        logger.info("Loaded ASR model from %s", self.cfg.asr_model_name)

        model_path = self.cfg.wav2vec2_path
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_path)
        self.wav2vec_model.eval()
        self.wav2vec_model.requires_grad_(False)
        self.wav2vec_model.to(device=self.device, dtype=torch.bfloat16)
        logger.info("Loaded Wav2Vec2 from %s", model_path)

        self.message_history = [{
            "role": "system",
            "content": (
                "You are a helpful TTS assistant. You need to generate a response "
                "in English based on the user's input, and the response will be "
                "converted to speech by a TTS model. Please make sure your "
                "response is suitable for TTS conversion."
            )
        }]
        self._prepare_online_stack()

    def _create_pipeline(self):
        return LiveAvatarPipeline(cfg=self.cfg, device=self.device)

    def _prepare_online_stack(self):
        self._warmup_asr()
        self._warmup_wav2vec()

    def _warmup_asr(self):
        if self.asr_model is None:
            return
        if os.getenv("WLLM_SKIP_ASR_WARMUP", "0") == "1":
            logger.info("Skipping ASR warmup due to WLLM_SKIP_ASR_WARMUP=1")
            return

        warmup_audio = np.zeros((int(self.cfg.audio_sample_rate * 0.5),), dtype=np.float32)
        try:
            self.asr_model.transcribe(
                audio=(warmup_audio, self.cfg.audio_sample_rate),
                language="English",
            )
            logger.info("ASR warmup completed")
        except Exception as exc:
            logger.warning("ASR warmup skipped due to runtime error: %s", exc)

    def _warmup_wav2vec(self):
        if self.wav2vec_model is None:
            return

        frames_per_latent = int(self.cfg.vae_config.scale_factor_temporal)
        target_frames = int(self.cfg.chunk_size) * frames_per_latent
        # Warm up on the same audio window a real DiT step uses (one tts_chunk_size).
        warmup_audio = np.zeros(
            (int(self.cfg.tts_chunk_size),),
            dtype=np.float32,
        )

        try:
            self.extract_audio_features(warmup_audio, target_frames=target_frames)
            logger.info("Wav2Vec2 warmup completed")
        except Exception as exc:
            logger.warning("Wav2Vec2 warmup skipped due to runtime error: %s", exc)
    
    def _estimate_tts_prompt_len(self, tts_params: dict) -> int:
        task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]
        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=tts_params,
            task_type=task_type,
            tokenize_prompt=lambda t: self.tts_tokenizer_for_prompt(t, padding=False)["input_ids"],
            codec_language_id=getattr(self.tts_talker_config, "codec_language_id", None),
            spk_is_dialect=getattr(self.tts_talker_config, "spk_is_dialect", None),
        )

    def extract_audio_features(
        self,
        audio_samples: np.ndarray,
        target_frames: int,
    ) -> torch.Tensor:
        inputs = self.wav2vec_processor(
            audio_samples,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device=self.device, dtype=torch.bfloat16)

        res = self.wav2vec_model(input_values, output_hidden_states=True)
        hidden_states = res.hidden_states

        stacked = torch.cat([h.squeeze(0) for h in hidden_states], dim=0)
        num_layers = len(hidden_states)
        channels = hidden_states[0].shape[-1]
        time_steps = hidden_states[0].shape[1]
        stacked = stacked.view(num_layers, time_steps, channels).permute(0, 2, 1)

        stacked = F.interpolate(
            stacked.float(),
            size=target_frames,
            mode="linear",
            align_corners=True,
        )

        return stacked.unsqueeze(0).to(dtype=torch.bfloat16)

    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=None,
            image_path=self.cfg.image_path,
        )

        num_warmup_steps = max(3, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 1)
        for _ in range(num_warmup_steps):
            self.pipe.step()
        self.pipe.reset()

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        custom_image_path = os.path.join(
            "/tmp",
            f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png",
        )
        image_path = custom_image_path if os.path.exists(custom_image_path) else self.cfg.image_path
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=image_path)
        self.ctrl_buffer.commit()
        logger.info("LiveAvatar session started")

    def reset(self):
        self.session_started = False
        self.pipe.reset()
        self.audio_input_buffer.clear()
        self.audio_output_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("LiveAvatar session reset")

    def terminate(self):
        # Idempotent: loop() calls this on a graceful ctrl-buffer TERM, and the
        # launcher's finally calls it on Ctrl+C / exit -- both must be safe.
        if getattr(self, "_terminated", False):
            return
        self._terminated = True
        self.session_started = False
        self.pipe.terminate_instance()
        if self.llm_engine is not None:
            self.llm_engine.shutdown()
        if self.tts_engine is not None:
            self.tts_engine.shutdown()
        if self._async_runner is not None:
            self._async_runner.close()
        self.ctrl_buffer.unlink()
        self.audio_input_buffer.unlink()
        self.audio_output_buffer.unlink()
        if self.signal_buffer is not None:
            self.signal_buffer.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _generate_llm_response(self) -> str:
        prompt = self.llm_tokenizer.apply_chat_template(
            self.message_history,
            tokenize=False,
            add_generation_prompt=True,
        )

        final_output = None
        async for output in self.llm_engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=int(self.cfg.llm_max_completion_tokens),
                temperature=float(self.cfg.llm_temperature),
                top_p=float(self.cfg.llm_top_p),
            ),
            request_id=str(uuid.uuid4()),
        ):
            final_output = output

        if final_output is None or not final_output.outputs:
            return ""
        return final_output.outputs[0].text or ""

    @staticmethod
    def _extract_multimodal_audio(stage_output, seen_audio_chunks: int, consumed_tts_samples: int):
        mm = getattr(stage_output, "multimodal_output", None)
        ro = None
        if not mm:
            ro = getattr(stage_output, "request_output", None)
            mm = getattr(ro, "multimodal_output", None) if ro else None
        if not mm:
            if ro is None:
                ro = getattr(stage_output, "request_output", None)
            outputs = getattr(ro, "outputs", None) if ro else None
            if outputs:
                for completion_output in outputs:
                    completion_mm = getattr(completion_output, "multimodal_output", None)
                    if completion_mm:
                        mm = completion_mm
                        break
        if not mm:
            return [], seen_audio_chunks, None

        audio_key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if audio_key is None:
            return [], seen_audio_chunks, None

        sr_raw = mm.get("sr", 24000)
        sr = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sample_rate = int(sr.item()) if hasattr(sr, "item") else int(sr)

        finished = bool(getattr(stage_output, "finished", False))

        audio_val = mm[audio_key]
        if isinstance(audio_val, list):
            new_chunks = audio_val[seen_audio_chunks:]
            seen_audio_chunks = len(audio_val)
        elif isinstance(audio_val, torch.Tensor):
            if finished and consumed_tts_samples > 0:
                flat = audio_val.reshape(-1)
                delta = flat[consumed_tts_samples:]
                new_chunks = [delta] if delta.numel() > 0 else []
            else:
                new_chunks = [audio_val]
            seen_audio_chunks += 1
        elif audio_val is not None:
            new_chunks = [audio_val]
            seen_audio_chunks += 1
        else:
            new_chunks = []

        return new_chunks, seen_audio_chunks, sample_rate

    async def _generate_tts_audio(self, content: str) -> np.ndarray:
        tts_params = {
            "task_type": ["CustomVoice"],
            "text": [content],
            "language": [self.cfg.tts_language],
            "speaker": [self.cfg.tts_voice],
            "instruct": [""],
            "max_new_tokens": [2048],
        }
        prompt_obj = {
            "prompt_token_ids": [1] * self._estimate_tts_prompt_len(tts_params),
            "additional_information": tts_params,
        }

        seen_audio_chunks = 0
        consumed_tts_samples = 0
        resampled_chunks: list[np.ndarray] = []

        async for stage_output in self.tts_engine.generate(
            prompt_obj,
            request_id=str(uuid.uuid4()),
            output_modalities=["audio"],
        ):
            new_chunks, seen_audio_chunks, sample_rate = self._extract_multimodal_audio(
                stage_output,
                seen_audio_chunks,
                consumed_tts_samples,
            )
            if not new_chunks:
                continue

            for chunk_tensor in new_chunks:
                if hasattr(chunk_tensor, "float"):
                    audio_array = chunk_tensor.float().detach().cpu().numpy()
                else:
                    audio_array = np.asarray(chunk_tensor)

                if audio_array.ndim > 1:
                    audio_array = audio_array.squeeze()
                audio_array = np.asarray(audio_array, dtype=np.float32).reshape(-1)
                if audio_array.size == 0:
                    continue

                consumed_tts_samples += int(audio_array.size)

                if float(self.cfg.audio_stretch_ratio) != 1.0:
                    audio_array = time_stretch_audio(audio_array, float(self.cfg.audio_stretch_ratio))

                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                if sample_rate != int(self.cfg.audio_sample_rate):
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor,
                        sample_rate,
                        int(self.cfg.audio_sample_rate),
                    )
                resampled = audio_tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                if resampled.size > 0:
                    resampled_chunks.append(resampled)

        if not resampled_chunks:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(resampled_chunks, axis=0)

    def _frame_audio(self, audio_samples: np.ndarray) -> np.ndarray:
        frame_samples = int(self.cfg.audio_frame_samples)
        if audio_samples.size == 0:
            return np.empty((0, frame_samples), dtype=np.float32)

        pad = (-audio_samples.size) % frame_samples
        if pad > 0:
            audio_samples = np.pad(audio_samples, (0, pad), mode="constant")
        return audio_samples.reshape(-1, frame_samples)

    def _run_liveavatar_reference(self, audio_samples: np.ndarray):
        frames_per_latent = int(self.cfg.vae_config.scale_factor_temporal)
        step_frames = int(self.cfg.chunk_size) * frames_per_latent
        step_samples = int(self.cfg.tts_chunk_size)

        audio_chunks: list[np.ndarray] = []
        video_chunks: list[np.ndarray] = []
        sample_offset = 0
        ran_step = False

        while True:
            if ran_step and sample_offset >= audio_samples.size:
                break

            chunk_audio = audio_samples[sample_offset:sample_offset + step_samples]
            actual_samples = int(chunk_audio.size)
            if actual_samples == 0 and ran_step:
                break
            if actual_samples < step_samples:
                chunk_audio = np.pad(chunk_audio, (0, step_samples - actual_samples), mode="constant")

            audio_features = self.extract_audio_features(chunk_audio, target_frames=step_frames)
            video_chunks.append(self.pipe.step(audio_input=audio_features))
            audio_chunks.append(np.asarray(chunk_audio, dtype=np.float32))

            sample_offset += step_samples
            ran_step = True

        if not video_chunks:
            return None, None
        return np.concatenate(video_chunks, axis=0), np.concatenate(audio_chunks, axis=0)

    def inference(self, audio: Tuple[np.ndarray, int]) -> None:
        asr_results = self.asr_model.transcribe(
            audio=(audio[0].flatten(), audio[1]),
            language="English",
        )
        text = asr_results[0].text
        logger.info("ASR output: %s", text)
        self.message_history.append({"role": "user", "content": text})

        content = self._async_runner.run(self._generate_llm_response())
        logger.info("LLM output: %s", content)

        content = truncate_to_last_sentence(content)
        self.message_history.append({"role": "assistant", "content": content})

        audio_samples = self._async_runner.run(self._generate_tts_audio(content))
        if audio_samples.size == 0:
            logger.warning("TTS produced no audio; skipping LiveAvatar generation")
            return

        video, emitted_audio = self._run_liveavatar_reference(audio_samples)
        if video is None:
            return
        audio_frames = self._frame_audio(emitted_audio)
        n = min(video.shape[0], audio_frames.shape[0])
        if n > 0:
            self.pipe._video_buffer.write(video[:n])
            self.audio_output_buffer.write(audio_frames[:n])

    def is_start(self):
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self):
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self):
        return int(self.ctrl_buffer.recv()) == 3

    def loop(self):
        while True:
            if self.is_terminate() and self.session_started:
                self.terminate()
                break
            if self.is_start() and not self.session_started:
                self.start()
            elif self.is_reset() and self.session_started:
                self.reset()

            if not self.session_started:
                time.sleep(0.005)
                continue

            self.num_read_input_chunks, audio_chunk = self.audio_input_buffer.read(
                x=self.num_read_input_chunks,
                n=1,
            )
            if audio_chunk is not None:
                should_infer, utterance_audio = self.stream_segmenter.process_chunk(audio_chunk)
                if should_infer and utterance_audio is not None:
                    logger.info(
                        "Detected utterance (%d samples), running sequential ASR->LLM->TTS->LiveAvatar pipeline",
                        len(utterance_audio),
                    )
                    try:
                        self.inference(
                            (
                                np.asarray(utterance_audio, dtype=np.float32).reshape(-1),
                                int(self.cfg.audio_sample_rate),
                            )
                        )
                    except Exception:
                        logger.exception("LiveAvatar sequential inference failed")

            time.sleep(0.001)
