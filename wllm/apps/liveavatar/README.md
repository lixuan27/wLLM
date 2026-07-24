# LiveAvatar

Face-to-face conversation with a talking avatar. You speak into your
mic; the audio runs through ASR (Qwen3-ASR), an LLM (Qwen3-4B), and TTS
(Qwen3-TTS), and the response audio drives the [LiveAvatar](https://huggingface.co/Quark-Vision/Live-Avatar) sound-to-video
model ([Wan2.2-S2V 14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)), producing avatar video played in sync with the
spoken response.
