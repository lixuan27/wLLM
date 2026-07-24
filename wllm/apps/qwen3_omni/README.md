# Qwen3-Omni

Text-to-speech conversation with the [Qwen3-Omni 30B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) model. You type a
prompt; the Thinker LLM writes the response, the Talker turns it into
audio codec frames, and the Code2Wav vocoder turns those into speech.
The optimized backends stream all three stages, so audio starts playing
about a third of a second after the prompt regardless of how long the
response is.
