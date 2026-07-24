# Krea-Realtime + SAM3

Live video background editing. Your webcam stream runs through two
models at once: the [Krea-Realtime](https://huggingface.co/krea/krea-realtime-video) video-to-video model restyles the
frames after a text prompt, while [SAM3](https://github.com/facebookresearch/sam3) segments you in each frame. The
two outputs are composited so your body stays original and only the
background is edited.
