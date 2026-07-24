import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

from wllm.apps.liveavatar.adapter import LiveAvatarAdapter
from wllm.serving.frontend.livekit_utils import (
    load_livekit_credentials,
    make_access_token,
    viewer_join_url,
)

try:
    from livekit import api, rtc  # type: ignore[import-not-found]
except ModuleNotFoundError:
    api = None
    rtc = None


logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLES_PER_CHUNK = 320
AUDIO_CHANNELS = 1
TICK_SECONDS = AUDIO_SAMPLES_PER_CHUNK / AUDIO_SAMPLE_RATE
VIDEO_FPS = 1.0 / TICK_SECONDS


def _require_livekit() -> None:
    if api is None or rtc is None:
        raise ModuleNotFoundError(
            "LiveKit frontend dependencies are not installed. "
            "Install `livekit` and `livekit-api` in the wllm environment."
        )


def to_rgba_frame(frame: np.ndarray):
    _require_livekit()
    if frame.ndim != 3:
        raise ValueError(f"video frame must be 3D [H, W, C], got shape={frame.shape}")

    height, width, channels = frame.shape
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if channels == 4:
        rgba = frame
    elif channels == 3:
        rgba = np.full((height, width, 4), 255, dtype=np.uint8)
        rgba[:, :, :3] = frame
    else:
        raise ValueError(f"video frame channel count must be 3 or 4, got {channels}")

    return rtc.VideoFrame(
        width=width,
        height=height,
        type=rtc.VideoBufferType.RGBA,
        data=rgba.tobytes(),
    )


def to_audio_frame(chunk: np.ndarray):
    _require_livekit()
    if chunk.ndim != 1:
        raise ValueError(f"audio chunk must be 1D [samples], got shape={chunk.shape}")

    if len(chunk) != AUDIO_SAMPLES_PER_CHUNK:
        raise ValueError(
            f"audio chunk length must be {AUDIO_SAMPLES_PER_CHUNK}, got {len(chunk)}"
        )

    if np.issubdtype(chunk.dtype, np.floating):
        pcm = np.clip(chunk, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
    else:
        pcm = chunk.astype(np.int16, copy=False)

    pcm = pcm.reshape(-1, 1)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=AUDIO_SAMPLE_RATE,
        num_channels=AUDIO_CHANNELS,
        samples_per_channel=pcm.shape[0],
    )


async def handle_remote_audio(
    participant,
    track,
    adapter: LiveAvatarAdapter,
) -> None:
    logger.info("receiving audio from participant=%s", participant.identity)
    audio_stream = rtc.AudioStream(
        track=track,
        sample_rate=AUDIO_SAMPLE_RATE,
        num_channels=1,
        frame_size_ms=20,
    )
    float_buffer = np.empty((0,), dtype=np.float32)
    try:
        async for event in audio_stream:
            pcm = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            float_buffer = np.concatenate([float_buffer, pcm])
            while len(float_buffer) >= AUDIO_SAMPLES_PER_CHUNK:
                chunk = float_buffer[:AUDIO_SAMPLES_PER_CHUNK].copy()
                float_buffer = float_buffer[AUDIO_SAMPLES_PER_CHUNK:]
                adapter.push_audio(chunk)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("error receiving audio from participant=%s", participant.identity)
    finally:
        await audio_stream.aclose()


async def push_adapter_stream(
    adapter: LiveAvatarAdapter,
    av_sync,
) -> None:
    tick_index = 0
    start_mono = time.monotonic()

    while True:
        vidx, video_np = adapter.get_frames_blocked()
        aidx, audio_np = adapter.get_audio_chunks_blocked()

        if video_np is None or audio_np is None:
            await asyncio.sleep(0.001)
            continue

        if not adapter.commit(frame_next_index=vidx, audio_next_index=aidx):
            logger.warning(
                "A/V index mismatch (video=%s audio=%s); waiting for next synchronized pair",
                vidx,
                aidx,
            )
            await asyncio.sleep(0.001)
            continue

        ts = tick_index * TICK_SECONDS
        await av_sync.push(to_rgba_frame(video_np), ts)
        await av_sync.push(to_audio_frame(audio_np), ts + TICK_SECONDS)

        tick_index += 1
        target = start_mono + tick_index * TICK_SECONDS
        sleep_s = target - time.monotonic()
        await asyncio.sleep(max(sleep_s, 0))


async def main(room, room_name: str, adapter: LiveAvatarAdapter, identity: str) -> None:
    _require_livekit()

    creds = load_livekit_credentials()
    url = creds.url
    token = make_access_token(creds, room_name, identity, agent=True)
    remote_audio_tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant) -> None:
        if isinstance(track, rtc.RemoteAudioTrack):
            key = f"{participant.identity}:{publication.sid}"
            task = asyncio.create_task(handle_remote_audio(participant, track, adapter))
            remote_audio_tasks[key] = task
            task.add_done_callback(lambda t, k=key: remote_audio_tasks.pop(k, None))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track, publication, participant) -> None:
        _ = track
        key = f"{participant.identity}:{publication.sid}"
        task = remote_audio_tasks.pop(key, None)
        if task is not None:
            task.cancel()

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant) -> None:
        for key in [k for k in remote_audio_tasks if k.startswith(f"{participant.identity}:")]:
            task = remote_audio_tasks.pop(key, None)
            if task is not None:
                task.cancel()

    logger.info("connecting to %s room=%s", url, room_name)
    try:
        await room.connect(url, token)
    except rtc.ConnectError as exc:
        logger.error("failed to connect: %s", exc)
        return
    logger.info("connected to room %s", room.name)

    join_url = viewer_join_url(creds, room_name, identity="user")
    print(
        "\n[frontend] room ready -- open this URL in your browser to talk to the avatar:\n"
        f"  {join_url}\n",
        flush=True,
    )

    logger.info("sending start signal to LiveAvatar worker")
    try:
        adapter.start()
    except Exception:
        logger.exception("failed to start LiveAvatar worker session")
        return

    first_video = adapter.get_first_image()
    first_audio = adapter.get_first_audio_chunk()
    video_height, video_width = first_video.shape[:2]
    queue_size_ms = 1000

    video_source = rtc.VideoSource(width=video_width, height=video_height)
    audio_source = rtc.AudioSource(
        sample_rate=AUDIO_SAMPLE_RATE,
        num_channels=AUDIO_CHANNELS,
        queue_size_ms=queue_size_ms,
    )

    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)

    await room.local_participant.publish_track(
        video_track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(max_framerate=50, max_bitrate=5_000_000),
        ),
    )
    await room.local_participant.publish_track(
        audio_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=VIDEO_FPS,
        video_queue_size_ms=queue_size_ms,
    )

    async def _log_fps(sync) -> None:
        while True:
            await asyncio.sleep(2)
            logger.info(
                "fps=%.2f video_t=%.3f audio_t=%.3f",
                sync.actual_fps,
                sync.last_video_time,
                sync.last_audio_time,
            )

    log_task: Optional[asyncio.Task] = None
    try:
        await av_sync.push(to_rgba_frame(first_video), 0.0)
        await av_sync.push(to_audio_frame(first_audio), TICK_SECONDS)

        log_task = asyncio.create_task(_log_fps(av_sync))
        await push_adapter_stream(adapter, av_sync)
    finally:
        if log_task is not None:
            log_task.cancel()
            await asyncio.gather(log_task, return_exceptions=True)
        for task in list(remote_audio_tasks.values()):
            task.cancel()
        if remote_audio_tasks:
            await asyncio.gather(*remote_audio_tasks.values(), return_exceptions=True)
        await av_sync.wait_for_playout()
        await av_sync.aclose()
        await audio_source.aclose()
        await video_source.aclose()


def main_cli(argv: list[str] | None = None) -> int:
    _require_livekit()

    parser = argparse.ArgumentParser(description="Run the LiveAvatar LiveKit frontend")
    parser.add_argument("--room", default="liveavatar_room", help="LiveKit room name")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"),
        help="app runtime config YAML (shared-memory buffer names)",
    )
    parser.add_argument("--identity", default="liveavatar-publisher",
                        help="identity the frontend publishes under")
    parser.add_argument("--log-file", default="liveavatar_lk.log")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    adapter = LiveAvatarAdapter(args.config)
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup() -> None:
        try:
            try:
                adapter.terminate()
            except Exception:
                logger.exception("failed to terminate LiveAvatar worker during shutdown")
            await room.disconnect()
        finally:
            loop.stop()

    asyncio.ensure_future(main(room, args.room, adapter, args.identity))
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
