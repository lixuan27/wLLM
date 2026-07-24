import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

from wllm.apps.longlive.adapter import LongLiveAdapter
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
TICK_SECONDS = AUDIO_SAMPLES_PER_CHUNK / AUDIO_SAMPLE_RATE  # 0.02
VIDEO_FPS = 1.0 / TICK_SECONDS  # 50.0


def _require_livekit() -> None:
    if api is None or rtc is None:
        raise ModuleNotFoundError(
            "LiveKit frontend dependencies are not installed. "
            "Install `livekit` and `livekit-api` in the wllm environment."
        )


def to_rgba_frame(frame: np.ndarray):
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


async def handle_remote_audio(
    participant: "rtc.RemoteParticipant",
    track: "rtc.RemoteAudioTrack",
    adapter: LongLiveAdapter,
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


async def push_adapter_video(
    adapter: LongLiveAdapter,
    video_source: "rtc.VideoSource",
    target_fps: float,
) -> None:
    tick_seconds = 1.0 / float(target_fps)
    last_publish: Optional[float] = None

    while True:
        frame = adapter.get_frames()
        if frame is None:
            await asyncio.sleep(0.001)
            continue

        if last_publish is not None:
            wait_until = last_publish + tick_seconds
            sleep_s = wait_until - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        video_source.capture_frame(to_rgba_frame(frame))
        adapter.commit(adapter.num_played_frames)
        last_publish = time.monotonic()


async def main(
    room: "rtc.Room",
    room_name: str,
    adapter: LongLiveAdapter,
    identity: str,
    target_fps: float,
) -> None:
    creds = load_livekit_credentials()
    token = make_access_token(creds, room_name, identity, agent=True)

    remote_audio_tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ) -> None:
        if isinstance(track, rtc.RemoteAudioTrack):
            key = f"{participant.identity}:{publication.sid}"
            task = asyncio.create_task(
                handle_remote_audio(participant, track, adapter)
            )
            remote_audio_tasks[key] = task
            task.add_done_callback(
                lambda t, k=key: remote_audio_tasks.pop(k, None)
            )

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ) -> None:
        key = f"{participant.identity}:{publication.sid}"
        t = remote_audio_tasks.pop(key, None)
        if t is not None:
            t.cancel()

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: "rtc.RemoteParticipant") -> None:
        for key in [k for k in remote_audio_tasks if k.startswith(f"{participant.identity}:")]:
            t = remote_audio_tasks.pop(key, None)
            if t is not None:
                t.cancel()

    url = creds.url
    logger.info("connecting to %s room=%s", url, room_name)
    try:
        await room.connect(url, token)
    except rtc.ConnectError as e:
        logger.error("failed to connect: %s", e)
        return
    logger.info("connected to room %s", room.name)

    join_url = viewer_join_url(creds, room_name, identity="user")
    print(
        "\n[frontend] room ready -- open this URL in your browser, allow the mic, "
        "and narrate:\n"
        f"  {join_url}\n",
        flush=True,
    )

    logger.info("sending start signal to LongLive worker")
    try:
        adapter.start()
    except Exception:
        logger.exception("failed to start LongLive worker session")
        return

    video_source = rtc.VideoSource(
        width=adapter.cfg.width, height=adapter.cfg.height,
    )
    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    await room.local_participant.publish_track(
        video_track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_framerate=int(target_fps), max_bitrate=5_000_000,
            ),
        ),
    )

    try:
        await push_adapter_video(adapter, video_source, target_fps=target_fps)
    finally:
        for t in list(remote_audio_tasks.values()):
            t.cancel()
        if remote_audio_tasks:
            await asyncio.gather(*remote_audio_tasks.values(), return_exceptions=True)
        await video_source.aclose()


def main_cli(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LongLive LiveKit frontend")
    parser.add_argument("--room", default="longlive_room", help="LiveKit room name")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"),
        help="app runtime config YAML (shared-memory buffer names)",
    )
    parser.add_argument("--identity", default="longlive-publisher",
                        help="identity the frontend publishes under")
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--log-file", default="longlive_lk.log")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    _require_livekit()

    adapter = LongLiveAdapter(args.config)
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup() -> None:
        try:
            try:
                adapter.terminate()
            except Exception:
                logger.exception("failed to terminate LongLive worker during shutdown")
            await room.disconnect()
        finally:
            loop.stop()

    asyncio.ensure_future(
        main(room, args.room, adapter, args.identity, args.target_fps)
    )
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(cleanup()))

    try:
        loop.run_forever()
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
