"""LiveKit publisher / subscriber frontend for the Krea+SAM application.

Subscribes to the user's webcam track, pushes RGB frames into the
backend through ``KreaSAMAdapter.push_frame``, and republishes the
backend's composited (Krea-stylised background + original body) output
as a local video track on the same LiveKit room.

The frontend is backend-agnostic: it only talks to ``KreaSAMAdapter``,
which in turn talks to whatever backend created the matching shared
buffers — the user reference under ``wllm/apps/krea_sam/reference/`` or any
agent-written backend under ``wllm/apps/krea_sam/backend/``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

import numpy as np

from wllm.apps.krea_sam.adapter import KreaSAMAdapter
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


def event_frame_to_rgb(frame) -> np.ndarray:
    _require_livekit()
    if frame.type != rtc.VideoBufferType.RGB24:
        frame = frame.convert(rtc.VideoBufferType.RGB24)
    return np.frombuffer(frame.data, dtype=np.uint8).reshape(
        frame.height, frame.width, 3,
    ).copy()


async def handle_remote_video(
    participant,
    track,
    adapter: KreaSAMAdapter,
    loop: asyncio.AbstractEventLoop,
) -> None:
    logger.info("receiving video from participant=%s", participant.identity)
    video_stream = rtc.VideoStream.from_track(
        track=track,
        loop=loop,
        format=rtc.VideoBufferType.RGB24,
    )
    try:
        async for event in video_stream:
            adapter.push_frame(event_frame_to_rgb(event.frame))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "error receiving video from participant=%s", participant.identity,
        )
    finally:
        await video_stream.aclose()


async def publish_adapter_stream(
    adapter: KreaSAMAdapter,
    video_source,
    fps: int,
) -> None:
    """Pull frames out of the adapter at the configured FPS and push
    them to the LiveKit local track. We hold the most recently produced
    frame so the published stream stays at a steady rate even when the
    backend hasn't emitted anything new this tick."""
    interval = 1.0 / float(max(1, fps))
    next_time = None
    last_frame = None

    while True:
        now = time.monotonic()
        if next_time is None:
            next_time = now
        next_time += interval

        frame = adapter.get_frames()
        if frame is not None:
            last_frame = frame

        # Don't publish anything until the worker has produced its first
        # composited frame from real webcam input.
        if last_frame is not None:
            video_source.capture_frame(
                to_rgba_frame(last_frame),
                timestamp_us=int(time.time_ns() // 1000),
            )

        sleep_s = next_time - time.monotonic()
        await asyncio.sleep(max(sleep_s, 0.0))


async def main(room, room_name: str, adapter: KreaSAMAdapter, identity: str) -> None:
    _require_livekit()

    creds = load_livekit_credentials()
    url = creds.url
    token = make_access_token(creds, room_name, identity, agent=True)
    loop = asyncio.get_running_loop()
    remote_video_tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant) -> None:
        if isinstance(track, rtc.RemoteVideoTrack):
            key = f"{participant.identity}:{publication.sid}"
            task = asyncio.create_task(
                handle_remote_video(participant, track, adapter, loop)
            )
            remote_video_tasks[key] = task
            task.add_done_callback(lambda _, k=key: remote_video_tasks.pop(k, None))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track, publication, participant) -> None:
        _ = track
        key = f"{participant.identity}:{publication.sid}"
        task = remote_video_tasks.pop(key, None)
        if task is not None:
            task.cancel()

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant) -> None:
        for key in [k for k in remote_video_tasks if k.startswith(f"{participant.identity}:")]:
            task = remote_video_tasks.pop(key, None)
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
        "\n[frontend] room ready -- open this URL in your browser and allow the webcam:\n"
        f"  {join_url}\n",
        flush=True,
    )

    logger.info("sending start signal to Krea+SAM worker")
    try:
        adapter.start()
    except Exception:
        logger.exception("failed to start Krea+SAM worker session")
        return

    video_height = int(adapter.cfg.height)
    video_width = int(adapter.cfg.width)
    fps = int(adapter.cfg.video_fps)

    video_source = rtc.VideoSource(width=video_width, height=video_height)
    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)

    await room.local_participant.publish_track(
        video_track,
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_framerate=fps,
                max_bitrate=5_000_000,
            ),
        ),
    )

    publish_task: Optional[asyncio.Task] = None
    try:
        publish_task = asyncio.create_task(
            publish_adapter_stream(adapter, video_source, fps)
        )
        await publish_task
    finally:
        if publish_task is not None:
            publish_task.cancel()
            await asyncio.gather(publish_task, return_exceptions=True)
        for task in list(remote_video_tasks.values()):
            task.cancel()
        if remote_video_tasks:
            await asyncio.gather(*remote_video_tasks.values(), return_exceptions=True)
        await video_source.aclose()


def main_cli(argv: list[str] | None = None) -> int:
    _require_livekit()

    parser = argparse.ArgumentParser(description="Run the Krea+SAM LiveKit frontend")
    parser.add_argument("--room", default="krea_sam_room", help="LiveKit room name")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"),
        help="app runtime config YAML (shared-memory buffer names)",
    )
    parser.add_argument("--identity", default="krea-sam-publisher",
                        help="identity the frontend publishes under")
    parser.add_argument("--log-file", default="krea_sam_lk.log")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    adapter = KreaSAMAdapter(args.config)
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)

    async def cleanup() -> None:
        try:
            try:
                adapter.terminate()
            except Exception:
                logger.exception("failed to terminate Krea+SAM worker during shutdown")
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
