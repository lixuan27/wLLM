"""Interactive web frontend for the Qwen3-Omni text -> speech backend.

Open the page, type a prompt, press Speak, and the generated speech plays
automatically as it streams in. The frontend talks to the backend only
through the shared adapter (``Qwen3OmniAdapter``), so it works with
whatever Qwen3-Omni backend is currently running.

Run (from the repo root), after a backend is up:
    bash wllm/apps/qwen3_omni/frontend/run_frontend.sh
then open http://localhost:8080 (forward the port if remote).

Audio is streamed to the browser as raw float32 PCM over a WebSocket and
played with the Web Audio API (gapless, real time). The server structure
(aiohttp ``make_app`` / ``main``) mirrors the WorldPlay frontend's.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time

import numpy as np
import aiohttp
from aiohttp import web

from wllm.apps.qwen3_omni.adapter import Qwen3OmniAdapter


# The audio pump polls this often for new frames (a cheap non-blocking
# shared-memory read). Coalesce up to MAX_FRAMES_PER_SEND frames per
# WebSocket message. Mark a response "done" after this much silence.
DEFAULT_POLL_INTERVAL = 0.005
DEFAULT_IDLE_TIMEOUT = 6.0
MAX_FRAMES_PER_SEND = 64


async def _send_json(ws: web.WebSocketResponse, lock: asyncio.Lock, obj: dict) -> None:
    """Serialize sends through ``lock`` so the control loop and the audio
    pump never interleave frames on the one socket."""
    async with lock:
        if not ws.closed:
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await ws.send_json(obj)


async def _send_bytes(ws: web.WebSocketResponse, lock: asyncio.Lock, payload: bytes) -> None:
    async with lock:
        if not ws.closed:
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await ws.send_bytes(payload)


def _drain_stale_audio(adapter: Qwen3OmniAdapter) -> None:
    """Discard any audio already buffered before we begin, so the first
    prompt's playback starts clean."""
    while adapter.get_audio_chunks() is not None:
        pass


async def _audio_pump(
    ws: web.WebSocketResponse,
    adapter: Qwen3OmniAdapter,
    state: dict,
    lock: asyncio.Lock,
    poll_interval: float,
    idle_timeout: float,
) -> None:
    """Stream audio frames to the browser and emit speaking/done status."""
    try:
        while not ws.closed:
            frames = []
            while len(frames) < MAX_FRAMES_PER_SEND:
                chunk = adapter.get_audio_chunks()
                if chunk is None:
                    break
                frames.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

            now = time.time()
            if frames:
                block = frames[0] if len(frames) == 1 else np.concatenate(frames)
                if state["prompt_in_flight"] and not state["first_audio_seen"]:
                    state["first_audio_seen"] = True
                    await _send_json(ws, lock, {
                        "type": "speaking",
                        "first_audio_latency_s": round(now - state["t_prompt"], 3),
                    })
                state["last_audio_time"] = now
                # Little-endian float32 so the browser's Float32Array reads
                # it correctly regardless of server byte order.
                await _send_bytes(ws, lock, np.ascontiguousarray(block, dtype="<f4").tobytes())
            else:
                if (state["prompt_in_flight"] and state["first_audio_seen"]
                        and now - state["last_audio_time"] > idle_timeout):
                    state["prompt_in_flight"] = False
                    await _send_json(ws, lock, {"type": "done"})
                await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        raise
    except Exception:
        await _send_json(ws, lock, {"type": "error", "message": "Audio playback error."})


def make_app(
    cfg_path: str,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    terminate_on_exit: bool = False,
) -> web.Application:
    app = web.Application()

    async def index(request: web.Request) -> web.StreamResponse:
        here = os.path.dirname(os.path.abspath(__file__))
        return web.FileResponse(
            os.path.join(here, "client.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        lock = asyncio.Lock()

        # Attach to the running backend (in a thread; blocks briefly if it
        # has to wait for the backend's buffers to appear).
        try:
            adapter = await asyncio.to_thread(Qwen3OmniAdapter, cfg_path)
        except Exception:
            await _send_json(ws, lock, {
                "type": "error",
                "message": "Couldn't reach the speech backend. Make sure it's running, then reload.",
            })
            await ws.close()
            return ws

        state = {"prompt_in_flight": False, "first_audio_seen": False,
                 "last_audio_time": 0.0, "t_prompt": 0.0}
        pump_task = None

        await _send_json(ws, lock, {"type": "connected"})

        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        break
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                cmd = data.get("cmd")

                if cmd == "start":
                    try:
                        await asyncio.to_thread(adapter.start, 600.0)
                        sample_rate = adapter.get_sample_rate()
                        await asyncio.to_thread(_drain_stale_audio, adapter)
                        if pump_task is None:
                            pump_task = asyncio.create_task(
                                _audio_pump(ws, adapter, state, lock, poll_interval, idle_timeout))
                        await _send_json(ws, lock, {"type": "ready", "sample_rate": sample_rate})
                    except Exception:
                        await _send_json(ws, lock, {
                            "type": "error", "message": "Couldn't start the session. Try reloading."})

                elif cmd == "prompt":
                    text = str(data.get("text", ""))
                    if not text.strip():
                        continue
                    if pump_task is None:
                        await _send_json(ws, lock, {
                            "type": "error", "message": "Press Start before sending a prompt."})
                        continue
                    state["prompt_in_flight"] = True
                    state["first_audio_seen"] = False
                    state["t_prompt"] = time.time()
                    state["last_audio_time"] = time.time()
                    try:
                        adapter.push_text(text)
                        await _send_json(ws, lock, {"type": "prompt_ack"})
                    except Exception:
                        state["prompt_in_flight"] = False
                        await _send_json(ws, lock, {
                            "type": "error", "message": "Couldn't send the prompt."})

                elif cmd == "ping":
                    await _send_json(ws, lock, {"type": "pong"})
        finally:
            if pump_task is not None:
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump_task
            if terminate_on_exit:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(adapter.terminate)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("WLLM_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("WLLM_PORT", "8080")))
    p.add_argument(
        "--cfg",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
        ),
        help="Backend YAML (shared-memory buffer names + audio params).",
    )
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    p.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT)
    p.add_argument("--terminate-on-exit", action="store_true",
                   help="Send TERMINATE to the backend when the browser disconnects "
                        "(default: leave it running for reconnects).")
    args = p.parse_args()

    if not os.path.isfile(args.cfg):
        raise SystemExit(f"config not found: {args.cfg}")

    print(f"[qwen3-omni-frontend] serving on http://{args.host}:{args.port}  "
          f"(a Qwen3-Omni backend must be running)")
    web.run_app(
        make_app(args.cfg, poll_interval=args.poll_interval,
                 idle_timeout=args.idle_timeout, terminate_on_exit=args.terminate_on_exit),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
