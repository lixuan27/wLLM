import argparse
import asyncio
import json
import os
import time
from typing import Dict, Any
import uuid
import av
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from wllm.apps.worldplay.adapter import WorldPlayAdapter as Adapter
from PIL import Image
import io

ICE_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]
)

DEFAULT_FPS = 30


async def wait_ice_complete(pc: RTCPeerConnection, timeout: float = 8.0):
    if pc.iceGatheringState == "complete":
        return
    ev = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _():
        if pc.iceGatheringState == "complete":
            ev.set()

    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        print("[WARN] ICE gathering timeout; SDP may miss candidates.")


class AdapterVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, adapter: Adapter, fps: int = DEFAULT_FPS):
        super().__init__()
        self.adapter = adapter
        self.started = False

        self._dc = None

        self._fps = int(fps)
        self._interval = 1.0 / float(self._fps)
        self._next_time = None

        self._waiting_img = adapter.waiting_img
        self._last_img = self._waiting_img

    def set_datachannel(self, dc):
        self._dc = dc

    def _send(self, payload: Dict[str, Any]):
        if self._dc is None:
            return
        if getattr(self._dc, "readyState", "") != "open":
            return
        try:
            self._dc.send(json.dumps(payload))
        except Exception:
            pass

    def reset_clock(self):
        self._next_time = None

    def set_started(self, flag: bool):
        self.started = flag
        self.reset_clock()

    def set_fps_if_waiting(self, fps: int) -> bool:
        """Allow changing FPS only when not started."""
        if self.started:
            return False
        fps_i = int(fps)
        fps_i = max(1, min(120, fps_i))
        self._fps = fps_i
        self._interval = 1.0 / float(self._fps)
        self.reset_clock()
        return True

    @property
    def fps(self) -> int:
        return self._fps

    async def recv(self):
        # steady pacing at current FPS (even while waiting)
        now = time.time()
        if self._next_time is None:
            self._next_time = now
        self._next_time += self._interval

        delay = self._next_time - now
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            if delay < -0.5:
                self._next_time = time.time()

        if not self.started:
            img = self._waiting_img
        else:
            frame = self.adapter.get_frames()
            if frame is None:
                img = self._last_img
            else:
                img = frame
                self._last_img = img

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


pcs = set()
sessions: Dict[str, Dict[str, Any]] = {}


def make_app(cfg_path: str):
    app = web.Application(client_max_size=20 * 1024 * 1024)

    async def index(request):
        here = os.path.dirname(os.path.abspath(__file__))
        return web.FileResponse(
            os.path.join(here, "client.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    async def offer(request):
        params = await request.json()
        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection(ICE_CONFIG)
        pcs.add(pc)

        adapter = Adapter(cfg_path)
        track = AdapterVideoTrack(adapter, fps=DEFAULT_FPS)
        pc.addTrack(track)

        session_id = str(uuid.uuid4())
        sessions[session_id] = {"adapter": adapter, "track": track}

        @pc.on("connectionstatechange")
        async def on_conn():
            print("connectionState:", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                sessions.pop(session_id, None)
                try:
                    adapter.terminate()
                except Exception:
                    pass
                await pc.close()
                pcs.discard(pc)

        @pc.on("datachannel")
        def on_datachannel(dc):
            print("DataChannel:", dc.label)
            track.set_datachannel(dc)

            async def do_start():
                try:
                    await asyncio.to_thread(adapter.start)
                    track.set_started(True)
                    track._send({"type": "started", "started": True, "fps": track.fps, "t_server_ms": int(time.time() * 1000)})
                except Exception as e:
                    track.set_started(False)
                    track._send({"type": "error", "reason": "start_failed", "message": str(e), "t_server_ms": int(time.time() * 1000)})

            async def do_restart_to_waiting():
                try:
                    await asyncio.to_thread(adapter.reset)
                    track._last_img = track._waiting_img
                    track.set_started(False)
                    track._send({"type": "restarted", "started": False, "fps": track.fps, "t_server_ms": int(time.time() * 1000)})
                except Exception as e:
                    track._send({"type": "error", "reason": "restart_failed", "message": str(e), "t_server_ms": int(time.time() * 1000)})

            @dc.on("message")
            def on_message(msg):
                if not isinstance(msg, str):
                    return
                try:
                    data = json.loads(msg)
                except Exception:
                    return

                cmd = data.get("cmd")

                if cmd == "start":
                    asyncio.create_task(do_start())
                    return

                if cmd == "restart":
                    asyncio.create_task(do_restart_to_waiting())
                    return

                if cmd == "set_fps":
                    fps = data.get("fps", DEFAULT_FPS)
                    ok = track.set_fps_if_waiting(int(fps))
                    track._send({
                        "type": "fps",
                        "ok": ok,
                        "fps": track.fps,
                        "started": track.started,
                        "t_server_ms": int(time.time() * 1000),
                        "message": "FPS updated (waiting state)." if ok else "Cannot change FPS while started.",
                    })
                    return

                if cmd == "action":
                    if not track.started:
                        track._send({
                            "type": "error",
                            "reason": "not_started",
                            "message": "Press Start before sending actions.",
                            "t_server_ms": int(time.time() * 1000),
                        })
                        return

                    action_key = str(data.get("key", ""))
                    action_code = int(data.get("code", 0))
                    action_id = str(data.get("id", ""))
                    t_send_ms = int(data.get("t_send_ms", 0))

                    try:
                        adapter.push_action(action_code)
                    except Exception:
                        pass

                    action_text = action_key.upper() if action_key else "ACTION"
                    track._send({
                        "type": "ack",
                        "id": action_id,
                        "action": action_text,
                        "code": action_code,
                        "t_send_ms": t_send_ms,
                        "t_server_ms": int(time.time() * 1000),
                    })
                    return

                if cmd == "ping":
                    track._send({
                        "type": "pong",
                        "t_client_ms": int(data.get("t_client_ms", 0)),
                        "t_server_ms": int(time.time() * 1000),
                    })
                    return

                if cmd == "stats":
                    track._send({
                        "type": "stats",
                        "started": track.started,
                        "fps": track.fps,
                        "num_played_frames": int(adapter.num_played_frames),
                        "t_server_ms": int(time.time() * 1000),
                    })
                    return

        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_ice_complete(pc)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "session_id": session_id})

    async def on_shutdown(app):
        await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
        pcs.clear()
        sessions.clear()

    async def upload_image(request):
        try:
            data = await request.post()
            session_id = data.get("session_id")
            image_field = data.get("image")

            if not session_id or not image_field:
                return web.json_response({"ok": False, "error": "Missing session_id or image"}, status=400)

            session = sessions.get(session_id)
            if not session:
                return web.json_response({"ok": False, "error": "Invalid session"}, status=404)

            image_data = image_field.file.read()

            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()
            except Exception:
                return web.json_response({"ok": False, "error": "Invalid image file"}, status=400)

            adapter = session["adapter"]
            track = session["track"]

            new_img = await asyncio.to_thread(adapter.set_custom_image, image_data)
            track._waiting_img = new_img
            if not track.started:
                track._last_img = new_img

            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/upload_image", upload_image)
    app.on_shutdown.append(on_shutdown)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--cfg",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
        ),
        help="YAML path for the WorldPlay adapter (shared-memory buffer names).",
    )
    args = p.parse_args()

    web.run_app(
        make_app(args.cfg),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
