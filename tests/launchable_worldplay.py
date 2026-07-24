"""Launchable-tier evidence for the worldplay app.

Launches the reference backend as a subprocess (1 GPU), waits for READY,
attaches the app adapter, starts a session, streams WASD-style action
codes (the worker is reactive: no actions -> no frames), polls the video
ring buffer until >= MIN_FRAMES frames arrive, then terminates cleanly.
Prints LAUNCHABLE_OK with the frame count, or a specific failure verdict.
Process hygiene: killpg in finally + post-run liveness assertion.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG = str(ROOT / "wllm/apps/worldplay/config.yaml")
MIN_FRAMES = 8
READY_TIMEOUT_S = 1500    # load + 18 warmup chunks at 704x1280
FRAMES_TIMEOUT_S = 600
# Discrete action codes 0..80; a small repeating motion pattern keeps the
# reactive worker fed one code per generated frame.
ACTION_PATTERN = [1, 1, 1, 1, 2, 2, 1, 1]
MAX_ACTIONS = 64          # < cfg.max_num_actions (72)


def main() -> int:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"

    log_path = ROOT / "logs/launchable_worldplay_backend.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m",
         "wllm.apps.worldplay.backend.launch", "--variant", "reference"],
        cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True)
    verdict, frames_seen = "FAIL_UNKNOWN", 0
    try:
        t0 = time.monotonic()
        ready = False
        while time.monotonic() - t0 < READY_TIMEOUT_S:
            if proc.poll() is not None:
                verdict = f"FAIL_BACKEND_EXIT rc={proc.returncode}"
                break
            # the launch banner quotes the READY phrase — only a real log
            # line (not the "[launch]" banner) counts
            if any("WorldPlay backend READY" in ln and "[launch]" not in ln
                   for ln in log_path.read_text(errors="replace").splitlines()):
                ready = True
                break
            time.sleep(5)
        else:
            verdict = "FAIL_READY_TIMEOUT"

        if ready:
            # Construct the adapter only after READY: the worker owns the
            # shared buffers (create=True); the adapter attaches to them.
            from wllm.apps.worldplay.adapter import WorldPlayAdapter
            ad = WorldPlayAdapter(CFG)
            ad.start()
            print("[harness] session started", flush=True)

            pushed = 0
            t1 = time.monotonic()
            while time.monotonic() - t1 < FRAMES_TIMEOUT_S:
                # keep the reactive worker fed, one code at a time
                if pushed < MAX_ACTIONS:
                    ad.push_action(ACTION_PATTERN[pushed % len(ACTION_PATTERN)])
                    pushed += 1
                frame = ad.get_frames()
                if frame is not None:
                    frames_seen += 1
                    if frames_seen in (1, MIN_FRAMES) or frames_seen % 16 == 0:
                        print(f"[harness] frames_total={frames_seen} "
                              f"shape={frame.shape}", flush=True)
                    if frames_seen >= MIN_FRAMES:
                        break
                if proc.poll() is not None:
                    verdict = f"FAIL_BACKEND_DIED rc={proc.returncode}"
                    break
                time.sleep(0.05)
            if frames_seen >= MIN_FRAMES:
                verdict = "OK"
            elif verdict == "FAIL_UNKNOWN":
                verdict = "FAIL_NO_FRAMES"
            print(f"[harness] actions_pushed={pushed}", flush=True)
            try:
                ad.terminate()
            except Exception as exc:  # noqa: BLE001
                print(f"[harness] terminate() raised: {exc!r}", flush=True)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log_f.close()
        assert proc.poll() is not None, "backend still alive after cleanup"

    print(f"frames_received={frames_seen}")
    print(f"LAUNCHABLE_{'OK' if verdict == 'OK' else verdict}")
    return 0 if verdict == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
