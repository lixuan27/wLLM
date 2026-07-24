"""Launchable-tier evidence for the longlive app.

Launches the reference backend as a subprocess (1 GPU), waits for READY,
attaches the app adapter, starts a session, polls the video ring buffer
until >= MIN_FRAMES frames arrive, then terminates everything cleanly.
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

CFG = str(ROOT / "wllm/apps/longlive/config.yaml")
MIN_FRAMES = 8
READY_TIMEOUT_S = 900     # first launch pays model load
FRAMES_TIMEOUT_S = 420


def main() -> int:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"

    log_path = ROOT / "logs/launchable_longlive_backend.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "-m",
         "wllm.apps.longlive.backend.launch", "--variant", "reference"],
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
            if any("LongLive backend READY" in ln and "[launch]" not in ln
                   for ln in log_path.read_text(errors="replace").splitlines()):
                ready = True
                break
            time.sleep(5)
        else:
            verdict = "FAIL_READY_TIMEOUT"

        if ready:
            from wllm.apps.longlive.adapter import LongLiveAdapter
            ad = LongLiveAdapter(CFG)
            ad.start()
            print("[harness] session started", flush=True)

            # the worker gates generation on the first spoken prompt (VAD +
            # ASR); stream a real 16 kHz speech fixture + trailing silence
            import numpy as np
            import soundfile as sf
            speech, sr = sf.read(ROOT / "tests/fixtures/speech_16k.wav",
                                 dtype="float32")
            assert sr == 16000, sr
            speech = np.concatenate(
                [speech, np.zeros(int(16000 * 2.0), dtype=np.float32)])
            # NOTE: do NOT call enable_microphone() here — the reference
            # worker never acks the optional signal buffer, and the control
            # send blocks forever waiting for that ack (r11 hang).  The
            # worker drains the audio ring unconditionally.
            step = 320  # cfg.audio_frame_samples
            for off in range(0, len(speech), step):
                ad.push_audio(speech[off:off + step])
                time.sleep(0.005)   # ~faster-than-realtime, bounded queue safe
            print(f"[harness] pushed {len(speech)/16000:.1f}s speech",
                  flush=True)
            t1 = time.monotonic()
            next_idx = 0
            while time.monotonic() - t1 < FRAMES_TIMEOUT_S:
                got = ad.get_frames()
                if got is not None:
                    idx, frames = got if isinstance(got, tuple) else (None, got)
                    n = 0 if frames is None else len(frames)
                    if n:
                        frames_seen += n
                        if idx is not None:
                            next_idx = idx
                            ad.commit(next_idx)
                        print(f"[harness] frames_total={frames_seen}",
                              flush=True)
                    if frames_seen >= MIN_FRAMES:
                        break
                if proc.poll() is not None:
                    verdict = f"FAIL_BACKEND_DIED rc={proc.returncode}"
                    break
                time.sleep(0.5)
            if frames_seen >= MIN_FRAMES:
                verdict = "OK"
            elif verdict == "FAIL_UNKNOWN":
                verdict = "FAIL_NO_FRAMES"
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
