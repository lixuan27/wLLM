import importlib, pkgutil, sys, traceback
sys.path.insert(0, "/public/home/lixuan/lixuan/wllm-infra")
import wllm
res = {"ok": 0, "fail": []}
roots = ["wllm.serving", "wllm.kernels_t", "wllm.native", "wllm.native_wm"]
mods = []
for root in roots:
    try:
        pkg = importlib.import_module(root)
        mods.append((root, None))
        for m in pkgutil.walk_packages(pkg.__path__, prefix=root + "."):
            mods.append((m.name, None))
    except Exception as e:
        res["fail"].append((root, f"ROOT: {type(e).__name__}: {str(e)[:120]}"))
import glob, os
for ad in glob.glob("/public/home/lixuan/lixuan/wllm-infra/wllm/apps/*/adapter.py"):
    app = os.path.basename(os.path.dirname(ad))
    mods.append((f"wllm.apps.{app}.adapter", None))
seen = set()
for name, _ in mods:
    if name in seen: continue
    seen.add(name)
    try:
        importlib.import_module(name)
        res["ok"] += 1
    except Exception as e:
        res["fail"].append((name, f"{type(e).__name__}: {str(e)[:120]}"))
print(f"OK={res['ok']} FAIL={len(res['fail'])}")
from collections import Counter
kinds = Counter(f.split(":")[0] for _, f in res["fail"])
print("fail kinds:", dict(kinds))
for n, f in res["fail"][:25]:
    print("  FAIL", n, "->", f)
