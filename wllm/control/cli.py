"""wllm control-plane CLI.

    wllm inspect  <root>                     project discovery -> manifest
    wllm plan     <root> --spec s.yaml       backend/pass planning (dry)
    wllm verify   --receipt r.json           promote-gate + fingerprint check
    wllm apply    --receipt r.json           promote (fail-closed)
    wllm rollback                            walk the fallback chain
    wllm report                              current deploy state + receipts

Designed to run identically with or without an agent driving it; every
subcommand prints human-readable text and writes machine-readable JSON
under `.wllm/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inspect import inspect_project
from .receipt import Receipt
from .registry import default_registry, legal_passes, rank_backends
from .spec import OptimizeSpec
from .state import DeployManager


def _workdir(root: str) -> Path:
    return Path(root) / ".wllm"


def cmd_inspect(args) -> int:
    man = inspect_project(args.root, probe_gpu=not args.no_gpu_probe)
    out = man.save(_workdir(args.root) / "manifests" / "project-manifest.json")
    print(f"manifest -> {out}")
    print(f"  entrypoints: {len(man.entrypoints)}, "
          f"model configs: {len(man.model_configs)}, "
          f"checkpoint refs: {len(man.checkpoint_refs)}, "
          f"gpus: {len(man.gpus)} ({man.gpu_probe})")
    for u in man.unknowns:
        print(f"  UNKNOWN: {u}")
    return 0


def cmd_plan(args) -> int:
    spec = OptimizeSpec.load(args.spec) if args.spec else OptimizeSpec(
        project=args.root)
    errs = spec.validate()
    if errs:
        for e in errs:
            print(f"SPEC ERROR: {e}", file=sys.stderr)
        return 2
    caps = default_registry()
    ranked = rank_backends(caps, args.model,
                           required_out=spec.contract.required_modalities)
    if not ranked:
        print(f"no registered backend supports model {args.model!r} with "
              f"required modalities {spec.contract.required_modalities}; "
              f"diagnose-only mode (nothing will be changed)")
        return 3
    context = {"num_gpus": args.num_gpus}
    if args.context:
        context.update(json.loads(args.context))
    plan_doc = {"model": args.model, "spec": spec.to_dict(), "backends": []}
    for cap, tier in ranked:
        decisions = legal_passes(cap, context, spec.quality.policy)
        kept = [d.name for d in decisions if d.kept]
        rejected = {d.name: d.reason for d in decisions if not d.kept}
        plan_doc["backends"].append({
            "backend": cap.backend, "support": tier,
            "passes": kept, "rejected_passes": rejected,
        })
        print(f"{cap.backend} [{tier}] passes: {', '.join(kept) or '(none)'}")
        for name, why in rejected.items():
            print(f"    reject {name}: {why}")
    out = _workdir(args.root) / "plans" / f"plan-{args.model.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan_doc, indent=1))
    print(f"plan -> {out}")
    if not any(b["passes"] for b in plan_doc["backends"]):
        print("diagnose-only: no optimizing backend offers a legal pass "
              "here; the reference path remains available, nothing will "
              "be changed")
        return 3
    return 0


def cmd_verify(args) -> int:
    rec = Receipt.load(args.receipt)
    problems = rec.promote_problems(args.quality_policy)
    speed = rec.speedup()
    print(f"receipt {rec.plan_id} fingerprint={rec.fingerprint()}")
    if speed is not None:
        print(f"  measured speedup vs baseline: {speed:.2f}x")
    if problems:
        for p in problems:
            print(f"  BLOCK: {p}")
        return 1
    print("  promote gate: PASS")
    return 0


def cmd_apply(args) -> int:
    rec = Receipt.load(args.receipt)
    mgr = DeployManager(_workdir(args.root))
    try:
        st = mgr.apply(rec, args.quality_policy)
    except PermissionError as exc:
        print(f"APPLY BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"active plan: {st.active} (last known good: {st.last_known_good})")
    return 0


def cmd_rollback(args) -> int:
    mgr = DeployManager(_workdir(args.root))
    st = mgr.rollback()
    print(f"active plan: {st.active} (last known good: {st.last_known_good})")
    return 0


def cmd_report(args) -> int:
    mgr = DeployManager(_workdir(args.root))
    st = mgr.state()
    print(f"active: {st.active}")
    print(f"last known good: {st.last_known_good}")
    rec = mgr.active_receipt()
    if rec is not None:
        speed = rec.speedup()
        print(f"active receipt: backend={rec.backend} "
              f"passes={rec.passes} "
              f"speedup={f'{speed:.2f}x' if speed else 'n/a'}")
    for ev in st.history[-args.last:]:
        print(f"  {ev['event']}: {ev['detail']} -> active={ev['active']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wllm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="discover project facts")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--no-gpu-probe", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("plan", help="rank backends + legal passes (dry)")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--model", required=True)
    p.add_argument("--spec", default="")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--context", default="", help="extra JSON context facts")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("verify", help="check a receipt's promote gate")
    p.add_argument("--receipt", required=True)
    p.add_argument("--quality-policy", default="exact")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("apply", help="promote a receipt-backed plan")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--receipt", required=True)
    p.add_argument("--quality-policy", default="exact")
    p.set_defaults(fn=cmd_apply)

    p = sub.add_parser("rollback", help="walk the fallback chain")
    p.add_argument("root", nargs="?", default=".")
    p.set_defaults(fn=cmd_rollback)

    p = sub.add_parser("report", help="deploy state + active receipt")
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--last", type=int, default=5)
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
