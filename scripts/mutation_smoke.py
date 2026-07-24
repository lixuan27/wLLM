"""Mutation smoke for the control plane's safety core.

Generates first-order mutants (comparison flips, and/or swaps, boolean
constant flips, dropped negations) of the fail-closed modules, runs the
focused test suite against each mutant in an isolated sandbox, and
requires the suite to kill at least ``--threshold`` of them. A surviving
mutant means a safety rule the tests do not actually pin down.

Usage (compute node):
    python scripts/mutation_smoke.py [--threshold 0.8] [--max-per-file 12]
"""

from __future__ import annotations

import argparse
import ast
import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "wllm/control/receipt.py",
    "wllm/control/registry.py",
    "wllm/control/state.py",
    "wllm/control/spec.py",
)
SANDBOX_FILES = (
    "wllm/__init__.py",
    "tests/test_control_plane.py",
)
SANDBOX_TREES = (
    "wllm/control",
)
TEST_CMD = [sys.executable, "tests/test_control_plane.py"]

_CMP_SWAP = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE,
             ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt}


class SiteCollector(ast.NodeVisitor):
    """Enumerate mutable sites; each site index maps to one mutant."""

    def __init__(self):
        self.sites: list[tuple[str, int]] = []   # (kind, occurrence index)
        self._counts: dict[str, int] = {}

    def _add(self, kind: str):
        idx = self._counts.get(kind, 0)
        self._counts[kind] = idx + 1
        self.sites.append((kind, idx))

    def visit_Compare(self, node):
        for op in node.ops:
            if type(op) in _CMP_SWAP:
                self._add("cmp")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self._add("boolop")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value is True or node.value is False:
            self._add("const")
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            self._add("not")
        self.generic_visit(node)


class Mutator(ast.NodeTransformer):
    def __init__(self, kind: str, target_idx: int):
        self.kind, self.target = kind, target_idx
        self._count = 0
        self.applied = False

    def _hit(self, kind: str) -> bool:
        if kind != self.kind:
            return False
        idx = self._count
        self._count += 1
        if idx == self.target:
            self.applied = True
            return True
        return False

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.kind == "cmp":
            new_ops = []
            for op in node.ops:
                if type(op) in _CMP_SWAP and self._hit("cmp"):
                    new_ops.append(_CMP_SWAP[type(op)]())
                else:
                    new_ops.append(op)
            node.ops = new_ops
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self._hit("boolop"):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_Constant(self, node):
        if node.value is True or node.value is False:
            if self._hit("const"):
                return ast.copy_location(
                    ast.Constant(value=not node.value), node)
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit("not"):
            return node.operand
        return node


def make_mutants(source: str, max_per_file: int) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    coll = SiteCollector()
    coll.visit(tree)
    sites = coll.sites
    if len(sites) > max_per_file:            # deterministic spread
        step = len(sites) / max_per_file
        sites = [sites[int(i * step)] for i in range(max_per_file)]
    mutants = []
    for kind, idx in sites:
        mut = Mutator(kind, idx)
        new_tree = mut.visit(copy.deepcopy(tree))
        if not mut.applied:
            continue
        ast.fix_missing_locations(new_tree)
        mutants.append((f"{kind}#{idx}", ast.unparse(new_tree)))
    return mutants


def build_sandbox(dst: Path) -> None:
    for rel in SANDBOX_TREES:
        shutil.copytree(ROOT / rel, dst / rel)
    for rel in SANDBOX_FILES:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)


def run_suite(sandbox: Path, timeout_s: float = 120.0) -> bool:
    """True == suite green (mutant survived)."""
    try:
        out = subprocess.run(TEST_CMD, cwd=sandbox, capture_output=True,
                             text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False                       # hang counts as killed
    return out.returncode == 0 and "ALL PASS" in out.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--max-per-file", type=int, default=12)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="wllm_mut_") as td:
        sandbox = Path(td)
        build_sandbox(sandbox)
        if not run_suite(sandbox):
            print("mutation_smoke: baseline suite is RED in the sandbox — "
                  "fix tests before measuring mutants")
            return 2
        total = killed = 0
        survivors: list[str] = []
        for rel in TARGETS:
            original = (ROOT / rel).read_text()
            for tag, mutated in make_mutants(original, args.max_per_file):
                total += 1
                (sandbox / rel).write_text(mutated)
                if run_suite(sandbox):
                    survivors.append(f"{rel} [{tag}]")
                else:
                    killed += 1
                (sandbox / rel).write_text(original)
        rate = killed / total if total else 0.0
        print(f"mutation_smoke: {killed}/{total} mutants killed "
              f"({rate:.0%}); threshold {args.threshold:.0%}")
        for s in survivors:
            print(f"  SURVIVOR: {s}")
        if rate < args.threshold:
            print("mutation_smoke: BLOCKED (kill rate below threshold)")
            return 1
        print("mutation_smoke: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
