#!/bin/bash
# Release gate: run before every push.
#   scripts/release_gate.sh          # scan files staged for commit
#   scripts/release_gate.sh --all    # scan every tracked file (pre-push audit)
# Checks:
# 1) Forbidden-term scan (naming discipline over published code).
# 2) Tracked-file sanity: no mirrors, no state dirs, no secrets.
set -euo pipefail
cd "$(dirname "$0")/.."

# Terms that must never appear in published code (case-insensitive).
# Assembled from pieces so this gate file itself passes the scan.
A="flash"; B="rt"; C="world"; D="foundry"; E="vllm"; F="omni"
G="m"; H="star"; I="sol"; J="engine"; K="nv"; L="labs"
PATTERNS="${A}${B}|${A}[-_]${B}|${C}${D}|${E}[-_ ]?${F}|\b${G}${H}\b|${I}[-_]${J}|${K}${L}"

if [ "${1:-}" = "--all" ]; then
  FILES=$(git ls-files)
else
  FILES=$(git diff --cached --name-only --diff-filter=ACM)
fi
[ -z "$FILES" ] && { echo "release_gate: nothing to scan"; exit 0; }

FAIL=0
for f in $FILES; do
  [ -f "$f" ] || continue
  if echo "$f" | grep -qiE "$PATTERNS"; then
    echo "release_gate: FORBIDDEN filename: $f"; FAIL=1
  fi
  if grep -qiE "$PATTERNS" "$f" 2>/dev/null; then
    echo "release_gate: FORBIDDEN term in: $f"
    grep -niE "$PATTERNS" "$f" | head -3
    FAIL=1
  fi
  case "$f" in
    upstream-*|ref/*|target-coverage/*|.deli/*|.venv/*)
      echo "release_gate: local-only path staged: $f"; FAIL=1;;
  esac
  if grep -qE "ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}" "$f" 2>/dev/null; then
    echo "release_gate: SECRET-LIKE token in: $f"; FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "release_gate: BLOCKED"; exit 1
fi
echo "release_gate: PASS ($(echo "$FILES" | wc -l) files)"
