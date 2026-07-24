#!/bin/bash
# Release gate: run before every push.
# 1) Forbidden-term scan over files staged for commit (naming discipline).
# 2) Tracked-file sanity: no mirrors, no state dirs, no secrets.
set -euo pipefail
cd "$(dirname "$0")/.."

# Terms that must never appear in published code (case-insensitive).
# Kept in a variable assembled from pieces so this gate file itself passes.
A="flash"; B="rt"
PATTERNS="${A}${B}|${A}_${B}|${A}-${B}"

FILES=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$FILES" ] && { echo "release_gate: nothing staged"; exit 0; }

FAIL=0
for f in $FILES; do
  [ -f "$f" ] || continue
  if echo "$f" | grep -qiE "$PATTERNS"; then
    echo "release_gate: FORBIDDEN filename: $f"; FAIL=1
  fi
  if grep -qiE "$PATTERNS" "$f"; then
    echo "release_gate: FORBIDDEN term in: $f"
    grep -niE "$PATTERNS" "$f" | head -3
    FAIL=1
  fi
  case "$f" in
    upstream-*|worldfoundry-upstream/*|WorldFoundry-main/*|ref/*|target-coverage/*|.deli/*|.venv/*)
      echo "release_gate: local-only path staged: $f"; FAIL=1;;
  esac
  if grep -qE "ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}" "$f"; then
    echo "release_gate: SECRET-LIKE token in: $f"; FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "release_gate: BLOCKED"; exit 1
fi
echo "release_gate: PASS ($(echo "$FILES" | wc -l) files)"
