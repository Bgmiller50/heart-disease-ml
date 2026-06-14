#!/usr/bin/env bash
# Reset practice_build/ to empty starter templates for another from-scratch attempt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRACTICE="$ROOT/practice_build"
TEMPLATES="$PRACTICE/templates"

if [[ ! -d "$TEMPLATES" ]]; then
  echo "Missing templates at $TEMPLATES"
  exit 1
fi

echo "Resetting practice_build (keeps .venv if present)..."
rm -rf "$PRACTICE/src"
rm -rf "$PRACTICE/data/raw/"* "$PRACTICE/data/processed/"* 2>/dev/null || true
rm -rf "$PRACTICE/models/"* "$PRACTICE/reports/figures/"* 2>/dev/null || true
rm -f "$PRACTICE/notebooks/01_eda_and_modeling.ipynb" 2>/dev/null || true

mkdir -p "$PRACTICE/src" "$PRACTICE/data/raw" "$PRACTICE/data/processed"
mkdir -p "$PRACTICE/models" "$PRACTICE/reports/figures" "$PRACTICE/notebooks"

cp -r "$TEMPLATES/src/"* "$PRACTICE/src/"
cp "$TEMPLATES/requirements.txt" "$PRACTICE/requirements.txt" 2>/dev/null || true
touch "$PRACTICE/src/__init__.py"

echo "Done. Start at BUILD_FROM_SCRATCH.md Phase 0."
echo "  cd \"$PRACTICE\""
