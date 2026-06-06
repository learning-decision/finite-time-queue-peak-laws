#!/usr/bin/env bash
# Reproduce all paper figures (appendix + main).
# See scripts/reproduce_fig_appendix.sh and scripts/reproduce_fig_main.sh
# for per-figure details and runtime estimates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ">>> Step 1: Appendix figures"
bash reproduce_fig_appendix.sh

echo ""
echo ">>> Step 2: Main figures (long run — ~30-45 min on 10 cores)"
bash reproduce_fig_main.sh

echo ""
echo "All figures reproduced."
