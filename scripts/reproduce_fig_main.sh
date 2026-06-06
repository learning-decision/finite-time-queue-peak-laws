#!/usr/bin/env bash
# Reproduce the main parallel-server figures (four epsilon values).
#
# Runtime: ~30-45 minutes on a 10-core machine with the native C++ fast path.
# Requires a C++17 compiler (g++ or clang++) on PATH.
# The first run compiles src/qpeak/simulation/fast_parallel_server.cpp automatically.
# To disable the native path: add '"fast_mode": "off"' to the simulation block.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== Main figures: long parallel-server epsilon sweep ==="
echo "Running four epsilon values (eps = 0.005, 0.001, 0.0005, 0.0001)..."

python main.py configs/two_phase_parallel_server_0.005.json
python main.py configs/two_phase_parallel_server_0.001.json
python main.py configs/two_phase_parallel_server_0.0005.json
python main.py configs/two_phase_parallel_server_0.0001.json

echo ""
echo "=== Plotting main figures ==="
PYTHONPATH=src python scripts/plot_logscale.py \
    _runs/two_phase_parallel_server \
    --out_dir _runs/two_phase_parallel_server_figs \
    --band_mult 1.96

echo ""
echo "Main figures written to: _runs/two_phase_parallel_server_figs/"
