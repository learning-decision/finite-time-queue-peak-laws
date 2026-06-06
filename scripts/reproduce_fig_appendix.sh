#!/usr/bin/env bash
# Reproduce the appendix large-slack figures (IQS and parallel-server).
# Runtime: a few minutes on a laptop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== Appendix: IQS large-slack experiment ==="
python main.py configs/two_phase_iqs.json
PYTHONPATH=src python scripts/plot_logscale.py \
    _runs/two_phase_iqs_large_eps \
    --out_dir _runs/two_phase_iqs_large_eps_figs \
    --band_mult 1.96

echo ""
echo "=== Appendix: Parallel-server large-slack experiment ==="
python main.py configs/two_phase_parallel_server_1.json
PYTHONPATH=src python scripts/plot_logscale.py \
    _runs/two_phase_parallel_server_large_eps_1 \
    --out_dir _runs/two_phase_parallel_server_large_eps_1_figs \
    --band_mult 1.96

echo ""
echo "Appendix figures written to:"
echo "  _runs/two_phase_iqs_large_eps_figs/"
echo "  _runs/two_phase_parallel_server_large_eps_1_figs/"
echo ""
echo "Key PDFs (referenced in the paper appendix):"
echo "  _runs/two_phase_iqs_large_eps_figs/peak_l2_vs_t_logscale.pdf"
echo "  _runs/two_phase_iqs_large_eps_figs/peak_l2_over_logT_vs_t.pdf"
echo "  _runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_vs_t_logscale.pdf"
echo "  _runs/two_phase_parallel_server_large_eps_1_figs/peak_l2_over_logT_vs_t.pdf"
