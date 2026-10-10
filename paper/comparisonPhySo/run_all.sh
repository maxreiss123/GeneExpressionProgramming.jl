#!/usr/bin/env bash
# The comparison end to end: data, GEP-SBP runs, PhySO runs, judging, figures, for each
# noise level in NOISES (default "0 0.05 0.1").
#
#   JULIA=julia PYTHON=python bash paper/comparisonPhySo/run_all.sh
#
# Each method runs on its own, WORKERS (default 4) single-threaded GEP runs at a time and
# PHYSO_WORKERS (default 3) PhySO runs. PYTHON needs physo 1.1.11 (with torch), sympy,
# numpy, pandas, matplotlib and SciencePlots. Runs whose result exists are skipped, so it
# resumes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
JULIA="${JULIA:-julia}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"
NOISES="${NOISES:-0 0.05 0.1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$HERE/results/logs"

cd "$HERE"
# shellcheck disable=SC2086
"$PYTHON" export_data.py --noise $NOISES

for noise in $NOISES; do
  # GEP-SBP: WORKERS Julia processes share the runs
  for k in $(seq 1 "$WORKERS"); do
    "$JULIA" --project="$ROOT" --threads=1 gep_run.jl "noise=$noise" "worker=$k/$WORKERS" \
      > "results/logs/gep_noise${noise}_w$k.out" 2> "results/logs/gep_noise${noise}_w$k.err" &
  done
  wait
done

for noise in $NOISES; do
  # PhySO: one process per run, PHYSO_WORKERS at a time; a PhySO process grows past
  # 3.5 GB, and four overrun 16 GB of memory
  "$PYTHON" physo_batch.py --noise "$noise" --workers "${PHYSO_WORKERS:-3}"
  "$PYTHON" judge.py --noise "$noise"
done

# the symbolic check against the models' numbers, then the figures (SciencePlots)
# shellcheck disable=SC2086
"$PYTHON" audit_symbolic.py --noise $NOISES
"$PYTHON" make_figures.py
