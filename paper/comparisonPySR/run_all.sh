#!/usr/bin/env bash
# The comparison end to end: data, GEP-SBP runs, PySR runs, judging, audit, figures, for
# each noise level in NOISES (default "0 0.05 0.1"), and GEP-SBP without units (no noise).
#
#   JULIA=julia PYTHON=python bash paper/comparisonPySR/run_all.sh
#
# Each method runs on its own, WORKERS (default 4) single-threaded processes at a time.
# PYTHON needs pysr 2.7.0 (with its Julia packages), physo 1.1.11 (with torch; for the
# AI-Feynman problems and the symbolic check), sympy, numpy, pandas, matplotlib and
# SciencePlots. Runs whose result exists are skipped, so it resumes. The whole-budget
# GEP-SBP runs of Figure 4 and the sensitivity runs are in the README.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
JULIA="${JULIA:-julia}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"
NOISES="${NOISES:-0 0.05 0.1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHON_JULIACALL_THREADS=1 JULIA_NUM_THREADS=1
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
# GEP-SBP without units (no SBP library, no repair), without noise, for Figure 3
for k in $(seq 1 "$WORKERS"); do
  "$JULIA" --project="$ROOT" --threads=1 gep_run.jl units=false "worker=$k/$WORKERS" \
    out=results/gep_nounits \
    > "results/logs/gep_nounits_w$k.out" 2> "results/logs/gep_nounits_w$k.err" &
done
wait

for noise in $NOISES; do
  # PySR: WORKERS Python processes share the runs
  for k in $(seq 1 "$WORKERS"); do
    "$PYTHON" pysr_run.py --noise "$noise" --worker "$k/$WORKERS" \
      > "results/logs/pysr_noise${noise}_w$k.out" \
      2> "results/logs/pysr_noise${noise}_w$k.err" &
  done
  wait
  "$PYTHON" judge.py --noise "$noise"
done
"$PYTHON" judge.py --runs-dir results/gep_nounits

# the symbolic check against the models' numbers, then the figures (SciencePlots)
# shellcheck disable=SC2086
"$PYTHON" audit_symbolic.py --noise $NOISES
"$PYTHON" make_figures.py
