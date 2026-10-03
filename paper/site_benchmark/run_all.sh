#!/usr/bin/env bash
# Full benchmark matrix behind the SITE comparison (arXiv:2507.01466v1).
#
# Everything is run strictly sequentially so that the wall-clock numbers of the two
# frameworks are measured under identical conditions on the same machine.  Stages whose
# result JSON already exists are skipped, so the script is safe to interrupt and re-run.
#
#   JULIA        path to the julia binary            (default: julia)
#   PYTHON       python with geppy + deap installed  (default: python3)
#   THREADS      julia threads                       (default: 4)
#   SEEDS        evolution seeds, scalar path        (default: 1 2 3 4 5)
#   TENSOR_SEEDS evolution seeds, tensor path        (default: 1 2 3)
#   SITE_REPO    clone of https://github.com/PistilReaper/SITE (optional)
#   SHADOW_ROOT  clone of GeneExpressionProgrammingShadow.jl, to run the tensor case
#                against the development package instead of this one (optional)
#
# Usage:  SITE_REPO=/path/to/SITE THREADS=4 bash paper/site_benchmark/run_all.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
JULIA="${JULIA:-julia}"
PYTHON="${PYTHON:-python3}"
THREADS="${THREADS:-4}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TENSOR_SEEDS="${TENSOR_SEEDS:-1 2 3}"
RES="$HERE/results"
LOGS="$RES/logs"
mkdir -p "$RES" "$LOGS"

jl() {  # jl <script> <name> <args...>
  local script="$1" name="$2"; shift 2
  [ -f "$RES/$name.json" ] && { echo "skip $name"; return; }
  echo "--- $name  ($(date +%H:%M:%S))"
  "$JULIA" --project="$ROOT" --threads="$THREADS" "$HERE/$script" "$@" \
      --out "results/$name.json" > "$LOGS/$name.log" 2>&1
  grep -hE "loss=|^size|mu S_ij" "$LOGS/$name.log" | tail -3
}

site() {  # site <case> <marker>
  [ -z "${SITE_REPO:-}" ] && { echo "SITE_REPO unset -- skipping SITE $1"; return; }
  [ -f "$RES/site_$2.json" ] && { echo "skip site $1"; return; }
  echo "--- SITE $1  ($(date +%H:%M:%S))"
  "$PYTHON" "$HERE/run_site_reference.py" "$SITE_REPO" --cases "$1" \
      --outdir "$RES" 2>&1 | tee -a "$LOGS/site_reference.log"
}

# ------------------------------------------------- case 1: Maxwell, scalar path ----
for s in $SEEDS; do
  jl maxwell_gep.jl "gep_dhc_s$s"      --data clean --config dhc   --seed "$s" --epochs 2000 --pop 1600
  jl maxwell_gep.jl "gep_nodhc_s$s"    --data clean --config nodhc --seed "$s" --epochs 2000 --pop 1600
  jl maxwell_gep.jl "gep_ls_clean_s$s" --data clean --config dhc   --seed "$s" --epochs 2000 --pop 1600 --scaling true
done
for lvl in 005 010 020; do
  for s in $SEEDS; do
    jl maxwell_gep.jl "gep_noise${lvl}_s$s"    --data "noise$lvl" --config dhc --seed "$s" --epochs 2000 --pop 1600
    jl maxwell_gep.jl "gep_ls_noise${lvl}_s$s" --data "noise$lvl" --config dhc --seed "$s" --epochs 2000 --pop 1600 --scaling true
  done
done

# ------------------------------------------------- case 1: Maxwell, tensor path ----
# The clean configurations are the cross of {order check, no check} with {fitted gene
# coefficients, none}, plus the variant that carries real SI units alongside the order.
for s in $TENSOR_SEEDS; do
  jl maxwell_tensor_shadow.jl "shadow_dhc_s$s"       --config dhc   --seed "$s" --epochs 2000 --pop 1600
  jl maxwell_tensor_shadow.jl "shadow_nodhc_s$s"     --config nodhc --seed "$s" --epochs 2000 --pop 1600
  jl maxwell_tensor_shadow.jl "shadow_ls_dhc_s$s"    --config dhc   --seed "$s" --epochs 2000 --pop 1600 --scaling true
  jl maxwell_tensor_shadow.jl "shadow_units_dhc_s$s" --config dhc   --seed "$s" --epochs 2000 --pop 1600 --scaling true --units true
done
for lvl in 005 010 020; do
  for s in $TENSOR_SEEDS; do
    jl maxwell_tensor_shadow.jl "shadow_units_noise${lvl}_s$s" --data "noise$lvl" \
        --config dhc --seed "$s" --epochs 2000 --pop 1600 --scaling true --units true
  done
done
# a fixed-budget run, so evaluation throughput is measured over the same number of
# candidates regardless of when the search happens to converge
jl maxwell_tensor_shadow.jl tensor_batched_s1 --config dhc --seed 1 --epochs 200 --pop 1600

# ---------------------------------------------------------- case 2: Reynolds ----
jl reynolds_gep.jl reynolds_gep --repeats 25 --epochs 300 --pop 800

# -------------------------------------------------------------- case 3: DSMC ----
if [ -f "$HERE/data/dsmc_compressible_s1.csv" ]; then
  jl dsmc_gep.jl dsmc_gep --cases incompressible,compressible --seeds 1,2,3 \
      --epochs 300 --pop 1200
else
  echo "DSMC CSVs missing -- run prepare_dsmc_data.py against a clone of the SITE repo"
fi

# ------------------------------------------------- SITE reference (python) ----
site tlr_and_rnc tlr_and_rnc
site only_tlr    only_tlr
site only_rnc    only_rnc

# ------------------------------------------------------ runtime decomposition ----
if [ ! -f "$RES/precompile.json" ]; then
  echo "--- precompile / JIT breakdown  ($(date +%H:%M:%S))"
  "$JULIA" --project="$ROOT" --threads="$THREADS" "$HERE/measure_precompile.jl" \
      --deps "${DEPS_PRECOMPILE_S:-205}" 2>&1 | tee "$LOGS/precompile.log"
fi

echo "ALL DONE $(date); results in $RES"
