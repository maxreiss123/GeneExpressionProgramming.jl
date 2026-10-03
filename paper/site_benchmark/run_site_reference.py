"""
Run the reference SITE implementation (https://github.com/PistilReaper/SITE) on the
same machine as the GeneExpressionProgramming.jl benchmark and record its timing.

The authors' scripts run from a copy, edited only where `run_case` says: `s_evol` is set
to the evolution seed (0 by default), `n_gen` and `NOISE` when requested, and the
`without_LINEAR` flag in SITE.py so that the RNC-only case runs without TLR. The wrapper

  * time-stamps every generation line the script prints (geppy's logbook, one line per
    generation), which yields loss against wall-clock without touching their code, and
  * stores the result as JSON next to the Julia results.

The paper reports its timings on a 13th Gen Intel Core i9-13900K. Numbers taken here are
measured on the same container as the Julia runs, so the two frameworks are directly
comparable, while the ratio to the paper's numbers shows the hardware offset.

Usage:
    python run_site_reference.py <site_repo_dir> [--cases tlr_and_rnc,only_tlr,only_rnc]
                                 [--outdir results] [--generations N]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

CASES = {
    "tlr_and_rnc": "Maxwell_tlr_and_rnc_test.py",
    "only_tlr": "Maxwell_only_tlr_test.py",
    "only_rnc": "Maxwell_only_rnc_test.py",
    "noise": "Maxwell_noise_test.py",
}

GEN_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$")


def run_case(case, src_dir, workroot, generations=None, noise=None, evol_seed=0,
             python=sys.executable):
    script = CASES[case]
    tag = case if noise is None else "noise%03d" % int(round(noise * 100))
    if evol_seed:
        tag += "_evol%d" % evol_seed
    work = os.path.join(workroot, "site_" + tag)
    os.makedirs(work, exist_ok=True)
    for fn in os.listdir(src_dir):
        if fn.endswith(".py"):
            shutil.copy(os.path.join(src_dir, fn), work)

    target = os.path.join(work, script)
    src = open(target).read()
    if generations is not None:
        src = re.sub(r"^n_gen = \d+", "n_gen = %d" % generations, src, flags=re.M)
    if noise is not None:
        src = re.sub(r"^NOISE = [0-9.]+", "NOISE = %s" % noise, src, flags=re.M)
    # the authors' header notes s_evol in {0,1,2,3,4}; sweeping it gives the
    # seed-to-seed reliability that the Julia side reports over its own seeds
    src = re.sub(r"^s_evol = \d+", "s_evol = %d" % evol_seed, src, flags=re.M)
    open(target, "w").write(src)

    # `Maxwell_only_rnc_test.py` is identical to the TLR+RNC script apart from the
    # output name; the tensor linear regression is switched off by a module-level flag
    # in SITE.py, which the release leaves at "TLR on". Set it here, so that this run
    # is the "RNC only" row of Table 1 and every other run keeps TLR.
    site_py = os.path.join(work, "SITE.py")
    flag = "True" if case == "only_rnc" else "False"
    src = open(site_py).read()
    src = re.sub(r"^without_LINEAR = (True|False)", "without_LINEAR = %s" % flag,
                 src, flags=re.M)
    open(site_py, "w").write(src)

    gens, losses, stamps = [], [], []
    t0 = time.time()
    proc = subprocess.Popen([python, "-u", script], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        m = GEN_RE.match(line)
        if m:
            gens.append(int(m.group(1)))
            losses.append(float(m.group(4)))     # column "min" = best fitness
            stamps.append(time.time() - t0)
    proc.wait()
    wall = time.time() - t0

    # the last equation the run wrote out
    eq_path = None
    outdir = os.path.join(work, "output")
    if os.path.isdir(outdir):
        cands = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".dat")]
        if cands:
            eq_path = max(cands, key=os.path.getmtime)
    equation = ""
    if eq_path:
        blocks = open(eq_path).read().strip().split("\n\n")
        equation = blocks[-1].strip() if blocks else ""

    best = min(losses) if losses else float("nan")
    converged = [g for g, l in zip(gens, losses) if l < 1e-6]
    res = dict(framework="SITE (reference implementation)", case="maxwell", variant=case,
               noise=noise, evolution_seed=evol_seed, wall_time_s=wall, generations_run=len(gens),
               converged_generation=(converged[0] if converged else -1),
               converged=bool(converged), best_loss=best,
               epoch_loss=losses, epoch_time_s=stamps, equation=equation,
               stdout_tail="\n".join(tail[-15:]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site_repo")
    ap.add_argument("--cases", default="tlr_and_rnc,only_tlr,only_rnc")
    ap.add_argument("--noise", default="")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--workroot", default="/tmp/site_runs")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--evol-seeds", default="0",
                    help="evolution seeds (s_evol) to sweep for the clean cases")
    args = ap.parse_args()

    src = os.path.join(args.site_repo, "1 Validation on benchmark problems", "1_Maxwell_Stress")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.workroot, exist_ok=True)

    jobs = [(c, None) for c in args.cases.split(",") if c]
    jobs += [("noise", float(n)) for n in args.noise.split(",") if n]

    seeds = [int(s) for s in args.evol_seeds.split(",") if s != ""]
    jobs = [(c, n, s) for (c, n) in jobs
            for s in (seeds if (n is None and c == "tlr_and_rnc") else [0])]

    for case, noise, seed in jobs:
        label = case if noise is None else "noise%03d" % int(round(noise * 100))
        if seed:
            label += "_evol%d" % seed
        print("=== SITE %s ===" % label, flush=True)
        res = run_case(case, src, args.workroot, generations=args.generations, noise=noise,
                       evol_seed=seed)
        path = os.path.join(args.outdir, "site_%s.json" % label)
        with open(path, "w") as f:
            json.dump(res, f)
        print("  wall %.1fs  gens %d  best loss %.3e  -> %s"
              % (res["wall_time_s"], res["generations_run"], res["best_loss"], path), flush=True)


if __name__ == "__main__":
    main()
