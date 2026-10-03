"""
Peak memory of the Maxwell identification, SITE against GeneExpressionProgramming.jl, and
the allocation of the scalar path as the data lengthens; writes results/memory.json.

Peak resident set size (VmHWM) is read from /proc every 0.1 s, from outside the process,
for the process and all its children, since the allocation counters of a Python and a
Julia process are not comparable. The runs, each 200 generations at population 1600 on
the 150-sample data unless the search reaches its tolerance first:

    SITE                 run_site_reference.py, TLR only (it does not converge early)
    GEP.jl scalar        maxwell_gep.jl --config nodhc
    GEP.jl tensor        maxwell_tensor_shadow.jl --config dhc

plus the idle footprint of each runtime with its packages loaded (the baseline), and
measure_allocation.jl for the allocation against the data size. The DynamicExpressions
figures of the earlier measurement are carried over, marked as such: that evaluator no
longer exists.

    python measure_memory.py <site_repo> [--julia julia] [--threads 4]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def vmhwm_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def descendants(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        try:
            with open(f"/proc/{p}/task/{p}/children") as f:
                kids = [int(k) for k in f.read().split()]
        except (FileNotFoundError, ProcessLookupError):
            kids = []
        out += kids
        todo += kids
    return out


def peak_rss_mb(cmd, cwd=ROOT):
    """Largest VmHWM of any process in the tree of `cmd`, in MB, and the exit code."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    peak = {}
    while proc.poll() is None:
        for p in [proc.pid] + descendants(proc.pid):
            peak[p] = max(peak.get(p, 0), vmhwm_kb(p))
        time.sleep(0.1)
    return round(max(peak.values(), default=0) / 1024), proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site_repo")
    ap.add_argument("--julia", default="julia")
    ap.add_argument("--threads", default="4")
    a = ap.parse_args()

    jl = [a.julia, f"--project={ROOT}", f"--threads={a.threads}"]
    tmp = tempfile.mkdtemp(prefix="gep_memory_")
    runs = {
        "SITE (geppy + deap)": [sys.executable, os.path.join(HERE, "run_site_reference.py"),
                                a.site_repo, "--cases", "only_tlr", "--generations", "200",
                                "--outdir", tmp, "--workroot", tmp],
        "GEP.jl scalar, batched buffers": jl + [os.path.join(HERE, "maxwell_gep.jl"),
                                                "--config", "nodhc", "--seed", "1",
                                                "--epochs", "200", "--pop", "1600",
                                                "--out", os.path.join(tmp, "scalar.json")],
        "GEP.jl tensor, batched buffers": jl + [os.path.join(HERE, "maxwell_tensor_shadow.jl"),
                                                "--config", "dhc", "--seed", "1",
                                                "--epochs", "200", "--pop", "1600",
                                                "--out", os.path.join(tmp, "tensor.json")],
    }
    baselines = {
        "python_with_imports": [sys.executable, "-c",
                                "import numpy, geppy, deap, time; time.sleep(2)"],
        "julia_with_package": jl + ["-e", "using GeneExpressionProgramming; sleep(2)"],
    }

    # load the package once, unmeasured: a stale precompile cache is rebuilt in a child
    # process, which the poller would otherwise count against the first measured run
    subprocess.run(jl + ["-e", "using GeneExpressionProgramming"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    base = {}
    for name, cmd in baselines.items():
        base[name], rc = peak_rss_mb(cmd)
        print(f"baseline {name}: {base[name]} MB (exit {rc})", flush=True)
    peak = {}
    for name, cmd in runs.items():
        peak[name], rc = peak_rss_mb(cmd)
        print(f"{name}: {peak[name]} MB (exit {rc})", flush=True)

    alloc_path = os.path.join(tmp, "allocation.json")
    subprocess.run(jl + [os.path.join(HERE, "measure_allocation.jl"), "--out", alloc_path],
                   cwd=ROOT, check=True)
    alloc = json.load(open(alloc_path))

    out = os.path.join(HERE, "results", "memory.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old_alloc = old.get("allocation_per_fit_mb", {})
    res = dict(
        note="Peak resident set size (VmHWM), polled from outside the process, on the "
             "150-sample Maxwell identification at 200 generations x 1600 (measure_memory.py). "
             "Both runtimes' figures include their own footprint, reported separately as "
             "the baseline.",
        julia_version=alloc.get("julia_version"), threads=int(a.threads),
        baseline_mb=base, peak_rss_mb=peak,
        allocation_per_fit_mb=dict(
            note="Bytes allocated by fit! over 25 generations at population 800, three "
                 "features, as the data lengthens (measure_allocation.jl), within Julia. "
                 "dynamic_expressions is the earlier measurement, taken before that "
                 "evaluator was removed; it cannot be rerun.",
            samples=alloc["samples"],
            batched_buffers=alloc["batched_buffers"],
            seconds_batched_buffers=alloc["seconds_batched_buffers"],
            dynamic_expressions=old_alloc.get("dynamic_expressions"),
            seconds_dynamic_expressions=old_alloc.get("seconds_dynamic_expressions")))
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
