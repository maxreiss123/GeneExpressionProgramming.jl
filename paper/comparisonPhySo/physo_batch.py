"""All PhySO runs of the comparison, `--workers` single-threaded processes at a time.

    python physo_batch.py [--noise 0] [--workers 3] [--epochs 30] [--attempts 3]

Runs physo_run.py for every equation and seed whose result file is missing, so an
interrupted batch resumes; each run's output goes to
results/logs/physo<tag>_<EQ>_s<SEED>.out (tag as common.tag).
A PhySO process grows to 3.5 GB and more over 30 epochs (10 000 candidates each), so four
at once overrun 16 GB and the kernel kills one, which leaves no result; such a run is
started again, up to `--attempts` times. Its wall time is that of the attempt that
finished, as physo_run.py measures it.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import common


def result(eq, seed, noise):
    return os.path.join(common.RESULTS_DIR, "physo" + common.tag(noise),
                        "%s_s%d.json" % (eq, seed))


def run(job, epochs, attempts, noise):
    eq, seed = job
    log = os.path.join(common.RESULTS_DIR, "logs",
                       "physo%s_%s_s%d.out" % (common.tag(noise), eq, seed))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for attempt in range(1, attempts + 1):
        if os.path.exists(result(eq, seed, noise)):
            return
        with open(log, "a") as f:
            f.write("--- attempt %d\n" % attempt)
            f.flush()
            p = subprocess.run([sys.executable, "physo_run.py", eq, str(seed), "--threads", "1",
                                "--epochs", str(epochs), "--noise", str(noise)],
                               cwd=common.HERE, stdout=f,
                               stderr=subprocess.STDOUT, env=env)
            f.write("--- exit code %d\n" % p.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(common.RESULTS_DIR, "logs"), exist_ok=True)
    jobs = [(e, s) for e, _ in common.EQUATIONS for s in common.SEEDS
            if not os.path.exists(result(e, s, a.noise))]
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda j: run(j, a.epochs, a.attempts, a.noise), jobs))
    missing = [j for j in jobs if not os.path.exists(result(*j, a.noise))]
    print("missing after %d attempts: %s" % (a.attempts, missing or "none"))


if __name__ == "__main__":
    main()
