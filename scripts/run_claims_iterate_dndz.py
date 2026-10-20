"""lit-claims-audit: iterate a dN/dz-matched variant's per-class tau factor until the refitted
catalog's line count in the matching window is within `tol` of the target (CLAIM_TEST_PLAN T1).

PYTHONPATH=. uv run python scripts/run_claims_iterate_dndz.py --key dndz13 --cls 4
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

P = Path("results/lit_claims_audit/matching_factors.json")
ART = Path("results/lit_claims_audit/artifacts")


def count(key: str, cls: int, lo: float, hi: float) -> int:
    N = np.load(ART / f"catalog_{key}_full" / f"class{cls}.npz")["logN"]
    return int(((N >= lo) & (N < hi)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--cls", type=int, required=True)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=9)
    args = ap.parse_args()
    hist = []                                               # (log10 A, rel_dev)
    for it in range(args.max_iter + 1):
        F = json.loads(P.read_text())
        d = F[args.key]
        lo, hi = d["window"]
        A = d["A"][args.cls - 1]
        dev = count(args.key, args.cls, lo, hi) / d["target"] - 1.0
        hist.append((np.log10(A), dev))
        print(f"[iterate] {args.key} class {args.cls} iter {it}: A={A:.4f} rel_dev={dev:+.4f}", flush=True)
        if abs(dev) <= args.tol or it == args.max_iter:
            break
        if len(hist) == 1:                                  # first step: analytic shift on the refit catalog
            N = np.sort(np.load(ART / f"catalog_{args.key}_full" / f"class{args.cls}.npz")["logN"])
            S = np.arange(-0.3, 0.3, 0.0005)
            n = np.searchsorted(N, hi - S) - np.searchsorted(N, lo - S)
            la = np.log10(A) + S[int(np.argmin(np.abs(n - d["target"])))]
        else:                                               # secant on (log A, deviation)
            (l0, e0), (l1, e1) = hist[-2], hist[-1]
            la = l1 - e1 * (l1 - l0) / (e1 - e0) if e1 != e0 else l1
        d.setdefault("iterations", []).append({"class": args.cls, "A_before": A, "rel_dev": dev, "A_after": 10 ** la})
        d["A"][args.cls - 1] = float(10 ** la)
        P.write_text(json.dumps(F, indent=2))
        src = ART / f"catalog_{args.key}_full" / f"class{args.cls}.npz"
        shutil.move(src, src.with_name(f"class{args.cls}_iter{it}.npz"))
        subprocess.run([sys.executable, "scripts/run_claims_catalogs.py", "--name", args.key, "--factors-key",
                        args.key, "--classes", str(args.cls), "--n-jobs", str(args.n_jobs)], check=True)


if __name__ == "__main__":
    main()
