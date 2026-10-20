"""lit-claims-audit: run literature-claim tests on the Voigt catalogs.

PYTHONPATH=. uv run python scripts/run_claims.py --tests t5,t4a,t6 --variants fiducial,taurescaled
"""

import argparse
import json
import time
from pathlib import Path

from src.core.claims import BlockBootstrap, load_catalogs, skewer_blocks, t4a_envelope, t5_winds_vs_agn, t6_paired
from src.core.provenance import provenance_header

ART = Path("results/eigen_voigt_embedding/artifacts")
OUT = Path("results/lit_claims_audit")
CAT_DIRS = {"fiducial": ART / "catalog_fiducial_full", "taurescaled": ART / "catalog_taurescaled_full"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="t5,t4a,t6")
    ap.add_argument("--variants", default="fiducial,taurescaled")
    ap.add_argument("--cat-root", default=str(OUT / "artifacts"), help="extra catalog dirs: catalog_<variant>_full")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = skewer_blocks()
    bb = BlockBootstrap(blocks, n_boot=500, seed=0)
    for variant in args.variants.split(","):
        cat_dir = CAT_DIRS.get(variant, Path(args.cat_root) / f"catalog_{variant}_full")
        cats = load_catalogs(cat_dir)
        for test in args.tests.split(","):
            t0 = time.time()
            if test == "t5":
                res = {"logN13": t5_winds_vs_agn(cats, bb, 13.0), "logN12": t5_winds_vs_agn(cats, bb, 12.0)}
            elif test == "t4a":
                res = t4a_envelope(cats, bb)
            elif test == "t6":
                res = {"dv10": t6_paired(cats, bb, blocks.size, logN_min=12.0, dv_max=10.0),
                       "dv5": t6_paired(cats, bb, blocks.size, logN_min=12.0, dv_max=5.0)}
            else:
                raise ValueError(test)
            res["provenance"] = provenance_header({"variant": variant, "test": test})
            (OUT / f"{test}_{variant}.json").write_text(json.dumps(res, indent=2, default=float))
            print(f"[claims] {test} {variant} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
