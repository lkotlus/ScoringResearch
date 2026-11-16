"""
Compare two runs' performance profiles: both runs' composite scores beside a
spider chart with run A in blue and run B in red.

Usage:
    python3 main.py --profile runs.json                      # two random runs
    python3 main.py --profile runs.json --index 12 207
    python3 main.py --profile runs.json --id HLHHL-03 LHHLH-07 -o compare.pdf
    python3 main.py --profile runs.json --show
"""

import argparse
import json
import random
from pathlib import Path
from scoring import plotting
import matplotlib.pyplot as plt

from scoring.constants import COMPONENTS, DEFAULT_FINDINGS, DEFAULT_ROE
from scoring.models.testbed import Params, Testbed
from scoring.scoring import score_run


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.strip().split("\n\n")[0]) # type: ignore
    ap.add_argument("runs", type=Path, help="runs JSON file")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--index", type=int, nargs=2, metavar=("A", "B"),
                      help="run indexes in the file (default: two random runs)")
    pick.add_argument("--id", nargs=2, metavar=("A", "B"), help="run ids, e.g. HLHHL-03 LHHLH-07")
    ap.add_argument("--seed", type=int, help="seed for choosing random runs")
    ap.add_argument("-o", "--output", type=Path,
                    help="figure path; the extension sets the format "
                         "(default: profile_<A>_vs_<B>.png)")
    ap.add_argument("--show", action="store_true", help="open the figure in a window instead of saving")
    ap.add_argument("--normalized", action="store_true",
                    help="plot each run's normalized profile instead of raw components")
    ap.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    ap.add_argument("--roe", type=Path, default=DEFAULT_ROE)
    ap.add_argument("--t-max", type=float, default=60.0)
    args = ap.parse_args(argv)

    data = json.loads(args.runs.read_text())
    runs = data["runs"] if isinstance(data, dict) else data
    if len(runs) < 2:
        ap.error("the runs file needs at least two runs")

    if args.id is not None:
        ids = [r.get("id") for r in runs]
        missing = [i for i in args.id if i not in ids]
        if missing:
            ap.error(f"no run with id {missing[0]!r}")
        indexes = [ids.index(i) for i in args.id]
    elif args.index is not None:
        for i in args.index:
            if not 0 <= i < len(runs):
                ap.error(f"indexes must be between 0 and {len(runs) - 1}")
        indexes = list(args.index)
    else:
        indexes = random.Random(args.seed).sample(range(len(runs)), 2)

    tb = Testbed(args.findings, args.roe)
    params = Params(t_max=args.t_max)
    results = [score_run(runs[i], tb, params) for i in indexes]
    for result, i in zip(results, indexes):
        result["id"] = result.get("id") or f"index-{i}"

    for name, result, i in zip("AB", results, indexes):
        archetype = f", archetype {result['archetype']}" if result.get("archetype") else ""
        print(f"{name}: run {result['id']} (index {i}){archetype}")
        print("   " + "  ".join(f"S_{k}={result['scores'][k]:.3f}" for k in ("sum", "geo", "auc")))
        print("   " + "  ".join(f"{c}={result['components'][c]:.3f}" for c in COMPONENTS))

    if not args.show:
        plotting.use_headless_backend()
    fig = plotting.comparison_figure(results[0], results[1], normalized=args.normalized)

    if args.show:
        plt.show()
    else:
        output = args.output or Path(f"profile_{results[0]['id']}_vs_{results[1]['id']}.png")
        plotting.save_figure(fig, output)
        print(f"wrote {output}")
    return 0
