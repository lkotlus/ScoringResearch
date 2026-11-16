"""
Generate synthetic runs for all 32 agent archetypes.

Usage:
    python3 main.py --generate -o runs.json
    python3 main.py --generate -o runs.json --seed 1593668362 --runs-per-archetype 10
    python3 main.py --generate -o runs.json --summary
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from scoring.constants import COMPONENTS, DEFAULT_FINDINGS, DEFAULT_ROE
from scoring.generation import ARCHETYPES, GenerationSpec, generate_runs
from scoring.models.testbed import Params, Testbed
from scoring.scoring import score_run

DEFAULT_SEED = 1593668362


def summarize(runs: list[dict], tb: Testbed, params: Params) -> str:
    """Realized component means by group, and overlap between the high and low groups."""
    scored = [(r["archetype"], score_run(r, tb, params)["components"]) for r in runs]
    lines = [f"{len(runs)} runs, {len(ARCHETYPES)} archetypes",
             "",
             f"  {'component':<10}{'low mean':>10}{'high mean':>11}{'P(low > high)':>16}{'P(tie)':>9}"]
    for j, c in enumerate(COMPONENTS):
        low = [comp[c] for code, comp in scored if code[j] == "L"]
        high = [comp[c] for code, comp in scored if code[j] == "H"]
        greater = sum(a > b for a in low for b in high)
        ties = sum(a == b for a in low for b in high)
        pairs = len(low) * len(high)
        lines.append(f"  {c:<10}{sum(low) / len(low):>10.3f}{sum(high) / len(high):>11.3f}"
                     f"{greater / pairs:>16.3f}{ties / pairs:>9.3f}")
    capped = sum(r["T"] >= params.t_max for r in runs)
    slow = sum(r["archetype"][COMPONENTS.index("T")] == "L" for r in runs)
    lines += ["", f"  runs reaching T_max: {capped} ({capped / slow:.0%} of low-time runs)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.strip().split("\n\n")[0]) # type: ignore
    ap.add_argument("-o", "--output", type=Path, required=True, help="write runs JSON here")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--runs-per-archetype", type=int, default=10)
    ap.add_argument("--t-max", type=float, default=60.0)
    ap.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    ap.add_argument("--roe", type=Path, default=DEFAULT_ROE)
    ap.add_argument("--summary", action="store_true",
                    help="score the generated runs and report realized means and overlap")
    args = ap.parse_args(argv)

    tb = Testbed(args.findings, args.roe)
    spec = GenerationSpec(runs_per_archetype=args.runs_per_archetype, t_max=args.t_max)
    runs = generate_runs(tb, spec, args.seed)

    meta = {"seed": args.seed, "findings": str(args.findings), "roe": str(args.roe),
            "spec": asdict(spec)}
    args.output.write_text(json.dumps({"meta": meta, "runs": runs}, indent=2) + "\n")
    print(f"wrote {len(runs)} runs to {args.output}", file=sys.stderr)

    if args.summary:
        print(summarize(runs, tb, Params(t_max=args.t_max)))
    return 0
