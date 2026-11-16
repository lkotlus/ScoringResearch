"""
CLI to actually use the damn tools.
"""

import argparse
import json
import sys
from pathlib import Path

from scoring.models.testbed import Testbed, Params
from scoring.constants import COMPONENTS, DEFAULT_FINDINGS, DEFAULT_ROE
from scoring.scoring import selftest, score_runs

def _load_runs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["runs"] if isinstance(data, dict) else data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0]) # type: ignore
    ap.add_argument("runs", nargs="?", type=Path, help="runs JSON file")
    ap.add_argument("-o", "--output", type=Path, help="write scores JSON here (default: stdout)")
    ap.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    ap.add_argument("--roe", type=Path, default=DEFAULT_ROE)
    ap.add_argument("--prices", type=Path, help="frozen price table JSON (USD per 1M tokens)")
    ap.add_argument("--t-max", type=float, default=60.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--weights", type=float, nargs=5, metavar=COMPONENTS,
                    help="omega in the order cov und pi T C (must sum to 1)")
    ap.add_argument("--selftest", action="store_true", help="run built-in checks and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0
    if args.runs is None:
        ap.error("a runs file is required unless --selftest is given")

    params = Params(t_max=args.t_max, beta=args.beta, gamma=args.gamma)
    if args.weights:
        params.weights = dict(zip(COMPONENTS, args.weights))
    for warning in params.validate():
        print(f"warning: {warning}", file=sys.stderr)

    tb = Testbed(args.findings, args.roe)
    prices = json.loads(args.prices.read_text()) if args.prices else None
    results = score_runs(_load_runs(args.runs), tb, params, prices)

    out = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(out + "\n")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
