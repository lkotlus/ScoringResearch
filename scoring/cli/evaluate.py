"""
Evaluate all three composite indicators on a generated dataset.

Usage:
    python3 main.py --evaluate runs.json
    python3 main.py --evaluate runs.json --draws 500 --latex
    python3 main.py --evaluate runs.json --json evaluation.json
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from scoring.cli.generate import DEFAULT_SEED
from scoring.constants import DEFAULT_FINDINGS, DEFAULT_PREFERENCES, DEFAULT_ROE
from scoring.evaluation import (MODELS, DominanceResult, PerturbationSpec, StabilityResult,
                                dominance, load_preferences, percentile,
                                preference_agreement, stability)
from scoring.models.testbed import Params, Testbed
from scoring.scoring import score_runs

GROUPS = [("weights", ["weights"]), ("penalties", ["penalties"]),
          ("half-points", ["half-points"]), ("auc", ["auc"]),
          ("all", ["weights", "penalties", "half-points", "auc"])]
NAMES = {"sum": "Sum", "geo": "Geo", "auc": "AUC"}


def report_dominance(d: DominanceResult) -> str:
    lines = [f"Dominance: {d.n_pairs} dominant pairs among {d.n_runs} runs "
             f"({d.n_archetype_level} also dominant at the archetype level)",
             f"  {'model':<8}{'ordered':>10}{'tied':>8}{'inverted':>10}"]
    for m in MODELS:
        n = d.n_pairs or 1
        lines.append(f"  {m:<8}{d.ordered[m] / n:>10.1%}{d.tied[m] / n:>8.1%}"
                     f"{d.inverted[m] / n:>10.1%}")
    return "\n".join(lines)


def report_preferences(agreement: dict[str, dict[str, float]]) -> str:
    width = max(len(name) for name in agreement)
    lines = ["Preference agreement: P(preferred archetype's run scores higher)",
             f"  {'preference':<{width}}" + "".join(f"{m:>8}" for m in MODELS)]
    for name, by_model in agreement.items():
        lines.append(f"  {name:<{width}}" + "".join(f"{by_model[m]:>8.3f}" for m in MODELS))
    return "\n".join(lines)


def report_stability(results: dict[str, StabilityResult], draws: int) -> str:
    lines = [f"Rank stability: Kendall's tau-b vs. baseline over {draws} draws (mean / 5th pct)",
             f"  {'perturbed':<13}" + "".join(f"{m:>16}" for m in MODELS)]
    for group, r in results.items():
        cells = "".join(f"{sum(r.taus[m]) / len(r.taus[m]):>9.3f} / {percentile(r.taus[m], 0.05):.3f}"
                        for m in MODELS)
        lines.append(f"  {group:<13}{cells}")
    lines += ["", "Edge-case robustness: share of draws passing every edge case",
              f"  {'perturbed':<13}" + "".join(f"{m:>8}" for m in MODELS)]
    for group, r in results.items():
        lines.append(f"  {group:<13}" + "".join(f"{r.edge_pass[m]:>8.0%}" for m in MODELS))
    return "\n".join(lines)


def latex_tables(d: DominanceResult, agreement: dict[str, dict[str, float]],
                 stab: dict[str, StabilityResult]) -> str:
    head = " & ".join(rf"\textbf{{{NAMES[m]}}}" for m in MODELS)
    n = d.n_pairs or 1
    out = [r"% Dominance consistency",
           r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
           r" & \textbf{Ordered} & \textbf{Tied} & \textbf{Inverted} \\", r"\midrule"]
    out += [rf"{NAMES[m]} & {100 * d.ordered[m] / n:.1f}\% & {100 * d.tied[m] / n:.1f}\% & "
            rf"{100 * d.inverted[m] / n:.1f}\% \\" for m in MODELS]
    out += [r"\bottomrule", r"\end{tabular}", "",
            r"% Preference agreement",
            r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", rf"\textbf{{Preference}} & {head} \\", r"\midrule"]
    out += [f"{name} & " + " & ".join(f"{v[m]:.3f}" for m in MODELS) + r" \\"
            for name, v in agreement.items()]
    out += [r"\bottomrule", r"\end{tabular}"]
    if stab:
        out += ["", r"% Rank stability (mean Kendall's tau-b, 5th percentile in parentheses)",
                r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", rf"\textbf{{Perturbed}} & {head} \\", r"\midrule"]
        for group, r in stab.items():
            cells = " & ".join(f"{sum(r.taus[m]) / len(r.taus[m]):.3f} ({percentile(r.taus[m], 0.05):.3f})"
                               for m in MODELS)
            out.append(f"{group} & {cells} \\\\")
        out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.strip().split("\n\n")[0]) # type: ignore
    ap.add_argument("runs", type=Path, help="runs JSON from --generate")
    ap.add_argument("--preferences", type=Path, default=DEFAULT_PREFERENCES)
    ap.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    ap.add_argument("--roe", type=Path, default=DEFAULT_ROE)
    ap.add_argument("--t-max", type=float, default=60.0)
    ap.add_argument("--draws", type=int, default=200, help="perturbation draws per group")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for perturbation draws")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--json", type=Path, help="write all results here")
    ap.add_argument("--latex", action="store_true", help="print results as LaTeX tables")
    args = ap.parse_args(argv)

    data = json.loads(args.runs.read_text())
    runs = data["runs"] if isinstance(data, dict) else data
    tb = Testbed(args.findings, args.roe)
    base = Params(t_max=args.t_max)
    for warning in base.validate():
        print(f"warning: {warning}", file=sys.stderr)

    baseline = score_runs(runs, tb, base)
    dom = dominance(baseline)
    prefs = load_preferences(args.preferences)
    agreement = {p.name: preference_agreement(baseline, p) for p in prefs}

    print(report_dominance(dom))
    print()
    print(report_preferences(agreement))

    stab: dict[str, StabilityResult] = {}
    if not args.skip_stability:
        rng = random.Random(args.seed)
        spec = PerturbationSpec()
        for label, groups in GROUPS:
            stab[label] = stability(runs, tb, base, groups, spec, args.draws, rng, baseline)
        print()
        print(report_stability(stab, args.draws))

    if args.latex:
        print("\n" + latex_tables(dom, agreement, stab))
    if args.json:
        args.json.write_text(json.dumps({
            "dominance": asdict(dom),
            "preferences": agreement,
            "stability": {g: asdict(r) for g, r in stab.items()},
            "perturbation": asdict(PerturbationSpec()),
            "draws": args.draws,
            "seed": args.seed,
        }, indent=2) + "\n")
    return 0
