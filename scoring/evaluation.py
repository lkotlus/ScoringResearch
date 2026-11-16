"""
Evaluation of the composite indicators against generated runs

    1. Dominance: dominant profiles should always score higher
    2. Preferences: archetype comparisons stated in advance; agreement is the
       probability a run from the preferred archetype outscores a run from the
       other (ties count one half).
    2. Preferences: find the probability of archetypal comparison leading to
       scores consistent with the stated archetypal preferences.
    3. Stability: Kendall's tau between baseline rankings and rankings under
       perturbed parameters, and how often every edge case still passes.
"""

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from scipy.stats import kendalltau

from scoring.cli.edge_cases import evaluate as evaluate_edge_cases
from scoring.constants import COMPONENTS
from scoring.generation import ARCHETYPES
from scoring.models.testbed import Params, Testbed
from scoring.scoring import score_runs

MODELS = ("sum", "geo", "auc")


#
# Dominance
#


def dominates(a: list[float], b: list[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def archetype_dominates(a: str, b: str) -> bool:
    return a != b and all(x == "H" or y == "L" for x, y in zip(a, b))


@dataclass
class DominanceResult:
    n_runs: int
    n_pairs: int                   # dominant run pairs found
    n_archetype_level: int         # of those, pairs whose archetypes also dominate
    ordered: dict[str, int]        # model ranks the dominant run strictly higher
    tied: dict[str, int]
    inverted: dict[str, int]


def dominance(scored: list[dict]) -> DominanceResult:
    vecs = [[s["components"][c] for c in COMPONENTS] for s in scored]
    pairs = [(i, j) for i in range(len(vecs)) for j in range(len(vecs))
             if i != j and dominates(vecs[i], vecs[j])]
    ordered = {m: 0 for m in MODELS}
    tied = {m: 0 for m in MODELS}
    inverted = {m: 0 for m in MODELS}
    for i, j in pairs:
        for m in MODELS:
            a, b = scored[i]["scores"][m], scored[j]["scores"][m]
            if a > b:
                ordered[m] += 1
            elif a == b:
                tied[m] += 1
            else:
                inverted[m] += 1
    archetype_level = sum(archetype_dominates(scored[i]["archetype"], scored[j]["archetype"])
                          for i, j in pairs)
    return DominanceResult(len(vecs), len(pairs), archetype_level, ordered, tied, inverted)


#
# Preferences
#


@dataclass
class Preference:
    """Pattern letters, in COMPONENTS order: H or L fixed, '=' same in both, '?' any."""
    name: str
    better: str
    worse: str

    def validate(self) -> None:
        for pattern in (self.better, self.worse):
            if len(pattern) != len(COMPONENTS) or set(pattern) - set("HL=?"):
                raise ValueError(f"preference {self.name!r}: bad pattern {pattern!r}")
        for x, y in zip(self.better, self.worse):
            if (x == "=") != (y == "="):
                raise ValueError(f"preference {self.name!r}: '=' must be in the same positions")

    def archetype_pairs(self) -> list[tuple[str, str]]:
        def matches(code: str, pattern: str) -> bool:
            return all(p in "=?" or p == c for c, p in zip(code, pattern))
        return [(x, y) for x in ARCHETYPES for y in ARCHETYPES
                if matches(x, self.better) and matches(y, self.worse)
                and all(x[k] == y[k] for k, p in enumerate(self.better) if p == "=")]


def load_preferences(path: Path) -> list[Preference]:
    prefs = [Preference(**p) for p in json.loads(path.read_text())["preferences"]]
    for p in prefs:
        p.validate()
    return prefs


def preference_agreement(scored: list[dict], pref: Preference) -> dict[str, float]:
    by_archetype: dict[str, list[dict]] = {}
    for s in scored:
        by_archetype.setdefault(s["archetype"], []).append(s["scores"])
    result = {}
    for m in MODELS:
        wins = total = 0.0
        for x, y in pref.archetype_pairs():
            for a in by_archetype.get(x, []):
                for b in by_archetype.get(y, []):
                    wins += 1.0 if a[m] > b[m] else 0.5 if a[m] == b[m] else 0.0
                    total += 1
        result[m] = wins / total if total else float("nan")
    return result


#
# Stability
#


@dataclass
class PerturbationSpec:
    """Relative perturbation half-width for each parameter group."""
    deltas: dict[str, float] = field(default_factory=lambda: {
        "weights": 0.2, "penalties": 0.2, "half-points": 0.5, "auc": 0.5})


def perturb(base: Params, groups: list[str], spec: PerturbationSpec, rng: random.Random) -> Params:
    """Multiply each parameter in the chosen groups by U(1 - d, 1 + d), keeping the
    design constraints (weights sum to 1, omega_und >= omega_cov, lambda_V >= lambda_H >= 0.5)."""
    def scale(x: float, group: str) -> float:
        d = spec.deltas[group]
        return x * rng.uniform(1.0 - d, 1.0 + d)

    p = replace(base, weights=dict(base.weights))
    if "weights" in groups:
        w = {c: scale(base.weights[c], "weights") for c in COMPONENTS}
        total = sum(w.values())
        w = {c: v / total for c, v in w.items()}
        if w["und"] < w["cov"]:
            w["und"], w["cov"] = w["cov"], w["und"]
        p.weights = w
    if "penalties" in groups:
        for _ in range(1000):
            r, h, v = (scale(x, "penalties") for x in (base.lambda_r, base.lambda_h, base.lambda_v))
            if v >= h >= 0.5:
                p.lambda_r, p.lambda_h, p.lambda_v = r, h, v
                break
        else:
            raise RuntimeError("could not draw penalty coefficients satisfying the ordering")
    if "half-points" in groups:
        p.pi_half_frac = scale(base.pi_half_frac, "half-points")
        p.t_half_frac = scale(base.t_half_frac, "half-points")
        p.r_half = scale(base.r_half, "half-points")
    if "auc" in groups:
        p.beta = scale(base.beta, "auc")
        p.gamma = scale(base.gamma, "auc")
    return p


@dataclass
class StabilityResult:
    taus: dict[str, list[float]]            # model -> tau per draw
    edge_pass: dict[str, float]             # model -> share of draws passing every edge case


def tau_b(x: list[float], y: list[float]) -> float:
    tau = kendalltau(x, y).statistic        # type: ignore | tau-b, which accounts for ties
    return 1.0 if tau != tau else float(tau)  # NaN only when a model is constant in both


def stability(runs: list[dict], tb: Testbed, base: Params, groups: list[str],
              spec: PerturbationSpec, draws: int, rng: random.Random,
              baseline: list[dict]) -> StabilityResult:
    taus = {m: [] for m in MODELS}
    edge_passes = {m: 0 for m in MODELS}
    for _ in range(draws):
        params = perturb(base, groups, spec, rng)
        scored = score_runs(runs, tb, params)
        for m in MODELS:
            taus[m].append(tau_b([s["scores"][m] for s in baseline],
                                 [s["scores"][m] for s in scored]))
        _, results = evaluate_edge_cases(tb, params)
        for m in MODELS:
            edge_passes[m] += all(r.passed for r in results if r.model == m)
    return StabilityResult(taus, {m: edge_passes[m] / draws for m in MODELS})


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]
