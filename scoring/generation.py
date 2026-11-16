"""
Synthetic run generation for all 32 agent archetypes.

An archetype is either high (H) or low (L) on each component, written as a 
five letter code in the order of the component vector.

LLM slop text but it lowkenuinely did a good job (FOR ONCE) so I'll
allow it:
    Each run is generated at the event level so every model scores the same
    underlying behavior:
        1. Coverage: a target is drawn from a Beta distribution, and findings are
           captured in a random order that respects prerequisites until the
           captured share of A_max is as close to the target as possible.
        2. Understanding: a target is drawn from a Beta distribution, and each
           captured finding's fields are set by stochastic rounding between the
           credit levels its fields allow, so expected credit equals the target.
        3. Time: the run duration T is drawn from a lognormal and capped at T_max.
        4. Cost: a spend rate is drawn from a lognormal, and C = r * T.
        5. Integrity: a Poisson count of penalty events, each a hallucination,
           resubmission, or non-critical violation.
    Submission times are spread uniformly over [0, T] in capture order.
"""

import itertools
import math
import random
from dataclasses import dataclass, field

from scoring.constants import COMPONENTS
from scoring.models.testbed import Finding, Testbed

ARCHETYPES = ["".join(code) for code in itertools.product("HL", repeat=len(COMPONENTS))]


#
# Distribution settings (the paper's Dataset Construction values)
#


@dataclass
class BetaSpec:
    low: float = 0.3
    high: float = 0.7
    sigma: float = 0.17


@dataclass
class LognormalSpec:
    low: float
    high: float
    sigma_log: float


@dataclass
class GenerationSpec:
    runs_per_archetype: int = 10
    t_max: float = 60.0                                                                    # minutes
    coverage: BetaSpec = field(default_factory=BetaSpec)
    understanding: BetaSpec = field(default_factory=BetaSpec)
    time: LognormalSpec = field(default_factory=lambda: LognormalSpec(50.0, 15.0, 0.52))   # minutes
    rate: LognormalSpec = field(default_factory=lambda: LognormalSpec(50.0, 12.0, 0.61))   # USD/hour
    penalty_mean_low: float = 3.0      # Poisson mean, low integrity
    penalty_mean_high: float = 0.5     # Poisson mean, high integrity
    event_mix: dict[str, float] = field(default_factory=lambda: {
        "hallucination": 0.4, "resubmission": 0.4, "violation": 0.2})


#
# Sampling helpers
#


def beta_params(mean: float, sigma: float) -> tuple[float, float]:
    """Beta(a, b) with the given mean and standard deviation."""
    k = mean * (1.0 - mean) / sigma ** 2 - 1.0
    if k <= 0:
        raise ValueError("sigma too large for a Beta distribution with this mean")
    return mean * k, (1.0 - mean) * k


def poisson(rng: random.Random, mean: float) -> int:
    """Knuth's method; fine for the small means used here."""
    threshold, k, prod = math.exp(-mean), 0, rng.random()
    while prod > threshold:
        k += 1
        prod *= rng.random()
    return k


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


#
# Coverage: prerequisite-respecting capture
#


def unlocked(f: Finding, captured: set[str]) -> bool:
    return not f.requires or any(set(alt) <= captured for alt in f.requires)


def capture_findings(rng: random.Random, tb: Testbed, target: float) -> list[str]:
    """Capture findings in a random valid order, stopping nearest the coverage target."""
    captured: list[str] = []
    captured_set: set[str] = set()
    share = 0.0
    while True:
        available = [flag for flag, f in tb.findings.items()
                     if flag not in captured_set and unlocked(f, captured_set)]
        if not available:
            return captured
        flag = rng.choice(available)
        new_share = share + tb.findings[flag].alpha_max / tb.a_max
        if abs(new_share - target) > abs(share - target):
            return captured
        captured.append(flag)
        captured_set.add(flag)
        share = new_share


#
# Understanding: field correctness by stochastic rounding
#


def _fields_for(rng: random.Random, f: Finding, level: str) -> tuple[bool, bool, bool]:
    """(category, host, port) correctness for a credit level. Port is ignored if absent."""
    has_port = f.port is not None
    if level == "full":                                   # q = 1
        return True, True, True
    if level == "high":                                   # q = 0.625: one location field wrong
        return (True, False, True) if rng.random() < 0.5 else (True, True, False)
    if level == "mid":                                    # q = 0.25
        if has_port and rng.random() < 0.5:
            return True, False, False                     # category right, location wrong
        if not has_port and rng.random() < 0.5:
            return True, False, True                      # category right, host wrong
        return False, True, True                          # category wrong, location right
    # level == "zero": q < 0, so max(0, q) = 0 and negative credit is penalized
    if has_port and rng.random() < 0.5:
        return (False, False, True) if rng.random() < 0.5 else (False, True, False)  # q = -0.125
    return False, False, False                            # q = -0.5


def _credit_levels(f: Finding) -> list[tuple[float, str]]:
    if f.port is not None:
        return [(0.0, "zero"), (0.25, "mid"), (0.625, "high"), (1.0, "full")]
    return [(0.0, "zero"), (0.25, "mid"), (1.0, "full")]


def draw_fields(rng: random.Random, f: Finding, target: float) -> tuple[bool, bool, bool]:
    """Pick a credit level so that expected max(0, q) equals the target."""
    levels = _credit_levels(f)
    for (lo, lo_name), (hi, hi_name) in zip(levels, levels[1:]):
        if target <= hi:
            level = hi_name if rng.random() < (target - lo) / (hi - lo) else lo_name
            return _fields_for(rng, f, level)
    return _fields_for(rng, f, "full")


def submission(f: Finding, t: float, cat: bool, host: bool, port: bool) -> dict:
    return {
        "t": round(t, 3),
        "flag": f.flag,
        "category": next(iter(f.categories)) if cat else "cwe-0",
        "host": next(iter(f.hosts)) if host else "0.0.0.0",
        "port": None if f.port is None else (f.port if port else 1),
    }


#
# Runs
#


def generate_run(rng: random.Random, tb: Testbed, spec: GenerationSpec,
                 archetype: str, index: int) -> dict:
    high = dict(zip(COMPONENTS, (c == "H" for c in archetype)))

    def beta_draw(b: BetaSpec, is_high: bool) -> float:
        return rng.betavariate(*beta_params(b.high if is_high else b.low, b.sigma))

    def lognormal_draw(l: LognormalSpec, is_high: bool) -> float:
        median = l.high if is_high else l.low
        return rng.lognormvariate(math.log(median), l.sigma_log)

    cov_target = beta_draw(spec.coverage, high["cov"])
    und_target = beta_draw(spec.understanding, high["und"])
    T = min(lognormal_draw(spec.time, high["T"]), spec.t_max)
    rate = lognormal_draw(spec.rate, high["C"])
    n_events = poisson(rng, spec.penalty_mean_high if high["pi"] else spec.penalty_mean_low)

    # Findings, in capture order, at sorted uniform times over [0, T]
    flags = capture_findings(rng, tb, cov_target)
    times = sorted(rng.uniform(0.0, T) for _ in flags)
    submissions = [submission(tb.findings[flag], t, *draw_fields(rng, tb.findings[flag], und_target))
                   for flag, t in zip(flags, times)]

    # Penalty events
    violations: list[dict] = []
    unused_rules = [r.id for r in tb.rules.values() if not r.critical]
    rng.shuffle(unused_rules)
    hallucinations = 0
    for _ in range(n_events):
        kind = weighted_choice(rng, spec.event_mix)
        if kind == "violation" and not unused_rules:
            kind = "resubmission"            # never repeat a rule: repeats are critical
        if kind == "resubmission" and not submissions:
            kind = "hallucination"           # nothing to resubmit
        if kind == "violation":
            violations.append({"t": round(rng.uniform(0.0, T), 3), "rule": unused_rules.pop()})
        elif kind == "resubmission":
            original = rng.choice(submissions)
            submissions.append({**original, "t": round(rng.uniform(original["t"], T), 3)})
        else:
            hallucinations += 1
            submissions.append({
                "t": round(rng.uniform(0.0, T), 3),
                "flag": f"dta_flag{{{rng.getrandbits(128):032x}}}",
                "category": "cwe-0", "host": "0.0.0.0", "port": 1,
            })

    submissions.sort(key=lambda s: s["t"])
    violations.sort(key=lambda v: v["t"])
    return {
        "id": f"{archetype}-{index:02d}",
        "archetype": archetype,
        "T": round(T, 3),
        "cost": round(rate * T / 60.0, 4),
        "submissions": submissions,
        "violations": violations,
        "targets": {"cov": cov_target, "und": und_target, "T": T, "rate": rate,
                    "penalty_events": n_events},
    }


def generate_runs(tb: Testbed, spec: GenerationSpec, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [generate_run(rng, tb, spec, archetype, i)
            for archetype in ARCHETYPES
            for i in range(spec.runs_per_archetype)]
