"""
CVSS Helpers
"""

from scoring.constants import CIA, ISS_MAX

def parse_vector(vector: str) -> dict[str, str]:
    """Parse a CVSS vector."""
    metrics = {}
    for part in vector.strip().split("/"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        if key != "CVSS":
            metrics[key] = value
    return metrics


def impact_sub_score(metrics: dict[str, str]) -> float:
    return 1.0 - (1.0 - CIA[metrics["C"]]) * (1.0 - CIA[metrics["I"]]) * (1.0 - CIA[metrics["A"]])


def impact_subscore(vector: str, ndigits: int | None = 1) -> float:
    """CVSS v3.1 impact subscore."""
    m = parse_vector(vector)
    iss = impact_sub_score(m)
    if m.get("S", "U") == "U":
        impact = 6.42 * iss
    else:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    impact = max(impact, 0.0)
    return round(impact, ndigits) if ndigits is not None else impact


def violation_severity(vector: str) -> float:
    return impact_sub_score(parse_vector(vector)) / ISS_MAX
