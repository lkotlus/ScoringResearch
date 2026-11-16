"""
Constants.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FINDINGS = HERE / "config" / "tb0_findings.json"
DEFAULT_ROE = HERE / "config" / "tb0_roe.json"

COMPONENTS = ("cov", "und", "pi", "T", "C")
CIA = {"H": 0.56, "L": 0.22, "N": 0.0}
ISS_MAX = 1.0 - (1.0 - CIA["H"]) ** 3
