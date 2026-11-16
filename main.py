"""
Entry point for everything because I'm just built like that.

The first argument selects a tool, subsequent arguments get
passed through.

Usage:
    python3 main.py --scoring runs.json -o scores.json
    python3 main.py --scoring --selftest
    python3 main.py --edge-cases --latex
    python3 main.py --TOOL --help
"""

import importlib
import sys

TOOLS = {
    "--scoring": ("scoring.cli.score", "score a file of runs with all three models"),
    "--edge-cases": ("scoring.cli.edge_cases", "check the edge-case pairs against all three models"),
}


def usage() -> str:
    width = max(len(flag) for flag in TOOLS)
    lines = ["usage: main.py --TOOL [TOOL OPTIONS]", "", "tools:"]
    lines += [f"  {flag:<{width}}  {desc}" for flag, (_, desc) in TOOLS.items()]
    lines += ["", "Run main.py --TOOL --help for a tool's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2

    tool, rest = argv[0], argv[1:]
    if tool not in TOOLS:
        print(f"main.py: unknown tool {tool!r}\n\n{usage()}", file=sys.stderr)
        return 2

    module_name, _ = TOOLS[tool]
    module = importlib.import_module(module_name)
    return module.main(rest, prog=f"main.py {tool}")


if __name__ == "__main__":
    sys.exit(main())
