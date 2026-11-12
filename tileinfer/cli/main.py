"""``tileinfer`` console script: ``doctor`` plus the plan subcommands.

Kept tiny and import-light so ``tileinfer --help`` works in a bare checkout.  Each command
imports its own module, which is what keeps ``doctor`` from pulling in the plan builder and
vice versa.
"""

from __future__ import annotations

from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else __import__("sys").argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: tileinfer <command> [options]\n\n"
            "commands:\n"
            "  doctor            check versions, plan, layer guard, bridge and graph counters\n"
            "  plan build ...    build a deployment plan from a checkpoint config\n"
            "  plan show ...     print a plan\n"
            "  plan verify ...   check a plan against this machine\n"
        )
        return 0
    if args[0] == "doctor":
        from .doctor import main as doctor_main

        return doctor_main(args[1:])
    if args[0] == "plan":
        from .plan import main as plan_main

        return plan_main(args[1:])
    print(f"unknown command {args[0]!r}; try 'tileinfer --help'", file=__import__("sys").stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
