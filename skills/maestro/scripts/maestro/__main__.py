"""CLI entry: `python3 <skill-dir>/scripts/maestro <cmd>` (a directory run) and `python3 -m maestro <cmd>`."""
from __future__ import annotations

import os
import sys

if not __package__:
    # Run as a directory: sys.path[0] is the package dir itself, so its modules would shadow others.
    # Put the parent (scripts/) first instead and import as the package.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.path.dirname(here)] + [p for p in sys.path if os.path.abspath(p or ".") != here]

from maestro.router import main  # noqa: E402



def _main():
    try:
        code = main()
        sys.stdout.flush()
        return code
    except BrokenPipeError:   # `maestro-router models --json | head`: the reader left early, which is fine
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())   # so the exit-time flush does not raise again
        except (OSError, ValueError):
            pass
        return 0


if __name__ == "__main__":
    sys.exit(_main())
