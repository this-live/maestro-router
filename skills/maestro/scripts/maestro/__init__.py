"""Maestro: route each task to the right model you already have.

`import maestro` re-exports the public router API (route, classify_rules, detect, ...). The logic lives in
`maestro.router`; provider, client and quota code are their own modules and are imported only when used,
so `route` stays fast and never touches the network.
"""
from __future__ import annotations

from . import router
from .router import *  # noqa: F401,F403  (the public names of router.py)
from .router import VERSION, main  # noqa: F401
