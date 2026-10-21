"""Shared test setup: put the maestro package on sys.path and provide inventory fixtures."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Never probe this machine's real model servers (Ollama :11434, vLLM :8000, ...): tests use fake servers on ephemeral ports.
# The env var reaches subprocesses that copy os.environ; PROBE_BUILTIN_PORTS (below) survives tests that clear os.environ.
os.environ["MAESTRO_NO_BUILTIN_PORTS"] = "1"
os.environ.pop("OLLAMA_HOST", None)
sys.path.insert(0, os.path.join(ROOT, "skills", "maestro", "scripts"))

from maestro import router as maestro  # noqa: E402  (tests patch module globals, so they use the module, not the package)

maestro.PROBE_BUILTIN_PORTS = False

TIER_TO_NATIVE = {"fast": "haiku", "standard": "sonnet", "strong": "opus", "frontier": "fable"}


def cls(job="code", difficulty=1, confidence=0.9, needs=()):
    return maestro.Classification(job, difficulty, confidence, tuple(needs))


def inventory(clis=(), local_models=None):
    """An inventory dict as detect() would produce, without touching the machine."""
    harnesses = {c: {"cli": c, "path": "/usr/bin/" + c, "creds_file": True, "available": "yes"} for c in clis}
    return {
        "version": 3, "generated_at": 9e12, "harnesses": harnesses, "api_keys": {},
        "local_endpoints": [], "local_models": local_models or [], "endpoints_sig": [],
        "hardware": {}, "warnings": [],
    }


def local(name, params_b, endpoint="this-machine", on_machine=None, latency=5.0, cfg=None, loaded=True):
    """A local-model record as detect() writes it (serves and specialist_for from the defaults or `cfg`)."""
    rec = {"id": "local:%s/%s" % (endpoint, name), "name": name, "endpoint_name": endpoint,
           "url": "http://%s:11434" % ("127.0.0.1" if endpoint == "this-machine" else endpoint), "kind": "ollama",
           "params_b": params_b, "on_this_machine": endpoint == "this-machine" if on_machine is None else on_machine,
           "latency_ms": latency, "loaded": loaded}
    rec["serves"], rec["specialist_for"] = maestro.rate_local(rec, cfg or {})
    return rec
