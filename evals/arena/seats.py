"""Contestant seats on this machine. A seat = {id, harness, model, effort, mode: agentic|chat, family, category, ...}.

    claude-code:<model>[@effort]   Claude Code (`claude -p --model M`)            agentic (also answers chat suites)
    codex:<model>[@effort]         Codex (`codex exec -m M`)                       agentic
    gemini-cli:<model>             Gemini CLI (`gemini -m M -p`)                  agentic
    opencode:<provider>/<model>    OpenCode (`opencode run -m provider/model`)    agentic; a served local model too
    hermes:<endpoint>/<model>      Hermes Agent on a served local model             agentic
    api:<provider>/<model>         an API model with a key, through maestro.clients chat only
    local:<endpoint>/<model>       a model a local / LAN server already serves      chat only (`maestro-router run`; never loaded)

Models per harness are the user's `models` config, else Maestro's suggestions (router.seat_models). evals/arena/seats.json
(optional) adds seats ({"add": [...]}), drops them ({"remove": [globs]}) and names the default judges ({"judges": [globs]}).
Family (the lab, from the catalog), price and strength G come from the bundled catalog; `maestro-router discover` is not needed.
"""
from __future__ import annotations

import os
import re
import shutil

from .common import SEATS_FILE, m, matches, read_json
from .harness import EXE, HARNESSES

API_SUGGESTED = ("anthropic", "openai", "google")     # API seats offered from the suggestions when their key is set
CLI_PROVIDER = {"claude-code": "anthropic", "codex": "openai", "gemini-cli": "google"}
SERVE_PORT = 4545       # `maestro-router serve` default
LOCAL_BONUS = 0.05     # a local judge wins its family's slot when within this much G of the strongest (it costs nothing)
JUDGE_PRIORITY = {"api": 0, "claude-code": 1, "codex": 1, "gemini-cli": 1, "local": 2, "opencode": 3, "hermes": 3}


def seat_id(harness, model, effort=None):
    if harness == "local":
        return model if str(model).startswith("local:") else "local:" + str(model)
    return "%s:%s%s" % (harness, model or "default", "@" + effort if effort else "")


def normalize(s, source="seats.json"):
    """A seat dict with defaults filled in. Raises ValueError on a seat that cannot work."""
    s = dict(s)
    h = s.get("harness")
    if h not in HARNESSES and h not in ("api", "local"):
        raise ValueError("seat %r: harness must be one of %s, api, local" % (s.get("id"), ", ".join(HARNESSES)))
    if h == "api" and not (s.get("provider") and s.get("model")):
        raise ValueError("seat %r: api seats need provider and model" % s.get("id"))
    if h == "local" and not (s.get("base_url") or str(s.get("model") or s.get("id") or "").startswith("local:")):
        raise ValueError("seat %r: local seats need base_url, or model local:<endpoint>/<model>" % s.get("id"))
    if h == "local" and not s.get("model"):
        s["model"] = s["id"]
    ep = s.get("endpoint")
    if ep is not None and not (isinstance(ep, dict) and ep.get("name") and ep.get("base_url") and ep.get("served")):
        raise ValueError("seat %r: endpoint needs name, base_url (ending in /v1) and served (the model id the server knows)"
                         % s.get("id"))
    s.setdefault("effort", None)
    s.setdefault("mode", "agentic" if h in HARNESSES else "chat")
    if s["mode"] not in ("agentic", "chat"):
        raise ValueError("seat %r: mode must be agentic or chat" % s.get("id"))
    s.setdefault("id", seat_id(h, s.get("model"), s.get("effort")))
    s.setdefault("source", source)
    return s


def annotate(seat, cat=None, ctx=None):
    """Fill family, price_in/out (USD per 1M tokens), strength (G) and category from the catalog when missing."""
    from maestro import catalog, score
    if cat is None:
        cat = catalog.load_snapshot()
        catalog._finish(cat)
        ctx = score.make_context(cat)
    h, model = seat["harness"], seat.get("model") or ""
    lid = None
    try:
        if seat.get("endpoint") or h == "local":
            name = (seat.get("endpoint") or {}).get("served") or seat.get("served") or model.split("/", 1)[-1]
            lid = catalog.resolve_local(cat, name)[0]
        elif h in CLI_PROVIDER:
            lid = catalog.resolve(cat, CLI_PROVIDER[h], model)[0] or catalog.resolve_alias(cat, model)
        elif h in ("opencode", "api"):
            prov, _, mid = (model.partition("/") if h == "opencode" else (seat.get("provider"), "/", model))
            lid = catalog.resolve(cat, prov, mid)[0]
    except Exception:  # an odd id never breaks the listing
        lid = None
    seat["lab_model"] = lid
    rec = catalog.rec_of(cat, lid) if lid else {}
    local = h == "local" or bool(seat.get("endpoint")) and (seat.get("endpoint") or {}).get("local", True)
    if not seat.get("family"):
        seat["family"] = rec.get("lab") or HARNESSES.get(h, {}).get("family") or CLI_PROVIDER.get(h) or _name_family(model)
    if seat.get("price_in") is None:
        seat["price_in"] = 0.0 if local else rec.get("price_in")
    if seat.get("price_out") is None:
        seat["price_out"] = 0.0 if local else rec.get("price_out")
    if seat.get("strength") is None and lid:
        sc = score.score_lab(ctx, cat, lid)
        seat["strength"] = sc["G"]
        seat.setdefault("category", "local" if local else sc["serves"])
    seat.setdefault("category", "local" if local else None)
    seat.setdefault("strength", None)
    return seat


def _name_family(model):
    tail = model.split("/")[-1].lower()
    mt = re.match(r"[a-z]+", tail)
    return mt.group(0) if mt else "unknown"


def quota_scope(seat):
    """The quota.py scope this seat spends: claude / codex / gemini for the subscription CLIs, the provider for API seats."""
    if seat.get("endpoint") or seat["harness"] == "local":
        return None
    if seat["harness"] == "api":
        return seat["provider"]
    if seat["harness"] == "opencode":
        return (seat.get("model") or "").partition("/")[0] or None
    return HARNESSES.get(seat["harness"], {}).get("quota")


def discover(env=None, home=None, cfg=None, inventory=None, which=None, seats_file=SEATS_FILE):
    """Every seat this machine can run: harness CLIs on PATH x their models, served local models (chat, and through
    OpenCode / Hermes when installed), API models with a key, then seats.json. Nothing here calls a model."""
    from maestro import providers as P
    env = os.environ if env is None else env
    home = home or m.maestro_home(env)
    cfg = cfg if cfg is not None else m.load_config(home)
    which = which or (lambda exe: shutil.which(exe, path=env.get("PATH")))
    inv = inventory if inventory is not None else m.ensure_inventory(home, env, cfg=cfg)
    out = []
    for h in ("claude-code", "codex", "gemini-cli", "opencode"):
        if not which(EXE[h]):
            continue
        seen = set()
        for tier, (model, effort, src) in sorted(m.seat_models(h, inv, cfg).items(), key=lambda kv: m.TIERS.index(kv[0])):
            if not model or (model, effort) in seen:
                continue
            seen.add((model, effort))
            out.append({"harness": h, "model": model, "effort": effort, "category": tier, "source": src})
    for rec in inv.get("local_models") or []:
        if rec.get("loaded") is False:     # installed but not loaded: Maestro never loads a model
            continue
        out.append({"harness": "local", "model": rec["id"], "id": rec["id"], "source": "detected"})
        ep = next((e for e in m.local_endpoints(cfg, env) if e["url"] == rec["url"]), {})
        endpoint = {"name": rec["endpoint_name"], "base_url": rec["url"] + "/v1", "served": rec["name"],
                    "api_key_env": ep.get("api_key_env")}
        for h in ("opencode", "hermes"):
            if which(EXE[h]):
                out.append({"harness": h, "model": "%s/%s" % (rec["endpoint_name"], rec["name"]), "endpoint": endpoint,
                            "id": "%s:%s/%s" % (h, rec["endpoint_name"], rec["name"]), "source": "detected"})
    if which(EXE["hermes"]):     # Hermes on Maestro's own router: needs `maestro-router serve` running; spends what Maestro picks
        out.append({"harness": "hermes", "model": "maestro/auto", "id": "hermes:maestro/auto", "family": "maestro",
                    "endpoint": {"name": "maestro", "base_url": "http://127.0.0.1:%d/v1" % SERVE_PORT, "served": "maestro/auto",
                                 "local": False}, "source": "maestro-router serve (start it first)"})
    for pid in API_SUGGESTED:
        spec = P.BY_ID.get(pid)
        if not spec or not P.get_key(spec["env"], env, home):
            continue
        for tier in m.TIERS:
            cell = m._cell(m.SUGGESTED_MODELS["opencode"][pid].get(tier))
            if cell and not any(s.get("provider") == pid and s["model"] == cell[0].split("/", 1)[1] for s in out):
                out.append({"harness": "api", "provider": pid, "model": cell[0].split("/", 1)[1], "category": tier,
                            "id": "api:" + cell[0], "source": "suggested"})
    seats = []
    for s in out:
        try:
            seats.append(normalize(s, s.get("source", "detected")))
        except ValueError:
            continue
    return apply_file(seats, seats_file)


def apply_file(seats, path=SEATS_FILE):
    """Apply seats.json: {"add": [seat, ...], "remove": [glob, ...]} (an added seat with an existing id replaces it)."""
    data = read_json(path) if path else None
    if not isinstance(data, dict):
        return seats
    rm = [g for g in data.get("remove") or [] if isinstance(g, str)]
    out = [s for s in seats if not matches(s["id"], rm)]
    for raw in data.get("add") or []:
        if not isinstance(raw, dict):
            continue
        s = normalize(raw)
        out = [x for x in out if x["id"] != s["id"]] + [s]
    return out


def file_judges(path=SEATS_FILE):
    data = read_json(path) if path else None
    return [g for g in (data or {}).get("judges") or [] if isinstance(g, str)] if isinstance(data, dict) else []


def select(seats, patterns):
    """The seats whose id matches any glob (all seats when there is no pattern)."""
    return [s for s in seats if not patterns or matches(s["id"], patterns)]


def default_judges(seats, want=4):
    """(judges, note). The strongest seat of each family (a local seat wins its family's slot when within LOCAL_BONUS of
    the strongest), families ordered by their best strength, up to `want`. With one family only: its two strongest seats,
    and a note that says the panel is single-family."""
    cands = [s for s in seats if s.get("strength") is not None] or list(seats)
    if not cands:
        return [], "no seat can judge"
    key = lambda s: (-((s.get("strength") or 0.0) + (LOCAL_BONUS if s["harness"] == "local" else 0.0)),   # noqa: E731
                     JUDGE_PRIORITY.get(s["harness"], 9), s["id"])
    best = {}
    for s in sorted(cands, key=key):
        best.setdefault(s.get("family") or "unknown", s)
    fams = sorted(best.values(), key=key)
    if len(fams) >= 2:
        return fams[:want], ""
    same = sorted(cands, key=key)[:2]
    return same, ("SINGLE-FAMILY JUDGING: every available judge is %s; rankings may carry that family's self-preference. "
                  "Add a seat from another family (seats.json or --judges)." % (same[0].get("family") or "unknown"))


def table(seats):
    """Rows for `seats` (text output)."""
    rows = [("id", "mode", "family", "category", "G", "$in/$out per 1M", "source")]
    for s in seats:
        price = "-" if s.get("price_in") is None else "%g/%g" % (s["price_in"], s.get("price_out") or 0)
        rows.append((s["id"], s["mode"], s.get("family") or "?", s.get("category") or "?",
                     "-" if s.get("strength") is None else "%.2f" % s["strength"], price, s.get("source") or ""))
    w = [max(len(r[k]) for r in rows) for k in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[k]) for k, c in enumerate(r)).rstrip() for r in rows)
