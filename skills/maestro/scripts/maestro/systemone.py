"""Maestro's own typed-decision API: `POST /v1/systemone` and `maestro-router decide`, in TypeSafe Jev's wire format (spec §11).

    handle(body, ctx) -> (status, dict)     200 {model, answers, usage, maestro} | 422/502/503 {"detail": [{loc, msg, type}]}

A question is native (S1's heads, about 1 ms, no LLM) only when it says `"intent": "maestro.<name>"` or its key is in ALIASES
and its criteria have that intent's shape; there is no similarity matching. For `maestro-s1+llm` (alias `jev-latest`) the
others go to config `systemone_delegate`, a model the user already serves, read through its answer logprobs (a cloud model
only with `systemone_cloud: true`; a model installed but not loaded is refused: Maestro never loads models).
ctx (all optional): home, env, cfg, mode, inv (inventory), models {id: models.json entry}, seats(names, task) -> {name:
candidate} (the server passes its own), decide_ctx (extra keys for decide.rank).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from collections import OrderedDict

from . import router as R

MODELS = {"maestro-s1": False, "maestro-s1+llm": True, "jev-latest": True}      # name -> may delegate
MODEL_LIST = [{"name": n, "description": d, "release_date": "2026-10-02"} for n, d in (
    ("maestro-s1", "Maestro S1: native routing questions only, about 1 ms, no LLM."),
    ("maestro-s1+llm", "Native when possible; other questions go to your own served model (systemone_delegate)."),
    ("jev-latest", "Alias of maestro-s1+llm."))]
ALIASES = {"job": "job", "model": "model", "route": "model", "seat": "model", "difficulty": "difficulty", "complexity": "difficulty",
           "reasoning": "difficulty", "stakes": "stakes", "risk": "stakes", "ambiguity": "ambiguity", "needs_tools": "needs_tools",
           "self_contained": "self_contained", "detectable": "detectable", "needs_vision": "needs_vision", "tokens": "tokens",
           "length": "tokens", "success": "success"}
INTENTS = frozenset(ALIASES.values())
HEADS = {"needs_tools": "tools", "self_contained": "self_contained", "detectable": "detectable"}
TOKEN_CUTS = (100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000)     # default buckets (the panel's for 5 levels)
B_QUANTILES = (-3.1, -1.2, 0.0, 1.3, 2.9)                   # placeholder s1.bin ordinal.b_quantiles, at p = i/(m+1)
STRATEGIES = ("logprobs", "prefill", "votes")
CACHE_FILE, CALIB_FILE, CACHE_SIZE, STATE_CHARS, MAX_QUESTIONS, TIMEOUT_S = "s1_cache.json", "systemone_calib.json", 2048, 6000, 64, 30.0
PROBE = ("Water boils at 100 degrees Celsius at sea level.",
         {"type": "choice", "instructions": "Is the statement true or false?", "criteria": {"true": "it is true", "false": "it is false"}})
_ND, _SEM, _LOCK, _CACHE = statistics.NormalDist(), threading.BoundedSemaphore(2), threading.Lock(), {}
_TOK_RX = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)


class Fail(Exception):
    def __init__(self, status, loc, msg, kind):
        Exception.__init__(self, msg)
        self.status, self.detail = status, {"loc": list(loc), "msg": msg, "type": kind}


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _engine(name):
    """The S1 engine (`s1`) or the decision layer (`decide`): one narrow import each (tests stub them in sys.modules)."""
    return importlib.import_module("%s.%s" % (__package__ or "maestro", name))


def validate(body):
    """[{loc, msg, type}] for a malformed request (the shape typesafe-sdk 0.7.2 validates); [] when it is well formed."""
    if not isinstance(body, dict):
        return [{"loc": ["body"], "msg": "Input should be a valid dictionary", "type": "dict_type"}]
    errs, any_ = [], (str, dict, list, type(None))
    for f, ok in (("model", str), ("state", (str, dict, list)), ("questions", dict)):
        if not isinstance(body.get(f), ok):
            errs.append({"loc": ["body", f], "msg": "Field required" if f not in body else "Input has the wrong type",
                         "type": "missing" if f not in body else "type_error"})
    qs = body.get("questions") if isinstance(body.get("questions"), dict) else {}
    if "questions" in body and isinstance(body["questions"], dict) and not 1 <= len(qs) <= MAX_QUESTIONS:
        errs.append({"loc": ["body", "questions"], "msg": "Dictionary should have 1 to %d items" % MAX_QUESTIONS,
                     "type": "too_short" if not qs else "too_long"})
    for name, q in qs.items():
        t = q.get("type") if isinstance(q, dict) else None
        if t not in ("choice", "score", "noul"):
            errs.append({"loc": ["body", "questions", name], "msg": "Unable to extract tag using discriminator 'type'" if t is None else
                         "Input tag %r found using 'type' does not match any of the expected tags: 'noul', 'choice', 'score'" % (t,),
                         "type": "union_tag_not_found" if t is None else "union_tag_invalid"})
            continue
        cr, bad = q.get("criteria"), []
        bad += [("instructions", "a string, object, list or null")] if not isinstance(q.get("instructions"), any_) else []
        bad += [("intent", "a string")] if "intent" in q and not isinstance(q["intent"], str) else []
        if t == "choice" and not (isinstance(cr, dict) and cr and all(isinstance(v, any_) for v in cr.values())):
            bad.append(("criteria", "a non-empty object of choice descriptions"))
        if t == "score" and not (isinstance(cr, list) and cr and all(isinstance(v, (str, dict, list)) for v in cr)):
            bad.append(("criteria", "a non-empty list of level descriptions"))
        if t == "noul" and cr is not None and not (isinstance(cr, dict) and all(isinstance(cr.get(k), any_) for k in ("true", "false"))):
            bad.append(("criteria", "an object with optional true and false"))
        errs += [{"loc": ["body", "questions", name, t, f], "msg": "Field required" if f == "criteria" and cr is None else
                  "Input should be " + m, "type": "missing" if f == "criteria" and cr is None else "value_error"} for f, m in bad]
    return errs


def task_of(state):
    """The task a state describes: a string is the task; an object reads task (or prompt), history, harness and images."""
    st = state if isinstance(state, dict) else {}
    text = state if isinstance(state, str) else st.get("task", st.get("prompt")) if st else None
    text = text if isinstance(text, str) else _txt(state if text is None else text)
    hist, imgs = st.get("history"), st.get("images")
    users = [m for m in hist if isinstance(m, dict) and m.get("role") == "user"] if isinstance(hist, list) else []
    prev = users[-1].get("content") if users else hist if isinstance(hist, str) else ""
    harness = st.get("harness") if isinstance(st.get("harness"), str) else None
    return {"text": text, "harness": harness, "seats": {}, "ineligible": set(), "images": len(imgs) if isinstance(imgs, list) else imgs if isinstance(imgs, int) else 0,
            "s1": {"text": text, "mode": "skill", "prev_user": _txt(prev or "")[-1024:], "n_user": len(users), "harness": harness}}


# ---- native answers (spec §11.2) ----

def _resolve(names, task, ctx):
    """{name: candidate} for the names that are seats: the server's (ctx seats), router seats, else models.json ids. A router seat
    that fails this task's hard filters (a chat-only model for a repo job) resolves but is marked ineligible (P(best) 0)."""
    todo = [n for n in names if n not in task["seats"]]
    if todo:
        if callable(ctx.get("seats")):
            got = ctx["seats"](todo, task) or {}
        else:
            task["cls"] = task.get("cls") or R.classify_rules(task["text"])
            seats = lambda cls: {c["name"]: c for c in R.build_candidates(cls, ctx["inv"], task["harness"] or "claude-code", ctx["cfg"])}  # noqa: E731
            got, chat = seats(task["cls"]), seats(R.Classification("ask", 0, 1.0))
            task["ineligible"].update(n for n in chat if n not in got)
            got = dict(chat, **got)
        task["seats"].update({n: got.get(n) or ctx["models"].get(n) for n in todo})
    return {n: task["seats"][n] for n in names if task["seats"].get(n)}


def _intent(name, q, task, ctx):
    """The native intent of a question, or None (delegate it). An explicit maestro.<x> intent that cannot be met is a 422."""
    it, t, cr = q.get("intent") or "", q["type"], q.get("criteria")
    explicit, n = it.startswith("maestro."), len(cr) if isinstance(cr, (list, dict)) else 0
    intent = it[8:] if explicit else ALIASES.get(name)
    if explicit and intent not in INTENTS:
        raise Fail(422, ["body", "questions", name, t, "intent"], "unknown intent %r (known: maestro.%s)" % (it, ", maestro.".join(
            sorted(INTENTS))), "unknown_intent")
    shape = {"job": lambda: t == "choice" and set(cr) <= set(R.JOBS), "model": lambda: t == "choice" and len(_resolve(list(cr), task, ctx)) == n,
             "difficulty": lambda: t == "score" and 2 <= n <= 6, "stakes": lambda: t == "score" and 2 <= n <= 4,
             "ambiguity": lambda: t == "score" and 2 <= n <= 4, "tokens": lambda: t == "score" and 2 <= n <= 10,
             "success": lambda: t == "noul" and isinstance((cr or {}).get("true"), str) and bool(_resolve([cr["true"]], task, ctx))}
    ok = bool(intent) and (shape[intent]() if intent in shape else t == "noul")      # the rest are the binary heads
    if explicit and not ok:
        raise Fail(422, ["body", "questions", name, t, "criteria"], "the question does not have the shape %s needs" % it, "intent_mismatch")
    return intent if ok else None


def _posterior(task):
    """S1's TaskPosterior (spec §2.1) plus the training-b quantiles of the s1.bin header (placeholders without one)."""
    try:
        S1 = _engine("s1")
    except ImportError:
        raise Fail(503, ["body", "questions"], "the S1 engine is not installed", "engine_unavailable")
    task["cls"] = task.get("cls") or R.classify_rules(task["text"])
    post = S1.posterior(task["s1"], task["cls"])
    try:
        qs = S1.load()[0]["header"]["ordinal"]["b_quantiles"]
    except (AttributeError, KeyError, TypeError, ValueError):
        qs = None
    return dict(post, b_quantiles=qs if isinstance(qs, list) and len(qs) >= 2 else post.get("b_quantiles") or list(B_QUANTILES))


def _bins(cuts, mu, sd):
    cdf = [0.0] + [_ND.cdf((c - mu) / max(sd, 1e-3)) for c in cuts] + [1.0]
    return [max(0.0, b - a) for a, b in zip(cdf, cdf[1:])]


def _token_cuts(levels):
    """Log-token cut points from legends with token ranges ("<100", "100-300", "1k-3k", ">3k"), else the default buckets."""
    rng, last = [], len(levels) - 1
    for i, s in enumerate(levels):
        v = [float(a.replace(",", "")) * (1000 if k else 1) for a, k in _TOK_RX.findall(_txt(s))]
        rng.append(v[:2] if len(v) >= 2 else [0.0, v[0]] if v and i == 0 else [v[0], 1e12] if v and i == last else None)
    cuts = [math.sqrt(max(1.0, a[1]) * max(1.0, b[0])) for a, b in zip(rng, rng[1:])] if all(rng) else []
    ok = len(cuts) == last and all(x < y for x, y in zip(cuts, cuts[1:]))
    return [math.log(c) for c in (cuts if ok else TOKEN_CUTS[:last])]


def _answer(intent, q, post, task, ctx, did):
    """The native answer; probabilities are tempered toward uniform with T = 1 + ood when ood > 0.5."""
    cr, choice = q.get("criteria"), None
    if intent == "job":
        p = [post["job_p"].get(k, 0.0) for k in cr]
        p = p if sum(p) > 0 else [1.0] * len(p)
    elif intent in ("model", "success"):
        names = list(cr) if intent == "model" else [cr["true"]]
        cands, D = _resolve(names, task, ctx), _engine("decide")
        dctx = dict(ctx.get("decide_ctx") or {}, mode=ctx["mode"], cfg=ctx["cfg"], home=ctx["home"], text=task["text"], decision_id=did)
        seats = [D.seat_of(cands[n], ctx["models"]) for n in names]
        ranked, ids = D.rank([s for n, s in zip(names, seats) if n not in task["ineligible"]], post, dctx), [s["id"] for s in seats]
        if intent == "success":
            p = [ranked[0]["p"] if ranked else 0.0]
        elif not ranked:
            raise Fail(422, ["body", "questions", q["_name"], "choice", "criteria"], "no seat in criteria qualifies for this task",
                       "no_seat_qualifies")
        else:
            pb = D.p_best(ranked, did, 64)
            p, choice = [pb.get(i, 0.0) for i in ids], ids.index(ranked[0]["seat"])
    elif intent == "difficulty":
        qs, n = post["b_quantiles"], len(cr)
        xs = [k * (len(qs) + 1.0) / n - 1 for k in range(1, n)]               # quantile positions of p = k/n
        cuts = [qs[i] + (qs[i + 1] - qs[i]) * max(0.0, min(1.0, x - i)) for x, i in ((x, max(0, min(len(qs) - 2, int(x)))) for x in xs)]
        p = _bins(cuts, post["b"][0], post["b"][1])
    elif intent in ("stakes", "ambiguity"):
        e = min(2.0, max(0.0, float(post.get(intent, 1.0)))) / 2
        p1, p2 = post.get(intent + "_p") or (1 - (1 - e) ** 2, e * e)       # [P(>=1), P(>=2)]; else Binomial(2, E/2)
        n = len(cr)                                                         # 3 levels spread evenly over n
        p = [sum(v * 3 * max(0.0, min((k + 1.0) / n, (j + 1) / 3.0) - max(k / float(n), j / 3.0)) for j, v in enumerate((1 - p1, p1 - p2, p2)))
             for k in range(n)]
    elif intent == "tokens":
        mu, sd = post["tok_out"]
        p = _bins(_token_cuts(cr), math.log(mu) if mu > 20 else mu, sd)     # mu > 20: a token count, not a log
    else:
        p = [float(task["images"] > 0) if intent == "needs_vision" else float(post.get(HEADS[intent], 0.5))]
    p = [p[0], 1 - p[0]] if q["type"] == "noul" else p
    return _build(q, _temper(p, 1 + post["ood"]) if post.get("ood", 0.0) > 0.5 else p, 1.0, choice)


def _temper(p, T):
    w = [max(0.0, x) ** (1.0 / T) for x in p]
    return [x / (sum(w) or 1.0) for x in w]


def _build(q, p, mass, choice=None):
    """A Jev answer from probabilities over the options (noul: [yes, no]); confidence × mass when mass < 0.5."""
    p = [x / (sum(p) or 1.0) for x in p]
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(p[0], 4)}
    k = max(range(len(p)), key=p.__getitem__) if choice is None else choice
    c = round(p[k] * (mass if mass < 0.5 else 1.0), 4)
    if q["type"] == "choice":
        return {"type": "choice", "choice": list(q["criteria"])[k], "confidence": c, "probabilities": {n: round(v, 4) for n, v in zip(q["criteria"], p)}}
    return {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4), "confidence": c,
            "legend": {str(i): v for i, v in enumerate(q["criteria"])}, "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)}}


# ---- delegation: an already-served model, read through its logprobs (spec §11.3) ----

def _messages(state, name, q):
    """(labels as normalized tokens, chat messages): the state (6,000 chars), the instructions, the options, the answer line."""
    t, cr = q["type"], q.get("criteria") or {}
    if t != "noul" and len(cr) > (26 if t == "choice" else 10):
        raise Fail(422, ["body", "questions", name, t, "criteria"], "a delegated %s takes at most %d options" % (t, 26 if t == "choice" else 10),
                   "too_many_options")
    if t == "noul":
        labels, close = ["yes", "no"], "Answer with Yes or No only."
        lines = ["Yes" + (": " + _txt(cr["true"]) if cr.get("true") else ""), "No" + (": " + _txt(cr["false"]) if cr.get("false") else "")]
    elif t == "choice":
        labels, close = [chr(97 + i) for i in range(len(cr))], "Answer with the letter only."
        lines = ["%s. %s%s" % (chr(65 + i), k, ": " + _txt(v) if v not in (None, "") else "") for i, (k, v) in enumerate(cr.items())]
    else:
        labels, close, lines = [str(i) for i in range(len(cr))], "Answer with the digit only.", ["%d. %s" % (i, _txt(v)) for i, v in enumerate(cr)]
    ins = q.get("instructions")
    text = "\n\n".join((_txt(state)[:STATE_CHARS], _txt(ins) if ins not in (None, "") else "Question: " + name, "\n".join(lines), close))
    return labels, [{"role": "user", "content": text}]


def _ready(dl):
    """Before the first request to a local delegate: it must answer and have the model loaded (Maestro never loads models)."""
    if not dl["ready"]:
        url, kind, key_env, model = dl["check"]
        live, loc = R.probe_endpoint({"url": url, "kind": kind, "api_key_env": key_env}, None, 1.0, dl["env"]), ["body", "questions"]
        if not live:
            raise Fail(503, loc, "the delegate %s is not reachable" % dl["id"], "delegate_unavailable")
        if not dl["cfg"].get("local_allow_load") and any(x["name"] == model and not x["loaded"] for x in live["models"]):
            raise Fail(503, loc, "the delegate %s is installed but not loaded, and Maestro never loads models" % dl["id"], "delegate_unavailable")
        dl["ready"] = True


def _query(dl, strategy, msgs, labels, loc):
    """(probabilities over the labels, mass on the labels, [in, out] tokens, whether the first token is a label)."""
    _ready(dl)
    body = {"messages": msgs, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20}
    if strategy == "logprobs":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif strategy == "prefill":     # an open assistant turn: skips gpt-oss's harmony analysis channel and Qwen3's thinking
        body.update(messages=msgs + [{"role": "assistant", "content": "Answer:"}], add_generation_prompt=False, continue_final_message=True)
    else:
        body = {"messages": msgs, "n": 8, "temperature": 1, "max_tokens": 4}
    with _SEM:
        status, _, out = dl["C"].complete(dl["spec"], dl["model"], body, env=dl["env"], home=dl["home"], cfg=dl["cfg"], timeout=TIMEOUT_S)
    if not 200 <= status < 300 or not isinstance(out, dict):
        err = out.get("error") if isinstance(out, dict) else None
        msg = dl["P"].scrub(str(err.get("message") if isinstance(err, dict) else err or "no detail"))
        raise Fail(502, loc, "the delegate %s answered HTTP %s: %s" % (dl["id"], status, msg), "delegate_error")
    u = out.get("usage") if isinstance(out.get("usage"), dict) else {}
    usage, idx, acc = [int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)], {k: i for i, k in enumerate(labels)}, [0.0] * len(labels)
    norm = lambda tok: str(tok or "").replace("Ġ", " ").replace("▁", " ").strip().lower()  # noqa: E731 (BPE/SentencePiece space marks)
    try:
        if strategy == "votes":
            for ch in out["choices"]:
                w = re.findall(r"[A-Za-z0-9]+", (ch.get("message") or {}).get("content") or "")
                if w and w[0].lower() in idx:
                    acc[idx[w[0].lower()]] += 1.0
            return [a / max(1.0, sum(acc)) for a in acc], sum(acc) / 8.0, usage, sum(acc) > 0     # a server that ignores n gets mass 1/8
        first = out["choices"][0]["logprobs"]["content"][0]
        tops = first.get("top_logprobs") or [first]
        tops = [{"token": k, "logprob": v} for k, v in tops.items()] if isinstance(tops, dict) else tops
        lps = [float(t["logprob"]) for t in tops]
        for t in tops:
            if norm(t["token"]) in idx:
                acc[idx[norm(t["token"])]] += math.exp(float(t["logprob"]))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise Fail(502, loc, "the delegate %s returned no usable logprobs" % dl["id"], "delegate_error")
    p = [a if a > 0 else math.exp(min(lps) - 2) for a in acc]               # a missing label gets min_logprob - 2
    return [x / sum(p) for x in p], sum(acc), usage, norm(first.get("token")) in idx


def probe(dl):
    """The first strategy whose first token on a fixed 2-option probe is an option label with >= 50% of the mass on the labels."""
    labels, msgs = _messages(PROBE[0], "probe", PROBE[1])
    _ready(dl)
    for s in STRATEGIES:
        try:
            _, mass, _, first = _query(dl, s, msgs, labels, ["probe"])
        except Fail:
            continue
        if first and mass >= 0.5:
            return s
    return None


def _store(home, save=False):
    """The delegate cache of this home ({"entries": LRU {sha256: raw answer}, "strategy": {delegate: name}}, s1_cache.json)."""
    with _LOCK:
        if home not in _CACHE:
            d = R._read_json(os.path.join(home, CACHE_FILE))
            d = d if isinstance(d, dict) and d.get("version") == 1 else {}
            _CACHE[home] = {"entries": OrderedDict((d.get("entries") or {}).items()), "strategy": dict(d.get("strategy") or {}), "dirty": False}
        st = _CACHE[home]
        if save and st["dirty"]:
            st["dirty"] = False
            _write(os.path.join(home, CACHE_FILE), {"version": 1, "strategy": st["strategy"], "entries": dict(st["entries"])})
    return st


def _write(path, data):
    R._write_json(path, data)   # created 0600


def delegate(ctx):
    """The configured delegate: resolved, its strategy probed once (then cached); served and loaded is checked before a request."""
    cfg, env, home, mid, loc = ctx["cfg"], ctx["env"], ctx["home"], ctx["cfg"].get("systemone_delegate"), ["body", "questions"]
    if not isinstance(mid, str) or not mid:
        raise Fail(422, loc, "a question is not native and no delegate is set: set systemone_delegate to a model you serve", "question_not_supported")
    from . import clients as C
    from . import providers as P
    m = re.match(r"local:([^/]+)/(.+)$", mid)
    if m:
        rec = next((r for r in ctx["inv"].get("local_models") or [] if r.get("id") == mid), None)
        eps = R.local_endpoints(cfg, env)
        ep = next((e for e in eps if rec and e["url"] == rec.get("url")), None) or next((e for e in eps if e["name"] == m.group(1)), None)
        url, key_env = (rec or {}).get("url") or (ep or {}).get("url"), (ep or {}).get("api_key_env")
        if not url:
            raise Fail(503, loc, "the delegate %s is on an unknown endpoint (see local_endpoints)" % mid, "delegate_unavailable")
        check = (url, (rec or {}).get("kind") or (ep or {}).get("kind"), key_env, m.group(2))
        spec, model = {"id": "local:" + m.group(1), "name": "Local: " + m.group(1), "style": "local", "base_url": url + "/v1",
                       "key_env": key_env, "overrides": {}}, m.group(2)
    elif not cfg.get("systemone_cloud"):
        raise Fail(422, loc, "the delegate %s is a cloud model: set systemone_cloud: true to send states to it" % mid, "question_not_supported")
    elif cfg.get("private") or "state" in ctx and _engine("decide").is_private(_txt(ctx["state"])):   # local delegates only (§4.5)
        raise Fail(503, loc, "the state is private (private mode or a key-like string): the cloud delegate %s may not see it" % mid,
                   "private_no_local")
    else:
        (spec, _, model), check = mid.partition("/"), None
    temps = ((R._read_json(os.path.join(home, CALIB_FILE)) or {}).get("temps") or {}).get(mid) or {}
    dl = {"id": mid, "spec": spec, "model": model, "env": env, "home": home, "cfg": cfg, "C": C, "P": P, "temps": temps, "check": check,
          "ready": check is None}
    st = _store(home)
    dl["strategy"] = (ctx["models"].get(mid) or {}).get("systemone_strategy") or st["strategy"].get(mid)
    if dl["strategy"] not in STRATEGIES:
        dl["strategy"] = probe(dl)
        if not dl["strategy"]:
            raise Fail(422, loc, "the delegate %s passed none of the logprob, prefill and vote probes" % mid, "delegate_unsupported")
        with _LOCK:
            st["strategy"][mid], st["dirty"] = dl["strategy"], True
    return dl


def strategy_of(model_id, ctx=None):
    """The strategy a served model passes (for `discover --live` to store as models.json systemone_strategy), or None."""
    ctx = context(ctx)
    ctx.update(cfg=dict(ctx["cfg"], systemone_delegate=model_id), models={})
    try:
        return delegate(ctx)["strategy"]
    except Fail:
        return None


def ask(dl, name, q, state, raw=False):
    """(answer, mass, fresh [in, out] tokens) of a delegated question; raw=True: the uncalibrated probabilities instead of the answer."""
    labels, msgs = _messages(state, name, q)
    key = hashlib.sha256(json.dumps([dl["id"], dl["strategy"], msgs]).encode("utf-8")).hexdigest()   # the prompt holds state and question
    st, usage = _store(dl["home"]), [0, 0]
    with _LOCK:
        hit = st["entries"].get(key)
        if hit:
            st["entries"].move_to_end(key)
    if not hit:
        p, mass, usage, _ = _query(dl, dl["strategy"], msgs, labels, ["body", "questions", name])
        hit = {"p": [round(x, 6) for x in p], "mass": round(mass, 4)}
        with _LOCK:
            st["entries"][key], st["dirty"] = hit, True
            while len(st["entries"]) > CACHE_SIZE:
                st["entries"].popitem(last=False)
    if raw:
        return hit["p"], hit["mass"], usage
    return _build(q, _temper(hit["p"], dl["temps"].get(q["type"], 1.0)), hit["mass"]), hit["mass"], usage


# ---- entry points ----

def context(ctx=None):
    ctx = dict(ctx or {})
    ctx["env"] = os.environ if ctx.get("env") is None else ctx["env"]
    ctx["home"] = ctx.get("home") or R.maestro_home(ctx["env"])
    ctx["cfg"] = ctx["cfg"] if ctx.get("cfg") is not None else R.load_config(ctx["home"])
    ctx.setdefault("mode", "skill")
    ctx["inv"] = ctx.get("inv") or R._read_json(os.path.join(ctx["home"], "inventory.json")) or {}
    if ctx.get("models") is None:
        ml = (R._read_json(os.path.join(ctx["home"], "models.json")) or {}).get("models")
        ctx["models"] = {m["id"]: m for m in ml if isinstance(m, dict) and m.get("id")} if isinstance(ml, list) else {}
    return ctx


def handle(body, ctx=None):
    """One /v1/systemone request -> (HTTP status, JSON body)."""
    t0, ctx = time.perf_counter(), context(ctx)
    errs = validate(body)
    if errs or body["model"] not in MODELS:
        return 422, {"detail": errs or [{"loc": ["body", "model"], "msg": "unknown model %r (use %s)" % (body["model"][:80], ", ".join(MODELS)),
                                         "type": "model_not_found"}]}
    llm, task, post, dl, answers, source, mass, usage = MODELS[body["model"]], task_of(body["state"]), None, None, {}, {}, {}, [0, 0]
    ctx["state"], raw = body["state"], None
    try:
        for name, q in body["questions"].items():
            q = dict(q, _name=name)
            intent = _intent(name, q, task, ctx)
            if intent:
                post = post or _posterior(task)
                raw = raw or hashlib.sha256(json.dumps(body["state"], sort_keys=True).encode("utf-8")).hexdigest()   # once per request
                did = hashlib.sha256((raw + name).encode("utf-8")).hexdigest()[:8]
                answers[name], source[name] = _answer(intent, q, post, task, ctx, did), "native"
            elif not llm:
                raise Fail(422, ["body", "questions", name], "maestro-s1 answers only native questions (%s); maestro-s1+llm delegates the "
                           "others" % ", ".join(sorted(ALIASES)), "question_not_supported")
            else:
                dl = dl or delegate(ctx)
                answers[name], mass[name], u = ask(dl, name, q, body["state"])
                source[name], usage = "delegate", [usage[0] + u[0], usage[1] + u[1]]
    except Fail as e:
        return e.status, {"detail": [e.detail]}
    finally:
        if dl:
            _store(ctx["home"], save=True)
    ext = dict({"source": source, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}, **({"mass": mass, "strategy": dl["strategy"]} if mass else {}))
    return 200, {"model": "maestro-s1+llm" if llm else "maestro-s1", "answers": answers,
                 "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}, "maestro": ext}


def calibrate(path, ctx=None):
    """{type: T}: one temperature per question type for the delegate, fitted on typed-eval rows {state, question, answer} (0.5..5)."""
    ctx = context(ctx)
    dl, pairs = delegate(ctx), {}
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, r in enumerate(rows):
        q, gold = r["question"], r["answer"]
        p, _, _ = ask(dl, r.get("name") or "q%d" % i, q, r["state"], raw=True)
        g = list(q["criteria"]).index(gold) if q["type"] == "choice" else int(gold) if q["type"] == "score" else 1 - int(bool(gold))
        pairs.setdefault(q["type"], []).append((p, g))
    _store(ctx["home"], save=True)
    temps = {t: round(min((0.5 + 0.05 * i for i in range(91)), key=lambda T: -sum(math.log(max(1e-12, _temper(p, T)[g])) for p, g in ps)), 2)
             for t, ps in pairs.items()}
    old = (R._read_json(os.path.join(ctx["home"], CALIB_FILE)) or {}).get("temps") or {}
    _write(os.path.join(ctx["home"], CALIB_FILE), {"version": 1, "temps": dict(old, **{dl["id"]: temps})})
    return temps


def cmd_decide(a):
    """`maestro-router decide [file|-] [--calibrate FILE]` (a.file, a.calibrate): print the answers, or the fitted temperatures."""
    try:
        if getattr(a, "calibrate", None):
            status, out = 200, {"temperatures": calibrate(a.calibrate)}
        else:
            src = getattr(a, "file", None) or "-"
            if src == "-":
                raw = sys.stdin.read()
            else:
                with open(src, "r", encoding="utf-8") as f:
                    raw = f.read()
            status, out = handle(json.loads(raw))
    except Fail as e:
        status, out = e.status, {"detail": [e.detail]}
    except (OSError, ValueError, KeyError) as e:
        print("decide: cannot read %s (%s)" % (getattr(a, "calibrate", None) or getattr(a, "file", None) or "stdin", e.__class__.__name__), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0 if status == 200 else 1
