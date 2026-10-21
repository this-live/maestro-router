"""Catalog: the bundled models.dev snapshot, an optional refresh, joining available models to it, and `models.json`.

Slow-path only (`discover`, `models`, the proxy); `route` never imports this module. The snapshot (data/catalog.json.gz,
built by evals/build_catalog.py) holds every lab model with its benchmarks and, for open-weight models, the registry fields
(Hugging Face repo, params_total / params_active); `prov` maps each provider's model ids to the lab model they are.

Join order for an available (provider, model id): the provider entry's canonical model, the exact id, `<provider>/<id>`,
the normalized id (dates, `-latest`, vendor prefixes and separators stripped). Local servers add the Hugging Face repo
(vLLM / SGLang report repo ids; quantization suffixes such as -FP8 or -AWQ are ignored) and Ollama tags (`gpt-oss:120b`).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time

from . import router as core

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.json.gz")
MODELS_FILE = "models.json"
REFRESH_FILE = "catalog_refresh.json"
MODELS_VERSION = 1

# embeddings, rerankers, speech, guard/safety classifiers and the like: not chat models a router can pick
NON_CHAT_RX = re.compile(r"embed|rerank|whisper|(?<![a-z])tts|transcri|moderation|guard|realtime|(?<![a-z])ocr(?![a-z])|-live\b|imagen|dall-e|-image-|"
                         r"voice|voxtral|(?<![a-z])stt|(?<![a-z])image(?![a-z])", re.I)
_QUANT = (r"nvfp4|mxfp4|fp8(?:[-_]dynamic|[-_]block)?|fp4|awq|gptq(?:[-_]int[48])?|int[48]|w[48]a(?:4|8|16)|gguf|bnb[-_]?[48]bit|"
          r"[48]bit|bf16|fp16|mlx|quantized|dflash|q[2-8](?:_k(?:_[msl])?|_[01])?")
QUANT_SUFFIX_RX = re.compile(r"(?:[-_.](?:%s))+$" % _QUANT, re.I)


# ════════════════════════════════════════════════════════════════════════════
# 1. names: parameter counts and normalization
# ════════════════════════════════════════════════════════════════════════════

def parse_params(text):
    """(total, active) parameters in billions parsed from a model or repo name, either may be None:
    120b, 2.4t, 270m, 8x7b (56), 30b-a3b (30, 3), 17b-16e (active 17: the total is not in the name)."""
    s = (text or "").lower()
    m = re.search(r"(?<![a-z\d.])(\d+)x(\d+(?:\.\d+)?)b(?![a-z\d])", s)
    if m:
        return round(int(m.group(1)) * float(m.group(2)), 2), None
    act = re.search(r"(?<![a-z\d.])a(\d+(?:\.\d+)?)b(?![a-z\d])", s)
    active = float(act.group(1)) if act else None
    totals = []
    for m in re.finditer(r"(?<![a-z\d.])(\d+(?:\.\d+)?)([bmt])(?![a-z\d])", s):
        v = float(m.group(1)) * {"b": 1.0, "t": 1000.0, "m": 0.001}[m.group(2)]
        if re.match(r"-\d+e\b", s[m.end():]):  # 17b-16e: 17B active, 16 experts
            active = active or v
        elif not (act and m.start() == act.start() + 1):
            totals.append(v)
    return (round(max(totals), 3) if totals else None), active


def norm_id(s):
    """A model id reduced to its identity: no vendor prefix, region, date, version tail or separators."""
    s = (s or "").lower().split("/")[-1]
    s = re.sub(r"^(us|eu|apac|global|au|jp)\.", "", s)
    s = re.sub(r"^(anthropic|openai|meta|mistral|amazon|google|xai|deepseek|moonshotai|qwen|zai|minimax)\.", "", s)
    s = re.sub(r"-v\d+:\d+$", "", s)
    s = re.sub(r"[@:].*$", "", s)
    s = re.sub(r"-((19|20)\d{6}|(19|20)\d\d-\d\d-\d\d)$", "", s)
    s = re.sub(r"-(latest|instruct|it|fp8|fp4|free|preview|exp)$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def strip_quant(repo):
    return QUANT_SUFFIX_RX.sub("", repo or "").lower()


# ════════════════════════════════════════════════════════════════════════════
# 2. the snapshot, the optional refresh, indices
# ════════════════════════════════════════════════════════════════════════════

def snapshot_sha(path=DATA_FILE):
    """sha256 of the bundled catalog snapshot: discover stamps it as models.json `catalog_sha`, fit_link.py next to the link it fits.
    A different sha at decision time means the link may be stale (S1 spec §3.2)."""
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def load_snapshot(path=DATA_FILE):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def text_only(output):
    return not output or list(output) == ["text"]


def _text_out(e):
    return text_only(e.get("om"))


def merge_api(cat, api):
    """Merge a models.dev api.json body ({provider: {models: {id: {...}}}}) into the catalog's provider table: prices, context
    and status of known models are updated, unknown text models are added with their own attributes. Benchmarks stay as
    built. Returns the delta (only what changed), which is what the cache file keeps."""
    delta = {}
    for pid, p in (api or {}).items():
        if not isinstance(p, dict):
            continue
        for mid, m in (p.get("models") or {}).items():
            if not isinstance(m, dict):
                continue
            mods = m.get("modalities") or {}
            if not text_only(mods.get("output")) or NON_CHAT_RX.search(mid):
                continue
            old = cat["prov"].get(pid, {}).get(mid)
            e = dict(old) if old else {"n": m.get("name") or mid, "d": _date(m.get("release_date"))}
            if not old:
                for k, key in (("f", "family"),):
                    if m.get(key):
                        e[k] = m[key]
                for k, key in (("r", "reasoning"), ("t", "tool_call"), ("so", "structured_output"), ("ow", "open_weights")):
                    if m.get(key):
                        e[k] = 1
                if (mods.get("input") or ["text"]) != ["text"]:
                    e["im"] = mods["input"]
            c = m.get("cost") or {}
            for k, key in (("ci", "input"), ("co", "output")):
                if isinstance(c.get(key), (int, float)):
                    e[k] = c[key]
            ctx = (m.get("limit") or {}).get("context")
            if ctx and (not e.get("c") or ctx != cat["models"].get(e["c"], {}).get("x")):
                e["x"] = int(ctx)
            if m.get("status"):
                e["st"] = m["status"]
            if e != old:
                cat["prov"].setdefault(pid, {})[mid] = e
                delta.setdefault(pid, {})[mid] = e
    return delta


def _date(d):
    d = str(d or "")
    return d + "-01" * (2 - d.count("-")) if d[:1].isdigit() else ""


def load(home=None, refresh=False, fetch=None, timeout=8.0, path=DATA_FILE):
    """(catalog, notes). The bundled snapshot, overlaid with the cached refresh in ~/.maestro/catalog_refresh.json if there
    is one; with refresh=True the models.dev API is fetched first ($MAESTRO_CATALOG_URL for a mirror). Offline or on any
    error the snapshot (and the cached refresh) is used and a note says so: discovery never fails on the network."""
    home = home or core.maestro_home()
    cat = load_snapshot(path)
    notes = []
    cache_path = os.path.join(home, REFRESH_FILE)
    cache = core._read_json(cache_path)
    if refresh:
        try:
            status, raw = (fetch or core._http_fetch)(os.environ.get("MAESTRO_CATALOG_URL", core.CATALOG_URL),
                                                      {"Accept": "application/json"}, None, timeout)
            if status != 200:
                raise ValueError("HTTP %s" % status)
            delta = merge_api(cat, json.loads(raw))
            cache = {"fetched_at": time.time(), "prov": delta}
            core._write_json(cache_path, cache, indent=None)
            notes.append("catalog refreshed from models.dev: %d providers changed" % len(delta))
            return _finish(cat), notes
        except Exception as e:  # offline, DNS, TLS, bad JSON: keep going with what we have
            notes.append("catalog refresh failed (%s); using the %s" % (str(e)[:80] or e.__class__.__name__,
                                                                          "cached copy" if isinstance(cache, dict) and cache.get("prov") else "bundled snapshot"))
    if isinstance(cache, dict) and isinstance(cache.get("prov"), dict):
        for pid, ms in cache["prov"].items():
            if isinstance(ms, dict):
                cat["prov"].setdefault(pid, {}).update({k: v for k, v in ms.items() if isinstance(v, dict)})
        notes.append("catalog: refreshed copy from %s" % time.strftime("%Y-%m-%d", time.localtime(cache.get("fetched_at") or 0)))
    return _finish(cat), notes


def _finish(cat):
    cat["_idx"] = build_index(cat)
    return cat


def build_index(cat):
    """Lookup tables for the join: normalized id -> lab ids (benchmarked first, then shortest), Hugging Face repo (and repo
    name) without quantization suffix -> lab ids of open-weight models."""
    norm, hf, hfname = {}, {}, {}
    for lid, m in cat["models"].items():
        norm.setdefault(norm_id(lid), []).append(lid)
        if m.get("hf"):
            k = strip_quant(m["hf"])
            hf.setdefault(k, []).append(lid)
            hfname.setdefault(k.split("/")[-1], []).append(lid)
    rank = lambda lid: (not cat["models"][lid].get("b"), len(lid), lid)  # noqa: E731
    for table in (norm, hf, hfname):
        for k in table:
            table[k].sort(key=rank)
    return {"norm": norm, "hf": hf, "hfname": hfname}


# ════════════════════════════════════════════════════════════════════════════
# 3. the join
# ════════════════════════════════════════════════════════════════════════════

_DATED = re.compile(r"-((19|20)\d{6}|(19|20)\d\d-\d\d-\d\d)$")


def resolve(cat, provider, model_id):
    """(lab model id or None, how, provider entry or None). how: canonical | exact | prefixed | normalized | dated | none.
    A dated snapshot with no benchmarks of its own (anthropic/claude-haiku-4-5-20251001) joins the benchmarked model the
    date-less id names (claude-haiku-4-5): `dated`."""
    entry = cat["prov"].get(provider, {}).get(model_id)
    models, idx = cat["models"], cat["_idx"]
    if entry and entry.get("c") in models:
        lid, how = entry["c"], "canonical"
    elif model_id in models:
        lid, how = model_id, "exact"
    elif "%s/%s" % (provider, model_id) in models:
        lid, how = "%s/%s" % (provider, model_id), "prefixed"
    else:
        hit = idx["norm"].get(norm_id(model_id))
        return (hit[0], "normalized", entry) if hit else (None, "none", entry)
    if not models[lid].get("b") and _DATED.search(lid):
        base = _DATED.sub("", lid)
        twin = base if models.get(base, {}).get("b") else next(
            (h for h in idx["norm"].get(norm_id(base)) or [] if models[h].get("b") and h != lid and not _DATED.search(h)), None)
        if twin:
            return twin, "dated", entry
    return lid, how, entry


_OLLAMA_NOISE = re.compile(r"(?:^|-)(?:instruct|it|chat|thinking|latest|q\d\w*|fp\d+|bf16|awq|gguf)(?=-|$)")


def resolve_local(cat, name, root=None, params_b=None):
    """(lab id or None, how) for a model a local server reports. how: hf-repo | hf-name | normalized | ollama | none.
    vLLM / SGLang ids are usually Hugging Face repo ids (the `root` field may carry the served path); Ollama uses
    `family:size-variant` tags."""
    idx = cat["_idx"]
    for cand in (name, root):
        if cand and "/" in cand and idx["hf"].get(strip_quant(cand)):
            return idx["hf"][strip_quant(cand)][0], "hf-repo"
    for cand in (name, root):
        last = strip_quant((cand or "").rstrip("/").split("/")[-1])
        if last and idx["hfname"].get(last):
            return idx["hfname"][last][0], "hf-name"
    for cand in (name, root):
        if cand and ":" not in cand.split("/")[-1]:
            hit = idx["norm"].get(norm_id(strip_quant(cand.rstrip("/").split("/")[-1])))
            if hit and cat["models"][hit[0]].get("ow"):
                return hit[0], "normalized"
    if name and ":" in name:
        base, _, tag = name.split("/")[-1].partition(":")
        tag = _OLLAMA_NOISE.sub("", tag.lower()).strip("-")
        size = re.match(r"\d+(?:\.\d+)?[bt]", tag)
        want = norm_id(base) + (norm_id(tag) if size else (("%gb" % params_b) if params_b else ""))
        if want and (size or params_b):
            best = [lid for k, lids in idx["norm"].items() if k.startswith(want) for lid in lids if cat["models"][lid].get("ow")]
            if best:
                best.sort(key=lambda lid: (not cat["models"][lid].get("b"), len(lid), lid))
                return best[0], "ollama"
    return None, "none"


def resolve_alias(cat, alias):
    """A `-latest` / short alias (haiku, sonnet, gemini-flash-latest) to the newest lab model whose id has those tokens plus
    only version numbers or the vendor word. None when nothing fits."""
    toks = [t for t in re.split(r"[-_.]", re.sub(r"-latest$", "", alias.lower())) if t]
    if not toks:
        return None
    best = None
    for lid, m in cat["models"].items():
        if m.get("st") == "deprecated" or not _text_out(m):
            continue
        lt = [t for t in re.split(r"[-_.]", lid.split("/")[-1].lower()) if t]
        it = iter(lt)
        if not all(t in it for t in toks):
            continue
        extra = [t for t in lt if t not in toks]
        if all(re.match(r"^\d+$", t) or t in (lid.split("/")[0].lower(), "claude", "preview", "latest", "exp") for t in extra):
            key = (m.get("d") or "", -len(extra))
            if best is None or key > best[0]:
                best = (key, lid)
    return best[1] if best else None


# ════════════════════════════════════════════════════════════════════════════
# 4. records: the shape the scorer reads
# ════════════════════════════════════════════════════════════════════════════

def rec_of(cat, lab_id=None, entry=None, model_id=""):
    """A scoring record from a lab model and/or a provider entry (either may be None). The lab supplies benchmarks,
    parameters and the representative price; a standalone provider entry supplies its own attributes and price."""
    lab = cat["models"].get(lab_id) or {}
    own = entry if entry and not entry.get("c") else {}
    src = lab or own
    price = lab.get("p") or [entry.get("ci") if entry else None, entry.get("co") if entry else None]
    ims = src.get("im") or ["text"]
    return {"id": lab_id or model_id, "name": src.get("n") or model_id, "family": src.get("f"), "release": src.get("d") or "",
            "reasoning": bool(src.get("r")), "tool": bool(src.get("t")), "so": bool(src.get("so")),
            "ctx": (entry or {}).get("x") or src.get("x") or 0, "open": bool(src.get("ow")), "vision": "image" in ims,
            "price_in": price[0] or None, "price_out": price[1] or None, "pt": src.get("pt"), "pa": src.get("pa"),
            "bench": lab.get("b") or [], "status": (entry or {}).get("st") or src.get("st"), "lab": (lab_id or "").split("/")[0]}


def rec_local(name, params_b=None, ctx=0):
    """A record for a local model the catalog does not know: only its size (from Ollama's tag or the name) is known."""
    total, active = parse_params(name)
    return {"id": name, "name": name, "family": None, "release": "", "reasoning": bool(re.search(r"think|reason|r1|qwq", name.lower())),
            "tool": True, "so": False, "ctx": ctx, "open": True, "vision": False, "price_in": None, "price_out": None,
            "pt": params_b or total, "pa": active, "bench": [], "status": None, "lab": ""}


# ════════════════════════════════════════════════════════════════════════════
# 5. models.json
# ════════════════════════════════════════════════════════════════════════════

def models_path(home=None):
    return os.path.join(home or core.maestro_home(), MODELS_FILE)


def write_models(data, home=None):
    """Save models.json (atomic): one model per line, so the file stays readable and diffable. Returns the path or None."""
    path = models_path(home)
    dump = lambda v: json.dumps(v, separators=(",", ":"))  # noqa: E731
    head = ['"%s": %s' % (k, dump(v)) for k, v in data.items() if k != "models"]
    body = '{\n%s,\n"models": [\n%s\n]\n}\n' % (",\n".join(head), ",\n".join(dump(m) for m in data.get("models") or []))
    try:
        core.ensure_home(os.path.dirname(path))
        fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # owner-only before it is moved into place
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(path + ".tmp", path)
        return path
    except OSError:
        return None


def overrides_of(home, cfg=None):
    """`models_overrides` from the config: {model id: {"category": ..., "enabled": ...}}. Read from the effective config,
    else straight from config.json (load_config only keeps keys it knows)."""
    ov = (cfg or {}).get("models_overrides")
    if not isinstance(ov, dict):
        raw = core._read_json(core.config_path(home))
        ov = raw.get("models_overrides") if isinstance(raw, dict) else None
    return ov if isinstance(ov, dict) else {}


def apply_overrides(data, overrides):
    """Apply user overrides to a models.json dict in place: a valid `category` replaces the discovered one (kept in
    `category_auto`), `enabled` switches the model on or off; `override` records what was applied."""
    for m in data.get("models") or []:
        ov = overrides.get(m.get("id"))
        applied = {}
        if isinstance(ov, dict):
            if ov.get("category") in core.CATEGORIES:
                applied["category"] = ov["category"]
            if isinstance(ov.get("enabled"), bool):
                applied["enabled"] = ov["enabled"]
        if applied:
            m["category_auto"] = m.get("category")
            m.update(applied)
        m["override"] = applied or None
    return data


def load_models(home=None, cfg=None):
    """The discovered model map (~/.maestro/models.json) with the user's overrides applied; what the proxy and the UI read.
    {"version", "generated_at", "catalog_as_of", "hardware", "models": [...], "local_fit", "present": True}. When there
    is no (readable) file: the same shape with "models": [] and "present": False, so the caller can fall back."""
    home = home or core.maestro_home()
    data = core._read_json(models_path(home))
    if not (isinstance(data, dict) and data.get("version") == MODELS_VERSION and isinstance(data.get("models"), list)):
        return {"version": MODELS_VERSION, "generated_at": None, "catalog_as_of": None, "hardware": {}, "models": [],
                "local_fit": {}, "present": False}
    data["present"] = True
    data.setdefault("hardware", {})
    data.setdefault("local_fit", {})
    allow = bool((cfg if cfg is not None else core.load_config(home)).get("local_allow_load"))
    for m in data["models"]:   # installed but not loaded: listed, not routable (unless the user allows the server to load it)
        if isinstance(m, dict) and m.get("access") == "local":
            m["routable"] = m.get("loaded") is not False or allow
    return apply_overrides(data, overrides_of(home, cfg))
