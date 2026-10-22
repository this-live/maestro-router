"""Discover: detect -> list live models -> join the catalog -> score -> ~/.maestro/models.json (Maestro v3).

`maestro-router discover [--refresh-catalog] [--backtest] [--live N] [--local-fit] [--json]`. Slow-path: it lists models over the
network (providers.detect_all) and reads the bundled catalog; the router's `route` never imports this module.
Also here: the offline backtest of the eval prompts over the discovered map, the optional live latency probe, hardware
detection and the "what fits on your machine" advice. Maestro uses what is already served (loaded) and only suggests models
when nothing is served; a model a server has installed but not loaded is listed, never routed to. Nothing is downloaded, loaded
or launched; keys are never read here
(providers.py reads them to list models, quota.py and clients.py to call).
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import json
import os
import re
import statistics
import sys
import threading
import time

from . import catalog, score
from . import router as core

ACCESS_ORDER = {"api": 0, "subscription-cli": 1, "local": 2}
CLI_VENDOR = {"claude": "anthropic", "codex": "openai", "gemini": "google"}
TOKENS_IN, TOKENS_OUT = score.DEFAULT_TOKENS   # assumed per request; output grows with difficulty: OUT * (1 + difficulty)
EVAL_PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "evals", "prompts.jsonl")


# ════════════════════════════════════════════════════════════════════════════
# 1. hardware
# ════════════════════════════════════════════════════════════════════════════

def _gpu_name(env=None):
    import shutil
    env = os.environ if env is None else env
    if not shutil.which("nvidia-smi", path=env.get("PATH")):
        return ""
    out = core._run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return (out or "").strip().splitlines()[0].strip() if (out or "").strip() else ""


# memory bandwidth in GB/s (decode speed is bandwidth-bound); best effort, matched on the device name
GPU_BANDWIDTH = ((r"GB10|spark", 273), (r"RTX 5090", 1792), (r"RTX 5080", 960), (r"RTX 4090", 1008), (r"RTX 4080", 717),
                 (r"RTX 3090", 936), (r"B200", 8000), (r"H200", 4800), (r"H100", 3350), (r"A100.*40\s?GB", 1555), (r"A100", 2039),
                 (r"L40S", 864), (r"RTX 6000 Ada", 960))
APPLE_BANDWIDTH = {"M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800, "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
                   "M3": 100, "M3 Pro": 150, "M3 Max": 400, "M3 Ultra": 819, "M4": 120, "M4 Pro": 273, "M4 Max": 546}


def bandwidth_of(name, cpu=""):
    """GB/s for a device name (GPU table, or an Apple chip such as "Apple M3 Max"), or None when unknown."""
    m = re.search(r"\b(M\d)(?: (Pro|Max|Ultra))?\b", "%s %s" % (cpu, name))
    if m and re.search(r"apple|\bmac", "%s %s" % (cpu, name), re.I):
        return APPLE_BANDWIDTH.get(" ".join(x for x in m.groups() if x))
    for rx, bw in GPU_BANDWIDTH:
        if re.search(rx, name or "", re.I):
            return bw
    return None


def _cpu_brand():
    out = core._run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return (out or "").strip()


def describe_hardware(hw, gpu="", cpu=""):
    """{"name", "memory_gb", "kind", "bandwidth_gbps"} for a machine from core.detect_hardware()'s numbers, the GPU name and (Apple)
    the CPU brand string. kind: blackwell (DGX Spark / GB10, 128 GB unified memory), nvidia (VRAM), apple (unified memory), cpu
    (RAM). bandwidth_gbps is None when the device is not in the table (speed is then unknown)."""
    hw = hw or {}
    if re.search(r"GB10|spark", gpu, re.I):
        return {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": bandwidth_of("spark")}
    if hw.get("apple_silicon"):
        name = cpu if cpu.startswith("Apple") else "Apple Silicon"
        return {"name": name, "memory_gb": hw.get("ram_gb"), "kind": "apple", "bandwidth_gbps": bandwidth_of(name, cpu)}
    if hw.get("vram_gb"):
        return {"name": gpu or "NVIDIA GPU", "memory_gb": hw["vram_gb"], "bandwidth_gbps": bandwidth_of(gpu),
                "kind": "blackwell" if re.search(r"RTX 50|B200|B100|GB200|Blackwell", gpu) else "nvidia"}
    return {"name": "CPU", "memory_gb": hw.get("ram_gb"), "kind": "cpu", "bandwidth_gbps": None} if hw.get("ram_gb") else {}


def hardware_map(cfg, inv, env=None, gpu=None, cpu=None):
    """{box: {"name", "memory_gb", "kind", "bandwidth_gbps", "assumed"?}}: "this-machine" plus each configured endpoint. An
    endpoint's `hardware: {name, memory_gb, bandwidth_gbps?}` in the config describes a remote box (bandwidth is looked up by
    name when omitted); without it an endpoint whose name or host has "spark" is assumed to be a DGX Spark with 128 GB
    (marked assumed)."""
    env = os.environ if env is None else env
    out = {}
    hw0 = (inv or {}).get("hardware") or {}
    me = describe_hardware(hw0, _gpu_name(env) if gpu is None else gpu, (_cpu_brand() if hw0.get("apple_silicon") else "") if cpu is None else cpu)
    if me.get("memory_gb"):
        out["this-machine"] = me
    for raw in (cfg or {}).get("local_endpoints") or []:
        if not isinstance(raw, dict) or not isinstance(raw.get("url"), str):
            continue
        name = raw.get("name") if isinstance(raw.get("name"), str) else core._host_of(core._norm_url(raw["url"]))
        hw = raw.get("hardware")
        if isinstance(hw, dict) and isinstance(hw.get("memory_gb"), (int, float)):
            box = {"name": str(hw.get("name") or name), "memory_gb": hw["memory_gb"]}
            box["kind"] = hw.get("kind") if hw.get("kind") in ("blackwell", "nvidia", "apple", "cpu") else describe_hardware({}, box["name"]).get("kind") or (
                "blackwell" if re.search(r"spark|GB10|RTX 50|B200", box["name"], re.I) else "apple" if re.search(r"apple|mac", box["name"], re.I) else "nvidia")
            if box["name"] and re.search(r"spark|GB10", box["name"], re.I):
                box["kind"] = "blackwell"
            bw = hw.get("bandwidth_gbps")
            box["bandwidth_gbps"] = bw if isinstance(bw, (int, float)) and bw > 0 else bandwidth_of(box["name"])
            out[name] = box
        elif re.search(r"spark", name + " " + raw["url"], re.I):
            out[name] = {"name": "DGX Spark", "memory_gb": 128, "kind": "blackwell", "bandwidth_gbps": 273, "assumed": True}
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. the model map
# ════════════════════════════════════════════════════════════════════════════

def _quota(provider, model, home):
    try:
        from . import quota
        s = quota.status(provider, model, home)
        return {"state": s["state"], "headroom": s["headroom"]}
    except Exception:  # quota state is advice; never let it break discovery
        return {"state": "unknown", "headroom": None}


def _entry(pid, mid, access, style, rec, res, lab, how, pentry, home, extra=None, quota_id=None):
    """One models.json record (the contract in docs/DESIGN.md). `rec` merges the lab's and the provider's attributes."""
    cost_in = (pentry or {}).get("ci", rec["price_in"])
    cost_out = (pentry or {}).get("co", rec["price_out"])
    e = {"id": "%s/%s" % (pid, mid), "provider": pid, "model": mid, "access": access, "style": style,
         "tools_ok": bool(rec["tool"]) if lab else access != "local", "vision": bool(rec["vision"]),
         "context": rec["ctx"] or None, "cost_in": cost_in, "cost_out": cost_out,
         "G": res["G"], "jobs": res["jobs"], "category": res["category"], "serves": res["serves"], "basis": res["basis"],
         "benchmarks": res["benchmarks"], "specialist": res["specialist"], "catalog_id": lab, "join": how, "family": rec["family"],
         "release_date": rec["release"] or None, "quota": _quota(quota_id or pid, mid, home) if access != "local" else {"state": "unknown", "headroom": None},
         "enabled": True, "override": None, "axes8": res.get("axes8"), "axes_n": res.get("axes_n")}
    e.update(extra or {})
    return e


def build_models(cat, ctx, inv, cfg, home, hw=None):
    """[(models.json record, provider spec for calling it)] for every usable model of the provider inventory. `hw` (the
    hardware map) names the box a local endpoint runs on: the endpoint's own name when it is described there."""
    out = []
    for p in inv.get("providers") or []:
        if p.get("status") != "connected" or not p.get("models"):
            continue
        access, pid = p["access"], p["id"]
        if access == "local":
            ep = pid.split(":", 1)[1]
            for name in p["models"]:
                info = (p.get("model_info") or {}).get(name) or {}
                lab, how = catalog.resolve_local(cat, name, info.get("root"), info.get("params_b"))
                rec = catalog.rec_of(cat, lab) if lab else catalog.rec_local(name, info.get("params_b"), info.get("max_model_len") or 0)
                res = score.score_lab(ctx, cat, lab) if lab else score.score_rec(ctx, rec)
                mid = "local:%s/%s" % (ep, name)
                loaded = info.get("loaded", True)
                lrec = {"id": mid, "name": name, "endpoint_name": ep}
                pins, ov = core.local_pins(cfg, lrec), core._lookup(cfg.get("local_models") or {}, lrec)
                serves = (max(pins, key=core.RANK.get) if pins else ov["serves"] if isinstance(ov, dict) and ov.get("serves") in core.CELLS
                          else score.local_serves(res["serves"]))
                res = dict(res, category="local", serves=serves)
                e = _entry("local:" + ep, name, "local", "local", rec, res, lab, how, None, home,
                           {"id": mid, "provider": "local:" + ep, "context": info.get("max_model_len") or rec["ctx"] or None,
                            "cost_in": 0.0, "cost_out": 0.0, "box": _box_of(p, hw or {}),
                            "loaded": loaded, "routable": loaded or bool((cfg or {}).get("local_allow_load")),
                            "params_b": info.get("params_b") or rec["pt"]})
                e["tools_ok"] = bool(lab and rec["tool"])
                out.append((e, p))
        elif access == "subscription-cli":
            vendor = CLI_VENDOR.get(pid)
            for name in p["models"]:
                # Pi models are "provider/model" from Pi's own catalog: resolve scores on the bare model name, and
                # keep quota per Pi provider (a Grok seat and a MiniMax seat behind Pi are separate subscriptions).
                bare = name.rsplit("/", 1)[-1] if pid == "pi" else name
                lab, how, pent = (catalog.resolve(cat, vendor, name) if vendor else (None, "none", None))
                if not lab:
                    lab = catalog.resolve_alias(cat, bare)
                    how = "alias" if lab else "none"
                rec = catalog.rec_of(cat, lab, None, bare) if lab else catalog.rec_local(bare)
                res = score.score_lab(ctx, cat, lab) if lab else score.score_rec(ctx, rec)
                extra = {"id": "cli:%s/%s" % (pid, name), "tools_ok": False}
                qid = None
                if pid == "pi" and "/" in name:
                    qid = "pi-" + name.split("/", 1)[0]
                    extra["qp"] = qid
                e = _entry(pid, name, "subscription-cli", "cli", rec, res, lab, how, None, home, extra, quota_id=qid)
                out.append((e, pid))
        else:
            for name in p["models"]:
                if catalog.NON_CHAT_RX.search(name):
                    continue
                lab, how, pent = catalog.resolve(cat, pid, name)
                if pent and not catalog.text_only(pent.get("om")):
                    continue
                rec = catalog.rec_of(cat, lab, pent, name)
                res = score.score_lab(ctx, cat, lab) if lab else score.score_rec(ctx, rec)
                out.append((_entry(pid, name, "api", p["style"], rec, res, lab, how, pent, home), pid))
    return out


def _box_of(p, hw):
    name = p["id"].split(":", 1)[1]
    return name if name in hw else "this-machine" if core._is_this_machine(p.get("endpoint_url") or "") else name


# ════════════════════════════════════════════════════════════════════════════
# 3. what fits on a box
# ════════════════════════════════════════════════════════════════════════════

BYTES = {"BF16": 2.0, "FP8": 1.0, "NVFP4": 0.5625, "MXFP4": 0.5625, "Q4": 0.6}   # bytes per parameter (weights)
QUANT_LOSS = {"BF16": 0.0, "FP8": 0.005, "NVFP4": 0.02, "MXFP4": 0.0, "Q4": 0.03}   # taken off G when ranking
NATIVE_4BIT = (r"^openai/gpt-oss-",)   # released in 4-bit (MXFP4): that is the checkpoint to serve, not a re-quantization
QUANTS = {"blackwell": ("BF16", "FP8", "NVFP4", "Q4"), "nvidia": ("BF16", "FP8", "Q4"), "apple": ("Q4",), "cpu": ("Q4",)}
RUNTIME = 1.15     # +15% over the weights for activations and the runtime
USABLE = 0.90      # of a box's memory; the rest is the OS, other processes and fragmentation
CTX_FIT = 32768
FIT_KEEP = 25       # entries per box kept in models.json
OLLAMA_TAGS = ((r"^openai/gpt-oss-(20b|120b)$", "gpt-oss:%s"), (r"^alibaba/qwen3-(\d+b)(?:-a\d+b)?$", "qwen3:%s"),
               (r"^alibaba/qwen3-coder-30b-a3b-instruct$", "qwen3-coder:30b"), (r"^meta/llama-3\.1-(8b|70b)-instruct$", "llama3.1:%s"),
               (r"^meta/llama-3\.3-70b-instruct$", "llama3.3:70b"), (r"^google/gemma-3-(4b|12b|27b)-it$", "gemma3:%s"),
               (r"^mistral/devstral-small-2507$", "devstral:24b"), (r"^deepseek/deepseek-r1-distill-qwen-32b$", "deepseek-r1:32b"))


def mem_gb(params_b, quant):
    """Weights at this quantization plus 15% for the runtime, in GB."""
    return round(params_b * BYTES[quant] * RUNTIME, 1)


def kv_gb(params_b):
    """Rough KV-cache headroom for a 32K context (fp8 KV): about 1 GB plus 0.09 GB per billion parameters up to 120B."""
    return round(1.0 + 0.09 * min(params_b, 120), 1)


def ollama_tag(lab_id):
    for rx, tag in OLLAMA_TAGS:
        m = re.match(rx, lab_id or "")
        if m:
            return tag % m.groups() if "%s" in tag else tag
    return None


def serve_cmd(repo, quant, lab_id):
    """A command that would serve the model at this quantization (advice only; Maestro never runs it)."""
    if quant == "Q4":
        tag = ollama_tag(lab_id)
        return "ollama run %s" % tag if tag else "llama-server -hf %s-GGUF --ctx-size %d   # needs a Q4_K_M GGUF build" % (repo, CTX_FIT)
    if quant == "MXFP4":
        return "vllm serve %s --max-model-len %d   # released in MXFP4" % (repo, CTX_FIT)
    if quant == "NVFP4":
        return "vllm serve %s-NVFP4 --max-model-len %d   # needs an NVFP4 build of this model" % (re.sub(r"-(?:BF16|FP8)$", "", repo, flags=re.I), CTX_FIT)
    return "vllm serve %s%s --max-model-len %d" % (repo, " --quantization fp8" if quant == "FP8" else "", CTX_FIT)


DECODE_EFFICIENCY = 0.6   # share of the memory bandwidth a decoder reaches in practice


def decode_tps(bandwidth_gbps, active_b, quant):
    """Estimated single-stream decode speed (tokens/s): each token reads the active weights once, so it is
    bandwidth * 0.6 / (active parameters * bytes per parameter). None when the bandwidth is unknown."""
    if not bandwidth_gbps or not active_b:
        return None
    return bandwidth_gbps * DECODE_EFFICIENCY / (active_b * BYTES[quant])


def speed_factor(tps):
    """Ranking weight for a decode speed: under 10 tok/s 0.3, 10 to 25 0.7, 25 and up 1. Unknown speed does not penalize."""
    return 1.0 if tps is None or tps >= 25 else 0.7 if tps >= 10 else 0.3


def fit_row(lid, m, res, hw):
    """The best way to run one open-weight catalog model on a box, or None if no precision fits. Every precision that fits
    (weights * 1.15 + 32K KV headroom within 90% of memory) is scored as
    (mean job strength - quantization loss) * speed factor and the best one wins, ties going to the higher precision."""
    best = None
    active = m.get("pa") or m["pt"]
    quants = QUANTS.get(hw.get("kind"), QUANTS["cpu"])
    if any(re.match(rx, lid) for rx in NATIVE_4BIT):
        quants = ("MXFP4",) if hw.get("kind") in ("blackwell", "nvidia") else quants
    for q in quants:
        if mem_gb(m["pt"], q) + kv_gb(m["pt"]) > hw["memory_gb"] * USABLE:
            continue
        tps = decode_tps(hw.get("bandwidth_gbps"), active, q)
        fit = (sum(res["jobs"].values()) / len(res["jobs"]) - QUANT_LOSS[q]) * speed_factor(tps)
        if best is None or fit > best[0] + 1e-9:
            best = (fit, q, tps)
    if best is None:
        return None
    fit, q, tps = best
    return {"model": lid, "hf_repo": m["hf"], "params_b": m["pt"], "active_params_b": active, "active_known": bool(m.get("pa")), "quant": q,
            "mem_gb": mem_gb(m["pt"], q), "kv_gb": kv_gb(m["pt"]), "tps": None if tps is None else round(tps, 1), "tps_est": True,
            "fit_score": round(fit, 4), "G": res["G"], "jobs": res["jobs"], "category": res["serves"], "specialist": res["specialist"],
            "basis": res["basis"], "serve_cmd": serve_cmd(m["hf"], q, lid), "params_source": m.get("ps")}


def local_fit(cat, ctx, hardware, top=None):
    """{box: [best open-weight models that fit, best first]}. Candidates are the catalog's open-weight text models with a known
    parameter count, one per Hugging Face repo, each at its best precision for the box (see `fit_row`). Ranked by fit score: the
    mean job strength, less the quantization loss, times a speed factor from the estimated decode tokens/s (a dense 120B model
    on a 273 GB/s box crawls; a mixture-of-experts model with 5B active parameters flies). Active parameters that the catalog
    does not know are taken to be all of them (dense), which is the pessimistic case; `tps_est` is always true."""
    cands = []
    for lid, m in cat["models"].items():
        if not (m.get("ow") and m.get("pt") and m.get("hf")) or m.get("st") == "deprecated" or catalog.NON_CHAT_RX.search(lid):
            continue
        cands.append((lid, m, score.score_lab(ctx, cat, lid)))
    cands.sort(key=lambda c: (-c[2]["G"], c[0]))
    out = {}
    for box, hw in hardware.items():
        rows, seen = [], set()
        for lid, m, res in cands:
            key = catalog.strip_quant(m["hf"])
            row = None if key in seen else fit_row(lid, m, res, hw)
            if row:
                seen.add(key)
                rows.append(row)
        rows.sort(key=lambda r: (-r["fit_score"], r["mem_gb"], r["model"]))
        out[box] = rows[:top]
    return out


def served_boxes(models):
    """{box: {"serving": [local models a server has loaded, routable], "idle": [installed but not loaded, not routable]}}."""
    out = {}
    for m in models:
        if m["access"] == "local":
            out.setdefault(m.get("box") or "this-machine", {"serving": [], "idle": []})["serving" if m.get("routable", True) else "idle"].append(m)
    return out


def serving_vs_fit(models, fits, hardware):
    """Per box: what it already serves and where each of those models would rank among the fits (by fit score; a served model
    that is one of the fits uses its row, others their mean job strength with no speed penalty)."""
    out = {}
    for box in hardware:
        served = [m for m in models if m["access"] == "local" and m.get("box") == box and m.get("routable", True)]
        ranked = fits.get(box) or []
        rows = []
        for m in served:
            row = next((r for r in ranked if r["model"] == m.get("catalog_id")), None)
            s = row["fit_score"] if row else sum(m["jobs"].values()) / len(m["jobs"])
            rows.append({"id": m["id"], "model": m["model"], "catalog_id": m["catalog_id"], "G": m["G"], "serves": m["serves"],
                         "basis": m["basis"], "fit_score": round(s, 4), "tps": row["tps"] if row else None,
                         "would_rank": 1 + sum(1 for r in ranked if r["fit_score"] > s), "of": len(ranked)})
        out[box] = rows
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. discover
# ════════════════════════════════════════════════════════════════════════════

def stamp_theta(models, header=None):
    """S1's seat ability (spec §3.2): `theta` {job: [mean, sd]} for every scored model, from the link in s1.bin's header (else
    decide.json's), stamped with that link's hash (`theta_link`) so decide ignores it once the link changes. Additive keys."""
    from . import decide, s1
    if header is None:
        md = s1.load()[0]
        header = md["header"] if md else None
    prm = decide.params(header)
    for e in models:
        if isinstance(e.get("jobs"), dict) and e["jobs"]:
            seat = decide.seat_of({k: v for k, v in e.items() if k not in ("theta", "theta_link")})
            e["theta"] = {j: [round(x, 3) for x in decide.theta(seat, {j: 1.0}, prm)[:2]] for j in core.JOBS}
            e["theta_link"] = prm["link"]["sha"]
    return models


def discover(home=None, env=None, cfg=None, refresh_catalog=False, providers=None, catalog_data=None, probe_quota=True,
             hardware=None, fetch=None, write=True, inventory=None):
    """Run discovery and write models.json. Returns (data, notes, specs) where specs maps a model id to the provider spec
    that calls it (for --live). `providers` (the detect_all result), `inventory`, `catalog_data` and `hardware` are
    injectable for tests."""
    env = os.environ if env is None else env
    home = home or core.maestro_home(env)
    cfg = cfg if cfg is not None else core.load_config(home)
    notes = []
    inv = inventory if inventory is not None else core.ensure_inventory(home, env, cfg=cfg)
    if providers is None:
        from . import providers as P
        providers = P.detect_all(env=env, home=home, cfg=cfg, inventory=inv)
    if probe_quota:
        try:
            from . import quota
            quota.refresh(env=env, home=home)
        except Exception as e:
            notes.append("quota probes skipped: %s" % str(e)[:80])
    if catalog_data is None:
        cat, cnotes = catalog.load(home, refresh=refresh_catalog, fetch=fetch)
        notes += cnotes
    else:
        cat = catalog_data
    ctx = score.make_context(cat)
    hw = hardware if hardware is not None else hardware_map(cfg, inv, env)
    built = build_models(cat, ctx, providers, cfg, home, hw)
    old = {m["id"]: m for m in (core._read_json(catalog.models_path(home)) or {}).get("models", []) if isinstance(m, dict)}
    models, specs = [], {}
    for e, spec in built:
        if old.get(e["id"], {}).get("latency"):
            e["latency"] = old[e["id"]]["latency"]  # measured by --live; kept until measured again
        models.append(e)
        specs[e["id"]] = spec
    models.sort(key=lambda m: (ACCESS_ORDER[m["access"]], m["provider"], -m["G"], m["model"]))
    ranked = local_fit(cat, ctx, hw) if hw else {}
    fits = {box: rows[:FIT_KEEP] for box, rows in ranked.items()}
    for box, sv in served_boxes(models).items():   # a box that serves models: every suggestion would replace them
        ids = [m["id"] for m in sv["serving"]]
        for r in (fits.get(box) or []) if ids else ():
            r.update(replaces=ids, serve_cmd="# stop the current server first: this replaces what %s serves, it does not add to it\n%s" % (box, r["serve_cmd"]))
    stamp_theta(models)
    data = {"version": catalog.MODELS_VERSION, "generated_at": int(time.time()), "catalog_as_of": cat.get("as_of"),
            "catalog_source": cat.get("source"), "catalog_sha": catalog.snapshot_sha(), "hardware": hw, "models": models, "local_fit": fits,
            "local_serving": serving_vs_fit(models, ranked, hw), "notes": notes,
            "providers": {p["id"]: {"status": p["status"], "reason": p.get("reason", "")} for p in providers.get("providers") or []
                          if p["status"] != "missing"}}
    if write:
        path = catalog.write_models(data, home)
        if not path:
            notes.append("could not write %s" % catalog.models_path(home))
    return data, notes, specs


# ════════════════════════════════════════════════════════════════════════════
# 5. backtest and live probe
# ════════════════════════════════════════════════════════════════════════════

eff_category = score.eff_category
expected_cost = score.expected_cost


def load_prompts(path=None):
    path = path or EVAL_PROMPTS
    if not os.path.exists(path):
        alt = os.path.join(os.getcwd(), "evals", "prompts.jsonl")
        path = alt if os.path.exists(alt) else path
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("job") in core.JOBS and r.get("difficulty") in (0, 1, 2, 3):
                    rows.append(r)
    return rows


def backtest(models, rows, cfg=None, classify=None, tools_needed=None):
    """Route every eval prompt over the map, offline. The classifier is `classify_rules` (what ships); the expected category
    is the ladder cell of the labelled job and difficulty. Reports:
      category_hit    the chosen model's category equals the expected one (within1, under_routed, severe_under: as in evals/run.py)
      cost_per_100    expected cost of 100 requests routed vs always using the strongest model (tokens assumed:
                      2,000 in, 500 x (1+difficulty) out; subscription and local models cost 0 at the margin)
      local_share     fraction of requests that go to a local model"""
    cfg = cfg if cfg is not None else core.load_config()
    classify = classify or core.classify_rules
    tools_needed = tools_needed or core.TOOL_JOBS
    live = [m for m in models if m.get("enabled", True) and m.get("routable", True)]
    api = [m for m in live if m["access"] == "api" and m["category"] != "specialist"] or [m for m in live if m["access"] == "api"]
    strongest = max(api, key=lambda m: (m["G"], expected_cost(m))) if api else None
    n = hits = w1 = under = severe = req_hit = local = 0
    cost = cost_top = 0.0
    used, lost, fallbacks = {}, 0, 0
    for r in rows:
        cls = classify(r["prompt"])
        cell = core.target_tier(cls, cfg)
        exp = core.tier_of(core.target_tier(core.Classification(r["job"], r["difficulty"], 1.0), cfg))
        ranked = score.pick(live, cls.job, cell, tools=cls.job in tools_needed, max_tier=cfg["max_tier"], prefer_local=cfg["prefer_local"],
                            tokens=(TOKENS_IN, TOKENS_OUT * (1 + cls.difficulty)), difficulty=cls.difficulty)
        fallbacks += bool(ranked and ranked[0]["below_floor"])
        m = ranked[0]["model"] if ranked else None
        n += 1
        if not m:
            lost += 1
            continue
        got = eff_category(m)
        d = core.RANK[got] - core.RANK[exp]
        hits += d == 0
        w1 += abs(d) <= 1
        under += d < 0
        severe += d <= -2
        req_hit += got == core.tier_of(cell)
        local += m["access"] == "local"
        toks = (TOKENS_IN, TOKENS_OUT * (1 + cls.difficulty))
        cost += expected_cost(m, *toks)
        cost_top += expected_cost(strongest, *toks) if strongest else 0.0
        used[m["id"]] = used.get(m["id"], 0) + 1
    served = max(n - lost, 1)
    return {"prompts": n, "unroutable": lost, "fallbacks": fallbacks, "category_hit": round(hits / served, 4), "within_1": round(w1 / served, 4),
            "under_routed": round(under / served, 4), "severe_under_routed": round(severe / served, 4),
            "requested_tier_served": round(req_hit / served, 4), "local_share": round(local / served, 4),
            "cost_per_100": round(cost / served * 100, 4), "cost_per_100_strongest": round(cost_top / served * 100, 4),
            "strongest": strongest["id"] if strongest else None,
            "saving": round(1 - cost / cost_top, 4) if cost_top else None,
            "models_used": dict(sorted(used.items(), key=lambda kv: -kv[1])[:12])}


def probeable(data, ids):
    """The ids of `ids` that may be sent a request: not a local model that is installed but not loaded."""
    ok = {m["id"] for m in data["models"] if m.get("routable", True)}
    return [i for i in ids if i in ok]


def live_probe(data, specs, ids, n, rows, cfg, env=None, home=None, workers=6, request_fn=None):
    """Send `n` short eval prompts to each model in `ids` (max_tokens 16, streamed) and record time to first byte, total time
    and failures under each model's `latency`. Costs a few tokens per call. Grading answers is not done here.
    Only routable models are probed (a local model that is installed but not loaded would be loaded by the request), and a
    local endpoint gets one request at a time."""
    from . import clients
    env = os.environ if env is None else env
    home = home or core.maestro_home(env)
    prompts = [r["prompt"][:400] for r in rows[:: max(1, len(rows) // max(n, 1))][:n]]
    ids = probeable(data, ids)
    locks = {m["provider"]: threading.Lock() for m in data["models"] if m["access"] == "local"}   # one request at a time per endpoint

    def one(mid):
        m = next(x for x in data["models"] if x["id"] == mid)
        with locks.get(m["provider"]) if m["access"] == "local" else contextlib.nullcontext():
            return probe_one(m, mid)

    def probe_one(m, mid):
        ttft, tot, fail, err = [], [], 0, ""
        for text in prompts:
            req = {"messages": [{"role": "user", "content": text}], "max_tokens": 16, "stream": True}
            t0 = time.perf_counter()
            try:
                status, _, body = (request_fn or clients.complete)(specs[mid], m["model"], req, stream=True, env=env, home=home, cfg=cfg, timeout=60.0)
                if status >= 400 or status == 0:
                    raise RuntimeError("HTTP %s" % status)
                it = iter(body)
                next(it)
                t1 = time.perf_counter()
                for _ in it:
                    pass
                ttft.append((t1 - t0) * 1000)
                tot.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                fail += 1
                err = str(e)[:60]
        return mid, {"n": len(prompts), "fail": fail, "ttft_ms": round(statistics.median(ttft)) if ttft else None,
                     "total_ms": round(statistics.median(tot)) if tot else None, "error": err, "at": int(time.time())}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(one, ids))
    for m in data["models"]:
        if m["id"] in results:
            m["latency"] = results[m["id"]]
    return results


# ════════════════════════════════════════════════════════════════════════════
# 6. rendering and the command
# ════════════════════════════════════════════════════════════════════════════

def _top_jobs(m, k=3):
    return " ".join("%s %.2f" % (j, v) for j, v in sorted(m["jobs"].items(), key=lambda kv: -kv[1])[:k])


def summary_lines(data, per_provider=12):
    """The readable table: by provider, the strongest models with category, G, top jobs and the basis of the score."""
    lines = []
    by = {}
    for m in data["models"]:
        by.setdefault(m["provider"], []).append(m)
    for prov, ms in by.items():
        acc = {"api": "api key", "subscription-cli": "subscription CLI", "local": "local"}[ms[0]["access"]]
        cats = {}
        for m in ms:
            cats[m["category"]] = cats.get(m["category"], 0) + 1
        lines.append("%s (%s, %d models: %s)" % (prov, acc, len(ms), ", ".join("%d %s" % (v, k) for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))))
        shown = sorted(ms, key=lambda m: (-m["G"], m["model"]))[:per_provider]
        for m in shown:
            cat = m["category"] if m["category"] not in ("local", "specialist") else "%s>%s" % (m["category"], m["serves"])
            lines.append("  %-38s %-16s G %.2f  %-26s %s%s" % (m["model"][:38], cat, m["G"], _top_jobs(m), m["basis"],
                                                              ("" if m["catalog_id"] else " (not in catalog)")
                                                              + ("" if m.get("routable", True) else " (installed, not loaded: not used)")))
        if len(ms) > len(shown):
            lines.append("  ... %d more (maestro-router models, or discover --all)" % (len(ms) - len(shown)))
    return lines or ["no models found: add a key (`maestro-router keys set <provider>`) or point Maestro at a model server you already run"]


def _speed(r):
    if r.get("tps") is None:
        return "?"
    return "~%d%s" % (round(r["tps"]), "" if r.get("active_known") else "*")


def serving_lines(data):
    """One line per box: what it serves (Maestro uses exactly those), or that nothing is served and `--local-fit` has advice."""
    boxes, hw = served_boxes(data["models"]), data.get("hardware") or {}
    lines = []
    for box in list(hw) + [b for b in boxes if b not in hw]:
        sv = boxes.get(box) or {"serving": [], "idle": []}
        names = [m["model"] for m in sv["serving"]]
        if names:
            lines.append("%s: serving %d model%s (%s) - using these" % (
                box, len(names), "s"[:len(names) != 1], ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")))
        else:
            lines.append("%s: nothing served%s; `maestro-router discover --local-fit` lists models that would fit (advice only)" % (
                box, " (%d installed but not loaded, ignored: Maestro never loads models)" % len(sv["idle"]) if sv["idle"] else ""))
    return lines


def fit_lines(data, top=10):
    lines = []
    boxes = served_boxes(data["models"])
    for box, hw in data["hardware"].items():
        rows = (data.get("local_fit") or {}).get(box) or []
        serving = [m["model"] for m in (boxes.get(box) or {}).get("serving", [])]
        lines.append("")
        if serving:   # one model server usually claims most of the memory, so these replace what is served
            lines.append("Alternatives on %s (would REPLACE what it serves now: %s): %s, %s GB%s%s" % (
                box, ", ".join(serving[:4]) + (", ..." if len(serving) > 4 else ""), hw["name"], "%g" % hw["memory_gb"],
                ", %g GB/s" % hw["bandwidth_gbps"] if hw.get("bandwidth_gbps") else ", memory speed unknown",
                " (assumed: set local_endpoints[].hardware to change)" if hw.get("assumed") else ""))
            lines.append("  The memory is in use: a model server usually claims most of it (vLLM reserves --gpu-memory-utilization, 0.9 by default), "
                         "so none of these fits alongside what is served. Sizes below assume the current server is stopped. Maestro will not stop it.")
        else:
            lines.append("Could run on %s: %s, %s GB%s%s; usable about %.0f GB" % (
                box, hw["name"], "%g" % hw["memory_gb"], ", %g GB/s" % hw["bandwidth_gbps"] if hw.get("bandwidth_gbps") else ", memory speed unknown",
                " (assumed: set local_endpoints[].hardware to change)" if hw.get("assumed") else "", hw["memory_gb"] * USABLE))
        for s in (data.get("local_serving") or {}).get(box) or []:
            lines.append("  serving now: %-34s G %.2f (%s, %s)%s, would rank #%d of %d that fit" % (
                s["model"][:34], s["G"], s["serves"], s["basis"], ", ~%d tok/s" % round(s["tps"]) if s.get("tps") else "", s["would_rank"], s["of"]))
        if not rows:
            lines.append("  nothing in the catalog fits")
            continue
        lines.append("  %-3s %-34s %7s %-6s %6s %6s %5s  %-12s %s" % ("#", "model", "params", "quant", "mem GB", "tok/s", "G", "category", "best jobs"))
        for i, r in enumerate(rows[:top], 1):
            lines.append("  %-3d %-34s %6.0fB %-6s %6.1f %6s %5.2f  %-12s %s" % (
                i, r["model"][:34], r["params_b"], r["quant"], r["mem_gb"] + r["kv_gb"], _speed(r), r["G"],
                r["category"] + ("*" if r["specialist"] else ""), _top_jobs(r, 2)))
            lines += ["      " + x for x in r["serve_cmd"].split("\n")]
        lines.append("  (mem GB includes 32K KV headroom. tok/s is an estimate of single-stream decode: bandwidth x 0.6 / active weights; * = active "
                     "parameters unknown, so all weights assumed (dense), MoE models are faster. Ranked by mean job strength less a quantization "
                     "loss, x0.3 under 10 tok/s and x0.7 under 25. Category * = job-specific. Advice only, nothing is downloaded or started)")
    return lines


def backtest_lines(bt):
    return ["backtest over %d eval prompts (offline, classifier = rules):" % bt["prompts"],
            "  category hit %.1f%%   within one %.1f%%   under-routed %.1f%%   severe %.1f%%   requested tier served %.1f%%" % (
                bt["category_hit"] * 100, bt["within_1"] * 100, bt["under_routed"] * 100, bt["severe_under_routed"] * 100,
                bt["requested_tier_served"] * 100),
            "  cost per 100 requests: $%.4f routed vs $%.4f always %s%s" % (
                bt["cost_per_100"], bt["cost_per_100_strongest"], bt["strongest"],
                "  (saves %.0f%%)" % (bt["saving"] * 100) if bt["saving"] is not None else ""),
            "  local share %.1f%%   most used: %s" % (bt["local_share"] * 100, ", ".join("%s x%d" % kv for kv in list(bt["models_used"].items())[:4]))]


def systemone_strategies(data, home, cfg):
    """`discover --live`: which /v1/systemone delegation strategy each loaded local model passes (spec §11.3), stored as
    models.json `systemone_strategy`. Probes only models that are already loaded (Maestro never loads one)."""
    from . import systemone
    out = []
    for e in data["models"]:
        if e["access"] == "local" and e.get("routable", True) and e.get("loaded", True) is not False:
            try:
                st = systemone.strategy_of(e["id"], {"home": home, "cfg": cfg})
            except Exception:  # a probe that breaks must not break discovery
                st = None
            e["systemone_strategy"] = st
            out.append("  systemone %-36s %s" % (e["id"][:36], st or "no strategy passes"))
    return ["", "systemone delegate probe (local models):"] + out if out else []


def models_view(home=None, cfg=None):
    """(data, lines) for `maestro-router models` from models.json, or None when discovery has not run or found nothing (the caller
    falls back to the harness table)."""
    home = home or core.maestro_home()
    data = catalog.load_models(home, cfg)
    if not data["models"]:
        return None
    age = (time.time() - (data.get("generated_at") or 0)) / 3600
    lines = ["discovered %s (%s ago), catalog as of %s; `maestro-router discover` refreshes it" % (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(data["generated_at"] or 0)), "%.0f h" % age if age >= 1 else "<1 h", data.get("catalog_as_of"))]
    pruned = dict(data, models=[m for m in data["models"] if m.get("enabled", True)])
    lines += summary_lines(pruned)
    off = [m["id"] for m in data["models"] if not m.get("enabled", True)]
    ovr = [m["id"] for m in data["models"] if m.get("override")]
    if off:
        lines.append("disabled by you: " + ", ".join(off[:8]) + (" ..." if len(off) > 8 else ""))
    if ovr:
        lines.append("overridden in config (models_overrides): %d" % len(ovr))
    lines.append("`maestro-router models --seats` shows the per-harness table")
    return data, lines


def cmd_discover(a):
    home = core.maestro_home()
    cfg = core.load_config(home)
    data, notes, specs = discover(home=home, cfg=cfg, refresh_catalog=a.refresh_catalog)
    out = {}
    lines = []
    if not a.json:
        lines += ["discovered %d models from %d providers; wrote %s" % (
            len(data["models"]), len({m["provider"] for m in data["models"]}), catalog.models_path(home))]
        lines += notes
        lines += summary_lines(data, per_provider=10**6 if a.all else 12)
        lines += [""] + serving_lines(data) if data["hardware"] or any(m["access"] == "local" for m in data["models"]) else []
    if a.local_fit and not a.json:
        lines += fit_lines(data) if data["hardware"] else ["", "no hardware detected; describe a box with local_endpoints[].hardware"]
    rows = None
    if a.backtest or a.live:
        try:
            rows = load_prompts(a.prompts)
        except (OSError, ValueError) as e:
            print("maestro: cannot read the eval prompts (%s); pass --prompts <file>" % e, file=sys.stderr)
            return 2
    if a.backtest:
        out["backtest"] = backtest(data["models"], rows, cfg)
        lines += [""] + backtest_lines(out["backtest"])
    if a.live:
        ids = probeable(data, out.get("backtest", {}).get("models_used") or [m["id"] for m in data["models"][:8]])
        lines += ["", "live probe: %d prompt(s) to each of %d model(s) (max_tokens 16; costs a few tokens each)" % (a.live, len(ids))]
        res = live_probe(data, specs, ids, a.live, rows, cfg)
        lines += systemone_strategies(data, home, cfg)
        catalog.write_models(data, home)
        out["live"] = res
        for mid, r in res.items():
            lines.append("  %-44s ttft %s ms  total %s ms  failed %d/%d %s" % (
                mid[:44], r["ttft_ms"], r["total_ms"], r["fail"], r["n"], r["error"]))
    if a.json:
        print(json.dumps(dict(data, **out), indent=2))
    else:
        print("\n".join(lines))
    return 0
