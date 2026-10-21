"""Score: benchmarks and heuristics to a general strength G, per-job strengths and a category (Maestro v3, `discover`).

Slow-path only; pure functions over the catalog (no network, no files). Deterministic: the same catalog and records
always give the same numbers (the reference date is taken from the catalog, never from the clock).

  1. Every benchmark row is grouped into an axis (code, term, reason, agent, browse, data, write, longctx).
  2. A benchmark value becomes a percentile among the recent cohort (models released within 7 months of the newest one,
     at least 4 reporters); variants of a benchmark are merged by the median. A benchmark with no cohort is ignored.
     Benchmarks reported by a lab with fewer than 3 benchmarked models count half (self-reported, unverified).
  3. A prior from parameters, price, reasoning flag, name-tier words, recency and context. Older than 18 months: capped 0.40.
  4. G = w*bench + (1-w)*prior, w = n/(n+3), n = the (trust-weighted) number of benchmarks used. Axes are blended
     the same way with their own counts, and axes with a single benchmark weigh less in the average (small-n damping).
  5. Job strength per job from axis weights, with small adjustments (reasoning, tool use, context, coder names).
  6. Category: frontier >= .68, strong >= .52, standard >= .35, else fast (a weak-tier name needs .42 for standard).
     Specialist: a text model that is job-specific (coder / sql / research / translate naming) or clearly lopsided
     (a job strength 0.12 or more above G). Local models are category `local` with a `serves` tier from G (never above strong).
"""
from __future__ import annotations

import math
import re
import statistics

AXES = ("code", "term", "reason", "agent", "browse", "data", "write", "longctx")
JOBS = ("plan", "code", "debug", "review", "ops", "research", "data", "write", "edit", "summarize", "ask")
FRONTIER, STRONG, STANDARD, WEAK_STANDARD = 0.68, 0.52, 0.35, 0.42
COHORT_MONTHS, MIN_REPORTERS, OLD_MONTHS, OLD_PRIOR_CAP = 7, 4, 18, 0.40
LIFT = 0.12       # a job this far above G makes a model a specialist
UNTRUSTED_W = 0.5  # weight of benchmarks from labs with fewer than 3 benchmarked models
MIN_LAB_MODELS = 3

# (regex on the normalized benchmark name) -> axes; the first match wins
AXIS_RULES = (
    (r"^terminalbench|^sreb|^osworld|^automationbench|^cybergym|^exploit|^androidworld|^mcpmark", ("term",)),
    (r"^swe|^deepswe|^nl2repo|^frontiercode|^frontierswe|^programbench|^cursorbench|^livecodebench|^aiderpolyglot|"
     r"^artificialanalysiscoding|^qwen(swe|qoder|react)|^codeforces|^mimocode|^mimocoding|^kimicode|^apexbench|^recreation|"
     r"^vision2web|^sweatlas|^scicode", ("code",)),
    (r"^gpqa|^humanitys?lastexam|^hle|^frontiermath|^arcagi|^aime|^hmmt|^imo|^matharena|^critpt|^artificialanalysisintelligence|"
     r"^mmlupro|^mmlu$|^frontierscience|^usamo|^arxivmath|^horizonmath|^phybench|^superchem|^longcontextreasoning", ("reason",)),
    (r"^mcpatlas|^toolathlon|^tool.?decathlon|^tau|^claw|^agentslastexam|^apexagents|^gdpval|^agenticif|^skillsbench|"
     r"^workspacebench|^pinchbench|^wildclaw|^jobbench|^coworkbench|^frontierbench|^forte", ("agent",)),
    (r"^browsecomp|^widesearch|^deepsearch|^finsearch|^financeagent|^webarena", ("browse",)),
    (r"^spreadsheet|^dsbench|^mlebench|^mlsbench|^posttrain|^gene|^labbench|^biomystery|^ctirealm|^fin(first|craft)", ("data",)),
    (r"^ifbench|^ifeval|^healthbench|^prbench|^simpleqa|^chinesesimpleqa|^aaomniscience|^plaw|^harvey|^onemillion|"
     r"^translation|^jfbench|^fairpolitics", ("write",)),
    (r"^mrcr|^openaimrcr|^gdmmrcr|^graphwalks|^aalcr|^longbench|^corpusqa|^clbench", ("longctx",)),
)
_AXIS_RX = [(re.compile(rx), ax) for rx, ax in AXIS_RULES]
# benchmark families reported under several names: one key, so that variants are merged by the median
_ALIASES = ((r"^humanit.*lastexam.*|^hle.*", "hle"), (r"^toolathlonverified", "toolathlon"), (r"^clawevalmm", "clawevalmm"),
            (r"^clawevalv.*", "claweval"), (r"^aime.*", "aime"), (r"^mrcr.*|^openaimrcr|^gdmmrcr", "mrcr"),
            (r"^tau.*", "taubench"), (r"^osworld.*", "osworld"))

# job -> (axis weights, requirements)
JOB_AXES = {
    "plan": ({"reason": .35, "agent": .25, "code": .15, "longctx": .15, "browse": .10}, ("r",)),
    "code": ({"code": .55, "term": .20, "agent": .15, "reason": .10}, ("tool",)),
    "debug": ({"code": .30, "term": .25, "reason": .30, "longctx": .15}, ("r",)),
    "review": ({"code": .35, "reason": .30, "longctx": .20, "write": .15}, ()),
    "ops": ({"term": .50, "agent": .30, "code": .20}, ("tool",)),
    "research": ({"browse": .35, "reason": .30, "longctx": .15, "write": .10, "agent": .10}, ("r", "ctx")),
    "data": ({"data": .35, "code": .25, "reason": .25, "agent": .15}, ("so",)),
    "write": ({"write": .50, "reason": .20, "agent": .20, "longctx": .10}, ()),
    "edit": ({"code": .50, "write": .30, "agent": .20}, ()),
    "summarize": ({"longctx": .50, "write": .30, "reason": .20}, ("ctx",)),
    "ask": ({"reason": .50, "write": .30, "browse": .20}, ()),
}
CODEY_JOBS = ("code", "debug", "review", "edit", "ops")
_STRONG_KW = re.compile(r"opus|fable|mythos|astra|\bsol\b|-sol|ultra|-pro\b|pro-|-max|max-|large|405b|thinking|reasoner|-pro$")
_WEAK_KW = re.compile(r"mini|nano|lite|haiku|small|-air|luna|instant|tiny|-8b|-7b|-3b|-1b|spark|highspeed|fast|turbo|flash")
_SMALL_KW = re.compile(r"mini|nano|lite|haiku|small|-air|luna|instant|tiny|-[1-9]b\b|-1[0-4]b\b")
_FAST_KW = re.compile(r"flash|highspeed|fast|turbo")
_CODER_RX = re.compile(r"codex|coder|devstral|codestral|(?<![a-z])code(?![a-z])|-build")
_SPECIALIST_RX = re.compile(r"(?<![a-z])sql|computer-use|deep-research|translate|robotics")


# ════════════════════════════════════════════════════════════════════════════
# 1. benchmarks: keys, axes, medians, cohort percentiles
# ════════════════════════════════════════════════════════════════════════════

def bn(s):
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("τ", "tau").replace("’", "'"))


def axes_of(name):
    n = bn(name)
    for rx, ax in _AXIS_RX:
        if rx.search(n):
            return ax
    return ()


def bench_key(name):
    n = bn(name)
    if n.startswith("terminalbench"):
        n = re.sub(r"(20|21|v\d)$", "", n)
    for rx, key in _ALIASES:
        if re.match(rx, n):
            return key
    return n


def medians(bench):
    """{benchmark key: (median value, first name)} over a record's benchmark rows; rows with no axis are dropped."""
    d = {}
    for row in bench or []:
        name, score = row[0], row[1]
        if isinstance(score, (int, float)) and axes_of(name):
            d.setdefault(bench_key(name), []).append((score, name))
    return {k: (statistics.median(v for v, _ in vs), vs[0][1]) for k, vs in d.items()}


def _months(later, earlier):
    """Whole months from `earlier` to `later` (YYYY-MM[-DD] strings); 0 if either is missing."""
    try:
        return (int(later[:4]) - int(earlier[:4])) * 12 + int(later[5:7]) - int(earlier[5:7])
    except (ValueError, TypeError, IndexError):
        return 0


def _shift(date, months):
    y, m = int(date[:4]), int(date[5:7]) - months
    while m <= 0:
        m, y = m + 12, y - 1
    return "%04d-%02d" % (y, m)


def make_context(cat, now=None):
    """Everything the scorer derives from the whole catalog: the reference date, the cohort's value lists per benchmark
    key and which labs are trusted. `now` (YYYY-MM-DD) overrides the reference date (tests)."""
    models = cat["models"]
    ref = now or max([cat.get("as_of") or ""] + [m.get("d") or "" for m in models.values()])
    from_ = _shift(ref, COHORT_MONTHS)
    labs = {}
    med = {}
    bvals = {}
    for lid, m in models.items():
        md = medians(m.get("b"))
        med[lid] = md
        if md:
            labs[lid.split("/")[0]] = labs.get(lid.split("/")[0], 0) + 1
            if (m.get("d") or "")[:7] >= from_:
                for k, (v, _) in md.items():
                    bvals.setdefault(k, []).append(v)
    return {"ref": ref, "bvals": bvals, "trusted": {o for o, n in labs.items() if n >= MIN_LAB_MODELS}, "med": med, "cache": {}}


def percentile(ctx, key, v):
    """(rank within the recent cohort, cohort size); None when fewer than 4 recent models report the benchmark."""
    xs = ctx["bvals"].get(key, [])
    n = len(xs)
    if n < MIN_REPORTERS:
        return None
    below = sum(1 for x in xs if x < v) + 0.5 * sum(1 for x in xs if x == v)
    return (below + 1) / (n + 2)


# ════════════════════════════════════════════════════════════════════════════
# 2. the prior
# ════════════════════════════════════════════════════════════════════════════

def tier_words(mid, name=""):
    """+1 for strong-tier words (pro, opus, ultra...), -1 for small ones (mini, haiku, lite...), -0.5 for flash / fast / turbo."""
    s = ("%s %s" % (mid, name)).lower()
    t = 1.0 if _STRONG_KW.search(s) else 0.0
    if _SMALL_KW.search(s):
        t -= 1
    elif _FAST_KW.search(s):
        t -= 0.5
    return t


def prior(ctx, rec):
    """A guess of G from what is known without benchmarks: size (active and total), price, reasoning, name tier, recency,
    context. Calibrated so that it correlates with the benchmark-derived G of the benchmarked models."""
    from . import catalog
    t, a = rec.get("pt"), rec.get("pa")
    if not t:
        t, a2 = catalog.parse_params(rec["id"].split("/")[-1])
        a = a or a2
        if not t:
            t, a3 = catalog.parse_params(rec.get("name") or "")
            a = a or a3
    eff = math.sqrt(a * t) if a and t else (t or a)
    s_size = 0.22 + 0.17 * math.log10(max(eff, 1) / 3) if eff else None
    out = rec.get("price_out")
    s_price = 0.15 + 0.20 * math.log10(max(min(out, 40), 0.05) / 0.1) if out else None
    if s_size is not None and s_price is not None:
        s = 0.6 * s_size + 0.4 * s_price
    else:
        s = s_size if s_size is not None else s_price if s_price is not None else 0.35
    s += 0.04 * bool(rec.get("reasoning")) + 0.04 * tier_words(rec["id"].split("/")[-1], rec.get("name") or "")
    age = _months(ctx["ref"], rec["release"]) if rec.get("release") else None
    if age is not None:
        s += 0.05 if age <= 6 else 0 if age <= 12 else -0.05 if age <= 24 else -0.10
    if (rec.get("ctx") or 0) >= 1_000_000:
        s += 0.02
    p = max(0.05, min(0.90, 0.089 + 0.729 * s))
    return min(p, OLD_PRIOR_CAP) if age is not None and age > OLD_MONTHS else p


# ════════════════════════════════════════════════════════════════════════════
# 3. scoring a record
# ════════════════════════════════════════════════════════════════════════════

def tier_of_g(g, weak=False):
    return "frontier" if g >= FRONTIER else "strong" if g >= STRONG else "standard" if g >= STANDARD and not (weak and g < WEAK_STANDARD) else "fast"


def score_rec(ctx, rec, med=None):
    """The score of one record (see catalog.rec_of): {"G", "jobs", "basis", "n", "benchmarks", "prior", "axes", "axes8", "axes_n",
    "serves", "category", "specialist"}; axes8 is the blended per-axis strength and axes_n its weight (S1 v1.1; unused in v1). `category` is a cloud category or "specialist"; callers turn it into "local" for local models."""
    med = medians(rec.get("bench")) if med is None else med
    trust = 1.0 if rec.get("lab") in ctx["trusted"] else UNTRUSTED_W
    axv, axw, used = {}, {}, []
    for k, (v, name) in med.items():
        p = percentile(ctx, k, v)
        if p is None:
            continue
        used.append((p, name, trust))
        for a in axes_of(name):
            axv[a] = axv.get(a, 0.0) + p * trust
            axw[a] = axw.get(a, 0.0) + trust
    n = sum(t for _, _, t in used)
    ax = {a: axv[a] / axw[a] for a in axv}
    pr = prior(ctx, rec)
    rel = {a: axw[a] / (axw[a] + 1.5) for a in axw}  # a lone benchmark says less than three
    gb = sum(rel[a] * ax[a] for a in ax) / sum(rel.values()) if ax else pr
    wb = n / (n + 3.0)
    g = wb * gb + (1 - wb) * pr
    s = {a: (rel[a] * ax[a] + (1 - rel[a]) * g) if a in ax else g for a in AXES}
    mid = rec["id"].split("/")[-1].lower()
    coder = bool(_CODER_RX.search(mid))
    jobs = {}
    for j, (w, req) in JOB_AXES.items():
        v = sum(w[a] * s[a] for a in w) / sum(w.values())
        if coder and "code" not in ax and j in CODEY_JOBS:
            v += 0.08
        if "r" in req and not rec.get("reasoning"):
            v -= 0.08
        if "tool" in req and not rec.get("tool"):
            v *= 0.6
        if "so" in req and rec.get("so"):
            v += 0.03
        if "ctx" in req and rec.get("ctx"):
            v += max(-0.06, min(0.06, 0.05 * math.log10(rec["ctx"] / 128000)))
        jobs[j] = round(max(0.0, min(1.0, v)), 3)
    weak = tier_words(mid, rec.get("name") or "") <= -1
    serves = tier_of_g(g, weak)
    named = bool(coder or _SPECIALIST_RX.search(mid))
    lifted = max(jobs.values()) - g >= LIFT
    if re.search(r"deep-research", mid):
        jobs["research"] = round(min(1.0, jobs["research"] + 0.15), 3)
    top = sorted(used, key=lambda x: -x[0])[:8]
    return {"G": round(g, 3), "jobs": jobs, "basis": "bench" if n >= 6 else "blend" if n else "prior", "n": round(n, 1),
            "benchmarks": [name for _, name, _ in top], "prior": round(pr, 3), "axes": {a: round(v, 3) for a, v in ax.items()},
            "axes8": {a: round(v, 3) for a, v in s.items()}, "axes_n": {a: round(v, 2) for a, v in axw.items()},   # blended; logged for v1.1
            "serves": serves, "specialist": named or lifted, "category": "specialist" if named or lifted else serves}


def score_lab(ctx, cat, lab_id):
    """Cached score of a lab model (the same for every provider that serves it)."""
    if lab_id not in ctx["cache"]:
        from . import catalog
        ctx["cache"][lab_id] = score_rec(ctx, catalog.rec_of(cat, lab_id), ctx["med"].get(lab_id))
    return ctx["cache"][lab_id]


def local_serves(serves):
    """The cloud tier a local model can stand in for: what its G says, never above strong."""
    return "strong" if serves == "frontier" else serves


# ════════════════════════════════════════════════════════════════════════════
# 4. picking a model from a discovered map (shared by the backtest and the proxy)
# ════════════════════════════════════════════════════════════════════════════

TIER_ORDER = ("fast", "standard", "strong", "frontier")
FLOORS = {"fast": 0.0, "standard": STANDARD, "strong": STRONG, "frontier": FRONTIER}   # lower threshold of each category
OBJECTIVES = ("balanced", "cost", "quality")
DEFAULT_TOKENS = (2000, 500)   # assumed input / output tokens of a request when the caller does not know


def eff_category(m):
    """The tier a model stands in for: its category, or for `local` / `specialist` models what its G says."""
    return m["serves"] if m["category"] in ("local", "specialist") else m["category"]


def expected_cost(m, tokens_in=DEFAULT_TOKENS[0], tokens_out=DEFAULT_TOKENS[1]):
    """Dollars for one request at list price; subscription and local models cost 0 at the margin."""
    if m["access"] != "api" or (m.get("cost_in") is None and m.get("cost_out") is None):
        return 0.0
    return ((m.get("cost_in") or 0) * tokens_in + (m.get("cost_out") or 0) * tokens_out) / 1e6


def pick(candidates, job, category, objective="balanced", prefer_local=True, tools=False, vision=False, min_context=0,
         max_tier="frontier", tokens=DEFAULT_TOKENS, difficulty=None):
    """Rank the models of a discovered map (models.json entries) for one job at one ladder cell. Pure and offline: the caller
    supplies `candidates` (e.g. `catalog.load_models()["models"]`, whose quota state may be fresher than the file's).

    category    the ladder cell: fast | standard | strong | frontier | local (a local model if one qualifies, else fast)
    objective   balanced (local first if prefer_local, then cheapest, then strongest) | cost (cheapest first, strength breaks
                ties) | quality (strongest qualifier first, cheapest breaks ties)
    tools/vision/min_context  filters: tool-capable, image-capable, context at least this many tokens (unknown context passes)
    max_tier    cloud models above this tier are out (local models are exempt)
    tokens      (input, output) estimate for the cost ranking
    difficulty  0..3; the walk never steps down from the cell when it is 2 or more

    A model qualifies for a cell only if its strength for `job` reaches the lower threshold of that category (fast 0, standard .35,
    strong .52, frontier .68) AND its own tier (`eff_category`) is at least the cell: a job score alone cannot pull a standard-tier
    model into a strong cell. Disabled models, models on cooldown or exhausted, free tiers (`:free` ids and API models priced 0) and
    specialists that are not lifted for this job are out. Qualifiers of a higher tier are also qualifiers of a lower one (both tests
    only get stricter upward), so a cell with no qualifier has none above it either: the walk goes down (tiers below the cell,
    strongest first) only when `difficulty` is below 2 or unknown. If nothing qualifies at all, the last resort is the strongest
    model for the job whatever its tier, flagged `below_floor`; the caller decides whether to use it.
    Returns [{"id", "model", "tier", "strength", "cost", "reasons", "fallback", "below_floor"}] best first: the qualifiers of the
    first tier that has any, then those of the next tiers in walk order (`fallback` is the tier they qualified at, None for the
    first), or the last resort alone."""
    if objective not in OBJECTIVES:
        objective = "balanced"
    want_local = category == "local"
    tier = "fast" if want_local else category
    cap = TIER_ORDER.index(max_tier)
    i = TIER_ORDER.index(tier)
    pool = []
    for m in candidates:
        st = (m.get("quota") or {}).get("state")
        if not m.get("enabled", True) or st in ("cooldown", "exhausted") or m["model"].endswith(":free") or (
                m["access"] == "api" and m.get("cost_in") == 0 and m.get("cost_out") == 0):   # free tiers are rate-limited: not counted on
            continue
        if (tools and not m.get("tools_ok")) or (vision and not m.get("vision")):
            continue
        if min_context and m.get("context") and m["context"] < min_context:
            continue
        if m["access"] != "local" and TIER_ORDER.index(eff_category(m)) > cap:
            continue
        if m["category"] == "specialist" and m["jobs"].get(job, 0) < m["G"] + 0.03:
            continue
        pool.append(m)
    order = [tier]
    if difficulty is None or difficulty < 2:
        order += list(reversed(TIER_ORDER[:i]))
    ranked, seen, first = [], set(), None
    for t in order:
        ok = [m for m in pool if m["jobs"].get(job, 0) >= FLOORS[t] and TIER_ORDER.index(eff_category(m)) >= TIER_ORDER.index(t)
              and m["id"] not in seen]
        first = first or (t if ok else None)

        def key(m):
            q = m.get("quota") or {}
            head = q.get("headroom")
            cost = round(expected_cost(m, *tokens), 8)
            low = q.get("state") == "low"
            bonus = not (prefer_local and m["access"] == "local")
            spare = -(1.0 if head is None else head) if cost == 0 else 0.0   # free seats: the one with more headroom left
            s = -m["jobs"].get(job, 0)
            if objective == "quality":
                return (low, s, cost, spare, m["id"])
            if objective == "cost":
                return (low, cost, spare, s, m["id"])
            return (low, bonus, cost, spare, s, m["id"])
        for m in sorted(ok, key=key):
            cost = expected_cost(m, *tokens)
            why = ["%s %.2f >= %.2f for %s" % (job, m["jobs"].get(job, 0), FLOORS[t], t),
                   ("%s, no marginal cost" % ("local" if m["access"] == "local" else "subscription")) if cost == 0 else "about $%.4f per request" % cost]
            if t != tier:
                why.append("no model at %s; stepped down to %s" % (tier, t))
            if (m.get("quota") or {}).get("state") == "low":
                why.append("quota low")
            ranked.append({"id": m["id"], "model": m, "tier": t, "strength": m["jobs"].get(job, 0), "cost": cost, "reasons": why,
                           "fallback": None if t == first else t, "below_floor": False})
            seen.add(m["id"])
    if not ranked and pool:
        m = max(pool, key=lambda x: (x["jobs"].get(job, 0), -expected_cost(x, *tokens), x["id"]))
        why = ["nothing reaches %s >= %.2f for %s; the strongest for the job" % (job, FLOORS[tier], tier), "%s %.2f" % (job, m["jobs"].get(job, 0))]
        ranked.append({"id": m["id"], "model": m, "tier": eff_category(m), "strength": m["jobs"].get(job, 0), "cost": expected_cost(m, *tokens),
                       "reasons": why, "fallback": eff_category(m), "below_floor": True})
    return ranked
