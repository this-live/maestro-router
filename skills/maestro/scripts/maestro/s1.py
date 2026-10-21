"""S1, Maestro's tiny local decision model (S1 spec §2, App. A): inference only, stdlib, no network. text -> normalize ->
crc32 features (Bloom known-feature filter) -> int8 folded embedding bag (SWAR) -> ReLU -> 24 outputs -> calibrated heads
-> TaskPosterior (a dict). Weights: data/s1.bin (evals/train_s1.py). Without usable weights the NB adapter answers
(`s1_ignored` says why); heads the weights were not trained on come from job priors (`trained` lists the others)."""
from __future__ import annotations

import json
import math
import mmap
import os
import re
import struct
import zlib
from operator import add, mul

from . import router as R

FEAT_VERSION = 1
FORMAT = 1
MAGIC = b"MS1\x00"
K = 24
SEED2 = 0x5BD1E995
BIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "s1.bin")
CLOSED = frozenset("lksmhxg")   # closed-vocabulary namespaces: in the Bloom filter once seen in training
CAPS = {"w": 256, "b": 255, "i": 64, "j": 128, "l": 2, "k": 12, "s": 10, "m": 6, "n": 32, "p": 48, "h": 6, "x": 6,
        "g": 3, "c": 300}
# Heads S1 has no weights for, and the NB and Jev adapters, use these (App. B tables; the rest are placeholders).
# data/decide.json's job_priors and nb_adapter override them when present.
PRIORS = {"tools": {"code": 1, "debug": 1, "review": 1, "ops": 1, "research": 0.6, "edit": 0.8},
          "detectable": {"code": 0.75, "debug": 0.75, "ops": 0.75, "edit": 0.8, "data": 0.6, "ask": 0.5,
                         "summarize": 0.5, "review": 0.4, "plan": 0.3, "write": 0.3, "research": 0.3},
          "tok_out": {"ask": 250, "edit": 300, "summarize": 400, "write": 1200, "plan": 1500, "code": 1200,
                      "debug": 900, "review": 900, "ops": 700, "research": 1000, "data": 800},
          "c": [-2.0, -0.5, 1.0, 2.5], "sd": 0.8, "self_contained": 0.5, "stakes": 1.0, "ambiguity": 0.5,
          "tok_sd": 0.8, "turns": [1.79, 0.8]}
SIZES = {"s": (300, 1), "m": (1000, 4), "l": (3000, 12), "xl": (8000, 30)}   # --size: output tokens, agent turns
_TOK = re.compile(r"https?://\S+|[^\W_][\w+#.'/-]*", re.I)
_NUM = re.compile(r"[0-9][0-9.,]*[a-z%+]{0,2}$")
_FILE = re.compile(r"[\w/.-]+\.[a-z]{1,4}$")
_IDSPLIT = re.compile(r"[_./-]+|(?<=[a-z0-9])(?=[A-Z])")
_CJK = ("[\u0e00-\u0e7f\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
                  "\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f]+")   # compiled on the first non-ASCII text (1.4 ms)
_LINE = re.compile(r"(?P<log>\s*\[?\d{4}-\d\d-\d\d[ T]\d\d:\d\d)"
                   r"|(?P<stacktrace>\s+at [\w.$<>]+\(|\s*File \".*\", line \d+|Traceback|panic:|error\[E\d+\])"
                   r"|(?P<diff>@@|\+\+\+ |--- |[+-][^+-])|(?P<json>\s*[{}\[\]]|\s*\"[^\"]*\"\s*:)|(?P<yaml>\s*[\w.-]+:\s)")
_LANG = {"py": "python", "js": "javascript", "ts": "typescript", "sh": "shell", "bash": "shell", "rb": "ruby", "rs": "rust",
         "yml": "yaml", "xml": "html", "patch": "diff", "text": "", "txt": "", "plaintext": "", "log": ""}
_TYPES = frozenset("json yaml sql csv html diff log stacktrace code prose".split())
_S1 = {}     # path -> (model or None, why)
_PRI = []
_b2 = lambda n: int(math.log2(1 + max(0, n)))  # noqa: E731   (a log2 bucket)
_sig = lambda v: 0.5 * (1.0 + math.tanh(0.5 * v))  # noqa: E731


def _lk(ln):
    m = _LINE.match(ln)
    return m.lastgroup if m else "csv" if ln.count(",") >= 3 else None


def _sym(s):
    r = sum(s.count(c) for c in "{}[]()<>;=$&|\\`") / max(1, len(s))
    return 0 if r < .01 else 1 if r < .03 else 2 if r < .06 else 3 if r < .12 else 4


def _runs(lines):
    """(start, end, type) of runs of >= 5 lines with the same material pattern; indented lines continue a stack or diff."""
    out, kind, start = [], None, 0
    for i, ln in enumerate(lines + ["\0"]):
        k = _lk(ln)
        if k == kind or (kind in ("stacktrace", "diff") and k is None and ln[:1] in " \t" and ln.strip()):
            continue
        if kind and i - start >= 5:
            out.append((start, i, kind))
        kind, start = k, i
    return out


def tokens(s, head=256, tail=0):
    """(tokens, identifier parts, file extensions, raw count) of the first `head` and last `tail` raw tokens of s:
    lowercased, trailing .,;:!?'" stripped, numbers, file names and URLs mapped to <n>, <f>, <u>."""
    raw = _TOK.findall(s)
    toks, idents, exts, n = [], [], [], len(raw)
    for t in raw if n <= head + tail else raw[:head] + raw[n - tail:]:
        lo = t.lower()
        if lo.startswith(("http://", "https://")):
            toks.append("<u>")
            continue
        t, lo = t.rstrip(".,;:!?'\""), lo.rstrip(".,;:!?'\"")
        if lo[:1].isdigit() and _NUM.match(lo):
            lo = "<n>"
        elif "." in lo and _FILE.match(lo):
            exts.append(lo.rsplit(".", 1)[1])
            lo = "<f>"
        toks.append(lo)
        parts = _IDSPLIT.split(t) if len(t) > 4 and not (t.isalnum() and t.islower()) else ()
        idents.extend(p.lower() for p in parts if len(p) >= 2 and len(parts) > 1)   # snake_case, a.b, paths, camelCase
    return toks, idents, exts, n


def normalize(text, prev_user=""):
    """Split out material (``` fences by str.split, then unfenced runs), apply the scan caps and tokenize (spec §2.2). Fences are
    typed from their first 8 KB, and only the first 32 fences and those starting in the first 10 KB of material (§2.8 budget)."""
    parts, types, size = (text or "").split("```"), [], 0
    instr, fences = parts[0::2] + [""] * (len(parts) % 2 == 0), len(parts) // 2   # an unclosed last fence runs to the end
    mat = [b[b.find("\n") + 1:] for b in parts[1::2]]                              # a fence's first line is its tag
    for k, blk in enumerate(mat[:32]):
        if size > 10240:
            break
        nl, size, blk = parts[2 * k + 1].find("\n"), size + len(blk), blk[:8192]
        tag = parts[2 * k + 1][:nl].strip().lower() if nl >= 0 else ""
        tag = _LANG.get(tag, tag) if len(tag) <= 20 and " " not in tag else ""
        if not tag:   # untagged: the line pattern of most of its first 20 lines, else code or prose
            ks = [x for x in map(_lk, blk.split("\n", 20)[:20]) if x]
            top = max(set(ks), key=ks.count) if ks else None
            tag = top if top and 2 * ks.count(top) >= min(20, blk.count("\n") + 1) else "code" if _sym(blk) else "prose"
        types.append(tag if tag in _TYPES else "code:" + tag)
    instr = " ".join(instr)
    if instr.count("\n") >= 4:
        lines = instr.split("\n", 400)
        keep = [True] * len(lines)
        for a, b, kind in _runs(lines[:400]):
            types.append(kind)
            mat.append("\n".join(lines[a:b]))
            keep[a:b] = [False] * (b - a)
        instr = "\n".join(ln for ln, k in zip(lines, keep) if k)
    mat, instr = "\n".join(mat), instr if len(instr) <= 5120 else instr[:4096] + "\n" + instr[-1024:]
    cjk = re.findall(_CJK, instr) if not instr.isascii() else []
    toks, idents, exts, n = tokens(re.sub(_CJK, " ", instr) if cjk else instr, 192, 64)
    return {"instr": instr, "toks": toks, "idents": idents, "ntok": n + sum(map(len, cjk)), "exts": exts, "cjk": cjk, "fences": fences, "chars": len(text or ""),
            "mat": mat if len(mat) <= 10240 else mat[:8192] + "\n" + mat[-2048:], "mat_chars": len(mat),
            "types": list(dict.fromkeys(types)), "prev": (prev_user or "")[-1024:]}


def _script(s):
    """g features: the majority script of the first 400 chars, `mixed`, and the ASCII share bucket."""
    s, cnt = s[:400], {}
    if s.isascii():
        return ["latin", "ascii=4"]
    for o in map(ord, filter(str.isalpha, s)):
        k = ("latin" if o < 0x250 or 0x1E00 <= o < 0x1F00 else "cyrillic" if 0x400 <= o < 0x530 else
             "arabic" if 0x600 <= o < 0x780 else "devanagari" if 0x900 <= o < 0x980 else "cjk"
             if 0x3040 <= o < 0xA000 or 0xAC00 <= o < 0xD7B0 or 0xF900 <= o < 0xFB00 or 0x1100 <= o < 0x1200 else "other")
        cnt[k] = cnt.get(k, 0) + 1
    top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) or [("other", 0)]
    mixed = len(top) > 1 and top[1][1] >= 0.2 * sum(cnt.values())
    return [top[0][0], "ascii=%d" % min(3, 4 * sum(c < "\x80" for c in s) // len(s))] + ["mixed"] * mixed


def features(nm, kw=None, ctx=None, char3=False):
    """Feature strings ns + "\\x1f" + content, deduped in order, capped per namespace (§2.3); kw = keyword_info() or None."""
    ctx, F, toks, instr, cjk = ctx or {}, [], nm["toks"], nm["instr"], nm["cjk"]

    def put(ns, items):
        F.extend([ns + "\x1f" + x for x in list(dict.fromkeys(items))[:CAPS[ns]]])
    put("w", toks)
    put("b", [a + " " + b for a, b in zip(toks, toks[1:])])
    put("i", nm["idents"])
    put("j", [c for r in cjk for c in r] + [r[k:k + 2] for r in cjk for k in range(len(r) - 1)])
    lead = next((t for t in toks if t not in R._FILLER), "")
    put("l", ["lead=" + lead] * bool(lead) + ["q"] * (instr.rstrip().endswith("?") or toks[:1] != [] and toks[0] in R._QUESTION_LEAD))
    if kw:
        put("k", ["%s=%d" % (j, min(int(v), 4)) for j, v in kw[0].items() if v > 0] + ["d=%d" % kw[1]])
    put("s", ["len=%d" % min(10, _b2(nm["ntok"])), "steps=%d" % min(3, len(R._STEPS_RE.findall(instr.lower()))),
              "files=%d" % min(3, toks.count("<f>")), "fences=%d" % min(2, nm["fences"]), "sym=%d" % _sym(instr)]
        + ["url"] * ("<u>" in toks) + ["ext=" + e for e in list(dict.fromkeys(nm["exts"]))[:3]])
    if nm["mat_chars"]:
        put("m", ["type=" + t for t in nm["types"][:5]] + ["size=%d" % min(20, _b2(nm["mat_chars"]))])
        put("n", tokens(nm["mat"][:2048], 32)[0])
    if nm["prev"]:
        put("p", tokens(nm["prev"], 0, 48)[0])
    if "n_user" in ctx:
        nu, nt = ctx.get("n_user") or 0, ctx.get("n_tool") or 0
        put("h", ["turns=" + ("0" if nu < 1 else "1" if nu < 2 else "2-3" if nu < 4 else "4-7" if nu < 8 else "8+"),
                  "tools=" + ("0" if nt < 1 else "1-3" if nt < 4 else "4-10" if nt < 11 else "11+"),
                  "asst=%d" % _b2(ctx.get("asst_len") or 0)] + ["last_tool"] * bool(ctx.get("last_tool"))
            + ["prev_job=%s" % ctx.get("prev_job")] * bool(ctx.get("prev_job")))
    est, img = ctx.get("est_in") or nm["chars"] / (3.0 if nm["fences"] else 3.6), ctx.get("images") or 0
    put("x", ["h=%s" % ctx.get("harness")] * bool(ctx.get("harness")) + ["mode=" + (ctx.get("mode") or "skill")]
        + ["tools"] * bool(ctx.get("tools")) + ["img=%d" % min(img, 2)] * bool(img) + ["in=%d" % min(20, _b2(est))]
        + ["max=%d" % _b2(ctx.get("max_tokens") or 0)] * bool(ctx.get("max_tokens")))
    put("g", _script(instr))
    if char3:
        put("c", [w[k:k + 3] for w in toks if len(w) > 3 and w[0] != "<" for k in range(len(w) - 2)])
    return F


def known(md, feats):
    """(unique bucket ids of the features that pass the Bloom filter, in order; the OOD score)."""
    bl, M1, B1, crc = md["bloom"], md["M"] - 1, md["B"] - 1, zlib.crc32
    ids, seen, nw, kw, nj, kj = [], set(), 0, 0, 0, 0
    for f in feats:
        fb = f.encode("utf-8", "surrogatepass")   # a non-UTF-8 argv byte arrives as a lone surrogate
        h1 = crc(fb)
        p, q = h1 & M1, (h1 + crc(fb, SEED2)) & M1
        ok = bl[p >> 3] >> (p & 7) & bl[q >> 3] >> (q & 7) & 1
        if f[0] == "w" and f[2] != "<":
            nw, kw = nw + 1, kw + ok
        elif f[0] == "j":
            nj, kj = nj + 1, kj + ok
        if ok and (h1 & B1) not in seen:
            seen.add(h1 & B1)
            ids.append(h1 & B1)
    return ids, 1.0 - (kw / nw if nw else kj / max(1, nj))


def bag(emb, ids, H=64):
    """Column sums of the uint8 rows `ids` of emb (B x H, row-major): SWAR over 16-bit lanes, 256 rows at a time."""
    even, fb, fmt = int.from_bytes(bytes([255, 0]) * (H // 2), "little"), int.from_bytes, "<%dH" % (H // 2)
    ae, ao = [0] * (H // 2), [0] * (H // 2)
    for s in range(0, len(ids), 256):
        xs = [fb(emb[i * H:i * H + H], "little") for i in ids[s:s + 256]]
        ae = list(map(add, ae, struct.unpack(fmt, sum([x & even for x in xs]).to_bytes(H, "little"))))
        ao = list(map(add, ao, struct.unpack(fmt, sum([(x >> 8) & even for x in xs]).to_bytes(H, "little"))))
    out = [0] * H
    out[0::2], out[1::2] = ae, ao
    return out


def infer(md, ids):
    """The K raw outputs: h = ReLU(scale * (S - 128 n) / n + b1); y = W2 h + b2."""
    n, S = len(ids), bag(md["emb"], ids, md["H"])
    h = [max(0.0, sc * (s - 128 * n) / max(n, 1) + b) for sc, s, b in zip(md["scale"], S, md["b1"])]
    return [sum(map(mul, row, h)) + c for row, c in zip(md["W2"], md["b2"])]


def load(path=None):
    """(model, "") or (None, why it is not used); opened once per process and path."""
    path = path or BIN_PATH
    if path not in _S1:
        try:
            _S1[path] = _open(path)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, ZeroDivisionError, struct.error):
            _S1[path] = (None, "s1.bin has a malformed header")
    return _S1[path]


def _open(path):
    """mmap (else a plain read); checks the format, FEAT_VERSION, JOBS, K and section sizes (sha256 in verify())."""
    try:
        with open(path, "rb") as f:
            try:
                buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                buf = f.read()
    except OSError:
        return None, "s1.bin not found"
    n = int.from_bytes(buf[4:8], "little")
    hd = json.loads(bytes(buf[8:8 + n]).decode("utf-8")) if buf[:4] == MAGIC and n <= len(buf) - 8 else None
    if not isinstance(hd, dict) or hd.get("format") != FORMAT:
        return None, "s1.bin has an unknown format"
    if hd.get("feat_version") != FEAT_VERSION:
        return None, "s1.bin feat_version %r != %d (retrain)" % (hd.get("feat_version"), FEAT_VERSION)
    if hd.get("jobs") != list(R.JOBS):
        return None, "s1.bin was trained on a different job list"
    B, H, M, sec = hd.get("B"), hd.get("H"), hd["bloom"].get("M"), hd["sections"]
    if hd.get("K") != K or not all(isinstance(v, int) and v > 0 and v & (v - 1) == 0 for v in (B, H, M)):
        return None, "s1.bin has bad sizes"
    size = {"bloom": M // 8, "emb": B * H, "dense": 4 * (2 * H + K * H + K)}
    bad = [k for k, v in size.items() if sec[k][1] != v or sec[k][0] % 64 or sec[k][0] + v > len(buf)]
    if bad:
        return None, "s1.bin section %s has the wrong size or offset" % bad[0]
    why = _bad_header(hd)
    if why:
        return None, "s1.bin has a malformed header (%s)" % why
    mv, o = memoryview(buf), sec["dense"][0]
    d = struct.unpack("<%df" % (size["dense"] // 4), mv[o:o + size["dense"]])
    return {"header": hd, "B": B, "H": H, "M": M, "path": path, "emb": mv[sec["emb"][0]:sum(sec["emb"])],
            "bloom": bytes(mv[sec["bloom"][0]:sum(sec["bloom"])]), "scale": list(d[:H]), "b1": list(d[H:2 * H]),
            "W2": [list(d[2 * H + r * H:3 * H + r * H]) for r in range(K)], "b2": list(d[2 * H + K * H:]),
            "trained": set(hd.get("trained") or ()), "calib": hd.get("calib") or {}}, ""


def _bad_header(hd):
    """What posterior, legacy and decide read from the header without further checks, validated once at load; None when fine."""
    num, c, o = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), hd.get("calib"), hd.get("ordinal")
    pair = lambda v: isinstance(v, (list, tuple)) and len(v) == 2 and all(map(num, v))  # noqa: E731
    tr = hd.get("train", {})
    if not isinstance(c, dict) or not isinstance(tr, dict) or not isinstance(tr.get("config", {}), dict):
        return "calib or train"
    if not all(num(c.get(k, 1.0)) and c.get(k, 1.0) > 0 for k in ("T_job", "c_sigma_b", "tok_sd", "turn_sd")) or \
            not (num(c.get("w_nb", 0.0)) and num(c.get("ood_gamma", 0.0))):
        return "calib scalars"
    if not isinstance(c.get("platt", {}), dict) or not all(pair(v) for v in (c.get("platt") or {}).values()):
        return "calib.platt"
    fin = c.get("final", {})
    if not isinstance(fin, dict) or not all(isinstance(v, dict) and pair(v.get("platt", (1, 0))) for v in fin.values()):
        return "calib.final"
    if o is not None and not (isinstance(o, dict) and isinstance(o.get("tau", ()), (list, tuple)) and len(o.get("tau", (0,) * 3)) == 3
                              and all(map(num, o.get("tau", ()))) and num(o.get("s_d", 1.0)) and o.get("s_d", 1.0) > 0):
        return "ordinal"
    lg = hd.get("legacy")
    return "legacy" if lg is not None and not (isinstance(lg, dict) and num(lg.get("U", 1.0)) and lg.get("U", 1.0) > 0) else None


def verify(path=None):
    """Sections whose sha256 does not match the header, or [why] when unusable (doctor and tests, not `route`)."""
    import hashlib
    md, why = load(path)
    if md is None:
        return [why]
    with open(md["path"], "rb") as f:
        raw = f.read()
    return [k for k, (o, n) in sorted(md["header"]["sections"].items())
            if hashlib.sha256(raw[o:o + n]).hexdigest() != md["header"]["sha256"].get(k)]


def priors():
    if not _PRI:
        p, d = dict(PRIORS), R._read_json(os.path.join(os.path.dirname(BIN_PATH), "decide.json"))
        d = d if isinstance(d, dict) else {}
        p.update({k: v for k, v in (d.get("job_priors") or {}).items() if k in p})
        _PRI.append(dict(p, **{k: v for k, v in (d.get("nb_adapter") or {}).items() if k in ("c", "sd")}))
    return _PRI[0]


def _softmax(xs):
    e = [math.exp(x - max(xs)) for x in xs]
    return [v / sum(e) for v in e]


def nb_job_p(cls):
    """The rules classifier's job distribution over JOBS (rounded entries floored at 0.02), or None."""
    sg = (cls.signals.get("rules") or cls.signals) if cls is not None else {}
    if isinstance(sg.get("job_p"), list) and len(sg["job_p"]) == len(R.JOBS):
        return sg["job_p"]
    v = [max(0.02, sg["job_probs"].get(j, 0.0)) for j in R.JOBS] if sg.get("job_probs") else None
    return v and [x / sum(v) for x in v]


def keyword_info(cls, text):
    """(keyword scores, heuristic difficulty) from the classify_rules pass; recomputed if it did not keep them."""
    sg = cls.signals if cls is not None else {}
    if "scores" in sg and cls.classifier == "rules":     # the heuristics path (or a model path that kept them)
        return sg["scores"], (sg.get("heuristic") or (None, cls.difficulty))[1]
    s = R._signals(text)
    return s["scores"], R._difficulty(s["lower"], s["demand"], R._pick(s["scores"], len(s["toks"]))[0])[0]


def _filled(job_p, b, source):
    """A posterior whose heads other than job and b come from job priors."""
    p, js = priors(), list(zip(R.JOBS, job_p))
    return {"job_p": dict(js), "b": list(b), "tools": sum(q * p["tools"].get(j, 0) for j, q in js),
            "self_contained": p["self_contained"], "stakes": p["stakes"],
            "detectable": sum(q * p["detectable"].get(j, 0.5) for j, q in js), "ambiguity": p["ambiguity"],
            "tok_out": [sum(q * math.log(p["tok_out"].get(j, 800)) for j, q in js), p["tok_sd"]],
            "turns": list(p["turns"]), "ood": 0.0, "source": source, "trained": []}


def _nb_b(cls):
    """NB adapter b: the moment-matched mixture of N(c_d, sd) over the rules classifier's difficulty probabilities."""
    p, pd = priors(), cls.signals.get("difficulty_probs") if cls is not None else None
    if not (isinstance(pd, list) and len(pd) == 4 and sum(pd) > 0):
        pd = [float(d == (cls.difficulty if cls is not None else 1)) for d in range(4)]
    mu = sum(q * c for q, c in zip(pd, p["c"])) / sum(pd)
    return mu, math.sqrt(max(sum(q * (p["sd"] ** 2 + c * c) for q, c in zip(pd, p["c"])) / sum(pd) - mu * mu, 1e-6))


def nb(cls):
    """Adapter: the rules (Naive Bayes or heuristic) classification as a posterior."""
    return _filled(nb_job_p(cls) or [float(j == cls.job) for j in R.JOBS], _nb_b(cls), "nb")


def jev(cls):
    """Adapter: Jev's job and four 0-3 dimensions (b from complexity() through c_d; stakes, ambiguity = score / 1.5)."""
    cx, p = (cls.jev or {}).get("complexity") or {}, priors()
    post = _filled([float(j == cls.job) for j in R.JOBS], (p["c"][max(0, min(3, int(cx.get("value", 1))))], p["sd"]), "jev")
    return dict(post, stakes=cx.get("risk", 1.5) / 1.5, ambiguity=cx.get("ambiguity", 0.75) / 1.5)


def heads(md, y, ood, cls=None):
    """Calibrated heads from the raw outputs (spec §2.6, §2.7); untrained heads from job priors (b then from NB)."""
    c, tr, pl = md["calib"], md["trained"], md["calib"].get("platt") or {}
    ex = lambda v, k: c.get(k, 1.0) * math.exp(max(-10.0, min(10.0, v)))  # noqa: E731
    pr = lambda k, v: _sig((pl.get(k) or (1.0, 0.0))[0] * v + (pl.get(k) or (1.0, 0.0))[1])  # noqa: E731
    p, w, q = _softmax([v / c.get("T_job", 1.0) for v in y[:11]]), c.get("w_nb", 0.0), nb_job_p(cls)
    if w and q:
        p = _softmax([(1 - w) * math.log(max(a, 1e-12)) + w * math.log(max(b, 1e-12)) for a, b in zip(p, q)])
    sd = ex(y[12], "c_sigma_b") * (1 + c.get("ood_gamma", 0.0) * ood)
    post = _filled(p, (y[11], sd) if "b" in tr else _nb_b(cls), "s1")
    s1, a1 = pr("stakes1", y[15]), pr("amb1", y[18])
    got = {"tools": pr("tools", y[13]), "self_contained": pr("self_contained", y[14]), "detectable": pr("detectable", y[17]),
           "stakes": s1 + min(s1, pr("stakes2", y[16])), "ambiguity": a1 + min(a1, pr("amb2", y[19])),
           "tok_out": [y[20], ex(y[21], "tok_sd")], "turns": [y[22], ex(y[23], "turn_sd")]}
    post.update({k: v for k, v in got.items() if k in tr}, ood=ood, trained=sorted(tr))
    return post


def posterior(task, cls=None, hints=None, path=None):
    """TaskPosterior of a task: a string (skill mode) or a dict with `text` and the proxy context (mode, harness, tools,
    images, est_in, max_tokens, prev_user, n_user, n_tool, last_tool, asst_len, prev_job); `cls` = classify_rules' result."""
    ctx = task if isinstance(task, dict) else {"text": task}
    text, h = ctx.get("text") or "", {k: v for k, v in (hints or {}).items() if v is not None}
    cls = cls if cls is not None else R.classify_rules(text)
    md, why = load(path)
    if cls.classifier == "jev" and cls.jev:
        post = jev(cls)
    elif md is None:
        post = dict(nb(cls), s1_ignored=why)
    else:
        cfg = md["header"].get("train", {}).get("config", {})
        kw = keyword_info(cls, text) if cfg.get("kw", True) else None
        ids, ood = known(md, features(normalize(text, ctx.get("prev_user") or ""), kw, ctx, cfg.get("char3", False)))
        post = heads(md, infer(md, ids), ood, cls)
    if h.get("job") in R.JOBS:
        jp = {j: 0.3 * v + 0.7 * (j == h["job"]) for j, v in post["job_p"].items()}
        post["job_p"] = {j: v / sum(jp.values()) for j, v in jp.items()}
    post.update({k: 0.95 if h[k] in (True, 1, "yes") else 0.05 for k in ("tools", "self_contained") if k in h})
    if h.get("stakes") in ("low", "normal", "high"):
        post["stakes"] = float(("low", "normal", "high").index(h["stakes"]))
    proxy = ctx.get("mode") == "proxy"
    post["group"] = "agentic" if (bool(ctx.get("tools")) if proxy else post["tools"] >= 0.5) else "chat"
    if h.get("size") in SIZES:
        post["tok_out"][0] = (post["tok_out"][0] + math.log(SIZES[h["size"]][0])) / 2
        if post["group"] == "agentic":
            post["turns"][0] = (post["turns"][0] + math.log(SIZES[h["size"]][1])) / 2
    if post["group"] == "chat":
        post["turns"] = [0.0, 0.0]
    return dict(post, hints=h) if h else post


def legacy(post, cls=None, path=None, U=None):
    """S1's job and difficulty decoded tier-risk-aware like classify_model (U from the header); else `cls` unchanged."""
    md, _ = load(path)
    if post.get("source") != "s1" or md is None or not {"job", "b"} <= md["trained"]:
        return cls
    o, U = md["header"].get("ordinal") or {}, U or (md["header"].get("legacy") or {}).get("U", R.UNDER_COST)
    a, b = (md["calib"].get("platt") or {}).get("d") or (1.0, 0.0)   # one Platt for the 3 thresholds keeps them ordered
    cum = [_sig(a * (post["b"][0] - t) / o.get("s_d", 1.0) + b) for t in o.get("tau", (-1.5, 0.2, 1.9))]
    pd, pc = [1 - cum[0], cum[0] - cum[1], cum[1] - cum[2], cum[2]], [post["job_p"][j] for j in R.JOBS]
    p_tier = [0.0] * 4
    for i in range(len(R.JOBS)):
        for d in range(4):
            p_tier[R._RISK_TIER[i][d]] += pc[i] * pd[d]
    risk = [sum(p * (U * (k - t) if k > t else t - k) for k, p in enumerate(p_tier)) for t in range(4)]
    best = min(range(4), key=risk.__getitem__)
    _, i, d = max((pc[i] * pd[d], i, d) for i in range(len(R.JOBS)) for d in range(4) if R._RISK_TIER[i][d] == best)
    return R.Classification(R.JOBS[i], d, max(0.35, min(0.95, p_tier[best])),
                            cls.needs if cls is not None else ("tools",) * (post["tools"] >= 0.5), "s1",
                            {"model": True, "job_probs": {j: round(v, 3) for j, v in zip(R.JOBS, pc) if v >= 0.05},
                             "difficulty_probs": [round(v, 3) for v in pd], "b": [round(v, 3) for v in post["b"]],
                             "ood": round(post["ood"], 3)})
