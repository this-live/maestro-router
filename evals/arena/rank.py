"""Ranking open-ended answers (S1 spec §7.5 rules, extended to a full ranking).

1. Criteria before answers: one call per item to the lead judge, from the request alone (alpha.CRITERIA_PROMPT): 3-6 MUST
   criteria, 0-3 disqualifiers, a reference sketch.
2. Blind judging: every judge sees the request, the criteria and all of that item's answers, anonymized (self-identifying
   phrases of any family stripped, markdown normalized, 6,000-character cap) and shuffled with a seeded order; it returns
   per-answer criterion booleans and a full ranking. Each judge runs twice, the second time with the order reversed, so a
   preference for a position cancels out.
3. Aggregation: every ranking gives pairwise wins; Bradley-Terry (MM algorithm, Hunter 2004, with `prior` pseudo-wins
   each way on every compared pair) turns them into a log-strength per seat (mean 0), overall and per item, with a
   cluster bootstrap over items for 95% intervals. `accepted` (all MUST met, no disqualifier) is averaged over judges and
   passes into y_soft; y = y_soft >= 0.5.
4. Checks: position bias, pass-to-pass consistency, judge agreement, length bias (within-item Spearman of score and
   length) and a judge-family x contestant-family matrix with each family's self-preference.
"""
from __future__ import annotations

import math
import random
import re

import alpha
from .common import ek, mean, st

OTHER_ID = re.compile(r"\b(?:as )?(?:an? )?(?:chatgpt|gpt-?oss|gpt-?\d[\w.-]*|openai|codex|gemini(?: \d[\w.]*)?(?: pro| flash)?|bard|"
                      r"google deepmind|deepmind|llama ?\d*|qwen\d*[\w.-]*|mistral|mixtral|deepseek|grok|hermes|nous research)\b", re.I)
RANK_PROMPT = (
    "You compare anonymous answers to one request. The answers are in random order. For each answer decide every MUST "
    "criterion (true = met) and every DISQUALIFIER (true = present), judging only what the answer text shows. Then rank ALL "
    "answers from best to worst: correctness first, then completeness and clarity; do not reward length for its own sake.\n\n"
    "{body}\n\nMUST criteria:\n{must}\nDISQUALIFIERS:\n{disq}\nReference sketch: {sketch}\n\n{answers}\n\n"
    "Reply with only JSON: {{\"answers\": {{{shape}}}, \"ranking\": [{labels}]}}")


def anonymize(text, limit=6000):
    """alpha.clean_answer (Claude names, markdown, truncation) plus other families' names, outside code fences."""
    parts = re.split(r"(```.*?(?:```|\Z))", alpha.clean_answer(text, limit), flags=re.S)
    for k in range(0, len(parts), 2):
        parts[k] = OTHER_ID.sub("", parts[k])
    return "".join(parts).strip()


def criteria_prompt(task):
    return alpha.CRITERIA_PROMPT.format(body=task.get("request") or "USER REQUEST:\n" + task["text"])


def parse_criteria(text):
    j = ek.parse_json_reply(text)
    must = [str(x) for x in (j.get("must") or []) if str(x).strip()] if isinstance(j, dict) else []
    if not 1 <= len(must) <= 8:
        raise ValueError("criteria need 1-8 MUST items")
    disq = [str(x) for x in j.get("disqualifiers") or [] if str(x).strip()][:5]
    return {"must": must, "disq": disq, "sketch": str(j.get("sketch") or "")}


def labels_for(n):
    return [chr(65 + k) for k in range(n)]


def rank_prompt(task, crit, texts):
    """`texts`: answer texts in the order shown (labels A, B, ...)."""
    labs = labels_for(len(texts))
    blocks = "\n\n".join("Answer %s:\n<<<\n%s\n>>>" % (lab, anonymize(t)) for lab, t in zip(labs, texts))
    shape = ", ".join('"%s": {"must": [%d booleans], "disq": [%d booleans]}' % (lab, len(crit["must"]), len(crit["disq"]))
                      for lab in labs)
    return RANK_PROMPT.format(body=task.get("request") or "USER REQUEST:\n" + task["text"], must=alpha._numbered(crit["must"]),
                              disq=alpha._numbered(crit["disq"]), sketch=crit["sketch"], answers=blocks, shape=shape,
                              labels=", ".join('"%s"' % lab for lab in labs))


def parse_judgment(text, order, crit):
    """(ranking as seat ids best first, {seat: accepted or None}). `order`: seats in the order shown. A ranking may omit
    one answer (it goes last); anything else that is not a permutation of the labels raises ValueError."""
    j = ek.parse_json_reply(text)
    if not isinstance(j, dict):
        raise ValueError("judgment is not a JSON object")
    labs = labels_for(len(order))
    seen = []
    for x in j.get("ranking") or []:
        x = str(x).strip().upper().replace("ANSWER ", "")
        if x in labs and x not in seen:
            seen.append(x)
    if len(seen) < len(labs) - 1:
        raise ValueError("the ranking does not cover the answers")
    seen += [x for x in labs if x not in seen]
    by = dict(zip(labs, order))
    acc = {}
    for lab in labs:
        d = (j.get("answers") or {}).get(lab) if isinstance(j.get("answers"), dict) else None
        mu, dq = (d.get("must"), d.get("disq")) if isinstance(d, dict) else (None, None)
        ok = isinstance(mu, list) and len(mu) == len(crit["must"]) and isinstance(dq, list) and len(dq) == len(crit["disq"])
        acc[by[lab]] = (all(v is True for v in mu) and not any(v is True for v in dq)) if ok else None
    return [by[x] for x in seen], acc


def orders(task_id, rep, judge, seats, seed=0):
    """The two orders a judge sees: a seeded shuffle, then the same reversed."""
    base = sorted(seats, key=lambda s: ek.unit("arena-order", seed, task_id, rep, judge, s))
    return [base, list(reversed(base))]


# ── Bradley-Terry ─────────────────────────────────────────────────────────────

def pairwise(ranking):
    """[(winner, loser)] for every pair in a ranking (best first)."""
    return [(ranking[i], ranking[k]) for i in range(len(ranking)) for k in range(i + 1, len(ranking))]


def bradley_terry(wins, players=None, prior=0.0, iters=2000, tol=1e-12):
    """{player: log-strength, mean 0} by the MM algorithm. `wins`: {(winner, loser): count}. `prior` adds that many wins
    each way to every pair that was compared (keeps unbeaten / winless players finite)."""
    players = sorted(set(players or ()) | {p for ab in wins for p in ab})
    W = {p: 0.0 for p in players}
    N = {}
    for (a, b), c in wins.items():
        if a == b or c <= 0:
            continue
        W[a] += c
        k = (a, b) if a < b else (b, a)
        N[k] = N.get(k, 0.0) + c
    if prior:
        for (a, b) in list(N):
            N[(a, b)] += 2 * prior
            W[a] += prior
            W[b] += prior
    nb = {p: [] for p in players}
    for (a, b), n in N.items():
        nb[a].append((b, n))
        nb[b].append((a, n))
    p = {x: 1.0 for x in players}
    for _ in range(iters):
        new = {}
        for i in players:
            den = sum(n / (p[i] + p[j]) for j, n in nb[i])
            new[i] = max(W[i] / den, 1e-12) if den > 0 else p[i]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new)) if new else 1.0
        new = {i: v / g for i, v in new.items()}
        delta = max((abs(math.log(new[i]) - math.log(p[i])) for i in players), default=0.0)
        p = new
        if delta < tol:
            break
    return {i: math.log(v) for i, v in p.items()}


def win_prob(scores):
    """{player: mean P(beats another player)} under the fitted model."""
    out = {}
    for i, si in scores.items():
        others = [1 / (1 + math.exp(sj - si)) for j, sj in scores.items() if j != i]
        out[i] = sum(others) / len(others) if others else None
    return out


def _wins_of(rows):
    w = {}
    for r in rows:
        for a, b in pairwise(r["ranking"]):
            w[(a, b)] = w.get((a, b), 0) + 1
    return w


def bt_with_ci(judgments, prior=0.5, n_boot=500, seed=0, players=None):
    """({seat: (score, lo, hi)}, win probabilities). Bootstrap clusters are tasks (repeats and judges of a task stay together)."""
    rows = [r for r in judgments if r.get("ok")]
    players = sorted(set(players or ()) | {s for r in rows for s in r["ranking"]})
    point = bradley_terry(_wins_of(rows), players, prior)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    tasks = sorted(by_task)
    rng = random.Random(seed)
    samples = {p: [] for p in players}
    for _ in range(n_boot if len(tasks) > 1 else 0):
        pick = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        w = {}
        for t in pick:
            for k, v in _wins_of(by_task[t]).items():
                w[k] = w.get(k, 0) + v
        sc = bradley_terry(w, players, prior, iters=300, tol=1e-9)
        for p in players:
            samples[p].append(sc[p])
    out = {}
    for p in players:
        xs = sorted(samples[p])
        lo = xs[int(0.025 * (len(xs) - 1))] if xs else None
        hi = xs[int(0.975 * (len(xs) - 1))] if xs else None
        out[p] = (point[p], lo, hi)
    return out, win_prob(point)


# ── acceptance and checks ─────────────────────────────────────────────────────

def norm_scores(ranking):
    """{seat: 1 for the best .. 0 for the worst}."""
    n = len(ranking)
    return {s: 1.0 - k / (n - 1) if n > 1 else 1.0 for k, s in enumerate(ranking)}


def acceptance(judgments):
    """{(task, rep, seat): y_soft} = mean of the judges' (and passes') accepted verdicts that could be read."""
    acc = {}
    for r in judgments:
        if not r.get("ok"):
            continue
        for s, a in (r.get("accepted") or {}).items():
            if a is not None:
                acc.setdefault((r["task"], r["rep"], s), []).append(1.0 if a else 0.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def kendall(a, b):
    """Kendall tau-a between two {key: score} maps over their common keys (None with fewer than 2)."""
    ks = sorted(set(a) & set(b))
    pairs = [(i, j) for x, i in enumerate(ks) for j in ks[x + 1:]]
    if not pairs:
        return None
    s = 0.0
    for i, j in pairs:
        d = (a[i] - a[j]) * (b[i] - b[j])
        s += 1.0 if d > 0 else -1.0 if d < 0 else 0.0
    return s / len(pairs)


def diagnostics(judgments, lengths, family):
    """Position bias, pass consistency, judge agreement, length bias and family self-preference.
    `lengths`: {(task, rep, seat): answer characters}; `family`: {seat or judge id: family}."""
    rows = [r for r in judgments if r.get("ok")]
    out = {"n_judgments": len(rows), "n_failed": sum(1 for r in judgments if not r.get("ok"))}
    firsts = [(1.0 if r["ranking"][0] == r["order"][0] else 0.0, 1.0 / len(r["order"])) for r in rows if r["order"]]
    out["position_first"] = (mean([f for f, _ in firsts]), mean([e for _, e in firsts]))
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["task"], r["rep"], r["judge"]), {})[r["pass"]] = norm_scores(r["ranking"])
    out["pass_tau"] = mean([kendall(v[0], v[1]) for v in by_cell.values() if 0 in v and 1 in v])
    judge_s = {}      # (task, rep) -> judge -> {seat: mean normalized score over passes}
    for (t, rep, j), v in by_cell.items():
        seats_ = set().union(*[set(x) for x in v.values()])
        judge_s.setdefault((t, rep), {})[j] = {s: mean([x[s] for x in v.values() if s in x]) for s in seats_}
    pair_tau = {}
    for per in judge_s.values():
        js = sorted(per)
        for x, a in enumerate(js):
            for b in js[x + 1:]:
                tau = kendall(per[a], per[b])
                if tau is not None:
                    pair_tau.setdefault((a, b), []).append(tau)
    out["judge_tau"] = {k: mean(v) for k, v in pair_tau.items()}
    acc_j = {}
    for r in rows:
        for s, a in (r.get("accepted") or {}).items():
            if a is not None:
                acc_j.setdefault((r["task"], r["rep"], s), {}).setdefault(r["judge"], []).append(a)
    agree = []
    for per in acc_j.values():
        votes = {j: sum(v) / len(v) >= 0.5 for j, v in per.items()}
        js = sorted(votes)
        agree += [1.0 if votes[a] == votes[b] else 0.0 for x, a in enumerate(js) for b in js[x + 1:]]
    out["accept_agreement"] = mean(agree)
    rhos = []
    for (t, rep), per in judge_s.items():
        seats_ = sorted(set().union(*[set(x) for x in per.values()]))
        cons = [mean([per[j][s] for j in per if s in per[j]]) for s in seats_]
        lens = [lengths.get((t, rep, s)) for s in seats_]
        if len(seats_) >= 3 and None not in lens and len(set(lens)) > 1 and len(set(cons)) > 1:
            rhos.append(st.spearman(lens, cons))
    out["length_rho"] = (mean(rhos), len(rhos))
    matrix, selfpref = {}, {}
    for (t, rep), per in judge_s.items():
        for j, sc in per.items():
            fj = family.get(j) or "unknown"
            for s, v in sc.items():
                fs = family.get(s) or "unknown"
                matrix.setdefault((fj, fs), []).append(v)
                if fs == fj:
                    others = [per[k][s] for k in per if (family.get(k) or "unknown") != fj and s in per[k]]
                    if others:
                        selfpref.setdefault(fj, []).append(v - mean(others))
    out["family_matrix"] = {k: (mean(v), len(v)) for k, v in matrix.items()}
    out["self_preference"] = {k: (mean(v), len(v)) for k, v in selfpref.items()}
    out["judge_families"] = sorted({family.get(r["judge"]) or "unknown" for r in rows})
    return out
