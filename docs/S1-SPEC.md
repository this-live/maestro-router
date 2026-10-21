# Maestro S1: the decision model (final spec)

Status: final spec for implementation. Date: 2026-10-02. Base: Maestro v3 at `556f97d`.
Inputs: the designs `design-decision-theory.md` (DT), `design-tiny-model-engineering.md` (TME) and
`design-alpha-and-data.md` (A&D), their three adversarial critiques, and the Maestro v3 code.

The spec is written so an engineer can build it without asking questions. Each number is one of three kinds:

- **Measured**: produced in this sandbox. The source is named.
- **Placeholder**: a prior that a named fitting step replaces. It is labelled where it is defined.
- **Fixed**: a chosen value, for example a size, a cap or a hyperparameter.

---

## 0. Verdict

### 0.1 Scores

Scale: 1 (poor) to 10 (excellent). "Robustness" means robustness to synthetic-label bias.

| criterion | DT | TME | A&D |
|---|---|---|---|
| expected alpha | 7 | 5 | 6 |
| latency | 8 | 9 | 8 |
| stdlib feasibility | 8 | 9 | 7 |
| measurability | 6 | 5 | 9 |
| robustness | 4 | 4 | 7 |
| simplicity | 4 | 5 | 6 |
| **total (of 60)** | **37** | **37** | **43** |

**DT (decision theory).**
- **Strongest:**
  - a factored success model behind an interface: task difficulty from text, seat ability from the catalog;
  - expected utility in dollars;
  - the under-routing asymmetry derived from detectability and stakes;
  - cascade planning with a Bayesian difficulty update on failure;
  - the pacing idea for quota.
- **Weakest:**
  - it overstates alpha because it compares against always-strongest at the wrong prices;
  - the quota price is wrong: a fresh window gets charged;
  - ECE gates that a perfectly calibrated model would fail;
  - about 30 hand-set constants;
  - an OOD bitmap that saturates;
  - stacking on in-sample NB outputs.

**TME (tiny-model engineering).**
- **Strongest:**
  - the best pure-Python engine: crc32 features into an int8 embedding bag with the first layer folded in, and SWAR summation;
  - measured p50 96 µs and p99 330 µs from raw text;
  - bit-exact parity between int8 and float;
  - the Unicode tokenizer fix.
- **Weakest:**
  - an 8-axis IRT with per-item discrimination that 4 ordered Claude anchors cannot identify;
  - the θ source (`score_rec.axes`) is unblended and its cohort-relative scale drifts;
  - the quota model needs `Cap_w`, which cannot be observed;
  - a 180-item test set with point-threshold gates inside the noise.

**A&D (alpha and data).**
- **Strongest:**
  - the measurement design: an offline replay, the constant-policy hull, a gold-label ladder baseline, a paired bootstrap, power analysis and an access ledger;
  - a difficulty head trained on the outcome likelihood, which handles censored items correctly;
  - checking panel labels against outcomes;
  - objective tests, measured self-preference and length bias, and a check of the generator's hints.
- **Weakest:**
  - in skill mode, EU only reorders seats inside the ladder walk;
  - the online loop never sees Claude Code or Codex traffic;
  - the quota formula has a units error and has no snapshot history to draw on;
  - inlined variants corrupt the difficulty target;
  - the split sizes contradict each other.

### 0.2 What this spec takes from each design, and what it fixes

| area | taken from | fixed per critique |
|---|---|---|
| Interface between the model and the decision (`TaskPosterior`) | DT | narrowed to the heads that feed decisions or hard constraints |
| Inference engine (crc32, folded int8 bag, SWAR, mmap) | TME | 2^15 buckets; a Bloom filter of known features instead of a bitmap of touched buckets |
| Difficulty b trained on the outcome likelihood | A&D | native self-contained items only, no inlined variants; a 1-D ability per job group |
| Success link `σ(a_g(θ − b))` with a probit shrinkage | DT, A&D | the slope β comes from public cross-vendor data; per-model offsets only for the anchors; a catalog hash guards against a stale link |
| Expected utility, with failure cost from detectability and stakes | DT | `V0` is calibrated on alpha-dev to a target retention, not hand-set |
| Quota as a pacing price | DT, A&D | projected use from the burn rate, with no price while there is slack; `Q = g · C_substitute` (units fixed); per-window snapshot history |
| Cache and switching cost, effort as part of the seat, inline as a seat | critiques | all three are new and in v1 |
| Replay, hull, paired bootstrap, ledger | A&D | ECE thresholds derived from a calibrated null; calibration measured over all (item, tier) cells; V is chosen on dev |
| Alpha-set collection | A&D | a pilot with a go/no-go rule; two-phase answering; one prompt per answering call; at least 50% verifiable items |
| Teacher panel | A&D | three distinct models (haiku, sonnet, opus); the generator's hint is never a label |
| Outcome signals | A&D, DT | v1 learns only from explicit outcomes and from tests and escalation signals. Skill mode gets an `outcome` command and a hook. |
| Typed-decision API compatible with Jev | all | intents matched by explicit name, never by cosine similarity; a thinking-safe probe for delegation |

### 0.3 Where the alpha comes from, honestly

The alpha has five sources. Each is measured separately (§9). Only the first depends on how well the model predicts
difficulty.

1. **Per-task difficulty routing.** Measured against the constant-policy hull at catalog Claude prices (1:2:4:10). The
   simulated alpha capture Α is 0.15 at ρ = 0.5, 0.30 at ρ = 0.7 and 0.62 at ρ = 0.9 (critique, `irt/alpha_ci2.py`).
   Here ρ is the correlation between predicted and true difficulty. A realistic text-only ρ is 0.5–0.75, so expect
   Α ≈ 0.15–0.35. The pilot in §7.1 decides whether this source is worth its labeling cost.
2. **Inline vs delegate, and detecting trivial work.** Skill mode only. Most of the gain is latency.
3. **Quota pacing.** Use subscription windows that would otherwise expire unused, avoid exhausting them, and shift work
   to Codex, local or API seats.
4. **Cache stickiness.** Proxy only. Never switch seats on a long conversation when staying is cheaper.
5. **Local seats.** Used for chat jobs they can do, priced by their real latency.

Sources 2–5 need the decision layer but no better difficulty predictor. They ship even if the pilot says no-go.

---

## 1. The decision problem

### 1.1 Objects

**Task** `x = (instr, material, history, harness, images, tools_param, tool_turns, max_tokens, stream, mode, hints)`

| field | meaning |
|---|---|
| `instr` | Skill mode: the description passed to `maestro route`. Proxy: the last user message. Fenced and unfenced material is removed (§2.2). |
| `material` | Pasted code, logs, traces, diffs and data. |
| `history` | Proxy only: the earlier user, assistant and tool messages, and the system prompt. |
| `harness` | `claude-code`, `codex`, `gemini-cli`, `opencode`, `hermes`, `api` or `unknown`. |
| `mode` | `skill` or `proxy`. |
| `hints` | Optional, skill mode: `job`, `tools`, `self_contained`, `size`, `stakes` and `context_tokens`, supplied by the calling agent (§5.6). |

**Seat** `s = (model, provider, harness, access, effort)`

| field | values |
|---|---|
| `access` | `inline`, `subscription`, `api` or `local` |
| `effort` | one of `router.EFFORTS`, only for seats that accept an effort setting: Codex, and proxy models with `reasoning` |
| attributes | ability θ for each job group and job with an sd; prices; context; `tools_ok`; `vision`; quota windows; latency; whether the seat is warm (holds this conversation's cache) |

The seats come from the existing builders, `router._candidates` in skill mode and `server.App.candidates` in the proxy.
Their **hard filters do not change**: routable, tools, vision, context, `max_tier`, cooldown or exhaustion, free tiers,
`prefer_local` and `_executable`. The decision layer only ranks what survives them, plus the new `inline` seat in skill
mode.

**Outcome** `y ∈ {0,1}`: the result is accepted without a redo. For an agentic task, that means the change works.

### 1.2 Output

The existing decision JSON keeps every key. These keys are added:

```json
{"decider": "utility",
 "posterior": {"job_p": {"debug": 0.81, "code": 0.12}, "b": [1.12, 0.71], "tools": 0.97, "self_contained": 0.35,
               "stakes": 0.9, "detectable": 0.82, "ambiguity": 0.4, "tok_out": [1450, 0.6], "turns": [9, 0.5],
               "ood": 0.04, "group": "agentic"},
 "options": [{"seat": "claude-code:opus", "p": 0.78, "p_lo": 0.71, "usd": 0.0, "quota_usd": 0.10, "secs": 95, "eu": 1.31},
             {"seat": "codex:gpt-6.1-sol", "p": 0.74, "p_lo": 0.62, "usd": 0.0, "quota_usd": 0.0, "secs": 120, "eu": 1.22}],
 "escalate_to": "claude-code:fable", "confidence": 0.71, "decision_id": "d7f3a2c1",
 "reason": "debug d2 (b +1.1±0.7): opus P .78 vs codex .74; Claude 5h window has slack (quota ×0.29)"}
```

The legacy fields are derived from the posterior:

- `difficulty` comes from the ordinal link (§3.6);
- `category` is the chosen seat's category;
- `confidence` is the analytic probability that the top seat beats the runner-up (§5.8).

### 1.3 Objective

Each eligible seat `s` gets an expected utility in dollars. The full definitions are in §5.

```
EU(s|x) = V(x)·P̃_s(x) − C_s(x) − Q_s(x) − κ·T_s(x) − (1 − P̃_s(x))·F_s(x)
```

| term | meaning |
|---|---|
| `P̃_s` | Pessimistic P(success). It is lowered by the uncertainty in the seat's ability, in proportion to the task's stakes. |
| `V` | The value of a success. It scales with the size of the task. |
| `C` | API dollars. |
| `Q` | The quota shadow price for subscription seats. |
| `T` | Wall-clock seconds. |
| `κ` | Dollars per second of waiting. |
| `F` | The cost of a failure: a retry if the failure is noticed, or a stakes-weighted loss if it goes unnoticed. |

**Decision.**

- **Choice.** `argmax EU`, subject to the floor `P̃ ≥ P_min(stakes)`. If no seat reaches the floor, take the seat with
  the highest `P̃` and flag `below_floor`.
- **Ties.** EU rounded to 1e-6, then lower cost, then the seat id.

---

## 2. S1, the tiny model

### 2.1 What S1 outputs: the TaskPosterior

```
TaskPosterior = {
  job_p[11]        calibrated softmax over router.JOBS (pooled with NB, §2.7)
  b = (mu, sd)     difficulty on the logit scale of group g's anchor abilities, after OOD inflation
  group            "agentic" if (mode == skill and tools >= 0.5) or (mode == proxy and tools_param), else "chat"
  tools            P(needs repo tools / commands / browsing)
  self_contained   P(everything needed is in the text)
  stakes           E[stakes] in [0, 2] = P(>=1) + P(>=2)
  detectable       P(a failure would be noticed within minutes)
  ambiguity        E[ambiguity] in [0, 2]  (systemone and explain only)
  tok_out = (mu, sd)  log-normal over output tokens of one answer (per turn for agentic work)
  turns = (mu, sd)    log-normal over agent turns (agentic only; 1 for chat)
  ood              0..1
  source           "s1" | "nb" | "jev" | "llm"
}
```

The decision layer depends only on this interface. Each backend has an adapter that fills it:

- `s1` is the default once its gate passes.
- `nb` maps today's Naive Bayes into the posterior (§2.9).
- `jev` maps Jev's four dimensions into it.
- `llm` uses delegated logprobs (§11).

### 2.2 Normalization and caps (`s1.normalize`)

1. **Split out material.** Rewrite `router.split_material` as a `str.find` loop, with no `re.S` regex. Today the regex
   costs 2.2 ms on a 200 KB paste (measured, DT). Then run an unfenced-material detector over the first 400 lines:
   - Any run of at least 5 consecutive lines that match the same pattern moves to `material`.
   - The patterns are:
     - log: `^\s*\[?\d{4}-\d\d-\d\d[ T]\d\d:\d\d`
     - stack frame: `^\s+at [\w.$<>]+\(`, `^\s*File ".*", line \d+`, `^Traceback`, `^panic:`, `^error\[E\d+\]`
     - diff: `^(@@|\+\+\+ |--- |[+-][^+-])`
     - JSON or YAML: a line starting with `{`, `[`, `"key":` or `key: value`
     - CSV: at least 3 commas in every line of the run
2. **Scan caps.**
   - The instruction is scanned over its first 4,096 chars plus its last 1,024 chars.
   - Material is scanned over its first 8 KB plus its last 2 KB.
   - The previous user turn is scanned over its last 1,024 chars.
   - Only the character count of the full input is taken over the whole text, by `len()`.
3. **Tokenizer.**
   - The pattern is `re.compile(r"[^\W_][\w+#.'/-]*")`, applied to the lowercased instruction. This fixes today's
     `_TOKEN_RE`, which drops non-ASCII letters (router.py:342).
   - A token matching `^[0-9][0-9.,]*[a-z%+]{0,2}$` becomes `<n>`.
   - A token matching `^[\w/.-]+\.[a-z]{1,4}$` becomes `<f>`.
   - `^https?://` becomes `<u>`.
   - Trailing `.,;:!?'"` characters are stripped.
   - Runs of CJK, Kana, Hangul or Thai characters are emitted per character into namespace `j`, not as word tokens.
   - Keep the first 192 tokens and the last 64.

### 2.3 Features (`FEAT_VERSION = 1`)

**Feature strings.** Each feature is the string `ns + "\x1f" + content`, encoded as UTF-8. Duplicates are removed while
order is kept.

**Hashing.**

```
h1     = zlib.crc32(f)
bucket = h1 & (B − 1)          # B = 2^15
h2     = zlib.crc32(f, 0x5BD1E995)
```

Python's `hash()` is never used, because it is salted per process.

| ns | content | cap |
|---|---|---|
| `w` | instruction unigrams | 256 |
| `b` | instruction bigrams over the capped token list | 255 |
| `i` | identifier parts of tokens longer than 4 chars containing `_`, camelCase, `.` or `/`. Split, keep parts of length ≥ 2, lowercase. | 64 |
| `j` | CJK, Kana, Hangul and Thai character unigrams and adjacent bigrams | 128 |
| `l` | `l:lead=<first token not in router._FILLER>`. Also `l:q` if the text ends with `?` or the first token is in `router._QUESTION_LEAD`. | 2 |
| `k` | `k:<job>=<min(int(score),4)>` for each job with a heuristic keyword score > 0 (from the `classify_rules` call that is already made, §2.7). Also `k:d=<heuristic difficulty 0..3>`. | 12 |
| `s` | `s:len=<min(10, int(log2(1+words)))>`, `s:steps=<0,1,2,3+>`, `s:url`, `s:files=<0,1,2,3+>`, `s:fences=<0,1,2+>`, `s:sym=<0..4>` (code-symbol ratio bucket), and `s:ext=<ext>` for up to 3 file extensions | 10 |
| `m` | `m:type=<stacktrace,diff,log,json,yaml,sql,csv,html,code:<lang>,prose>` for each type detected, and `m:size=<min(20,int(log2(1+chars)))>` | 6 |
| `n` | the first 32 material tokens, as unigrams | 32 |
| `p` | the previous user turn: unigrams of its last 48 tokens | 48 |
| `h` | `h:turns=<0,1,2-3,4-7,8+>` (user turns), `h:tools=<0,1-3,4-10,11+>` (tool messages), `h:last_tool`, `h:asst=<log2 bucket of the last assistant turn's length>`, `h:prev_job=<job>` from affinity (proxy) | 6 |
| `x` | `x:h=<harness>`, `x:mode=<skill,proxy>`, `x:tools` (tools param), `x:img=<0,1,2+>`, `x:in=<min(20,int(log2(1+est_in)))>`, `x:max=<log2 bucket of max_tokens>` | 6 |
| `g` | `g:<latin,cyrillic,cjk,arabic,devanagari,other>` (the majority script of the first 400 instruction chars, by Unicode block ranges), `g:mixed` if a second script makes up ≥ 20%, and `g:ascii=<0..4>` | 3 |

At most about 830 features. The median eval prompt has about 70.

**Known-feature filter.**

- A Bloom filter with M = 2^20 bits (128 KB), built from every training feature that occurs at least twice in train.
- Probes: `k = 2`, at positions `(h1 + j·h2) & (M − 1)` for j = 0, 1.
- A feature that fails the filter is **dropped**. It is neither summed nor counted, and training applies the same filter.
- The closed-vocabulary namespaces (`l, k, s, m, h, x, g`) are always inserted.

This replaces the "touched buckets" bitmap, which saturates, and keeps collision noise from novel identifiers out of
the mean.

**OOD score.**

```
ood = 1 − |known w-features| / max(1, |w-features excluding <n>, <f>, <u>|)
```

- If there are no `w` features (CJK-only text), use the `j` namespace instead.
- It is validated on persona, translated and gibberish probes (§9.2).

### 2.4 Architecture

```
ids (known, unique) ──► S[c] = Σ_ids U8[id, c]              (SWAR lane sums; U8 = int8 + 128)
h_pre[c] = scale[c] · (S[c] − 128·n) / max(n,1) + b1[c]       c = 0..63 ;  n = number of known ids
h = ReLU(h_pre)                                                64
y = W2 · h + b2                                                K = 24 raw outputs
heads = calibrate(y)                                           §2.6
```

- E' = E·W1 is folded at export (exact, TME). E is B × d, trained with d = 32. W1 is d × 64.
- **Sizes.** U8 is 2^15 × 64 = 2,097,152 bytes. The dense block is (64 + 64 + 24·64 + 24) float32 = 6,752 bytes.
  With the Bloom filter (131,072 bytes) and the header, the total is **≈ 2.25 MB**.
- **Hard gate.** ≤ 3 MiB.

**Output layout (K = 24).**

| index | head | link |
|---|---|---|
| 0–10 | job | softmax |
| 11, 12 | b μ, log σ | identity, exp |
| 13 | tools | sigmoid |
| 14 | self_contained | sigmoid |
| 15, 16 | stakes ≥ 1, ≥ 2 | sigmoid, monotone fix `p2 = min(p1, p2)` |
| 17 | detectable | sigmoid |
| 18, 19 | ambiguity ≥ 1, ≥ 2 | sigmoid, monotone fix |
| 20, 21 | tok_out μ (log tokens), log σ | identity, exp |
| 22, 23 | turns μ (log turns), log σ | identity, exp |

**Not in v1:**

- the 8-axis demand vector π, which outcomes from Claude-only anchors cannot identify;
- per-item discrimination;
- pass@anchor heads, which duplicate the IRT;
- scope;
- turn type.

**SWAR summation** (from the TME prototype; checked bit-exact).

- `EVEN = int.from_bytes(bytes([255, 0]) * 32, "little")`.
- For each block of 256 ids:
  - `xs = [int.from_bytes(mm[64*i: 64*i+64], "little") for i in block]`;
  - `te = sum(x & EVEN for x in xs)` and `to = sum((x >> 8) & EVEN for x in xs)`;
  - decode both with `array('H').frombytes(te.to_bytes(64, 'little'))`, byte-swapping on big-endian machines.
- Accumulate across blocks in Python ints.
- Lane bound: 256 rows × 255 = 65,280 < 2^16.

**Weights access.**

- Read the weights through `mmap.mmap(fd, 0, access=ACCESS_READ)`. If mmap raises, fall back to `open().read()`.
- Open the file once per process (`_S1 = []` cache, like `router._MODEL`).

### 2.5 Heads, targets, losses

`group` is defined in §2.1. `θ_{t,g}` are the anchor abilities for tiers t ∈ {haiku, sonnet, opus, fable}, learned jointly
(§3.5).

| head | target | loss | λ |
|---|---|---|---|
| job | panel soft distribution (§6.4), label smoothing 0.05 | soft CE | 1.0 |
| b (outcome) | alpha outcomes `y_it ∈ [0,1]` for each observed tier t | `−Σ_t [y log p + (1−y) log(1−p)]`, with `p = σ(a_g(θ_{t,g} − μ)/√(1 + π a_g² σ²/8))`, divided by the number of observed tiers | 2.0 |
| b (ordinal, panel difficulty) | panel difficulty 0–3, soft | cumulative logit `P(d ≥ k) = σ((μ − τ_k)/s_d)`, k = 1..3 | λ_d (§8.2) |
| b (ordinal, panel min_tier) | panel `min_tier` ∈ {haiku, sonnet, opus, fable, none}, soft | interval-censored: `P(min_tier ≤ t) = σ((θ_{t,g} − μ − c_mt)/s_mt)` | λ_d |
| tools | panel (soft) | BCE | 0.5 |
| self_contained | the corpus field when present, else the panel | BCE | 0.5 |
| stakes ≥ 1, ≥ 2 | panel (soft) | BCE on each threshold | 0.3 |
| detectable | panel | BCE | 0.3 |
| ambiguity ≥ 1, ≥ 2 | panel | BCE on each threshold | 0.2 |
| tok_out | alpha: measured output tokens per tier (§7.4); panel buckets elsewhere | Gaussian NLL on `log tout_it = μ + v_t + e` (alpha, v_t learned per tier); interval-censored NLL on the bucket (panel) | 0.5 alpha / 0.2 panel |
| turns | agentic fixtures: measured tool-call count; panel `agent_turns` bucket elsewhere (tool rows only) | Gaussian NLL / interval-censored NLL on log turns | 0.3 |

- A head with no label for a row is masked out.
- The outcome term is the **only** source of the scale and location of b. Panel ordinal terms only shape b for unlabeled
  rows.

### 2.6 Calibration (fitted on the calib split only; stored in the header)

| head | method |
|---|---|
| job | Temperature T_job (grid 0.5–5.0, step 0.05, minimizing NLL), then pooled with NB (§2.7). |
| binary and ordinal thresholds | Platt `(A, B)` per output. |
| b | Variance scale `c_σ`, so that `σ_b ← c_σ·σ_raw`. It maximizes the outcome log-likelihood on calib (golden-section search over [0.5, 3]). |
| OOD inflation | `σ_b ← σ_b·(1 + γ·ood)`. γ ∈ [0, 2] maximizes the calib outcome log-likelihood. |
| tok_out, turns | Variance scale, so that calib z-scores have unit variance. |
| final P(success) | Per group g: Platt `(A_g, B_g)` on `logit P_raw`, fitted over **all** observed (item, tier) cells of calib. It becomes isotonic (≤ 16 knots, evaluated with `bisect` and linear interpolation) once calib has ≥ 2,000 cells in a group. |

### 2.7 Pooling with the shipped Naive Bayes (v1)

`route` and `plan` already call `classify_rules`, which takes 0.2–0.3 ms. It returns the NB job distribution and the
heuristic keyword scores. S1 reuses both, so there is no second scan.

```
job_p ∝ p_S1^(1 − w) · p_NB^w
```

- `w ∈ {0, 0.25, 0.5}` is chosen on calib by NLL.
- The shipped `maestro_model.json` is unchanged. It was trained on `train.jsonl` only, so calib is out-of-sample for it.
- S1 never takes NB probabilities as input features. This avoids the in-sample stacking problem.

### 2.8 Latency budget (in process, CPython 3.11, 4 vCPU)

| step | p50 | p99 | source |
|---|---|---|---|
| `classify_rules` (unchanged; provides NB and the `k` features) | 0.22–0.31 ms | 1.05–1.45 ms | measured, TME and A&D |
| normalize + features + crc32 + Bloom | 0.10 ms | 0.35 ms | crc32 of 569 features takes 0.10 ms (measured, critique); the Bloom check is estimated |
| bag (SWAR) + heads + calibration | 0.07 ms | 0.20 ms | measured, TME: 63/151 µs |
| `decide.rank` (≤ 40 seats, cascade, reasons) | 0.06 ms | 0.15 ms | measured, DT/TME |
| **total** | **≈ 0.5 ms** | **≈ 1.8 ms** | |
| cold: mmap + header parse | 0.1 ms + 0.3 ms | | estimated |

**Budgets (CI-gated, §9.5):**

| quantity | budget |
|---|---|
| in-process p50 | ≤ 1 ms |
| in-process p99 | ≤ 3 ms |
| p99 on the 60k-char long-input fixtures | ≤ 5 ms |
| `route` CLI wall time | today's +3 ms (today 38–41 ms) |

### 2.9 Adapters

- **NB.**
  - Uses the NB difficulty probabilities `p_d`, d = 0..3.
  - Builds a b mixture with component means `c_d` and sd 0.8. Placeholder `c = (−2.0, −0.5, 1.0, 2.5)`, replaced by the
    fitted `τ` midpoints.
  - Moment-matches the mixture to give `(μ, σ)`.
  - Other heads come from job priors in `decide.json` (`job_priors`): tools = 1 for TOOL_JOBS, detectable by job, stakes
    1, and `tok_out` by job.
- **Jev.**
  - `b` comes from the existing `complexity()` value through the same `c_d`.
  - stakes = risk/1.5. ambiguity = ambiguity/1.5.
  - The job is Jev's job.

---

## 3. The success model

### 3.1 Equation

```
θ̄_s(x)  = Σ_j job_p_j · θ_{m(s), j}  +  ε_{effort(s)}  +  η_{h(s), g}  −  ω·(1 − self_contained)·[s delegated, skill mode]  +  o_s
sd_s²    = sd_link,m² + sd_η,h² + 0.2²·[link stale]
P_s      = Cal_g( σ( a_g·(θ̄_s − μ_b) / √(1 + (π/8)·a_g²·(σ_b² + sd_s²)) ) )
P̃_s      = Cal_g( σ( a_g·(θ̄_s − z_r·sd_s − μ_b) / √(1 + (π/8)·a_g²·σ_b²) ) )     z_r = risk·0.5·E[stakes]
```

`Cal_g` is the final Platt or isotonic map for group g (§2.6). Ranking uses `P̃`. Reports use `P`.

### 3.2 Seat ability θ_{m,j}

There are two job groups (§2.1): `chat` and `agentic`. For model `m` and job `j` in group `g`:

```
z_{m,j} = Φ⁻¹(clip(strength_{m,j}, 0.02, 0.98))          strength = models.json "jobs"[j] (score.py)
z̄_m     = mean_j z_{m,j}
anchor m (haiku, sonnet, opus, fable models of the alpha set):
    θ_{m,j} = θ_{t(m),g} + β·(z_{m,j} − z̄_m)              θ_{t,g} learned on outcomes (§3.5)
any other model:
    θ_{m,j} = α_g + β·z_{m,j}                             α_g, β: the link (§3.4)
sd_link,m = 0.15 (anchor) | 0.35 (basis bench) | 0.50 (blend) | 0.75 (prior)
```

- `discover` precomputes `theta: {job: [mean, sd]}` for every model in `models.json`. It is an additive key;
  `MODELS_VERSION` is unchanged. It is computed from the link in the `s1.bin` header, and discover stamps
  `catalog_sha`, the sha256 of the bundled `catalog.json.gz`.
- If that sha differs from the header's `link.catalog_sha`, every non-anchor sd gets +0.2 and the decision is flagged
  `link_stale`.
- **Fallback.** When `models.json` has no `theta`, `decide` computes it lazily with `statistics.NormalDist().inv_cdf` and
  caches it per model id.
- **No `models.json`.** Skill mode uses `data/decide.json:seat_prior`, which holds θ for the `SUGGESTED_MODELS` ids in
  each harness.

### 3.3 Effort, harness, context, personal offset

| term | value (v1) | source |
|---|---|---|
| `ε_effort` | low −0.27, medium 0, high +0.09, xhigh +0.13, max +0.15 (logits) | Aider polyglot gpt-5 low/medium/high (measured, DT), extrapolated. Placeholder for xhigh and max. |
| effort token multiplier | 0.6 / 1.0 / 1.65 / 2.5 / 3.5 | same |
| `η_{h,g}`, skill mode, agentic | native subagent 0 (sd 0.3); `codex exec` 0 (sd 0.4); headless `claude -p`, `gemini -p`, `opencode run` −0.1 (sd 0.4) | Placeholder priors. sd width from the SWE-bench same-model scaffold spread (median range 1.0 logit; a range is biased upward, so sd ≈ 0.3–0.5). |
| `η`, proxy | 0 (sd 0) | The client harness runs the loop for every candidate, so η cancels in the ranking (DT). |
| `η`, chat group | 0 (sd 0.2) | |
| `ω` (context penalty for a delegated seat) | 0.5 (sd folded into `sd_η`) | Placeholder. It is kept only if the alpha-dev ablation shows gain (§9.6). |
| `o_s` | personal seat offset, 0 at install | §10 |

Plain chat seats (local `maestro run`, proxy API without tools) are **hard-filtered** for jobs that need tools. This
stays as today in `_executable`.

### 3.4 The link (α_g, β), fitted cross-vendor

The problem the link solves: with one family per Claude tier, a free per-family offset τ on the link leaves α and β
pinned by their priors alone (critique DT #4).

**Step 1: β_pub (`evals/fit_link.py`, numpy).** Match public systems to catalog models by display name:

| source | matched systems | quantity |
|---|---|---|
| SWE-bench Verified per-instance results (`pubdata/swev`) | mini-SWE-agent / bash-only runs with a fixed scaffold, plus vendor runs | the system's 1PL θ |
| Aider polyglot (`pubdata/polyglot_leaderboard.yml`) | `diff` edit format only | logit(pass@1) |

Regress each quantity on `z = Φ⁻¹(catalog code strength)`. Pool the two slopes by inverse variance into `β_pub` and
`se_pub`. Require at least 15 matched models; if fewer match, use `β_pub = 1.0`, `se = 0.5`.

**Step 2: scale conversion.** The public logits are on an accuracy scale with discrimination 1. The prior for this
model's β is `N(β_pub / â_chat, max(se_pub, 0.3)²)`, where `â_chat` is the current estimate of `a_chat` (iterate twice).

**Step 3: joint MAP in `train_s1.py` (§8).**

- `α_g` gets a flat prior.
- β gets the prior from step 2.
- The 4 anchors give 4 points for each group, and β is shared.
- Identifiability: `θ_{haiku,chat} := 0` is fixed. `a_g ~ logN(log 1.2, 0.3²)` for chat and `logN(log 1.5, 0.4²)` for
  agentic.

**Validation (reported, not gated).**

- Leave-one-model-out on the matched public systems: AUC with the catalog-linked θ vs the fitted θ.
- Target ≤ 0.02 AUC loss.

**User-run extra anchors (§7.7).**

- Any seat answered on the verifiable linking set gets a per-seat θ fitted against frozen item b's.
- These are stored in `~/.maestro/anchors.json` with sd 0.25 and override the link for that seat.
- This is how the user's Spark, local models and Codex become identified.

### 3.5 Anchor parameters learned jointly with S1

These parameters are trained in `train_s1.py` as plain scalars alongside the network:

| parameter | count |
|---|---|
| `θ_{t,g}` | 7 (8 minus the fixed `θ_{haiku,chat}`) |
| `a_g` | 2 |
| `α_g` | 2 |
| `β` | 1 |
| `τ_1..3`, `s_d`, `c_mt`, `s_mt` | 6 |
| verbosity `v_t` | 4 |

- **Priors for the anchors.** `θ_{t,g} ~ N(α_g + β·z̄_{t}, 0.5²)`. The placeholder starting values are 0, 1.1, 2.2 and
  3.3 for haiku, sonnet, opus and fable. They are placeholders: the 1.1-logit step was an assumption in `sim2.py`,
  not a fit.
- **Without enough agentic fixtures.** When fewer than 40 agentic fixtures have outcomes, `θ_{t,agentic}` keep their
  priors and the tool-job gate in §12.4 keeps the ladder for agentic skill-mode routing.

### 3.6 Legacy difficulty and min_tier

- **Legacy difficulty.** `P(d ≥ k) = σ((μ_b − τ_k)/s_d)`. It is decoded with today's risk-aware rule (`UNDER_COST`,
  `_RISK_TIER`) into the legacy `difficulty`. That field feeds `run.py`, the ladder and the explain output. It does not
  depend on the inventory (critique DT #15).
- **min_tier.** The cheapest anchor tier with `P ≥ 0.7`, or `none`.

### 3.7 Escalation (cascade planning and Bayesian update)

**Posterior after a failure on seat s1** (probabilists' Gauss–Hermite, 7 points):

```
nodes x = (−3.750439717725742, −2.366759410734541, −1.154405394739968, 0, +…)
weights w = (0.000548268855972, 0.030757123967586, 0.240123178605013, 0.457142857142857, …) (symmetric, sum 1)
b_i = μ_b + σ_b·x_i ;   L_i = 1 − σ(a_g(θ̄_s1 − b_i))
μ' = Σ w_i L_i b_i / Σ w_i L_i ;   σ'² = Σ w_i L_i b_i² / Σ w_i L_i − μ'²
```

Check: μ = 0, σ = 1, a = 1.2, P = 0.8 at the prior mean gives μ' = 0.724 and σ' = 0.884. This matches a numerical
integral (measured).

**`escalate_to`.**

- Compute `argmax_{s2 ≠ s1} EU(s2 | posterior')`, restricted to seats with `θ̄_s2 ≥ θ̄_s1` or the same seat with feedback.
- **Same seat with feedback.** `θ + φ_fb`, where `φ_fb = 1.0` (sd 0.5) for detectable tasks only. Placeholder from Aider
  pass@2 vs pass@1, median +1.46 logits, discounted.

**Where it is applied.**

- Skill mode: `maestro outcome <id> fail`, or a detected re-delegation.
- Proxy: explicit feedback `bad`, or a conversation re-sent naming a stronger model.
- The updated `(μ, σ)` is stored in affinity (proxy) or `decisions.jsonl` (skill), and the next route of the same task
  uses it. The same task is detected by MinHash Jaccard ≥ 0.6 within 15 minutes.

---

## 4. Tokens, cost, quota, latency

### 4.1 Tokens

```
tok_ref   = exp(μ_tok + σ_tok²/2)                                     one answer (chat) or one turn (agentic)
tout_s    = tok_ref · exp(v_family(s)) · effort_mult(s) · (1 + r_m)       r_m: hidden reasoning tokens per visible token
turns     = exp(μ_turn + σ_turn²/2) if group == agentic and mode == skill, else 1
```

| quantity | value |
|---|---|
| `v_family` | Anchors: `v_t` from the alpha fit. Others: 0 at first, then learned online as an EWMA of `log(actual/predicted)` from proxy `usage.completion_tokens` (half-life 200 requests, clipped to ±1). |
| `r_m` | 0 for non-reasoning models. For reasoning models: from `usage.completion_tokens_details.reasoning_tokens` when reported (EWMA); otherwise a placeholder prior of 1.0. |
| `est_in` | Proxy: exact, `chars/4 + 800·images + tool schema chars/4`, as in today's `features()`. Skill: `len(task)/3.6` (code-heavy: `/3.0`). |

**Agentic input tokens (skill mode, delegated seats).**

```
O_h       = harness bootstrap overhead: claude-code 18k, codex 10k, opencode 12k, gemini-cli 12k, local maestro run 0.3k   (placeholders; v1.1 learns from transcripts via mine.py)
base      = est_in + 20,000·(1 − self_contained)                     re-reading files a delegated worker must do (placeholder)
tin_total = turns·(base + O_h) + 1,500·turns·(turns − 1)/2
c_hit     = (turns − 1)/turns   (providers with prompt caching), else 0
```

**Inline seat (skill mode).**

- Same formula with `O_h = 0` and `base = est_in + context_tokens`.
- `context_tokens` comes from the hint if given, otherwise 30,000 (placeholder; the caller's conversation is carried
  along).

**Proxy: the decision is per request.**

- `tin = est_in`, `tout = tout_s`, turns = 1.
- **Context feasibility** (hard).
  - Proxy: `est_in + tout_p80 ≤ 0.9·context`, where `tout_p80 = tout·exp(0.84·σ_tok)`.
  - Skill mode: the final-turn context `base + O_h + 1,500·(turns_p80 − 1) ≤ 0.9·context`.

### 4.2 Cost in dollars

| access | C_s |
|---|---|
| api | `[tin·(1 − c_hit)·p_in + tin·c_hit·p_cache_read + tout_total·p_out]/1e6`, where `p_cache_read` is the catalog value or `p_in·r_read` (Anthropic 0.1, OpenAI 0.5, others 1.0) |
| subscription | 0 (its cost is Q) |
| local | `local_usd_per_hour·T_s/3600` (default 0) |
| inline | the caller's access: API pricing if `ANTHROPIC_API_KEY`-style billing is configured for the harness (`inline_access: api`), otherwise subscription (Q) |

**Switching cost (proxy, every turn).**

- Affinity keeps the last seat for each `conv_key` on **every** turn, not only tool turns, with TTL 30 min.
- The prefix is `tin − tokens(last user message)`:
  - The warm seat pays `p_cache_read` on the prefix.
  - Any other API seat pays `p_in` on the prefix, plus the cache-write premium where the provider charges one
    (Anthropic ×1.25).
  - A local seat pays `κ·prefix/prefill_tps`. `prefill_tps` comes from discover, or defaults to 3,000 tok/s for ≤ 30B
    active parameters and 1,000 otherwise.
  - Subscription seats use the same token arithmetic inside `C_sub`.

This replaces the "tool turns only" affinity rule (`server.plan`, `aff_get(...) if f["tool_turn"]`). Stickiness now
follows from the economics.

### 4.3 Subscription quota: forecast pacing price

**Inputs per window w.**

| window kind | source |
|---|---|
| Claude `five_hour`, `seven_day`, and any `seven_day_<family>` key present in `rate_limits` | the statusline cache `claude_limits.json` |
| Codex primary and secondary | rollouts, `quota.codex_limits` |

For each window: `used_w ∈ [0, 1]`, `τ_w` (seconds to reset), `L_w` (window length: 18,000 s, 604,800 s, or Codex
`window_minutes·60`) and the sample history `H_w`.

**History (new).**

- The statusline script appends one JSON line `{"t": epoch, "five_hour": pct, "seven_day": pct}` to
  `~/.maestro/claude_limits.log` whenever a value changes. When the file passes 4,000 lines it is truncated to the last
  2,000 (`tail`).
- `quota.refresh_subscriptions` appends Codex samples to `~/.maestro/quota_hist.json`
  (`{"version": 1, "windows": {"codex/primary": [[t, used], ...]}}`), at most one sample per 300 s, keeping 2,016 per
  window.
- `decide.read_windows(home)` reads the last 64 KB of the log plus `quota_hist.json`, with an mtime cache. It is a plain
  read: `quota.py` is never imported on the `route` path.

**Price.**

```
r_window = used_w / max(600, L_w − τ_w)                                 mean burn since the window opened
r_recent = least-squares slope of used over H_w samples in the last 60 min (≥ 2 samples spanning ≥ 10 min), else None
r̂_w      = r_window                                      if r_recent is None
         = 0.5·r_recent + 0.5·r_window                    if L_w ≤ 6 h
         = 0.15·r_recent + 0.85·r_window                  otherwise                (nights and weekends: critique A&D #7c)
Δu_s(x)  = k_{w,family} · API$_s(x)                       share of window w this task uses (k learned, §4.4)
U_w      = used_w + r̂_w·τ_w + Δu_s(x)                     projected use at reset
S_w      = (1 − reserve_w) − U_w                          slack; reserve_w = quota_reserve (0.15) for windows shared with the user's own session
                                                          (Claude five_hour/seven_day, Codex primary), 0.05 otherwise
m_w      = 0.10 if L_w ≤ 6 h else 0.20
g_w      = 0                     if S_w ≥ m_w
         = g_max·(m_w − S_w)/m_w   if 0 ≤ S_w < m_w
         = g_max                 if S_w < 0
g_max    = 1.0 if an API seat with θ̄ within 0.3 logit of s is configured, else 3.0
Q_s      = max_w g_w · C_sub(s, x)
C_sub    = API cost (§4.2 formula, same tokens) of the cheapest API seat with θ̄ within 0.3 logit of s; if none, s's own model at its
           catalog API list price; if unknown, the median API price of s's category
hard skip: used_w + Δu_s(x) > 1 − reserve_w/3        reason "would exhaust the <window> window"
```

**Behaviour, from the reference implementation `design/ref/decide_ref.py`:**

| situation | U | S | g |
|---|---|---|---|
| fresh 5 h window | | | 0 |
| 42% used, 70 min to reset, burning 20%/h | 0.61 | 0.24 | 0 ("use it or lose it") |
| weekly window 80% used with 4 days left | 1.87 | | g_max |
| weekly window 30% used with 2 days left | 0.42 | | 0 |

This fixes DT's fresh-window bug: there, ρ = 1.18 on a fresh window.

**Stale or missing data.**

- If `claude_limits.json` is older than 30 min or absent, every Claude window uses `g = 0.25` and the decision is flagged
  `quota_unknown`.
- Gemini CLI has no source, so it always gets `g = 0.25`.

### 4.4 Learning `k_{w,family}`

- **Prior:** `k = 0.02` per API-$ for 5 h windows (a $50 window) and `0.003` for weekly windows ($330). These are
  placeholders, flagged `estimated`.
- **Update.** For each pair of consecutive samples bracketing at least one routed task of that family:
  `k ← EWMA(Δused / Σ predicted API$ of the routed tasks)`, half-life 20 intervals, clipped to [0.25×, 4×] of the prior.
- **Bias.** Usage by the user's own session is attributed to routed tasks, so k is biased upward. That is conservative:
  pacing starts early.
- **Where it matters.** k only enters `Δu` and the hard skip. The price itself does not depend on window capacity.

### 4.5 API rate limits and local seats

- **API token windows** (`x-ratelimit-remaining-tokens`):
  - hard skip when the estimated tokens exceed what remains;
  - when less than 10% remains, add `0.2·V(x)` to Q (replaces `quota.penalty`'s 0.8);
  - cooldown or exhaustion: hard filter, unchanged.
- **Local seats:**
  - `C = local_usd_per_hour·T/3600`;
  - latency from discover's `est_decode_tps` or the measured `--live` TPS;
  - queue congestion `κ·max(0, ttft_ewma − ttft_idle)`.
  - Maestro still never loads models. `routable` stays a hard filter.
- **Privacy.** `private: true` in the config, or a hit on a key-like pattern from `evals/mine.py:_REDACT` (copied into
  `decide.py` as `_SECRET_RX`), makes `access == local` a hard constraint. If no local seat qualifies: inline in skill
  mode, or 503 `private_no_local` in the proxy.

### 4.6 Latency

```
T_s = turns·(ttft_s + startup_s) + tout_total/tps_s + [local] tin_uncached/prefill_tps
```

| parameter | value |
|---|---|
| startup | subagent 2 s, CLI 3 s, inline 0, proxy 0 (placeholders) |
| `ttft` and `tps` priors by category | fast 0.6 s / 150, standard 0.9 / 80, strong 1.5 / 50, frontier 2.0 / 40 (placeholders) |
| overrides | proxy EWMA of observed TTFT and decode rate per model (`server.lat`); discover `--live` |

---

## 5. Utility and the decision

### 5.1 Value of a success

```
u(x)  = clip( √( tok_ref/1000 · max(1, turns) ), 0.5, 8 )
V(x)  = V0[objective] · u(x)        V0 = config wtp_usd if set, else header decide.V0[objective]
```

**Fitting V0** (`evals/alpha.py calibrate-v`, on alpha **dev**). For each objective, find the smallest `V0` on a log
grid (120 points, 0.005 to 50) at which replay success is at least the target share of always-fable success. The targets
are:

| objective | target share |
|---|---|
| quality | 0.99 |
| balanced | 0.97 |
| cost | 0.90 |

The values are written to the `s1.bin` header. Uncalibrated defaults (placeholders): cost 0.05, balanced 0.5,
quality 5.0.

### 5.2 Cost of a failure

```
d       = detectable head
ℓ(x)    = risk · interp(E[stakes]; 0→0.3, 1→1.0, 2→3.0)
F_s(x)  = d·(κ·(T_s + 45) + C_s) + (1 − d)·ℓ(x)·V(x)
```

- The implied penalty for under-routing relative to over-routing is `1 + F/V`. It is per task, not today's fixed
  `UNDER_COST = 1.5`.
- `ℓ` and `risk` are a **user risk preference**, not something learned. `alpha.py` reports how many decisions flip when
  they change (§9.6).

### 5.3 Time value

| constant | default |
|---|---|
| `κ` (interactive) | `latency_usd_per_s` = 0.002 $/s |
| `κ` (background) | `background_latency_usd_per_s` = 0.0002 $/s |

A request counts as background when:

- it carries `X-Maestro-Background: 1`; or
- the requested model is `maestro/<job>` from a Hermes `auxiliary.*` config written by `onboard.py`.

### 5.4 Floor

`P_min = interp(E[stakes]; 0→0.45, 1→0.55, 2→0.70)`. Seats below the floor are ranked after all seats at or above it.

### 5.5 Soft tools penalty

When `0.2 ≤ tools < 0.5` and the seat has no repo tools: `P̃ ← P̃·(1 − tools)`. At `tools ≥ 0.5` the existing hard
`_executable` rule applies.

### 5.6 Hints (skill mode)

| hint | effect |
|---|---|
| `--job J` | `job_p ← normalize(0.7·onehot(J) + 0.3·job_p)` |
| `--tools yes\|no` | tools ← 0.95 / 0.05 |
| `--self-contained yes\|no` | ← 0.95 / 0.05 |
| `--size s\|m\|l\|xl` | `μ_tok ← (μ_tok + log(300, 1000, 3000, 8000))/2`. For an agentic task, also `μ_turn ← (μ_turn + log(1, 4, 12, 30))/2`. |
| `--stakes low\|normal\|high` | `E[stakes] ← 0, 1, 2` |
| `--context-tokens N` | inline seat context |

- The fusion weights are fixed in v1.
- They are evaluated on the agent-written-subtask slice (§6.2).
- Hints are echoed in `decision.hints`.

### 5.7 Skill mode: `route()` with `decider == "utility"`

1. `cls = classify_rules(text)`, as today. Call `s1.posterior(...)`. If S1 is missing or invalid, use the NB adapter and
   set `signals.s1_ignored`.
2. `cands = _candidates(cls, inv, harness, cfg)`. If `--current-tier` is given, add the inline seat: θ is the current
   tier's anchor for Claude Code, otherwise the catalog θ of `seat_models(harness)[tier]`.
3. `ranked = decide.rank(cands_as_seats, posterior, ctx)`.
4. Choose `ranked[0]`. Build the seat result with the existing `_seat_result`, keeping the `fallback` semantics. Its
   order becomes the EU order, skipping the chosen seat's seat name.
5. **Tool-job gate (v1).**
   - When `posterior.group == "agentic"` and `decide.json:agentic_ready` is false, the ladder decides, exactly as today,
     with S1's job and difficulty.
   - `decide` still reports its options in `explain.shadow`.
   - `agentic_ready` is set to true only by the release step in §12.4.

### 5.8 Confidence and P(best)

**`confidence`** is the analytic probability that the top seat beats the runner-up:

```
Φ( (EU_1 − EU_2) / ((V + F̄)·√(p1(1−p1)·v1 + p2(1−p2)·v2)) )
v_i = a_g²·(σ_b² + sd_i²)/(1 + (π/8)·a_g²·(σ_b² + sd_i²))
```

**P(best)** is computed only for `--explain` and `/v1/systemone`. Draw 64 samples of `(b, θ_s)` with
`random.Random(seed=crc32(decision_id))`, and report the share of draws in which each seat is the EU-argmax.

### 5.9 Reasons

Template:

```
"{job} d{difficulty} (b {μ:+.1f}±{σ:.1f}): {seat} P {p:.2f} vs {alt} {p_alt:.2f}; {cost_phrase}{; quota_phrase}{; flags}"
```

**`cost_phrase`** names the largest of the three terms ΔC, ΔQ and κΔT. For example:

- "+$0.03 for +9 pts success"
- "same price, 40 s faster"

**`quota_phrase`** examples:

- "Claude 5h window has slack (quota ×0)"
- "weekly 80% used: priced at API ×1.0"

**Flags:**

- `below_floor`
- `link_stale`
- `quota_unknown`
- `unfamiliar task: routed conservatively` (when ood > 0.5)
- `private: local only`

---

## 6. Training data and labels

### 6.1 Corpora

| id | set | size | role | trains S1? |
|---|---|---|---|---|
| C | generated corpus (`prompt, history, context, images, lang, self_contained, hint`) | ~6,000 (1,498 so far: 25% self-contained, 2% with history, 98% English, median 21 words; measured) | panel labels; split by cluster | train split only |
| T | `evals/train.jsonl` | 1,536 | panel relabel | yes, all rows (cluster-checked against the eval sets) |
| L | `evals/prompts.jsonl` | 474 | legacy gate, untouched; panel relabel is used only to measure label noise | **no** |
| P | persona OOD set | ~600 | sealed release test; panel labels; outcomes on its self-contained part | **never** |
| V | verifiable alpha items, generated with checkers (§7.2) | 1,400 | alpha pool | train split only |
| S | agent-written subtasks: for 1,800 C items, opus writes the one-paragraph description plus hints an orchestrating agent would pass to `maestro route` | 1,800 | skill-mode distribution | train split only; test-split items are the skill-mode test slice |
| X | translations: 10% of C (600 items) into es, de, fr, pt, ja, zh, ko, ru, hi (sonnet); labels copied | 600 | multilingual | same cluster as source |
| R | paraphrase pairs: terse and verbose rewrites of 500 C-train items (haiku); labels copied | 1,000 | anti-cue augmentation | train only |
| A | alpha set (outcome-labeled subset of C ∪ V) | 2,400 pool (+200 persona) | outcomes | train split only |
| F | agentic fixtures (§7.6) | 80 generated → ≥ 60 kept | agentic outcomes, turns | by fixture, 50/50 |
| U | the user's mined prompts (`evals/mine.py`) | the user's own | local personal report only | never committed |

### 6.2 Splits and leakage (`evals/split.py`)

**Cluster ids.** Union-find over three relations:

- (a) the same generator `scenario` or persona id, when present;
- (b) MinHash over word 5-shingles: 64 permutations as `zlib.crc32(shingle, seed_i)`, LSH with 16 bands × 4 rows, union
  at an estimated Jaccard ≥ 0.5;
- (c) derived rows (S, X, R) inherit their source's cluster.

**Split function:**

```
split = int(sha256(cluster_id).hexdigest(), 16) % 100
```

| range | split |
|---|---|
| 0–54 | train |
| 55–64 | calib |
| 65–74 | dev |
| 75–99 | test |

C, T, V, S, X, R, A and F all use this one function, so no cluster crosses splits. A test-split cluster removes all of
its derived rows from training. L and P keep their own roles.

**Disjointness tests** (extending `tests/test_evals.py`):

- No train row may have estimated Jaccard ≥ 0.6 (the existing rule) with any row of L, P, or any calib, dev or test row.
- Normalized-prompt sha256 sets must be disjoint.
- Generator scenario ids in test must not appear in train.

**Frozen manifest.** `evals/manifest.json` records every id, split and cluster, plus the sha256 of every frozen file.
CI checks it.

**Ledger.** `evals/ledger.jsonl` gets one row each time a test or persona split is scored outside the CI gate:
`{ts, split, weights_sha, metrics}`.

### 6.3 Panel label schema (`evals/labels/schema.json`, version 1)

One row per item per labeler:

```json
{"id": "cc-debug-0004", "labeler": "sonnet", "schema": 1, "rubric_sha": "…",
 "job": {"debug": 0.8, "research": 0.2},
 "difficulty": 2, "min_tier": "opus",
 "needs_tools": 1, "self_contained": 0, "stakes": 1, "detectable": 1, "ambiguity": 1,
 "output_tokens": "300-1k", "agent_turns": "4-10",
 "rationale": "goroutine leak in fanout; needs reading code and pprof, a fix plus a test"}
```

**Field rules:**

- `job` has at most 3 nonzero entries, in multiples of 0.1, summing to 1.
- Labelers never see the generator `hint`, Maestro's prediction, other labels, or the slice name.

**Labeler prompt** (`evals/labels/rubric.md`). This is the exact text that defines the fields. Every label row records
its sha256.

> You label requests that a user or an AI agent sends to a model router. Label what the request needs, not what you would
> do. Use only the text shown (and its history, if any). Output one JSON object per item with exactly the fields below.
>
> **job**: a distribution over at most 3 of these jobs (multiples of 0.1, summing to 1):
> plan (architecture, system design, tradeoffs, roadmaps, migration plans) · code (write or change code: features,
> refactors, tests) · debug (find and fix what is broken: errors, stack traces, flaky tests, incidents, perf regressions) ·
> review (review code, PRs or diffs; security or quality audits) · ops (run and set things up: commands, CI, deploys,
> infra, environments, dependency upgrades across a repo) · research (look things up or investigate: docs, the web, how or
> where something works in a codebase, comparisons) · data (SQL, data analysis, spreadsheets, metrics, number crunching)
> · write (docs, specs, READMEs, emails, long-form prose) · edit (small mechanical edits: rename, format, typo, commit
> message, version bump, add a comment) · summarize (summarize, convert or pull facts out of text or data pasted into the
> request) · ask (a quick question answerable from general knowledge, or a conversational turn).
>
> **difficulty** (0–3), for a capable model working with the access it needs:
> 0 trivial: any small model gets it right first try (a lookup, a one-line edit, a greeting; e.g. "rename getUser to
> fetchUser in this snippet"). 1 routine: a competent mid-level engineer or writer gets it right first try without
> research (one file or a few clear steps; e.g. "add a --verbose flag to this argparse CLI"). 2 hard: needs careful
> multi-step reasoning, several files, or subtle constraints; good models often get part of it wrong (e.g. "this test is
> flaky only on CI, 1 run in 5; find out why"). 3 frontier: expert or novel work; the best models often fail
> (e.g. "design a lock-free MPMC queue with proofs of linearizability for our allocator").
>
> **min_tier**: the cheapest of haiku, sonnet, opus, fable that you expect to get this right at least 3 times in 4,
> given the same text (and, for repo tasks, the repo with tools). "none" if you expect all four to fail more often than not.
>
> **needs_tools** (0/1): 1 if doing it well requires reading or writing files, running commands, or browsing beyond the
> text given.
>
> **self_contained** (0/1): 1 if everything needed to complete the task is in the text (including pasted material and
> history); 0 if it depends on a repo, files, systems or facts that are not shown.
>
> **stakes** (0–2): 0 throwaway, exploratory or personal; 1 normal work that gets reviewed or is easy to revert;
> 2 production systems, security, money, data loss, legal or medical consequences, or irreversible actions.
>
> **detectable** (0/1): 1 if a wrong result would likely be noticed within minutes without expert review (fails tests,
> compile or run; obviously wrong; the user can check it directly); 0 if a plausible wrong answer could go unnoticed
> (advice, analysis, plans, subtle numbers).
>
> **ambiguity** (0–2): 0 fully specified; 1 small gaps a reasonable assumption fills; 2 underspecified: a careful
> assistant should ask before doing it.
>
> **output_tokens**: the length of a good final answer, not counting hidden reasoning: "<100", "100-300", "300-1k",
> "1k-3k", ">3k".
>
> **agent_turns**: for tasks needing tools, how many tool-using steps a capable agent takes: "0" (no tools), "1-3",
> "4-10", "11-30", ">30".
>
> **rationale**: at most 25 words, for auditing only.

### 6.4 Panel protocol and aggregation (`evals/label.py`)

**Labelers.** Three distinct models: **haiku, sonnet, opus**. Two Sonnets would have correlated errors, which Dawid–Skene
assumes away. **fable** adjudicates.

**Batches.**

- 20 items per call, mixed across slices, with the job list order shuffled for each batch.
- Each batch includes 2 honeypots from a 100-item gold set.
  - Gold set: fable labels with deliberation, then the user checks a CSV (blank = agree, about 45 min).
  - If the user does not review it, the gold set is flagged `gold_unverified`.
- A batch that gets both honeypots wrong on job is discarded and relabeled.

**Volume.** C + T + L + P + V + S ≈ 11,800 items, giving ≈ 1,770 labeling calls plus about 120 adjudication calls
(20 items per call).

**Aggregation.**

| field kind | rule |
|---|---|
| categorical (job top-1, difficulty, min_tier, stakes, ambiguity, the two buckets) | Dawid–Skene EM with one confusion matrix per labeler, Dirichlet(0.5) smoothing, 30 iterations, majority initialization. The soft label is the posterior. |
| job (final) | 0.5 × the DS posterior + 0.5 × the mean of the labelers' distributions |
| binary fields | the DS posterior |

**Adjudication.**

- It is triggered by any of:
  - no 2-of-3 job majority;
  - a difficulty spread ≥ 2;
  - a min_tier spread ≥ 2 tiers;
  - disagreement on `needs_tools` or `self_contained`;
  - a random 5% sample.
- Expect 15–25% of items.
- fable sees the item and the three anonymized labels with their rationales, and outputs the schema.
- The final label is 0.7 × fable + 0.3 × the aggregate.

**Agreement report** (`evals/labels/agreement.json`). Each field has a statistic and a threshold. A field below
threshold is trained at half weight and flagged.

| field | statistic | threshold |
|---|---|---|
| job | Fleiss κ | 0.6 |
| difficulty, stakes, ambiguity | Krippendorff α (ordinal) | 0.4 |
| tools, self_contained | Cohen κ (pairwise mean) | 0.6 |

Agreement measures consistency, not validity. Validity is measured against outcomes (§8.2).

**Hint check.** Report generator-hint vs panel agreement. If it is ≥ 95%, the panel may be reading the generator's cues,
so persona results become the headline numbers.

**Legacy noise.** The panel relabels L. Report the share of rows where the single author's labels disagree with the
panel. The legacy gate margins in §9.5 stay as they are; this is reported only.

---

## 7. The alpha set (outcomes)

### 7.1 Pilot and go/no-go (M1, before the full spend)

1. **Pilot set.** 200 pool items, stratified by job and panel difficulty, at least 50% of them verifiable.
2. **Answering.** All 4 tiers answer every item (800 answers). Haiku and sonnet answer every non-unanimous item twice.
   Grading follows §7.4 and §7.5.
3. **Measurements:**
   - the solve rate per tier;
   - the share of items solved by all four tiers or by none;
   - the repeat κ on non-unanimous items;
   - the grader false-pass and false-fail rates per tier on verifiable items;
   - the Spearman correlation of panel `min_tier` and panel difficulty with `t*`, the cheapest tier that passed;
   - ρ for NB and for keyword-only logistic regression, against IRT b fitted on the pilot.
4. **Projection** (`alpha.py pilot-report`).
   - Fit a 1PL to the pilot.
   - Resample items from the fitted b distribution, at catalog prices.
   - Simulate routers at ρ ∈ {0.4, 0.5, 0.6, 0.7}.
   - Report Α against the constant hull at each ρ.
   - Report the oracle headroom `1 − C_oracle(S_hull@fable)/C_fable`.
5. **Go/no-go rule.**
   - **Go** for difficulty-trained S1 (the full alpha set and outcome loss) if both:
     - projected Α ≥ 0.10 at ρ = 0.6;
     - oracle headroom ≥ 0.20.
   - **No-go:**
     - build the alpha set at 900 items (enough for calibration and V0);
     - ship S1 with the panel ordinal b and the decision layer;
     - spend the remaining effort on sources 2–5 in §0.3.
6. **Free extra measurement** (`evals/swe_text.py`, optional).
   - Fetch SWE-bench Verified issue texts, by either route:
     - the PR, then its linked issue, through `gh api repos/{owner}/{repo}/pulls/{n}`;
     - the Hugging Face connector, if available to the orchestrator.
   - Measure ρ(b̂_NB, b_SWE) and ρ(panel difficulty, b_SWE) against `irt/swe_irt.json`.
   - This gives a lower bound on agentic text-to-b without spending any answers.

### 7.2 Pool (2,400 self-contained items)

**Verifiable (V, 1,400).** Generated by opus. Each item comes with a checker and a reference solution, written in the
voice of the corpus personas. The interface or output contract is stated in the prompt.

| type | n | checker |
|---|---|---|
| code-fn: implement a function with a given signature | 300 | hidden unit tests (Python `unittest`) |
| debug-fix: buggy snippet + traceback/symptom → return the fixed code | 250 | hidden tests |
| data: compute over a pasted CSV/JSON table, or SQL against a given schema | 250 | result-set equality on an in-memory sqlite fixture, or numeric tolerance 1e-6 relative |
| edit: exact transformation (rename, reformat, convert) | 150 | normalized string equality |
| extract: pasted text → JSON with a given schema | 200 | field-wise equality after normalization |
| reason: math, logic or estimation with a unique answer, ending `ANSWER: <value>` | 150 | parse and compare |
| review: "return the corrected function" for a snippet with a planted defect | 100 | hidden tests |

**Validation of each item:**

- the reference solution passes;
- an empty answer and a "copy the input" answer fail;
- the checker is deterministic over 3 runs.

Items that fail validation are regenerated.

**Open (1,000).**

- Native self-contained items from C (any split; they keep their own split).
- Stratified over job × panel difficulty, capped at 40 per cell, covering plan, write, review-of-pasted-code, research
  from provided text, ask, and summarize.
- **No inlined variants.** Context invented by opus would set the outcome (critique A&D #11).

**Persona alpha.** 200 self-contained persona items, scored at release only.

### 7.3 Answering (two phases, one prompt per call)

- **Phase 1.** haiku and fable answer all 2,400 items, giving 4,800 answers.
- **Phase 2.** sonnet and opus answer:
  - every item where haiku's and fable's `y` differ;
  - plus a uniform random 25% of the agreeing items (seeded). These carry IPW weight 4.
  - Expect about 1,100 items, giving 2,200 answers.
- **Repeats.** haiku and sonnet answer again on 200 non-unanimous items, giving 400 answers, to measure repeat κ.

**Answer prompt** (fixed; one subagent call per (item, tier); never batched):

> Answer the request below as a helpful assistant in a chat. Do not use any tools and do not read or write files. If the
> request needs information you do not have, say what you would need. {output_contract}
>
> {history rendered as "Earlier user message: …" / "Earlier assistant reply: …", if any}
>
> USER REQUEST:
> {prompt}

**Recorded per answer:**

- the answer text (stored outside git, as a release asset with its sha256 in the manifest);
- `tout = round(len(answer.encode()) / (3.2 if code_ratio > 0.3 else 3.9))`.

The token count is estimated from the text, because subagent usage includes harness overhead. Answering latency is
**not** recorded: subagent wall time is not comparable. Latency comes from the priors in §4.6.

**Invalid answers.** A transcript showing tool use, or a refusal caused by the harness, is retried once. A second failure
marks the cell invalid and excludes it.

**Concurrency and quota.**

- At most 8 subagents at once.
- The orchestrator reads Maestro's own quota state and pauses while Claude `five_hour` used > 70% or `seven_day` used
  > 60%.
- Expect the work to spread over 5–10 days.

### 7.4 Grading verifiable items

- The checker is ground truth: `y ∈ {0, 1}`.
- **Grader validity subset.** A random 300 verifiable items are also LLM-graded (§7.5). This measures each grader's
  false-pass and false-fail rate per answering tier, and the grader × tier interaction (self-preference).

### 7.5 Grading open items

1. **Criteria before answers.** One opus call per item. From the prompt alone, it writes:
   - 3–6 **MUST** criteria: necessary, binary, checkable from the answer text;
   - 0–3 **DISQUALIFIERS**: for example, a factual error about X, or ignoring constraint Y;
   - a reference sketch of 1–3 sentences.

   These are stored in `evals/alpha/criteria.jsonl`.
2. **Two graders: sonnet and opus.** One call per (item, grader).
   - The grader sees the prompt, the criteria, the sketch, and all of that item's answers.
   - Answers are anonymized and shuffled as A–D. Self-identifying phrases are stripped by regex (`\b(as )?(claude|an ai
     (model|assistant) (made|created|developed) by anthropic|haiku|sonnet|opus|fable)\b`), markdown is normalized, and
     each answer is truncated at 6,000 chars with a note.
   - For each answer and criterion the grader returns yes/no, and yes/no for each disqualifier.
   - **No holistic score.** `accepted = all MUST yes ∧ no disqualifier`, computed by the script.
3. **Combination.** `y_soft` is the mean of the two graders' `accepted`. When they disagree, fable adjudicates. It sees
   both graders' criterion-level judgments, and its verdict sets `y`.
4. **Bias control.**
   - Self-preference: from the validity subset, if grader g's acceptance on its own tier exceeds its acceptance on other
     tiers by more than 0.05 relative to the checker, then on open items g's vote on its own tier's answers is replaced
     by fable's.
   - False-pass correction: let `FP_t` and `FN_t` be the per-tier false-pass and false-fail rates. If haiku's false-pass
     rate exceeds fable's by more than 3 points, replace `y` on open items by
     `P(true success | y, t) = y·(1 − FDR_t) + (1 − y)·FOR_t`, where FDR and FOR are estimated on the validity subset.
     The result is a soft label.
   - Length bias: report the within-item logistic slope of `accepted` on log(answer length).
5. **Human audit.** The user grades 150 cells blind:
   - 50 where the graders disagreed;
   - 50 from non-monotone items;
   - 50 at random.

   Report raw agreement and κ for each stratum. The **release** is blocked if raw agreement on the random stratum is
   below 80%.

### 7.6 Agentic fixtures (F)

**Generation.**

- opus writes 80 mini-repos in Python or TypeScript, each 3–15 files.
- Each has a `TASK.md` in corpus style (a bug report or a feature request), hidden tests in `.hidden/` that are not
  copied into the workspace, and a reference patch.
- **Validation:** the hidden tests fail before the reference patch and pass after it, 3 runs each.

**Runs.**

- Each tier runs as a subagent with tools in a fresh temp copy, with a limit of 40 tool calls. That is 320 runs.
- Afterwards the hidden tests are copied in and run, giving `y`.
- The number of tool calls and the answer tokens are recorded.

**Kept.** All fixtures enter training. Fixtures with all-pass or all-fail outcomes are flagged uninformative.

**Uses:**

- `θ_{t,agentic}` and `a_agentic`;
- the turns head;
- the `agentic_ready` gate (§12.4).

**Split.** By fixture cluster, 50/50.

This set is reported with confidence intervals. It is **never** a CI gate.

### 7.7 Linking set and user-run anchors

**Linking set.** `evals/alpha/linking.jsonl` holds 150 verifiable test-split items with their checkers and frozen
`(b, a_g)` from the global fit.

**Command:**

```
python evals/alpha.py collect --seat <seat id> [--n 150]
```

- Run on the user's machine, it sends only these synthetic prompts to that seat. Allowed seats: a local or network
  model through `maestro run`, `codex exec`, or an API model.
- It grades the answers locally with the checkers. Checkers run in a subprocess:
  - `python -I`;
  - empty environment;
  - temp cwd;
  - `resource` limits on POSIX: CPU 10 s, address space 1 GB;
  - the user confirms on the first run.
- It fits one θ per seat by 1-D Newton MAP, with the prior N(link θ, 0.5²).
- It writes `~/.maestro/anchors.json` (`{"version": 1, "seats": {"local:spark/qwen3-27b": {"theta_chat": 1.4, "sd": 0.22, "n": 150, "at": …}}}`).
- `decide` reads that file through an mtime cache. These are the identified anchors for non-Claude seats (DT's "5th rung").

### 7.8 Derived fields and file format

`evals/alpha/outcomes.jsonl` is committed. It holds **no answer text**, about 1 MB. One row:

```json
{"id": "v-code-0412", "cluster": "k0457", "split": "test", "source": "verifiable|open|persona|fixture", "group": "chat",
 "job": "code", "lang": "en", "phase2": "disagree|sampled|unsampled", "ipw": 1.0,
 "y": {"haiku": 0, "sonnet": 1, "opus": 1, "fable": 1}, "y_soft": {"sonnet": 1.0},
 "y2": {"haiku": 0, "sonnet": 1},
 "tin": 612, "tout": {"haiku": 410, "sonnet": 655, "opus": 720, "fable": 980},
 "graders": {"sonnet": {"haiku": 0, "...": 1}, "opus": {"...": 1}}, "adjudicated": false,
 "t_star": "sonnet", "nonmonotone": false}
```

**Sizes, budget and power.**

| quantity | value |
|---|---|
| test split | 25% of the pool ≈ 600 items, of which ≈ 210 are disagreement items |
| minimum detectable success difference (paired, 80% power) | ≈ 4.5 pts |
| subagent calls (all of M0–M2) | ≈ 14,000: answers 7,600; grading and criteria 3,750; panel 1,890; generation 430; fixture runs 320 |
| API-equivalent cost | ≈ $600–800 |

---

## 8. Training

### 8.1 `evals/train_s1.py` (numpy only; never shipped)

**Rows.**

- C-train, T, S-train, X-train, R, V-train and A-train, with masks per head.
- Alpha outcomes enter as per-tier observations. Unsampled phase-2 items contribute only haiku and fable.
- F-train contributes outcomes and turns.

**Model.**

| component | shape | init |
|---|---|---|
| E | 2^15 × 32 float32 | N(0, 0.1) |
| W1 | 32 × 64 | Xavier |
| b1 | 64 | |
| W2 | 24 × 64 | |
| b2 | 24 | |

The IRT scalars of §3.5 are trained alongside.

**Optimizer.**

- Lazy Adam on the rows of E touched in each batch: lr 0.01, β = (0.9, 0.999).
- Adam on the dense weights and scalars: lr 0.003.
- Batch size 32, at most 30 epochs.
- Early stopping (patience 4) on the joint loss of a 10% cluster holdout of train.

**Regularization.**

- Dropout 0.2 on h.
- Feature dropout 0.1.
- L2 1e-5 on touched E rows, lazily applied.
- L2 1e-4 on W1 and W2.
- Gaussian priors on the IRT scalars, as in §3.4 and §3.5.

**Hyperparameter search.**

- 5-fold grouped CV on train.
- Grid:
  - B ∈ {2^14, 2^15};
  - d ∈ {16, 32};
  - dropout ∈ {0.1, 0.3};
  - `k` features on or off;
  - char 3-grams of instruction words longer than 3 chars (namespace `c`, cap 300) on or off.
- Selection criterion: held-out outcome log-likelihood on A-train items, plus 0.5 × job NLL.
- A configuration with `c` on is adopted only if it wins by ≥ 1% of the criterion **and** the latency gate still holds.
- Estimated cost: 32 configurations × 5 folds × ~40 s ≈ 2 h on 4 CPUs.

**Determinism.**

- Seed 0. The numpy version, the seed and `data_sha256` go in the header.
- `--check` retrains and requires dev metrics within 0.5 pt and at least 99.9% identical int8 weights. Bytes are not
  compared, because of BLAS.

### 8.2 Panel validity

λ_d is fitted against outcomes:

1. On A-train, fit `τ_k`, `s_d`, `c_mt` and `s_mt` with b taken from outcomes alone (1PL with anchors fixed).
2. Validity = Spearman(panel difficulty, t*) on A-train.
3. Set λ_d:

   | validity | λ_d |
   |---|---|
   | ≥ 0.5 | 0.5 |
   | 0.3–0.5 | 0.25 |
   | < 0.3 | 0.1 |

4. Report validity, plus the AUC of panel difficulty for predicting haiku failure.

### 8.3 Calibration and export

1. Fit §2.6 on calib.
2. Fold `E' = E·W1`.
3. Compute the per-column scale as the 99.99th percentile of |E'[:, c]| / 127.
4. Quantize to int8 and store as uint8 (+128).
5. Zero the rows of buckets that no Bloom-known feature maps to.
6. Write `s1.bin` (Appendix A).

**Parity check** (stdlib inference vs the numpy float model, on 500 dev rows):

- max |Δ logit| ≤ 0.05;
- job argmax agreement ≥ 99.5%.

**Golden file.** `tests/data/s1_golden.json` holds 20 prompts with their full calibrated outputs at a tolerance of 1e-6.
It covers short and long input, CJK, an empty string, a 60k-char paste, and history.

### 8.4 Replacing NB as the legacy classifier

S1's job and difficulty replace NB in `classify_rules` only if, on prompts **dev**, at an under-routing rate matched to
NB's (sweep `U` so that under-routed ≤ NB's value):

- category accuracy ≥ NB − 1 pt;
- job accuracy ≥ NB − 1 pt.

Otherwise S1 is used only by the decider, and the legacy path stays NB.

---

## 9. Proving alpha

### 9.1 Policies (all scored on the same frozen outcomes)

| id | policy |
|---|---|
| B1 | always haiku |
| B2 | always fable |
| B3 | always sonnet |
| B3' | always opus |
| Bhull | lower convex hull of the constant (cost, success) points: a randomized mix of adjacent tiers |
| Bbest | the best constant tier at the target retention (the gain from picking the right tier, with no model) |
| B0 | uniform random over the 4 tiers |
| B4 | today's router: `classify_rules` → ladder → mapped to the 4 tiers |
| B5 | the ladder fed panel gold job and difficulty (separates classifier error from ladder error) |
| B6 | Jev, from an offline cached run when a key exists (not in CI) |
| B7 | a local-LLM logprob router (user-run) |
| R | S1 + decide, at V0[balanced] fixed from dev, plus the V-sweep curve for reporting |
| R-nb | NB adapter + decide (the value of S1) |
| R-lad | S1 + ladder (the value of decide) |
| R-noσ | point estimates only |
| O_irt | IRT oracle: true item b from a fit on the test outcomes |
| O_real | realized oracle: the cheapest tier that passed |

**Replay.**

- The 4 anchors use Claude Code subagent seats, with catalog API prices for the dollar accounting
  (`evals/alpha/inventory.json`).
- Policies run through the real code path:
  - `route()` with that inventory, for skill mode;
  - `server.App.plan()` with a stub upstream, for proxy mode.
- Quota windows are relaxed, so `Q = 0`. Pacing is evaluated separately (§9.4).

**Cost of a choice.** `(tin·p_in + tout_tier·p_out)/1e6`, using that tier's recorded tokens on the item.

**Time.** `κ·T` from §4.6 using the recorded `tout` and category priors. It is reported as a second currency:
`$ + κ·time`.

### 9.2 Metrics

All averages use the IPW weights from §7.3.

| metric | definition |
|---|---|
| S, C | mean success, mean $ per task |
| **α_$** (headline) | `1 − C_R / C_hull(S_R)` at V0[balanced]. If `S_R` exceeds the best constant, report `α_S = S_R − S_hull(C_R)` instead. |
| decomposition | Bbest saving vs B2 (from picking the tier) and α_$ vs hull (from per-task routing), shown side by side |
| Α (alpha capture) | `(AUC_R − AUC_hull)/(AUC_Oirt − AUC_hull)`, with AUC of success over log cost on [C_B1, C_B2], from the V-sweep. Reported only (an envelope over V is biased upward). |
| APGR, CPT90/95 | reported only (CPT is a step function) |
| outcome under-routing | chosen tier failed and a pricier tier passed |
| severe under-routing | chosen tier failed and a tier ≥ 2 steps up passed |
| over-spend | $ wasted where a cheaper tier passed |
| calibration | log loss, Brier, ECE (10 equal-mass bins) and calibration slope and intercept (logistic regression of y on logit P), over **all** observed (item, tier) cells. Per-group reliability tables. |
| discrimination | AUC of P over cells; Spearman ρ(μ_b, b_fit), with b_fit from a 1PL on the test outcomes alone (anchors fixed); ρ corrected for the reliability of b_fit (split-half over tiers) |
| heads | job accuracy and NLL against panel soft labels on C-test; tools F1; tokens log-RMSE and share within ×2; turns MAE on F-test |
| OOD | AUROC of `ood` separating persona, translated and gibberish probes from C-test; ECE on persona with and without inflation |
| latency | in-process p50, p99 and max over L + C-test + 50 long fixtures (60k chars) |
| slices | by job, panel difficulty, context, language, verifiable vs open, length bucket, skill-mode subtasks, persona. A slice with n < 30 shows "n/a". |

### 9.3 Statistics

- Cluster bootstrap with 2,000 resamples and seed 0, for every CI.
- Paired bootstrap for any comparison of two policies, plus McNemar for success at a fixed operating point.
- **ECE null.** In the same run, simulate y ~ Bernoulli(P) on the same cells 200 times, giving ECE_null_p95.
  - Measured null p95 for perfectly calibrated P: 0.119 at n = 150, 0.081 at 300, 0.058 at 600, 0.040 at 1,200, 0.028
    at 2,400 cells.
  - The ECE gate is `ECE ≤ ECE_null_p95 + 0.02`.

### 9.4 Pacing simulation (`evals/alpha.py pacing`)

**Setup.**

- Replay a stream of alpha-test items.
- Arrivals: bursts of 20–60 tasks per working hour on weekdays, sparse at night.
- Windows: a Claude-like 5 h plus weekly window, with `k` drawn from [0.5×, 2×] of the prior.
- Seats: the 4 Claude seats on the subscription, one Codex-like seat with its own windows, and one API seat per tier.

**Policies compared:**

- (a) subscription until exhausted, then API;
- (b) the pacing price of §4.3;
- (c) API only.

**Metrics:** success, API $, hours with an exhausted shared window, and window share wasted at reset.

**Required:**

- (b) has no more exhausted hours than (a);
- (b) spends no more API $ than (c);
- (b)'s success is within 1 pt of (a).

### 9.5 CI gate (fast job, stdlib only, no network, < 60 s)

Added to `.github/workflows/ci.yml`:

1. **Unit tests** (unchanged).
2. **Legacy gate** (unchanged command and floors):

   ```
   python evals/run.py --classifier rules --split test ...
   ```

   Once §8.4 passes, also run `--classifier s1` with floors set by the CLAUDE.md rule (test − 5, under-routing + 3).
3. **Alpha gate.** `python evals/alpha.py gate --split test` reads `evals/alpha/outcomes.jsonl`,
   `evals/alpha/gates.json` and `evals/alpha/baseline_preds.jsonl` (the committed predictions of the previous release).
   It fails if any of these holds:

   | check | fails when |
   |---|---|
   | ΔS_bal (paired, current − baseline) | lower 95% bound < −0.02 |
   | ΔC_bal / C_baseline | upper 95% bound > +0.05 |
   | Δ log loss over all cells | upper 95% bound > +0.01 nats |
   | severe outcome under-routing | point estimate > committed + 1.0 pt |
   | ECE (all cells) | > ECE_null_p95 + 0.02 |
   | calibration slope | outside [0.8, 1.25] |
   | quota monotonicity (3 fixed scenarios: relaxed, pressured, fable-exhausted) | more pressure on a window does not mean a lower or equal share routed to it |
   | `s1.bin` size | > 3 MiB |
   | integrity | header sha256 mismatch, or golden outputs off by more than 1e-6 |
   | latency | `p99(classify+S1+decide) / p99(classify_rules)` on the same runner > committed ratio × 1.5, or absolute p99 > 5 ms (in-process, after 50 warm-up calls) |

4. **Manifest and leakage checks** (§6.2).

`alpha.py gate --update` rewrites `gates.json` and `baseline_preds.jsonl`. The PR must show the paired deltas with their
CIs. CLAUDE.md and README get the new numbers, as today.

### 9.6 Release gate (`evals/alpha.py release`; manual; appends to the ledger)

**Required to set `decider: "utility"` as the default:**

- The 95% CI of α_$ against Bhull excludes 0 on alpha-test, **and** the 95% CI of α_$ against B4 excludes 0.
- The persona-alpha point estimate is ≥ 0.
- Severe outcome under-routing is ≤ 3% in every slice with n ≥ 30.
- On the skill-mode subtask slice, job accuracy is ≥ the ladder's − 2 pts.
- Human audit raw agreement on the random stratum is ≥ 80%.
- Proxy shadow mode on the user's own traffic for ≥ 14 days shows:
  - no increase in explicit-bad or escalate rates for decisions where utility and the ladder disagreed;
  - a paired comparison on the subset that had outcomes.

**Sensitivity table** (printed, not gated). For each knob (`κ`, `ℓ`, `risk`, `ω`, `V0`, `g_max`, `m_w`) at ×0.5 and ×2:
the share of decisions that flip, ΔS and ΔC. Any knob whose ablation does not improve alpha-dev log-utility is set to its
neutral value in the release (ω = 0, η = 0).

---

## 10. Personalization and online learning

### 10.1 Logs (local only, never text unless `log_prompts`)

**Skill mode.** `route` appends one line to `~/.maestro/decisions.jsonl`. Failures are ignored. It is disabled by
`log_decisions: false` and rotated at 10 MB.

```json
{"id": "d7f3a2c1", "ts": 1790000000.1, "harness": "claude-code", "job_p": {"debug": 0.81, "code": 0.12},
 "b": [1.12, 0.71], "group": "agentic", "chosen": "claude-code:opus", "p": 0.78, "options": [["claude-code:opus", 0.78, 1.31], ["codex:gpt-6.1-sol", 0.74, 1.22]],
 "hints": {"job": "debug"}, "decider": "utility", "shadow": "claude-code:sonnet", "minhash": [16 uint32], "policy": "s1:<sha8>"}
```

`minhash` holds 16 uint32 values computed over word 3-shingles of the description. It is not invertible.

**Proxy.**

- New `decisions` columns, added with `alter table ... add column` when missing:
  - `s1` (JSON ≤ 400 B)
  - `p_success`
  - `eu`
  - `decider`
  - `shadow`
  - `explore`
  - `propensity`
  - `policy`
- New table:

  ```sql
  create table if not exists outcomes (
    id integer primary key autoincrement, decision_id integer, ts real,
    signal text,   -- explicit | tests | escalate | reask | thanks | override
    value real,    -- 0..1
    weight real,
    detail text);  -- counts only, never content
  ```

### 10.2 Outcome signals

| signal | where | rule | value | weight v1 |
|---|---|---|---|---|
| explicit | both | `maestro outcome <id\|last> pass\|partial\|fail`; `maestro feedback <id> good\|bad`; UI buttons; header `X-Maestro-Feedback: <id>=good` | 1 / 0.5 / 0 | 1.0 |
| tests | skill (hook), proxy (tool messages) | a test-runner summary (`(\d+) passed`, `(\d+) failed`, `FAILED`, `Tests: … passed`, exit code) after the routed work | passed/total | 0.6 |
| escalate | both | skill: a re-route of the same task (MinHash Jaccard ≥ 0.6, ≤ 15 min) whose chosen seat is stronger, or an explicit stronger hint. Proxy: the same `conv_key` re-sent naming a stronger category or model. | 0 for the earlier decision | 0.7 |
| reask, thanks, override | proxy | lexicon and Jaccard rules (A&D §7) | 0 / 1 / 1 | **0 in v1**, logged only |

**Implicit signals stay off until validated.** `maestro learn --audit` samples 100 decisions that carry an implicit
signal and asks the user to rate them. A signal's weight becomes its audited precision once that precision is ≥ 0.8.

**Skill-mode hooks.**

- `maestro install --hooks` adds a Claude Code **SubagentStop** hook that runs `maestro outcome --from-hook`.
- That command reads the hook JSON from stdin, takes `transcript_path`, scans the last 200 transcript lines for
  test-runner summaries (counts only), and records `tests` for the most recent `subagent` decision within 30 min.
- SKILL.md tells the agent to call `maestro outcome <decision_id> pass|fail` after it checks a delegated result.

### 10.3 Fit (`learn.py`, stdlib)

**Schedule.** `maestro learn` runs it on demand. `serve` runs it in its background thread once a day.

**Method.** A replay from the last 90 days of outcomes; never incremental. Weighted logistic MAP by Newton's method over
at most 40 parameters:

| parameter | prior |
|---|---|
| seat-family offsets `o_{family,group}` | N(0, 0.5²) |
| per-job b shifts `Δb_j` | N(0, 0.4²) |
| global Platt `(A, B)` | prior at the shipped values, sd 0.2 |

The likelihood uses each logged decision's stored `(μ_b, σ_b, θ̄)`, so a refit does not depend on the current weights.

**Guards.**

- `|o| ≤ 1.0` and `|Δb| ≤ 0.5`.
- No parameter moves more than 0.2 until it has ≥ 30 weighted outcomes.
- **Time-split guard.** Fit on the first 80% of the window. If log loss on the last 20% is worse than the global model's,
  keep the previous `personal.json`. Three snapshots are kept.
- **Replay guard.** Applying the personal Platt to alpha-dev must not worsen log loss by more than 0.02 nats.
- **Reset on change.** When the weights sha changes or a seat's link θ moves by more than 0.3, that seat's offsets reset.
- Hard filters, `max_tier`, `quota_reserve` and privacy are never learned.

**Output.** `~/.maestro/personal.json`, mode 0600, at most 20 KB. `decide` reads it through an mtime cache.
`personalize: false` disables it, and `maestro learn --reset` deletes it.

### 10.4 Exploration (opt-in, proxy only)

- `explore: 0.0` by default. With `explore = ε`:
  - pick the second-best seat with probability ε;
  - only when the EU gap is < 0.05·V, `E[stakes] < 0.5` and `detectable ≥ 0.5`;
  - log the propensity, which is exact under ε-greedy.
- Off-policy evaluation is cut from v1.

### 10.5 v2 (not v1)

- Per-user ΔW2 fine-tuning on stored 64-dimension int8 hidden vectors, using explicit corrections.
- Distilling delegated systemone schemas into native heads.

---

## 11. "Our own Jev": `POST /v1/systemone`

### 11.1 Wire format

Same as TypeSafe's (checked against `typesafe-sdk` 0.7.2, DT, TME, A&D).

**Request:**

```
{model, state, questions: {name: {type: choice|score|noul, instructions?, criteria, intent?}}}
```

**Response:**

```
{model, answers: {name: {type, choice, confidence, probabilities} | {type, score, confidence, legend, probabilities} | {type, noul}},
 usage: {input_tokens, output_tokens},
 maestro: {source: {name: "native"|"delegate"}, latency_ms}}
```

The `maestro` key is extra. Strict clients still validate because it sits at the top level as an extension.

**Errors:** 422 `HTTPValidationError` with `{detail: [{loc, msg, type}]}`.

**Endpoints:**

- `maestro serve` serves `/v1/systemone` with the same authentication, Host and Origin checks as `/v1/*`.
- The CLI is `maestro decide [file|-]`.
- `GET /v1/models` adds `"models": [{name, description, release_date}]` next to OpenAI's `data`.

**Models:**

| model | behaviour |
|---|---|
| `maestro-s1` | native only; anything else gets 422 `question_not_supported` |
| `maestro-s1+llm` | native when possible, otherwise delegated |
| `jev-latest` | alias of `maestro-s1+llm` |

### 11.2 Native questions (≈ 1 ms, no LLM)

A question is native **only** in two cases:

- it has `"intent": "maestro.<name>"`; or
- its key is in the alias table below **and** its criteria have the expected shape.

There is no similarity matching.

| intent (aliases) | type | answer |
|---|---|---|
| `job` | choice, keys ⊆ JOBS | the job head, renormalized over the keys |
| `model` (`route`, `seat`) | choice, keys resolve to seat ids or `models.json` ids | `probabilities` = P(best) from 64 draws (§5.8); `choice` = argmax EU; `confidence` = P(best) of the choice |
| `difficulty` (`complexity`, `reasoning`) | score with 2–6 levels | `P(level k) = Φ((t_k − μ)/σ_eff) − Φ((t_{k−1} − μ)/σ_eff)`, where `t_k` are the training-b quantiles at k/n |
| `stakes` (`risk`) | score with 2–4 levels | the stakes ordinal, re-binned |
| `ambiguity` | score | the ambiguity ordinal |
| `needs_tools`, `self_contained`, `detectable`, `needs_vision` (from images in the state) | noul | the binary heads |
| `tokens` (`length`) | score whose legend contains token ranges (parsed with `(\d[\d,]*)\D+(\d[\d,]*)`), else default buckets | the log-normal integrated over each bucket |
| `success` | noul whose `criteria.true` names a seat | P_s |

**State.** A string becomes the task. An object reads `task` (or `prompt`), `history` and `harness`.

**OOD.** When `ood > 0.5`, native probabilities are tempered toward uniform with `T = 1 + ood`.

### 11.3 Delegated questions

The delegate is set by config `systemone_delegate`, a served local model id. It defaults to null.

**Cloud.** A cloud delegate is used only when `systemone_cloud: true` is set **and** the delegate is a cloud model id.

**Prompt.**

- The state, truncated to 6,000 chars.
- Then the instructions.
- Then the options: letters A–Z for a choice, digits 0–9 for a score, Yes/No for a noul.
- Then "Answer with the letter only." (or the digit, or Yes/No).

**Strategy probe at discover**, cached per delegate in `models.json` as `systemone_strategy`. The strategies are tried in
this order:

1. chat completions with `chat_template_kwargs: {"enable_thinking": false}`, `max_tokens: 1`, `logprobs: true`,
   `top_logprobs: 20`, `temperature: 0`;
2. chat with `add_generation_prompt: false, continue_final_message: true` and an assistant message prefilled with
   "Answer:". This form handles gpt-oss's harmony final channel and Qwen3's thinking.
3. Vote sampling: `n = 8`, `temperature: 1`, `max_tokens: 4`, with probability = vote frequency. It is slower and flagged
   `votes`.

A strategy is accepted when, on a fixed 2-option probe, the first token is an option label and at least 50% of the
probability mass is on the labels. The probe also checks that each label is a single token by comparing the variants
" A" and "A".

**Probabilities.**

- Sum the probabilities of each label's case and space variants, then renormalize over the options.
- A missing label gets `min_logprob − 2`.
- `mass` (the total probability on the labels) is reported under `maestro.mass`. When `mass < 0.5`, confidence is
  multiplied by `mass`.

**Calibration.** One temperature per delegate and question type, fitted by
`maestro decide --calibrate evals/typed_eval.jsonl`.

**Limits.**

- At most 2 concurrent requests to the delegate.
- An LRU cache of 2,048 entries, keyed by sha256 of (state, question), in `~/.maestro/s1_cache.json`.

### 11.4 Measuring "better Jev"

`evals/typed_eval.jsonl` holds about 1,500 typed questions:

- job choice, difficulty score, needs and self_contained noul, from panel labels on C-test;
- plus 200 seat-choice questions from alpha-test, where the correct answer is the cheapest tier that passed.

`python evals/typed_eval.py` reports accuracy, ECE, and p50 and p99 latency for:

- s1 native;
- the delegate;
- Jev (cached, if a key exists);
- Laya (user-run).

This is reported, not gated.

---

## 12. Integration

### 12.1 New modules (stdlib, `from __future__ import annotations`, Python ≥ 3.9)

| module | lines (target) | contents | imported by |
|---|---|---|---|
| `s1.py` | ≤ 380 | `FEAT_VERSION`, `normalize`, `features`, `load` (mmap, header validation, fallback reason), `infer`, `posterior(task, cls, hints)`, the NB and Jev adapters, `legacy(posterior)` → Classification | `router.route` (lazy), `server`, `systemone` |
| `decide.py` | ≤ 450 | `seat_of(candidate)`, `theta`, `p_success`, `tokens`, `cost`, `read_windows(home)`, `quota_price`, `latency`, `utility`, `rank`, `escalate`, `confidence`, `p_best`, `reasons`, `log_decision`, `_SECRET_RX`; reads `data/decide.json`, `anchors.json` and `personal.json` (mtime cache) | `router.route` (lazy), `server`, `systemone` |
| `learn.py` | ≤ 300 | outcome signals, `fit`, guards, `personal.json`, `audit`, `outcome` and `feedback` commands, the hook handler | `server` (thread), `commands` |
| `systemone.py` | ≤ 300 | validation, native registry, delegation strategies, cache | `server`, `commands` |
| `commands.py` | ≤ 150 | `register(sub)` for the new subcommands `s1`, `decide`, `outcome`, `feedback`, `learn`; the handlers lazily import their modules | `router.main` |
| `nb.py` | ~160 (moved) | `MODEL_*` constants, `load_model`, `model_problem`, `_nb_logits`, `classify_model`, `_RISK_TIER`. Re-exported by `router` under the same names. | `router` |
| `jev.py` | ~170 (moved) | `JevClassifier`, `JevError`, cache, `complexity`, criteria constants, `choose_jev`. Re-exported by `router`, imported lazily on the Jev path. | `router` |

**Package data.**

- `data/s1.bin` (≈ 2.25 MB).
- `data/decide.json` (≈ 4 KB, Appendix B).
- `pyproject.toml` package-data gains `"data/*.bin"`.

**Import rules.**

- `route` still never imports `providers`, `clients`, `quota`, `catalog`, `score`, `discover`, `server`, `onboard`,
  `learn` or `systemone`.
- `s1` and `decide` import only `array`, `bisect`, `hashlib`, `json`, `math`, `mmap`, `os`, `random`, `re`,
  `statistics`, `struct`, `sys`, `time` and `zlib`.
- The existing forbidden-import tests are extended with `learn` and `systemone`.

### 12.2 Changes to existing modules

**`router.py`** (net change about −300 lines, by moving `nb.py` and `jev.py` out):

- `DEFAULT_CONFIG` gains the keys in §12.3.
- `route()`:
  - When `decider == "utility"`, or `s1_shadow` is on: lazily `import s1, decide`.
  - Build the posterior, rank as in §5.7, and fill the decision keys from §1.2.
  - Write `explain.shadow` with the other decider's choice and `explain.options`.
  - Call `decide.log_decision`.
  - About 30 lines in total.
- `classify_rules` is unchanged. When §8.4 passes, it calls `s1.legacy` for job and difficulty if S1 is valid.
- `--explain` prints the posterior and the options table.
- The parser gains the `route` flags `--job`, `--tools`, `--self-contained`, `--size`, `--stakes`, `--context-tokens`
  and `--decider`, plus one line, `commands.register(sub)`.

**`server.py`:**

- `features()` adds `prev_user` (the previous user message), `n_user`, `n_tool`, `last_tool`, `asst_len` and `prefix_in`
  (estimated tokens before the last user message).
- `plan()`:
  - With `decider == "utility"`: call `s1.posterior` and keep `candidates()`'s hard filters.
  - Replace the floor test with `P̃ ≥ P_min` and `rank_key` with `decide.rank`. The `jev` path stays.
  - Affinity stores `{seat, job, μ_b, σ_b, ts}` on **every** turn; `decide` applies the switching cost (§4.2).
  - Shadow: compute the other decider and store it in `shadow`.
- `x-maestro-decision` gains `p_success`, `escalate_to` and `decision_id`.
- New routes:
  - `POST /v1/systemone`;
  - `POST /api/feedback {id, value}`;
  - `GET /api/alpha` (shadow agreement and outcome counts for the UI).
- The `learn.fit` thread runs daily.
- The `decisions` columns and the `outcomes` table from §10.1.

**`quota.py`:**

- `refresh_subscriptions` appends history samples (§4.3).
- `window_view(provider)` returns `[{name, used, reset_at, length_s}]` for `decide`, as plain dicts.
- `STATUSLINE_SCRIPT` appends the log line (§4.3).

**`score.py`:**

- `score_rec` also returns `axes8` (the blended `s`) and `axes_n` (`axw`).
- They are logged for v1.1 and not used in v1. This fixes the "unblended axes" gap.

**`discover.py`:**

- Writes `theta` per model (§3.2), `catalog_sha`, and `systemone_strategy` for local models (probed only with
  `--live`).

**`ui/index.html`:**

- good and bad buttons on the Activity rows;
- a "shadow agreement" card;
- a "Decider" setting (ladder or utility) under Routing.

**`SKILL.md`:**

- Step 2 passes the hints (`--job`, `--tools`, `--size`, `--stakes`) when the agent knows them.
- New step: after checking a delegated result, run `maestro outcome <decision_id> pass|fail`.

**`docs/DESIGN.md`:**

- A new section, "v3 Stage 3: S1 and expected-utility routing".
- New deviation entries:
  - `UNDER_COST` is superseded by F/V when `decider == "utility"`;
  - affinity on every turn, with a switching cost;
  - `models.json` gains additive keys (`theta`, `catalog_sha`);
  - the tokenizer change;
  - `route` writes `decisions.jsonl`;
  - the tool-job gate.

### 12.3 Config keys (added to `DEFAULT_CONFIG`; settable ones validated in `server.CONFIG_KEYS`)

| key | default | values |
|---|---|---|
| `decider` | `"ladder"`. Flips to `"utility"` after the release gate (§9.6). | `ladder`, `utility` |
| `classifier` | `"auto"`: Jev if `TYPESAFE_API_KEY` is set (unchanged), else S1 if it passes §8.4 and is valid, else rules | `auto`, `rules`, `s1`, `jev` |
| `wtp_usd` | `null`, meaning use the header V0 for the objective | number > 0 |
| `latency_usd_per_s` | 0.002 | ≥ 0 |
| `background_latency_usd_per_s` | 0.0002 | ≥ 0 |
| `risk` | 1.0 | 0–3 |
| `quota_reserve` | 0.15 | 0–0.5 |
| `local_usd_per_hour` | 0 | ≥ 0 |
| `inline_access` | `"subscription"` | `subscription`, `api` |
| `private` | false | bool |
| `log_decisions` | true | bool |
| `s1_shadow` | true | bool |
| `personalize` | true | bool |
| `explore` | 0.0 | 0–0.1 |
| `systemone_delegate` | null | model id |
| `systemone_cloud` | false | bool |

### 12.4 Tool-job gate (`decide.json:agentic_ready`)

The gate is set to true by the release step only if, on F-test:

- the utility router's mean cost is ≤ the ladder's;
- its success is ≥ the ladder's − 5 pts (point estimates);
- the calibration slope CI includes 1.

Until then, agentic skill-mode routing uses the ladder with S1's classification, and the utility choice is shown as
shadow.

### 12.5 CLI summary

| command | what it does |
|---|---|
| `maestro route [--job --tools --self-contained --size --stakes --context-tokens --decider] [--explain] "<task>"` | route a task; `--explain` prints the options table |
| `maestro s1 "<text>" [--json]` | print the heads |
| `maestro decide [file\|-] [--calibrate FILE]` | systemone |
| `maestro outcome <id\|last> pass\|partial\|fail [--tests P/T] [--from-hook]` | record an outcome |
| `maestro feedback <id> good\|bad\|partial` | proxy decision ids |
| `maestro learn [--report\|--audit\|--reset\|--dry-run]` | personalization |
| `maestro install --hooks` | Claude Code SubagentStop hook |

**Evals** (repo only; numpy allowed in the training scripts):

- `evals/split.py`
- `evals/label.py {batch,ingest,aggregate,agreement}`
- `evals/alpha.py {pilot,pilot-report,select,answer,grade,check,fit,replay,report,gate,release,calibrate-v,pacing,collect}`
- `evals/fit_link.py`
- `evals/train_s1.py [--check]`
- `evals/typed_eval.py`
- `evals/fetch_public.py --source swebench|aider|tbench|routerbench --path FILE`
- `evals/swe_text.py`
- `evals/run.py --classifier s1`

The subagent steps (`answer`, `grade`, `label batch`) print manifests of prompts for the orchestrating session and
ingest its replies. They never call a model API themselves.

### 12.6 Tests (stdlib `unittest`)

- **`test_s1.py`:**
  - feature hashing is stable under different `PYTHONHASHSEED` values (subprocess);
  - golden outputs to 1e-6;
  - SWAR equals the naive sum;
  - a missing or mismatched file, or a wrong `feat_version`, falls back to NB with `s1_ignored`;
  - caps hold: a 60k-char paste produces ≤ 830 features;
  - CJK text yields `j` features;
  - the non-ASCII tokenizer fix.
- **`test_decide.py`:**
  - P rises with θ and falls with b;
  - a cheaper seat with equal P wins;
  - higher stakes or lower detectability picks a stronger seat;
  - a fresh window gets g = 0 and a pressured one g > 0;
  - the hard skip on exhaustion;
  - `quota_reserve` is respected;
  - stale limits give `quota_unknown`;
  - the warm seat wins on a long prefix;
  - escalation moves μ up (μ' = 0.724, σ' = 0.884 for the §3.7 case);
  - confidence stays in [0, 1];
  - private mode means local only;
  - `decide.rank` p99 ≤ 2 ms on 40 seats.
- **`test_learn.py`:**
  - replay determinism;
  - caps, guards and reset;
  - no text in logs.
- **`test_systemone.py`:**
  - the Jev shape against `typesafe-sdk` when it is installed;
  - native intents;
  - no similarity matching (an unknown "difficulty"-like question without an alias gets 422);
  - the three delegation strategies against a fake logprob server on 127.0.0.1.
- **`test_alpha_replay.py`:**
  - a 24-item fixture with 4 tiers;
  - O_real ≥ R ≥ B1 on success;
  - the hull computation;
  - IPW weights.
- **Extended existing tests:**
  - forbidden imports for `route`;
  - `test_latency.py`: 200 routes in under 1 s with `decider: "utility"`;
  - `test_evals.py`: disjointness of the new splits.

---

## 13. Rollout and the v1 cut line

### 13.1 Milestones

| milestone | contents | exit criterion |
|---|---|---|
| M0 data foundations | `split.py`, manifest and leakage tests; panel labels for C, T, L, P, V and S, plus the gold set; agreement report; β_pub from public data; optional SWE text-ρ | job κ ≥ 0.6; manifest frozen |
| M1 pilot | 200 items × 4 tiers + grading + pilot report | go/no-go decision recorded in the ledger |
| M2 alpha set | pool, two-phase answers, grading, human audit, linking set, 80 fixtures | grader validity report; human audit ≥ 80% on the random stratum |
| M3 model | `fit_link.py`, `train_s1.py`, calibration, export, offline report with ablations | dev: ρ(μ_b, b_fit) ≥ NB's ρ + 0.05 with a CI excluding 0 (if no-go: ≥ NB's); ECE ≤ null + 0.02 |
| M4 ship behind flag | `s1.py`, `decide.py`, `nb.py` and `jev.py` moves, router and server hooks, quota history, `decisions.jsonl`, `outcome`, hooks, alpha CI gate, DESIGN.md | CI green; latency gates |
| M5 learn and systemone | `learn.py`, `systemone.py` native and delegate, typed eval | tests green; typed-eval report |
| M6 default flip | `decider: "utility"` | §9.6 release gate, including 14 days of proxy shadow |

### 13.2 Ships in v1

- **S1** with 24 outputs:
  - job, b ± σ, tools, self_contained, stakes, detectable, ambiguity, tok_out, turns;
  - crc32 features with the Bloom known-feature filter;
  - the folded int8 bag at 2^15 × 64;
  - mmap and SWAR inference;
  - calibration for every head;
  - pooling with NB.
- **decide:**
  - 1-D IRT per job group with the catalog link and anchor offsets;
  - expected utility with V0 calibrated on dev;
  - F from detectability and stakes;
  - the floor;
  - pessimism under stakes;
  - the inline seat;
  - effort as part of the seat for Codex and reasoning proxy models;
  - the switching cost;
  - forecast pacing for quota with snapshot history;
  - privacy as a hard constraint;
  - `escalate_to` and the Bayesian update on explicit failure;
  - reasons and confidence.
- **Data and evaluation:**
  - the panel (haiku, sonnet, opus, with fable adjudicating) and the agreement report;
  - the pilot with its go/no-go rule;
  - the alpha set: 2,400 pool items, two-phase, at least 50% verifiable, checker-graded or criteria-graded;
  - the 150-cell human audit;
  - 80 agentic fixtures, reported only;
  - the linking set and `alpha.py collect`;
  - the replay harness, CI gate, release gate and ledger;
  - the pacing simulation.
- **Feedback:** explicit, tests and escalate signals; `decisions.jsonl`; the `outcome` command and hook; `learn.py`
  offsets with guards.
- **systemone:** `/v1/systemone` with native intents and the probed logprob delegate.

### 13.3 Cut from v1 (later)

- The 8-axis demand π and per-axis abilities. In v1.1, once the outcomes cover ≥ 2 vendors or local seats with different
  axis profiles.
- Per-item discrimination and per-job `a_j`.
- Pass@anchor heads.
- Scope and turn-type heads.
- Implicit signals with nonzero weight, until an audit validates them.
- Exploration by default, and OPE.
- Per-user ΔW2.
- Distilled multilingual initialization (Model2Vec-style).
- NVIDIA, DeBERTa or Laya teacher distillation, and RouterBench, RouterEval or LMSYS pretraining. Each is adopted later
  only if dev ρ improves by ≥ 0.05.
- Automatic cascade *execution*. Only the plan ships.
- Terminal-Bench ingestion. The script exists and is run by the user; until then η stays at its priors.
- Learning delegation overhead from transcripts.
- Distilling systemone schemas.
- Agentic utility routing in skill mode, until `agentic_ready`.

---

## 14. Risks and failure modes

| risk | how it shows up | detection | mitigation |
|---|---|---|---|
| Little per-task alpha exists (easy distribution, ceiling) | α_$ against the hull is about 0 | pilot projection; the release gate | go/no-go; sources 2–5 of §0.3 do not depend on b |
| Claude grading Claude (self-preference, over-credit, length) | inflated success for strong or cheap tiers, fake alpha | validity subset vs checkers; grader × tier interaction; length slope; human audit | ≥ 50% checker-graded items; binary criteria with no holistic score; per-tier FP correction; cross-grader replacement |
| Synthetic ≠ real (generator cues, short prompts) | good on C, weak on the user's own prompts | persona alpha; hint-agreement check; mined-prompt personal report | paraphrase pairs; panel blind to hints; headline uses persona if the gap > 50% |
| Agentic decisions rest on priors | wrong tool-job tiers | F-test calibration; outcome hook | the `agentic_ready` gate keeps the ladder; wider σ |
| Non-Claude θ comes from the link only | wrong vendor or local choices | LOMO AUC on public data; `collect` anchors | β from public data; sd by basis; anchors.json; personal offsets |
| Catalog re-ranking drifts the link | stale θ | `catalog_sha` mismatch | +0.2 sd, `link_stale` flag, refit at release |
| Subscription capacity unknown | early or late pacing | predicted vs actual Δused in logs | the price does not depend on capacity; k only gates near exhaustion and is biased conservative |
| Hash collisions, novel identifiers | confident noise | OOD AUROC; golden tests | Bloom filter of known features; feature dropout in training |
| Latency creep (long inputs) | p99 grows | relative and absolute p99 gates including 60k fixtures | scan caps; `str.find` split; bounded features |
| Feedback loop and drift | personalization confirms its own picks | time-split and replay guards | explicit, test and escalate signals only; caps; resets; no implicit weight until audited |
| Teacher budget burns the user's windows | the user's own session is blocked | the orchestrator reads quota | pause at 70% (5 h) and 60% (weekly); spread over days |
| Git growth from binary weights | about 2.25 MB per retrain | size gate | retrain at most once per release; `.gitattributes` binary |
| Delegate is a thinking or harmony model | `max_tokens: 1` reads a reasoning token | discover probe | three strategies; 422 if none passes |
| Privacy | task text leaks | tests assert no text in logs | `decisions.jsonl` and the DB hold heads and MinHash only; delegation local-only by default; `private` mode |

---

## Appendix A. `s1.bin` format

```
offset 0      b"MS1\x00"
offset 4      u32 LE  header_len
offset 8      header JSON (UTF-8), padded with spaces to a multiple of 64 bytes
then sections at the offsets the header lists (each 64-byte aligned):
  bloom   M/8 bytes                     (bit i of byte i>>3, LSB first)
  emb     B × H bytes, uint8 = int8+128, row-major
  dense   float32 LE: scale[H], b1[H], W2[K×H] row-major, b2[K]
```

Header JSON:

```json
{"format": 1, "feat_version": 1, "B": 32768, "H": 64, "K": 24, "bloom": {"M": 1048576, "k": 2, "seed2": 1540483477},
 "sections": {"bloom": [O1, 131072], "emb": [O2, 2097152], "dense": [O3, 6752]}, "sha256": {"bloom": "…", "emb": "…", "dense": "…"},
 "jobs": ["plan","code","debug","review","ops","research","data","write","edit","summarize","ask"],
 "heads": {"job": [0, 11], "b": [11, 12], "tools": 13, "self_contained": 14, "stakes": [15, 16], "detectable": 17,
           "ambiguity": [18, 19], "tok": [20, 21], "turns": [22, 23]},
 "calib": {"T_job": 1.1, "w_nb": 0.25, "platt": {"tools": [1.0, 0.0], "self_contained": [1.0, 0.0], "stakes1": [1, 0], "stakes2": [1, 0],
           "detectable": [1, 0], "amb1": [1, 0], "amb2": [1, 0]}, "c_sigma_b": 1.2, "ood_gamma": 0.8, "tok_sd": 1.0, "turn_sd": 1.0,
           "final": {"chat": {"platt": [1.0, 0.0], "iso": null}, "agentic": {"platt": [1.0, 0.0], "iso": null}}},
 "ordinal": {"tau": [-1.5, 0.2, 1.9], "s_d": 0.9, "c_mt": 0.0, "s_mt": 0.8, "b_quantiles": [-3.1, -1.2, 0.0, 1.3, 2.9]},
 "irt": {"theta": {"chat": {"haiku": 0.0, "sonnet": 1.1, "opus": 2.2, "fable": 3.3}, "agentic": {"haiku": 0.0, "sonnet": 1.1, "opus": 2.2, "fable": 3.3}},
         "a": {"chat": 1.2, "agentic": 1.5}, "anchor_models": {"haiku": "anthropic/claude-haiku-4-5", "...": "..."},
         "verbosity": {"haiku": 0.0, "sonnet": 0.1, "opus": 0.15, "fable": 0.3}},
 "link": {"alpha": {"chat": -2.0, "agentic": -2.4}, "beta": 1.6, "catalog_sha": "…", "sd_basis": {"anchor": 0.15, "bench": 0.35, "blend": 0.5, "prior": 0.75},
          "beta_pub": [1.9, 0.3], "sources": ["swebench-verified 2026-09", "aider-polyglot 2026-09"]},
 "decide": {"V0": {"cost": 0.05, "balanced": 0.5, "quality": 5.0}, "V0_calibrated": false},
 "train": {"data_sha256": "…", "n": {"panel": 9100, "alpha_items": 1320, "fixtures": 30}, "numpy": "2.x", "seed": 0,
           "config": {"d": 32, "dropout": 0.2, "kw": true, "char3": false}, "metrics": {"dev": {}}}}
```

Every numeric value above is a placeholder, written by the fitting steps. The loader validates these and falls back to
NB with a reason if any check fails:

- `format`;
- `feat_version == s1.FEAT_VERSION`;
- `jobs == router.JOBS`;
- the section sizes;
- `K`.

Section sha256 values are checked in `doctor` and the tests, not in `route`.

## Appendix B. `data/decide.json` (shipped; placeholders marked)

```json
{"version": 1, "agentic_ready": false,
 "groups": {"agentic_jobs": ["code", "debug", "review", "ops", "research", "edit"]},
 "effort": {"low": [-0.27, 0.6], "medium": [0.0, 1.0], "high": [0.09, 1.65], "xhigh": [0.13, 2.5], "max": [0.15, 3.5],
            "_source": "aider polyglot gpt-5 low/medium/high (measured); xhigh/max placeholder"},
 "harness": {"subagent": [0.0, 0.3], "codex-cli": [0.0, 0.4], "cli": [-0.1, 0.4], "proxy": [0.0, 0.0], "chat_group_sd": 0.2,
             "_source": "placeholder; sd from SWE-bench same-model scaffold spread"},
 "context_penalty": 0.5, "retry_feedback": [1.0, 0.5],
 "overhead_tokens": {"claude-code": 18000, "codex": 10000, "opencode": 12000, "gemini-cli": 12000, "local": 300, "inline": 0},
 "reread_tokens": 20000, "turn_step_tokens": 1500, "inline_context_tokens": 30000,
 "cache_read_ratio": {"anthropic": 0.1, "openai": 0.5, "default": 1.0}, "cache_write_ratio": {"anthropic": 1.25},
 "latency_prior": {"fast": [0.6, 150], "standard": [0.9, 80], "strong": [1.5, 50], "frontier": [2.0, 40],
                   "startup_s": {"subagent": 2, "cli": 3, "inline": 0, "proxy": 0}},
 "prefill_tps": {"small": 3000, "large": 1000},
 "stakes_loss": [0.3, 1.0, 3.0], "p_min": [0.45, 0.55, 0.70], "retry_secs": 45,
 "pacing": {"reserve_shared": 0.15, "reserve_other": 0.05, "margin_short": 0.10, "margin_long": 0.20, "g_max_sub": 1.0, "g_max": 3.0,
            "g_unknown": 0.25, "k_prior": {"short": 0.02, "long": 0.003}, "stale_s": 1800},
 "job_priors": {"tools": {"code": 1, "debug": 1, "review": 1, "ops": 1, "research": 0.6, "edit": 0.8},
                "detectable": {"code": 0.75, "debug": 0.75, "ops": 0.75, "edit": 0.8, "data": 0.6, "ask": 0.5, "summarize": 0.5,
                               "review": 0.4, "plan": 0.3, "write": 0.3, "research": 0.3},
                "tok_out": {"ask": 250, "edit": 300, "summarize": 400, "write": 1200, "plan": 1500, "code": 1200, "debug": 900,
                            "review": 900, "ops": 700, "research": 1000, "data": 800}},
 "nb_adapter": {"c": [-2.0, -0.5, 1.0, 2.5], "sd": 0.8},
 "seat_prior": {"claude-code:haiku": {"chat": 0.0, "agentic": 0.0}, "claude-code:sonnet": {"chat": 1.1, "agentic": 1.1},
                "claude-code:opus": {"chat": 2.2, "agentic": 2.2}, "claude-code:fable": {"chat": 3.3, "agentic": 3.3}},
 "gh7": {"x": [-3.750439717725742, -2.366759410734541, -1.154405394739968, 0.0, 1.154405394739968, 2.366759410734541, 3.750439717725742],
         "w": [0.000548268855972, 0.030757123967586, 0.240123178605013, 0.457142857142857, 0.240123178605013, 0.030757123967586, 0.000548268855972]}}
```

## Appendix C. Other file formats

**`evals/alpha/gates.json`:**

```json
{"split": "test", "n_items": 600, "n_cells": 1700, "weights_sha": "…", "V0_balanced": 0.5,
 "baseline": {"S_bal": 0.0, "C_bal": 0.0, "logloss": 0.0, "severe": 0.0}, "latency_ratio_p99": 0.0,
 "bin_max_bytes": 3145728, "abs_p99_ms": 5.0}
```

**`evals/alpha/baseline_preds.jsonl`:** one row per test item:

```json
{"id": "...", "chosen": "sonnet", "p": {"haiku": .., "sonnet": .., "opus": .., "fable": ..}}
```

**`~/.maestro/personal.json`:**

```json
{"version": 1, "weights_sha": "…", "fitted_at": 1790000000, "n_outcomes": 412,
 "seat_offsets": {"anthropic/opus|agentic": 0.21, "local:spark/gpt-oss-120b|chat": -0.35},
 "job_b_shift": {"debug": 0.3}, "platt": [1.04, -0.08], "verbosity": {"anthropic": 0.12},
 "k": {"claude/five_hour|opus": 0.018}, "guard": {"global_ll": 0.0, "personal_ll": 0.0}, "snapshots": 3}
```

**`~/.maestro/outcomes.jsonl`** (skill mode):

```json
{"decision_id": "d7f3a2c1", "ts": 1790000300, "signal": "explicit", "value": 1.0, "weight": 1.0, "detail": {"tests": [12, 13]}}
```

## Appendix D. Evidence used

| item | value | source |
|---|---|---|
| SWE-bench Verified IRT (133 × 500) | 1PL held-out AUC 0.943, ECE 0.010, against AUC 0.707 from system accuracy alone; sd(b) 2.95, sd(θ) 1.99; 6.4% of items unsolved | `irt/fit_swe.py` |
| Alpha capture against the hull at catalog prices | Α = 0.15 / 0.30 / 0.62 at ρ = 0.5 / 0.7 / 0.9 | `irt/alpha_ci2.py` |
| Easy-mix caveat | with b shifted −3, the best constant tier keeps 95% of success at 40% of the cost | critique, A&D |
| S1 engine | p50 96 µs and p99 330 µs from raw text; int8 parity 237/237 | `s1/bench2.py`, `s1/exp2.py` |
| Prototype S1 vs NB on dev (1.5k synthetic labels) | difficulty +4–10 pts; job on par once pooled; under-routing worse at matched U; still climbing about 4 pts per doubling of data | `s1/exp*.py`, `alpha/exp_out.txt` |
| ECE of a perfectly calibrated model, by n | the table in §9.3 | measured here |
| Posterior after a failure (§3.7 case) | (0.724, 0.884) | measured here; Gauss–Hermite matches a numerical integral |
| Pacing behaviour | the table in §4.3 | `design/ref/decide_ref.py` |
| Corpus facts (1,498 rows) | 25% self-contained; 2% with history; 98% English; median 21 words | measured here |
