"""The proxy end to end: fake upstream servers (OpenAI and Anthropic styles), fake CLI seats, the admin API, the decision log."""
import http.client
import json
import os
import sqlite3
import stat
import tempfile
import threading
import time
import unittest
from unittest import mock

import helpers  # noqa: F401
from helpers import maestro as core
from fake_http import FakeServer, fake_cli, json_reply

from maestro import clients as C
from maestro import providers as P
from maestro import quota as Q
from maestro import server as S

OPENAI_KEY, GROQ_KEY, ANT_KEY = "sk-openai-SECRET-1111", "gsk-groq-SECRET-2222", "sk-ant-SECRET-3333"
ALL_KEYS = (OPENAI_KEY, GROQ_KEY, ANT_KEY)


def E(mid, cat="standard", G=0.5, cost=(1.0, 4.0), access=None, **kw):
    """A models.json entry (spec section 9) with job strengths all equal to G unless `jobs` is given."""
    access = access or ("local" if mid.startswith("local:") else "subscription-cli" if mid.startswith("cli:") else "api")
    provider, _, model = mid.partition("/")
    if access == "local":
        provider = provider
    e = {"id": mid, "provider": provider, "model": model, "access": access, "style": kw.pop("style", {"local": "local", "subscription-cli": "cli"}.get(access, "openai")),
         "tools_ok": access != "subscription-cli", "vision": False, "context": 200000, "cost_in": 0.0 if access != "api" else cost[0],
         "cost_out": 0.0 if access != "api" else cost[1], "G": G, "jobs": {j: G for j in core.JOBS}, "category": cat, "serves": cat, "basis": "bench",
         "benchmarks": ["SWE-Bench Pro"], "enabled": True, "override": None}
    e.update(kw)
    return e


def sse(*objs):
    return [b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs] + [b"data: [DONE]\n\n"]


def chunk(model, delta, finish=None, **kw):
    return dict({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, **kw)


def completion(model, text="pong", usage=(5, 2)):
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


def parse_sse(raw):
    out, done = [], False
    for block in raw.decode().split("\n\n"):
        block = block.strip()
        if block == "data: [DONE]":
            done = True
        elif block.startswith("data: "):
            out.append(json.loads(block[6:]))
    return out, done


class Up:
    """A fake upstream: one FakeServer with a scriptable answer. `mode`: ok | 429 | 500 | empty; `stream_chunks` for SSE."""

    def __init__(self, name, text="pong", mode="ok", stream=None, headers=None, custom=None):
        self.name, self.text, self.mode, self.stream_chunks, self.headers, self.custom = name, text, mode, stream, headers or {}, custom
        self.srv = FakeServer(self.handle)

    @property
    def url(self):
        return self.srv.url

    @property
    def requests(self):
        return [r for r in self.srv.requests if r["method"] == "POST"]

    def handle(self, req):
        body = req["json"] or {}
        if self.custom:
            return self.custom(req)
        if self.mode == "429":
            return json_reply({"error": {"message": "slow down", "type": "rate_limit"}}, 429, {"retry-after": "30"})
        if self.mode == "500":
            return json_reply({"error": {"message": "boom"}}, 500)
        if self.mode == "401":
            return json_reply({"error": {"message": "bad key " + req["headers"].get("authorization", "")}}, 401)
        if body.get("stream"):
            if self.mode == "empty":
                return 200, {"Content-Type": "text/event-stream"}, b""
            chunks = self.stream_chunks or sse(chunk(body["model"], {"role": "assistant", "content": self.text}),
                                               chunk(body["model"], {}, "stop"))
            return 200, dict({"Content-Type": "text/event-stream"}, **self.headers), iter(chunks)
        return json_reply(completion(body.get("model", "?"), self.text), 200, self.headers)

    def close(self):
        self.srv.close()


class AnthropicUp(Up):
    def handle(self, req):
        body = req["json"] or {}
        if body.get("stream"):
            ev = self.stream_chunks
            return 200, {"Content-Type": "text/event-stream"}, iter([("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e))).encode() for e in ev])
        return json_reply({"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": self.text}],
                           "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 3}})


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    raw = r.read()
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, raw)
    conn.close()
    return out


def jcall(port, method, path, body=None, headers=None):
    st, h, raw = call(port, method, path, body, headers)
    try:
        return st, h, json.loads(raw)
    except ValueError:
        return st, h, raw


class ServerCase(unittest.TestCase):
    """A running server over a temp MAESTRO_HOME. Subclasses call self.start(models, cfg, ups)."""
    api_key = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.bin = os.path.join(self.home, "bin")
        os.makedirs(self.bin)
        self.ups, self.httpd, self.app, self.serving = [], None, None, False
        self.addCleanup(self.stop)

    def up(self, cls=Up, *a, **kw):
        u = cls(*a, **kw)
        self.ups.append(u)
        return u

    def stop(self):
        if self.httpd:
            self.app.stop.set()
            if self.serving:
                self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        for u in self.ups:
            u.close()

    def launch(self, host="127.0.0.1", env=None, **kw):
        """Build a server over self.home and serve it in a thread."""
        env = dict({"HOME": self.home, "MAESTRO_HOME": self.home, "PATH": self.bin}, **(env or {}))
        self.httpd, self.app = S.make_server(host, 0, home=self.home, env=env, refresh_s=kw.pop("refresh_s", 0), **kw)
        self.serving = True
        threading.Thread(target=lambda: self.httpd.serve_forever(poll_interval=0.02), daemon=True).start()
        self.port, self.token = self.app.port, self.app.token

    def write(self, name, obj):
        with open(os.path.join(self.home, name), "w") as f:
            json.dump(obj, f)

    def start(self, models, cfg=None, hosts=None, host="127.0.0.1", env_extra=None, **kw):
        self.write("models.json", {"version": 1, "generated_at": time.time(), "catalog_as_of": "2026-09-30", "hardware": {}, "models": models,
                                   "local_fit": kw.pop("local_fit", {})})
        self.write("config.json", dict({"providers": hosts or {}}, **(cfg or {})))
        env = {"OPENAI_API_KEY": OPENAI_KEY, "GROQ_API_KEY": GROQ_KEY, "ANTHROPIC_API_KEY": ANT_KEY}
        env.update(env_extra or {})
        Q.reset_state(self.home)
        self.launch(host, env, **kw)

    def chat(self, body, headers=None, **kw):
        return jcall(self.port, "POST", "/v1/chat/completions", dict({"model": "maestro/auto"}, **body), headers)

    def admin(self, method, path, body=None, token=True, headers=None):
        h = dict(headers or {})
        if token:
            h["X-Maestro-Token"] = self.token
        return jcall(self.port, method, path, body, h)

    def rows(self):
        db = sqlite3.connect(os.path.join(self.home, "maestro.db"))
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("select * from decisions order by id")]

    def decision(self, headers):
        return json.loads(headers["x-maestro-decision"])


def fixed(job="debug", difficulty=2, confidence=0.9, needs=()):
    """Patch the rules classifier so a test does not depend on the wording of its prompt."""
    return mock.patch.object(core, "classify_rules", lambda text: core.Classification(job, difficulty, confidence, needs))


MSG = {"messages": [{"role": "user", "content": "Why does the login test fail?"}]}


class Routing(ServerCase):
    def two(self, **cfg):
        a, b = self.up(name="mini", text="from mini"), self.up(name="big", text="from big")
        models = [E("openai/gpt-mini", "fast", 0.40, (0.15, 0.6)), E("openai/gpt-big", "strong", 0.75, (10.0, 40.0)),
                  E("groq/llama-strong", "strong", 0.60, (2.0, 6.0))]
        self.start(models, dict(cfg, providers={"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}))
        return a, b

    def test_models_lists_meta_and_concrete_models_but_not_disabled_ones(self):
        self.two(models_overrides={"openai/gpt-mini": {"enabled": False}})
        st, _, j = jcall(self.port, "GET", "/v1/models")
        ids = [m["id"] for m in j["data"]]
        self.assertEqual(st, 200)
        for n in ("maestro/auto", "maestro/local", "maestro/fast", "maestro/standard", "maestro/strong", "maestro/frontier", "maestro/code", "maestro/debug"):
            self.assertIn(n, ids)
        self.assertIn("openai/gpt-big", ids)
        self.assertNotIn("openai/gpt-mini", ids)
        self.assertEqual(next(m for m in j["data"] if m["id"] == "openai/gpt-big")["context_length"], 200000)
        self.assertTrue(all(m["context_length"] for m in j["data"] if m["id"].startswith("maestro/")))

    def test_auto_picks_the_cheapest_model_that_reaches_the_floor(self):
        mini, big = self.two()
        with fixed("debug", 2):   # debug d2 -> strong (floor .52): gpt-mini (.40) is too weak, llama-strong is 10x cheaper than gpt-big
            st, h, j = self.chat(MSG)
        self.assertEqual(st, 200, j)
        self.assertEqual(j["model"], "groq/llama-strong")
        self.assertEqual(j["choices"][0]["message"]["content"], "from big")   # the fake at the groq base url
        d = self.decision(h)
        self.assertEqual((d["job"], d["difficulty"], d["model"], d["fallbacks"]), ("debug", 2, "groq/llama-strong", 0))
        self.assertEqual(big.requests[0]["json"]["model"], "llama-strong")   # the upstream id, not maestro/auto

    def test_easy_work_goes_to_the_cheap_model(self):
        mini, _ = self.two()
        with fixed("edit", 0):
            st, h, j = self.chat(MSG)
        self.assertEqual(j["model"], "openai/gpt-mini")
        self.assertEqual(self.decision(h)["category"], "fast")

    def test_category_and_job_model_names_narrow_the_choice(self):
        self.two()
        with fixed("ask", 0):
            self.assertEqual(self.chat(dict(MSG, model="maestro/strong"))[2]["model"], "groq/llama-strong")
            st, h, j = self.chat(dict(MSG, model="maestro/plan"))
        d = self.decision(h)
        self.assertEqual((d["job"], d["cell"] if "cell" in d else "standard"), ("plan", "standard"))   # the job name overrides the classifier's job
        self.assertEqual(j["model"], "groq/llama-strong")   # plan d0 -> standard: gpt-mini reaches the .35 floor but is a fast-tier model, so the cheapest model of standard tier or above

    def test_frontier_floor_falls_to_the_only_model_that_reaches_it(self):
        self.two()
        st, h, j = self.chat(dict(MSG, model="maestro/frontier"))
        self.assertEqual(j["model"], "openai/gpt-big")

    def test_concrete_model_passes_through_and_unknown_model_is_a_404(self):
        mini, big = self.two()
        st, h, j = self.chat(dict(MSG, model="openai/gpt-big"))
        self.assertEqual((st, j["model"]), (200, "openai/gpt-big"))
        self.assertEqual(self.decision(h)["requested"], "openai/gpt-big")
        st, _, j = self.chat(dict(MSG, model="nope/none"))
        self.assertEqual(st, 404)
        self.assertEqual(j["error"]["code"], "model_not_found")

    def test_quality_objective_prefers_the_stronger_model(self):
        self.two(objective="quality")
        with fixed("debug", 2):
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/gpt-big")

    def test_max_tier_hides_stronger_cloud_models(self):
        self.two(max_tier="standard")
        st, _, j = self.chat(dict(MSG, model="maestro/frontier"))
        self.assertEqual(st, 200)
        self.assertNotEqual(j["model"], "openai/gpt-big")

    def test_category_override_changes_where_a_model_is_used(self):
        self.two()
        st, _, j = self.admin("POST", "/api/models/openai%2Fgpt-mini", {"category": "frontier"})
        self.assertEqual(st, 200, j)
        self.assertEqual(self.chat(dict(MSG, model="maestro/frontier"))[2]["model"], "openai/gpt-mini")   # the user's word beats the score
        self.assertEqual(core.load_config(self.home)["models_overrides"]["openai/gpt-mini"], {"category": "frontier"})
        self.admin("POST", "/api/models/openai%2Fgpt-mini", {"category": None, "enabled": False})
        self.assertNotIn("openai/gpt-mini", [m["id"] for m in jcall(self.port, "GET", "/v1/models")[2]["data"]])
        self.assertEqual(core.load_config(self.home)["models_overrides"], {"openai/gpt-mini": {"enabled": False}})
        self.admin("POST", "/api/models/openai%2Fgpt-mini", {"enabled": True})
        self.assertEqual(core.load_config(self.home)["models_overrides"], {})   # back to defaults: the entry is gone

    def test_prefer_local_uses_a_local_model_that_is_strong_enough(self):
        loc, cloud = self.up(name="spark", text="from spark"), self.up(name="cloud", text="from cloud")
        self.start([E("openai/gpt-big", "strong", 0.75, (10.0, 40.0)), E("local:spark/gpt-oss:120b", "local", 0.58, serves="standard")],
                   {"providers": {"openai": {"base_url": cloud.url}}, "local_endpoints": [{"name": "spark", "url": loc.url, "kind": "openai"}]})
        with fixed("debug", 0):   # debug d0 = standard: the local model's size earns it standard
            st, h, j = self.chat(MSG)
            self.assertEqual(j["model"], "local:spark/gpt-oss:120b")
            self.assertEqual(j["choices"][0]["message"]["content"], "from spark")
            self.assertEqual(loc.requests[0]["path"], "/v1/chat/completions")
            self.assertEqual(loc.requests[0]["json"]["model"], "gpt-oss:120b")
        with fixed("debug", 3):   # strong is beyond what it serves, even though its score for debug is above the floor
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/gpt-big")
        with fixed("debug", 0):
            self.admin("POST", "/api/config", {"prefer_local": False})
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/gpt-big")   # cloud only
            self.assertEqual(self.chat(dict(MSG, model="maestro/local"))[2]["model"], "local:spark/gpt-oss:120b")   # asked for explicitly

    def test_small_local_models_only_take_local_cells_and_specialists_are_judged_by_their_job(self):
        loc, cloud = self.up(name="spark", text="from spark"), self.up(name="cloud", text="from cloud")
        tiny = E("local:spark/tiny:3b", "local", 0.3, serves="local")
        coder = E("local:spark/coder:30b", "specialist", 0.3, serves="fast", jobs=dict({j: 0.3 for j in core.JOBS}, code=0.7))
        self.start([E("openai/gpt-mini", "fast", 0.40, (0.15, 0.6)), E("openai/gpt-big", "strong", 0.75, (10.0, 40.0)), tiny, coder],
                   {"providers": {"openai": {"base_url": cloud.url}}, "local_endpoints": [{"name": "spark", "url": loc.url, "kind": "openai"}]})
        with fixed("summarize", 1):   # fast cell: the tiny model is not enough, the coder is not tuned for this job (.3 >= .15, but see below)
            self.assertNotEqual(self.chat(MSG)[2]["model"], "local:spark/tiny:3b")
        with fixed("code", 2):   # strong cell: the specialist's code score (.7) reaches the floor, its size does not matter
            self.assertEqual(self.chat(MSG)[2]["model"], "local:spark/coder:30b")
        with fixed("plan", 2):
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/gpt-big")

    def test_local_ladder_cell_prefers_local_and_falls_back_to_fast(self):
        loc = self.up(name="spark", text="from spark")
        mini = self.up(name="mini", text="from mini")
        self.start([E("openai/gpt-mini", "fast", 0.40, (0.15, 0.6)), E("local:spark/tiny:3b", "local", 0.2, serves="local")],
                   {"providers": {"openai": {"base_url": mini.url}}, "local_endpoints": [{"name": "spark", "url": loc.url, "kind": "openai"}]})
        with fixed("ask", 0):   # ask d0 = local
            self.assertEqual(self.chat(MSG)[2]["model"], "local:spark/tiny:3b")
        self.admin("POST", "/api/models/" + "local%3Aspark%2Ftiny%3A3b", {"enabled": False})
        with fixed("ask", 0):
            st, h, j = self.chat(MSG)
        self.assertEqual(j["model"], "openai/gpt-mini")
        self.assertIn("using fast", self.decision(h)["reason"])

    def test_prefer_local_off_turns_the_ladders_local_cell_into_fast(self):
        loc, mini = self.up(name="spark", text="from spark"), self.up(name="mini", text="from mini")
        self.start([E("openai/gpt-mini", "fast", 0.40, (0.15, 0.6)), E("local:spark/tiny:3b", "local", 0.2, serves="local")],
                   {"providers": {"openai": {"base_url": mini.url}}, "local_endpoints": [{"name": "spark", "url": loc.url, "kind": "openai"}],
                    "prefer_local": False})
        with fixed("ask", 0):   # ask d0 = the ladder's local cell
            st, h, j = self.chat(MSG)
            self.assertEqual(j["model"], "openai/gpt-mini")
            self.assertEqual(self.admin("POST", "/api/route", {"prompt": "hello"})[2]["cell"], "fast")
            self.assertEqual(self.chat(dict(MSG, model="maestro/local"))[2]["model"], "local:spark/tiny:3b")   # asked for by name
            self.assertEqual(self.chat(dict(MSG, model="local:spark/tiny:3b"))[2]["model"], "local:spark/tiny:3b")   # concrete id
            self.assertEqual(self.chat(dict(MSG, model="maestro/ask"))[2]["model"], "openai/gpt-mini")   # a job name is not a category
        self.assertEqual(len(loc.requests), 2)

    def test_free_api_models_are_skipped_unless_requested_or_pinned(self):
        free, paid = self.up(name="free", text="from free"), self.up(name="paid", text="from paid")
        self.start([E("openai/plan-model", "strong", 0.9, (0.0, 0.0)), E("openai/x:free", "strong", 0.9, (1.0, 2.0)),
                    E("groq/paid", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"openai": {"base_url": free.url}, "groq": {"base_url": paid.url}}})
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
            self.assertEqual(j["model"], "groq/paid")
            r = self.admin("POST", "/api/route", {"prompt": "why does this fail"})[2]
            reasons = {x["id"]: x["reason"] for x in r["skipped"]}
            self.assertIn("free tier", reasons["openai/plan-model"])
            self.assertIn("free tier", reasons["openai/x:free"])
            self.assertEqual(len(free.requests), 0)
            self.assertEqual(self.chat(dict(MSG, model="openai/plan-model"))[2]["model"], "openai/plan-model")   # concrete id works
            self.admin("POST", "/api/models/openai%2Fplan-model", {"category": "strong"})   # a category pin is the user's word
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/plan-model")

    def test_free_api_models_are_the_last_resort_not_a_503(self):
        # A first run with one coding-plan key (models.dev lists plans at price 0): maestro/auto uses it instead of refusing.
        plan = self.up(name="plan", text="from plan")
        self.start([E("openai/plan-model", "standard", 0.6, (0.0, 0.0)), E("openai/x:free", "fast", 0.3, (1.0, 2.0))],
                   {"providers": {"openai": {"base_url": plan.url}}})
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
            self.assertEqual(st, 200)
            self.assertEqual(j["model"], "openai/plan-model")
            self.assertIn("only price-0 models", json.loads(h["x-maestro-decision"])["reason"])
            self.assertEqual(self.chat(dict(MSG, model="maestro/standard"))[0], 200)

    def test_keys_go_only_to_their_own_provider(self):
        mini, big = self.two()
        self.chat(dict(MSG, model="openai/gpt-mini"))
        self.chat(dict(MSG, model="groq/llama-strong"))
        self.assertEqual(mini.requests[0]["headers"]["authorization"], "Bearer " + OPENAI_KEY)
        self.assertEqual(big.requests[0]["headers"]["authorization"], "Bearer " + GROQ_KEY)
        for u in (mini, big):
            for r in u.requests:
                for other in ALL_KEYS:
                    if other not in (r["headers"].get("authorization") or ""):
                        self.assertNotIn(other.encode(), r["body"] + json.dumps(r["headers"]).encode())

    def test_bad_requests_are_openai_shaped_errors(self):
        self.two()
        st, _, j = self.chat({"messages": []})
        self.assertEqual((st, j["error"]["type"]), (400, "invalid_request_error"))
        self.assertEqual(call(self.port, "POST", "/v1/chat/completions", b"not json")[0], 400)
        self.assertEqual(call(self.port, "POST", "/v1/chat/completions", b"[1]")[0], 400)
        st, _, j = jcall(self.port, "GET", "/v1/embeddings")
        self.assertEqual((st, j["error"]["code"]), (404, "not_found"))


class Filters(ServerCase):
    def test_tools_exclude_cli_seats_and_say_why(self):
        fake_cli(self.bin, "claude", "cat >/dev/null; echo '{\"result\": \"from the cli\", \"usage\": {\"input_tokens\": 3, \"output_tokens\": 4}}'")
        api = self.up(name="api", text="from api")
        self.start([E("cli:claude/sonnet", "standard", 0.6), E("openai/gpt-mini", "standard", 0.5, (0.15, 0.6))],
                   {"providers": {"openai": {"base_url": api.url}}})
        with fixed("edit", 1):
            st, h, j = self.chat(MSG)   # no tools: the subscription seat is (almost) free, so it wins
            self.assertEqual((st, j["model"], j["choices"][0]["message"]["content"]), (200, "cli:claude/sonnet", "from the cli"))
            tools = {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]}
            st, h, j = self.chat(dict(MSG, **tools))
            self.assertEqual(j["model"], "openai/gpt-mini")
            st, _, plan = self.admin("POST", "/api/route", {"prompt": "hi", "tools": True})
        self.assertIn({"id": "cli:claude/sonnet", "reason": "subscription CLI seats cannot use tools"}, plan["skipped"])
        st, _, j = self.chat(dict(MSG, model="cli:claude/sonnet", **tools))
        self.assertEqual((st, j["error"]["code"]), (400, "model_unsuitable"))

    def test_vision_and_context_filters(self):
        a, b = self.up(name="a", text="from text model"), self.up(name="b", text="from vision model")
        self.start([E("openai/text", "standard", 0.6, (0.1, 0.3), context=8000),
                    E("groq/eyes", "standard", 0.5, (2.0, 8.0), vision=True, context=1000000)],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
        img = {"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
        with fixed("ask", 1):
            self.assertEqual(self.chat(img)[2]["model"], "groq/eyes")   # only one sees images
            big = {"messages": [{"role": "user", "content": "word " * 20000}]}   # ~25k tokens: over the 8k model's window
            self.assertEqual(self.chat(big)[2]["model"], "groq/eyes")
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/text")

    def test_nothing_eligible_is_a_503_that_explains(self):
        api = self.up(name="api")
        self.start([E("openai/gpt-mini", "fast", 0.4, tools_ok=False)], {"providers": {"openai": {"base_url": api.url}}})
        st, _, j = self.chat(dict(MSG, tools=[{"type": "function", "function": {"name": "f"}}]))
        self.assertEqual((st, j["error"]["code"]), (503, "no_route"))
        self.assertIn("no tool support", j["error"]["message"])

    def test_quota_low_goes_last_and_exhausted_is_skipped(self):
        a, b = self.up(name="a", text="from a"), self.up(name="b", text="from b")
        self.start([E("openai/cheap", "standard", 0.6, (0.1, 0.2)), E("groq/dear", "standard", 0.6, (1.0, 8.0))],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
        with fixed("edit", 1):
            self.assertEqual(self.chat(MSG)[2]["model"], "openai/cheap")
            Q.record("openai", "cheap", 200, {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "4", "x-ratelimit-reset-requests": "60s"}, self.home)
            self.assertEqual(self.chat(MSG)[2]["model"], "groq/dear")   # under 10% left: strong penalty
            Q.record("groq", "dear", 200, {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "60s"}, self.home)
            st, h, j = self.chat(MSG)
        self.assertEqual(j["model"], "openai/cheap")   # the other is exhausted: ineligible, not just penalised

    def test_session_headers_are_honoured_and_latency_is_recorded(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        self.chat(MSG)
        self.assertIn("openai/x", self.app.lat)


class Fallback(ServerCase):
    def pair(self, first_mode, second_mode="ok", **kw):
        a, b = self.up(name="a", text="from a", mode=first_mode, **kw), self.up(name="b", text="from b", mode=second_mode)
        self.start([E("openai/a", "strong", 0.7, (1.0, 2.0)), E("groq/b", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
        return a, b

    def test_429_before_the_first_byte_falls_back_and_cools_the_seat_down(self):
        a, b = self.pair("429")
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
            self.assertEqual((st, j["model"], self.decision(h)["fallbacks"]), (200, "groq/b", 1))
            self.assertEqual(self.decision(h)["attempted"], ["openai/a", "groq/b"])
            n = len(a.requests)
            self.assertEqual(self.chat(MSG)[2]["model"], "groq/b")
            self.assertEqual(len(a.requests), n)   # on cooldown (retry-after 30): not even tried
        self.assertFalse(Q.is_available("openai", "a", self.home)[0])
        attempts = json.loads(self.rows()[0]["attempts"])
        self.assertEqual([x["status"] for x in attempts], [429, 200])

    def test_5xx_and_connection_errors_fall_back(self):
        a, b = self.pair("500")
        with fixed("debug", 2):
            self.assertEqual(self.chat(MSG)[2]["model"], "groq/b")
        a.close()
        self.ups.remove(a)
        self.app.lat.clear()
        Q.reset_state(self.home)
        with fixed("debug", 2):
            self.assertEqual(self.chat(MSG)[2]["model"], "groq/b")   # connection refused = status 0

    def test_streaming_falls_back_when_the_first_byte_never_comes(self):
        a, b = self.pair("empty")
        with fixed("debug", 2):
            st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="maestro/auto", stream=True))
        chunks, done = parse_sse(raw)
        self.assertEqual(st, 200)
        self.assertEqual(self.decision(h)["fallbacks"], 1)
        self.assertEqual(chunks[0]["model"], "groq/b")
        self.assertTrue(done)

    def test_all_candidates_failing_returns_the_last_upstream_error(self):
        a, b = self.pair("429", "500")
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
        self.assertEqual(st, 500)
        self.assertEqual(j["error"]["message"], "boom")
        self.assertEqual(self.decision(h)["fallbacks"], 1)
        self.assertEqual(self.rows()[0]["status"], 500)

    def test_a_400_is_the_callers_problem_and_is_not_retried(self):
        a = self.up(name="a", custom=lambda req: json_reply({"error": {"message": "bad param"}}, 400))
        b = self.up(name="b")
        self.start([E("openai/a", "strong", 0.7, (1.0, 2.0)), E("groq/b", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
        self.assertEqual((st, j["error"]["message"]), (400, "bad param"))
        self.assertEqual(len(b.requests), 0)

    def test_no_fallback_once_bytes_have_streamed(self):
        broken = [{"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3, "output_tokens": 1}}},
                  {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                  {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}]   # then the connection ends
        ant = self.up(AnthropicUp, name="ant", stream=broken)
        other = self.up(name="other", text="from other")
        self.start([E("anthropic/claude-x", "strong", 0.9, (3.0, 6.0), style="anthropic"), E("openai/o", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"anthropic": {"base_url": ant.url}, "openai": {"base_url": other.url}}})
        with fixed("debug", 2):
            st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="maestro/auto", stream=True))
        self.assertEqual(st, 200)   # the status line was already sent
        chunks, done = parse_sse(raw)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Hel")
        self.assertIn("error", chunks[-1])
        self.assertTrue(done)   # the error event is followed by [DONE]
        self.assertEqual(len(other.requests), 0)   # never retried elsewhere
        self.assertEqual(self.decision(h)["fallbacks"], 0)
        time.sleep(0.1)
        self.assertEqual(self.rows()[0]["status"], 502)


class FallbackDiversity(ServerCase):
    def test_a_429_from_one_provider_skips_its_other_models_and_the_next_provider_answers(self):
        a, b = self.up(name="a", mode="429"), self.up(name="b", text="from b")
        self.start([E("openai/a1", "strong", 0.9, (1.0, 2.0)), E("openai/a2", "strong", 0.85, (1.1, 2.0)), E("openai/a3", "strong", 0.8, (1.2, 2.0)),
                    E("groq/b", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
        self.assertEqual((st, j["model"]), (200, "groq/b"))
        self.assertEqual(len(a.requests), 1)
        self.assertEqual(self.decision(h)["attempted"], ["openai/a1", "groq/b"])

    def test_an_outage_or_bad_key_skips_the_provider_and_the_attempt_cap_still_holds(self):
        for mode in ("500", "401"):
            Q.reset_state(self.home)
            a, b = self.up(name="a", mode=mode), self.up(name="b", text="from b")
            self.start([E("openai/a1", "strong", 0.9, (1.0, 2.0)), E("openai/a2", "strong", 0.85, (1.1, 2.0)), E("groq/b", "strong", 0.6, (5.0, 20.0))],
                       {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
            with fixed("debug", 2):
                st, h, j = self.chat(MSG)
            self.assertEqual((st, j["model"], len(a.requests)), (200, "groq/b", 1), mode)
            self.stop()
        # 4 models on 4 failing providers: three real attempts, then the last error goes to the caller
        ups = [self.up(name=n, mode="500") for n in "abcd"]
        provs = ["openai", "groq", "mistral", "xai"]
        self.start([E("%s/m" % pv, "strong", 0.9 - i / 100.0, (1.0, 2.0)) for i, pv in enumerate(provs)],
                   {"providers": {pv: {"base_url": u.url} for pv, u in zip(provs, ups)}}, env_extra={"MISTRAL_API_KEY": "k1", "XAI_API_KEY": "k2"})
        with fixed("debug", 2):
            st, h, j = self.chat(MSG)
        self.assertEqual(st, 500)
        self.assertEqual(sum(len(u.requests) for u in ups), 3)

    def test_a_401_puts_the_provider_on_cooldown_so_its_other_models_are_not_tried(self):
        a, b = self.up(name="a", mode="401"), self.up(name="b", text="from b")
        self.start([E("openai/a1", "strong", 0.9, (1.0, 2.0)), E("openai/a2", "strong", 0.85, (1.1, 2.0)), E("groq/b", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}})
        with fixed("debug", 2):
            self.chat(MSG)
            st, h, j = self.chat(MSG)
        self.assertEqual((st, j["model"], len(a.requests)), (200, "groq/b", 1))   # the second request never touched openai
        ok, why = Q.is_available("openai", "a2", self.home)
        self.assertFalse(ok)
        self.assertIn("auth failed", why)
        self.assertNotIn(OPENAI_KEY, json.dumps(self.rows()))


class Streaming(ServerCase):
    def test_a_stream_cut_off_upstream_ends_with_an_error_event_and_done(self):
        cut = sse(chunk("gpt-x", {"role": "assistant", "content": "Hel"}))[:-1]   # no finish_reason, no [DONE]: the connection just ends
        a = self.up(name="a", stream=cut)
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="maestro/auto", stream=True))
        chunks, done = parse_sse(raw)
        self.assertEqual(st, 200)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Hel")
        self.assertEqual(chunks[-1]["error"]["type"], "upstream_error")
        self.assertIn("ended early", chunks[-1]["error"]["message"])
        self.assertTrue(done)
        time.sleep(0.1)
        self.assertEqual(self.rows()[0]["status"], 502)

    def test_a_stream_with_a_finish_reason_but_no_done_is_complete(self):
        ok = sse(chunk("gpt-x", {"role": "assistant", "content": "Hi"}), chunk("gpt-x", {}, "stop"))[:-1]
        a = self.up(name="a", stream=ok)
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="maestro/auto", stream=True))
        chunks, done = parse_sse(raw)
        self.assertNotIn("error", json.dumps(chunks))
        time.sleep(0.1)
        self.assertEqual(self.rows()[0]["status"], 200)

    def test_sse_is_relayed_with_the_real_model_and_usage_is_logged(self):
        ch = sse(chunk("gpt-x", {"role": "assistant", "content": "Hel"}), chunk("gpt-x", {"content": "lo"}), chunk("gpt-x", {}, "stop"),
                 chunk("gpt-x", {}, choices=[], usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}))
        a = self.up(name="a", stream=ch)
        self.start([E("openai/gpt-x", "standard", 0.5)], {"providers": {"openai": {"base_url": a.url}}})
        body = dict(MSG, model="maestro/auto", stream=True, stream_options={"include_usage": True})
        st, h, raw = call(self.port, "POST", "/v1/chat/completions", body)
        chunks, done = parse_sse(raw)
        self.assertEqual((st, h["content-type"], done), (200, "text/event-stream", True))
        self.assertEqual("".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"]), "Hello")
        self.assertTrue(all(c["model"] == "openai/gpt-x" for c in chunks))
        self.assertEqual(a.requests[0]["json"]["stream_options"], {"include_usage": True})   # passed through
        self.assertEqual(self.decision(h)["model"], "openai/gpt-x")
        time.sleep(0.1)
        row = self.rows()[0]
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"], row["usage_est"], row["stream"]), (11, 7, 0, 1))

    def test_hermes_tool_loop_streams_valid_tool_call_deltas_and_stays_on_one_model(self):
        m = "gpt-big"
        tc = [chunk(m, {"role": "assistant", "content": None, "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": ""}}]}),
              chunk(m, {"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": "}}]}),
              chunk(m, {"tool_calls": [{"index": 0, "function": {"arguments": "\"src/a.py\"}"}}]}), chunk(m, {}, "tool_calls")]
        big, small = self.up(name="big", stream=sse(*tc)), self.up(name="small", text="from small")
        tools = [{"type": "function", "function": {"name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
        models = [E("openai/gpt-big", "strong", 0.8, (10.0, 40.0)), E("groq/small", "strong", 0.6, (0.5, 1.0))]
        self.start(models, {"providers": {"openai": {"base_url": big.url}, "groq": {"base_url": small.url}}})
        sys_msg = {"role": "system", "content": "You are Hermes, an agent."}
        user = {"role": "user", "content": "Find why the login test fails in src/a.py"}
        req1 = {"model": "maestro/auto", "stream": True, "stream_options": {"include_usage": True}, "tools": tools, "messages": [sys_msg, user]}
        # step 1: the cheap model would win on price, so first make gpt-big the choice by pinning its category up and the other down
        self.admin("POST", "/api/models/groq%2Fsmall", {"enabled": False})
        with fixed("debug", 2):
            st, h, raw = call(self.port, "POST", "/v1/chat/completions", req1)
        chunks, done = parse_sse(raw)
        self.assertEqual((st, done), (200, True))
        self.assertTrue(all(c["object"] == "chat.completion.chunk" and c["model"] == "openai/gpt-big" for c in chunks))
        calls = [t for c in chunks for t in c["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(calls[0]["id"], "call_1")
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads("".join(t["function"].get("arguments", "") for t in calls)), {"path": "src/a.py"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(big.requests[0]["json"]["tools"], tools)   # tools reached the upstream untouched
        # step 2: the tool result comes back. Now the cheap model is enabled again and would win on price, but this is the same turn.
        self.admin("POST", "/api/models/groq%2Fsmall", {"enabled": True})
        req2 = dict(req1, stream=False, messages=[sys_msg, user, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/a.py\"}"}}]},
                                                   {"role": "tool", "tool_call_id": "call_1", "content": "def login(): ..."}])
        big.stream_chunks = None
        with fixed("edit", 0):   # a fresh classification would say something else entirely
            st, h, j = jcall(self.port, "POST", "/v1/chat/completions", req2)
        self.assertEqual(j["model"], "openai/gpt-big")
        d = self.decision(h)
        self.assertTrue(d["affinity"])
        self.assertEqual((d["job"], d["difficulty"]), ("debug", 2))   # the classification of the user turn, reused
        # a new user message in the same conversation is routed afresh
        with fixed("debug", 2):
            j3 = self.chat(dict(req1, stream=False, messages=[sys_msg, user, {"role": "assistant", "content": "done"}, {"role": "user", "content": "thanks, now rename it"}]))[2]
        self.assertEqual(j3["model"], "groq/small")

    def test_session_id_header_keys_the_conversation(self):
        a, b = self.up(name="a", text="from a"), self.up(name="b", text="from b")
        prices = lambda pa, pb: [E("openai/a", "strong", 0.7, pa), E("groq/b", "strong", 0.6, pb)]
        self.start(prices((1.0, 2.0), (5.0, 20.0)), {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}, "session_affinity_header": "X-Hermes-Session"})

        def tool_turn(first_user):
            return {"messages": [{"role": "user", "content": first_user}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                                 {"role": "tool", "tool_call_id": "c", "content": "r"}]}
        with fixed("debug", 2):
            self.assertEqual(self.chat({"messages": [{"role": "user", "content": "q1"}]}, headers={"X-Hermes-Session": "s1"})[2]["model"], "openai/a")
            self.write("models.json", {"version": 1, "generated_at": time.time(), "models": prices((5.0, 20.0), (1.0, 2.0))})   # now b is cheaper
            # same session id, but a different first message: only the header can say this is the same conversation
            st, h, j = self.chat(tool_turn("q2 (the session header decides)"), headers={"X-Hermes-Session": "s1"})
            self.assertEqual((j["model"], self.decision(h)["affinity"]), ("openai/a", True))
            st, h, j = self.chat(tool_turn("q2 (the session header decides)"), headers={"X-Hermes-Session": "s2"})   # another session: routed afresh
            self.assertEqual((j["model"], self.decision(h)["affinity"]), ("groq/b", False))
            st, h, j = self.chat(tool_turn("q3"), headers={"X-Session-Id": "s1"})   # the generic header names the same conversation
            self.assertEqual((j["model"], self.decision(h)["affinity"]), ("openai/a", True))
            st, h, j = self.chat(tool_turn("q3"), headers={"X-Session-Id": "s9"})
            self.assertEqual((j["model"], self.decision(h)["affinity"]), ("groq/b", False))

    def test_anthropic_upstream_streams_tool_use_as_openai_chunks(self):
        ev = [{"type": "message_start", "message": {"id": "msg_9", "usage": {"input_tokens": 12, "output_tokens": 1}}},
              {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
              {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Paris\"}"}},
              {"type": "content_block_stop", "index": 0},
              {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}, {"type": "message_stop"}]
        ant = self.up(AnthropicUp, name="ant", stream=ev)
        self.start([E("anthropic/claude-x", "strong", 0.9, (3.0, 6.0), style="anthropic", vision=True)], {"providers": {"anthropic": {"base_url": ant.url}}})
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="anthropic/claude-x", stream=True, tools=tools))
        chunks, done = parse_sse(raw)
        calls = [t for c in chunks for t in c["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual((st, done, calls[0]["id"], calls[0]["function"]["name"]), (200, True, "toolu_1", "get_weather"))
        self.assertEqual(json.loads("".join(t["function"].get("arguments", "") for t in calls)), {"city": "Paris"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(ant.requests[0]["headers"]["x-api-key"], ANT_KEY)
        self.assertEqual(ant.requests[0]["json"]["tools"][0]["name"], "get_weather")
        st, _, j = self.chat(dict(MSG, model="anthropic/claude-x"))   # and non-streaming
        self.assertEqual((j["choices"][0]["message"]["content"], j["usage"]["prompt_tokens"]), ("pong", 9))

    def test_cli_seat_fakes_streaming_as_one_chunk(self):
        fake_cli(self.bin, "claude", "cat >/dev/null; echo '{\"result\": \"cli answer\"}'")
        self.start([E("cli:claude/haiku", "fast", 0.4)])
        st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="maestro/fast", stream=True))
        chunks, done = parse_sse(raw)
        self.assertEqual((st, done, chunks[0]["choices"][0]["delta"]["content"], chunks[0]["model"]), (200, True, "cli answer", "cli:claude/haiku"))

    def test_concurrent_requests_all_succeed(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        res = []
        ts = [threading.Thread(target=lambda: res.append(self.chat(MSG)[0])) for _ in range(12)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(res, [200] * 12)
        self.assertEqual(len(self.rows()), 12)


class Security(ServerCase):
    def base(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})

    def test_binding_beyond_loopback_requires_an_api_key(self):
        with self.assertRaises(ValueError) as cm:
            S.make_server("0.0.0.0", 0, home=self.home, env={"HOME": self.home, "PATH": self.bin}, refresh_s=0)
        self.assertIn("MAESTRO_API_KEY", str(cm.exception))
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}}, host="0.0.0.0", env_extra={"MAESTRO_API_KEY": "fleet-key"})
        st, _, j = self.chat(MSG)
        self.assertEqual((st, j["error"]["code"]), (401, "invalid_api_key"))
        self.assertEqual(self.chat(MSG, headers={"Authorization": "Bearer nope"})[0], 401)
        self.assertEqual(self.chat(MSG, headers={"Authorization": "Bearer fleet-key"})[0], 200)
        self.assertEqual(jcall(self.port, "GET", "/v1/models", headers={"Authorization": "Bearer fleet-key"})[0], 200)
        self.assertEqual(jcall(self.port, "GET", "/v1/models")[0], 401)
        self.assertEqual(jcall(self.port, "GET", "/health")[0], 200)   # liveness needs no key
        self.assertEqual(call(self.port, "GET", "/")[0], 200)   # the UI shell is static; its API needs the admin token
        self.assertEqual(self.admin("GET", "/api/state", token=False)[0], 401)

    def test_admin_api_rejects_missing_wrong_token_and_foreign_origin(self):
        self.base()
        for path in ("/api/state", "/api/decisions", "/api/stats"):
            self.assertEqual(self.admin("GET", path, token=False)[0], 401, path)
        st, _, j = self.admin("POST", "/api/keys", {"provider": "openai", "value": "x"}, token=False)
        self.assertEqual((st, j["error"]["code"]), (401, "invalid_token"))
        self.assertEqual(jcall(self.port, "GET", "/api/state", headers={"X-Maestro-Token": "wrong"})[0], 401)
        self.assertEqual(self.admin("GET", "/api/state")[0], 200)
        # right token, wrong Origin: a page on another site cannot use it
        st, _, j = self.admin("POST", "/api/config", {"prefer_local": False}, headers={"Origin": "http://evil.example"})
        self.assertEqual((st, j["error"]["type"]), (403, "forbidden"))
        self.assertEqual(self.admin("POST", "/api/config", {"prefer_local": False}, headers={"Origin": "null"})[0], 403)
        self.assertEqual(self.admin("POST", "/api/config", {"prefer_local": False}, headers={"Origin": "http://127.0.0.1:%d" % self.port})[0], 200)
        # DNS rebinding: a foreign Host name on a loopback server
        self.assertEqual(self.admin("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        # the same protection on the model endpoints (a web page must not spend the user's money)
        self.assertEqual(self.chat(MSG, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.chat(MSG)[0], 200)
        self.assertEqual(config_of(self.home).get("prefer_local"), False)

    def test_admin_token_file_is_private_and_stable(self):
        self.base()
        path = os.path.join(self.home, "admin_token")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        with open(path) as f:
            self.assertEqual(f.read().strip(), self.token)
        self.assertEqual(S.load_admin_token(self.home), self.token)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.home, "maestro.db")).st_mode), 0o600)

    def test_keys_are_write_only_everywhere(self):
        self.base()
        secret = "sk-NEW-SECRET-key-987654321"
        st, _, j = self.admin("POST", "/api/keys", {"provider": "mistral", "value": secret})
        self.assertEqual((st, j["ok"], j["name"], j["source"]), (200, True, "MISTRAL_API_KEY", "keys.env"))
        path = os.path.join(self.home, "keys.env")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        with open(path) as f:
            self.assertIn(secret, f.read())
        self.assertEqual(P.get_key(("MISTRAL_API_KEY",), {}, self.home), secret)
        self.chat(MSG)
        seen = [self.admin("GET", "/api/state"), self.admin("GET", "/api/decisions"), jcall(self.port, "GET", "/v1/models"), jcall(self.port, "GET", "/health"),
                self.admin("POST", "/api/route", {"prompt": "hi"}), self.admin("POST", "/api/test", {"provider": "mistral"})]
        for st, h, body in seen:
            text = json.dumps(body) + json.dumps(h)
            for s in ALL_KEYS + (secret,):
                self.assertNotIn(s, text)
        row = [p for p in self.admin("GET", "/api/state")[2]["providers"] if p["id"] == "mistral"][0]
        self.assertEqual((row["key_source"], row["key_env"]), ("keys.env", "MISTRAL_API_KEY"))
        self.assertNotIn("value", row)
        # unsetting removes it; an env var name nobody reads is refused
        self.assertEqual(self.admin("POST", "/api/keys", {"provider": "mistral", "value": None})[2]["removed"], ["MISTRAL_API_KEY"])
        self.assertEqual(self.admin("POST", "/api/keys", {"provider": "PATH", "value": "x"})[0], 400)
        self.assertEqual(self.admin("POST", "/api/keys", {"provider": "openai", "value": "bad\nvalue"})[0], 400)

    def test_an_upstream_that_echoes_the_key_is_scrubbed(self):
        a = self.up(name="a", mode="401")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        st, h, raw = call(self.port, "POST", "/v1/chat/completions", dict(MSG, model="openai/x"))
        self.assertEqual(st, 401)
        self.assertNotIn(OPENAI_KEY.encode(), raw + json.dumps(h).encode())

    def test_config_endpoint_validates(self):
        self.base()
        self.assertEqual(self.admin("POST", "/api/config", {"objective": "cheapest"})[0], 400)
        self.assertEqual(self.admin("POST", "/api/config", {"ladder": {"code": ["fast"]}})[0], 400)
        self.assertEqual(self.admin("POST", "/api/config", {"local_endpoints": []})[0], 400)   # not changeable from the UI
        st, _, j = self.admin("POST", "/api/config", {"objective": "cost", "max_tier": "strong", "ladder": {"code": ["fast", "fast", "standard", "strong"]}})
        self.assertEqual((st, j["config"]["objective"], j["config"]["max_tier"]), (200, "cost", "strong"))
        self.assertEqual(config_of(self.home)["ladder"]["code"], ["fast", "fast", "standard", "strong"])
        self.admin("POST", "/api/config", {"ladder": None})
        self.assertNotIn("ladder", config_of(self.home))
        self.assertEqual(config_of(self.home)["providers"], {"openai": {"base_url": self.ups[0].url}})   # other keys untouched


def config_of(home):
    with open(os.path.join(home, "config.json")) as f:
        return json.load(f)


class DecisionLog(ServerCase):
    def test_rows_have_the_routing_facts_and_no_prompt_text_by_default(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6, (2.0, 10.0)), E("groq/y", "strong", 0.7, (5.0, 20.0))], {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": a.url}}})
        with fixed("debug", 0):
            self.chat({"messages": [{"role": "user", "content": "my secret plan: the password is hunter2"}]})
        row = self.rows()[0]
        self.assertEqual((row["job"], row["difficulty"], row["chosen"], row["status"], row["requested"]), ("debug", 0, "openai/x", 200, "maestro/auto"))
        self.assertIsNone(row["prompt"])
        self.assertEqual(json.loads(row["candidates"]), ["openai/x", "groq/y"])
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"], row["usage_est"]), (5, 2, 0))
        self.assertGreater(row["latency_ms"], 0)
        self.assertGreater(row["cost"], 0)
        self.assertGreater(row["baseline"], row["cost"])   # the strongest model would have cost more
        self.assertEqual(len(row["conv"]), 12)
        self.assertNotIn("hunter2", json.dumps(row))
        self.assertNotIn("hunter2", json.dumps(self.admin("GET", "/api/decisions")[2]))

    def test_prompt_snippet_only_when_log_prompts_is_on_and_is_120_chars(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}, "log_prompts": True})
        self.chat({"messages": [{"role": "user", "content": "q" * 300}]})
        self.assertEqual(self.rows()[0]["prompt"], "q" * 120)

    def test_stats_and_the_admin_state(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6, (2.0, 10.0)), E("groq/big", "frontier", 0.9, (15.0, 60.0))], {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": a.url}}})
        with fixed("edit", 1):
            for _ in range(3):
                self.chat(MSG)
        st, _, s = self.admin("GET", "/api/state")
        self.assertEqual(st, 200)
        self.assertEqual((s["stats"]["requests"], s["counts"]["models"], s["models_source"]), (3, 2, "models.json"))
        self.assertGreater(s["stats"]["saved"], 0)
        self.assertEqual(len(s["decisions"]), 3)
        self.assertEqual(sorted(s["jobs"]), sorted(core.JOBS))
        self.assertIn("maestro/auto", s["snippets"]["hermes"])
        self.assertTrue(any(p["id"] == "openai" and p["status"] == "detected" for p in s["providers"]))
        self.assertEqual(len(self.admin("GET", "/api/decisions?limit=2")[2]["decisions"]), 2)


class Admin(ServerCase):
    def test_route_dry_run_lists_candidates_with_scores_and_skips(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6, (2.0, 10.0)), E("groq/dead", "standard", 0.9, (1.0, 1.0))], {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": a.url}}})
        Q.record("groq", "dead", 429, {"retry-after": "120"}, self.home)
        with fixed("plan", 1):
            st, _, p = self.admin("POST", "/api/route", {"prompt": "design the thing"})
        self.assertEqual((st, p["job"], p["chosen"]), (200, "plan", "openai/x"))
        self.assertEqual(p["candidates"][0]["id"], "openai/x")
        self.assertEqual(p["candidates"][0]["strength"], 0.6)
        self.assertEqual(len(a.requests), 0)   # a dry run calls nothing
        self.assertTrue(p["skipped"][0]["reason"].startswith("unavailable: cooling down"))

    def test_send_runs_it_for_real(self):
        a = self.up(name="a", text="the answer")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        st, _, r = self.admin("POST", "/api/send", {"prompt": "hi"})
        self.assertEqual((st, r["status"], r["answer"], r["model"]), (200, 200, "the answer", "openai/x"))
        self.assertEqual(len(self.rows()), 1)

    def test_providers_are_listed_and_keys_shown_by_presence(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        rows = {p["id"]: p for p in self.admin("GET", "/api/state")[2]["providers"]}
        self.assertEqual((rows["openai"]["status"], rows["openai"]["key_source"]), ("detected", "env"))
        self.assertEqual(rows["mistral"]["status"], "missing")
        self.assertIn("claude", rows)
        self.assertGreater(len(rows), 100)

    def test_test_button_lists_models_with_the_providers_own_key(self):
        a = self.up(name="a", custom=lambda req: json_reply({"data": [{"id": "m1"}, {"id": "m2"}]}))
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        st, _, j = self.admin("POST", "/api/test", {"provider": "openai"})
        self.assertEqual((st, j["ok"], j["message"]), (200, True, "key works: 2 models"))
        self.assertEqual(a.srv.requests[0]["headers"]["authorization"], "Bearer " + OPENAI_KEY)
        self.assertFalse(self.admin("POST", "/api/test", {"provider": "nope"})[2]["ok"])

    def test_discover_endpoint_runs_the_hook_and_reloads_models(self):
        a = self.up(name="a")
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})

        def fake_discover(home, env, cfg, refresh_catalog=False):
            self.write("models.json", {"version": 1, "generated_at": time.time(), "models": [E("openai/x", "standard", 0.6), E("openai/y", "strong", 0.8)]})
            return True, "discovery finished (fake)"
        with mock.patch.object(S, "run_discover", fake_discover):
            st, _, j = self.admin("POST", "/api/discover", {})
        self.assertEqual((st, j["ok"], j["models"]), (200, True, 2))
        self.assertIn("openai/y", [m["id"] for m in jcall(self.port, "GET", "/v1/models")[2]["data"]])

    def test_discovery_runs_on_start_only_when_models_json_is_missing_or_older_than_a_day(self):
        calls = []
        with mock.patch.object(S, "run_discover", lambda *a, **k: calls.append(1) or (True, "ok")):
            self.start([E("openai/x", "standard", 0.6)], discover_on_start=True)   # fresh file
            time.sleep(0.2)
            self.assertEqual(calls, [])
            self.stop()
            path = os.path.join(self.home, "models.json")
            os.utime(path, (time.time() - 3600 * 25, time.time() - 3600 * 25))
            self.launch(discover_on_start=True)   # 25 h old
            time.sleep(0.3)
            self.assertEqual(len(calls), 1)
            self.stop()
            os.remove(path)
            self.launch(discover_on_start=True)   # missing
            time.sleep(0.3)
        self.assertEqual(len(calls), 2)

    def test_quota_refresh_runs_periodically(self):
        calls = []
        with mock.patch.object(Q, "refresh", lambda *a, **k: calls.append(1)):
            self.launch(refresh_s=0.05)
            time.sleep(0.4)
        self.assertGreaterEqual(len(calls), 3)


class NotLoaded(ServerCase):
    """A model a server has installed but not loaded is listed, never routed to (a request would make the server load it)."""
    def models(self):
        return [E("openai/gpt-mini", "fast", 0.40, (0.15, 0.6)),
                E("local:spark/gpt-oss:120b", "local", 0.58, serves="standard", loaded=False, routable=False),
                E("local:spark/tiny:3b", "local", 0.2, serves="local", loaded=True, routable=True)]

    def boot(self, **cfg):
        loc, cloud = self.up(name="spark", text="from spark"), self.up(name="cloud", text="from cloud")
        self.start(self.models(), dict({"providers": {"openai": {"base_url": cloud.url}},
                                        "local_endpoints": [{"name": "spark", "url": loc.url, "kind": "openai"}]}, **cfg))
        return loc, cloud

    def test_auto_routing_skips_it_with_the_reason(self):
        loc, cloud = self.boot()
        with fixed("debug", 0):   # debug d0 = standard: the 120B model would be the pick if it were loaded
            st, h, j = self.chat(MSG)
            self.assertEqual((st, j["model"]), (200, "openai/gpt-mini"))
            plan = self.admin("POST", "/api/route", {"prompt": "why does this fail"})[2]
        reasons = {x["id"]: x["reason"] for x in plan["skipped"]}
        self.assertEqual(reasons["local:spark/gpt-oss:120b"], "installed but not loaded (Maestro never loads models)")
        self.assertEqual(loc.requests, [])

    def test_a_concrete_request_gets_409_and_nothing_is_sent(self):
        loc, cloud = self.boot()
        st, _, j = self.chat(dict(MSG, model="local:spark/gpt-oss:120b"))
        self.assertEqual((st, j["error"]["code"]), (409, "model_not_loaded"))
        self.assertIn("installed but not loaded", j["error"]["message"])
        self.assertIn("local_allow_load", j["error"]["message"])
        self.assertEqual(loc.requests, [])
        self.assertEqual(self.chat(dict(MSG, model="local:spark/tiny:3b"))[0], 200)   # a loaded one is fine

    def test_v1_models_and_the_ui_state_mark_it(self):
        self.boot()
        ids = [m["id"] for m in jcall(self.port, "GET", "/v1/models")[2]["data"]]
        self.assertNotIn("local:spark/gpt-oss:120b", ids)
        self.assertIn("local:spark/tiny:3b", ids)
        rows = {m["id"]: m for m in self.admin("GET", "/api/state")[2]["models"]}
        self.assertEqual((rows["local:spark/gpt-oss:120b"]["loaded"], rows["local:spark/gpt-oss:120b"]["routable"]), (False, False))
        self.assertTrue(rows["local:spark/tiny:3b"]["routable"])

    def test_local_allow_load_makes_installed_models_routable(self):
        loc, cloud = self.boot(local_allow_load=True)
        with fixed("debug", 0):
            st, h, j = self.chat(MSG)
        self.assertEqual(j["model"], "local:spark/gpt-oss:120b")
        self.assertEqual(self.chat(dict(MSG, model="local:spark/gpt-oss:120b"))[0], 200)
        self.assertIn("local:spark/gpt-oss:120b", [m["id"] for m in jcall(self.port, "GET", "/v1/models")[2]["data"]])
        self.assertEqual(self.admin("POST", "/api/config", {"local_allow_load": True})[0], 400)   # a file-only switch, not the UI's

    def test_the_fallback_built_from_providers_json_carries_loaded(self):
        a = self.up(name="a")
        self.write("providers.json", {"version": 1, "generated_at": time.time(), "verified": True, "providers": [
            {"id": "local:spark", "name": "Local: spark", "access": "local", "style": "local", "status": "connected", "reason": "",
             "models": ["big:120b", "on:8b"], "base_url": a.url + "/v1", "model_info": {"big:120b": {"loaded": False}, "on:8b": {"loaded": True}}}]})
        by = {m["id"]: S._norm_model(m, core.load_config(self.home)) for m in S._load_models(self.home)["models"]}
        self.assertEqual((by["local:spark/big:120b"]["routable"], by["local:spark/on:8b"]["routable"]), (False, True))
        self.write("config.json", {"local_allow_load": True})
        by = {m["id"]: S._norm_model(m, core.load_config(self.home)) for m in S._load_models(self.home)["models"]}
        self.assertTrue(by["local:spark/big:120b"]["routable"])


class Fallback_models(ServerCase):
    def test_without_models_json_the_server_builds_entries_from_providers_json(self):
        a = self.up(name="a", text="from a")
        self.write("providers.json", {"version": 1, "generated_at": time.time(), "verified": True, "providers": [
            {"id": "anthropic", "name": "Anthropic", "access": "api", "style": "anthropic", "status": "connected", "reason": "", "models": ["claude-opus-5-5", "claude-haiku-4-5"]},
            {"id": "openai", "name": "OpenAI", "access": "api", "style": "openai", "status": "connected", "reason": "", "models": ["some-new-model"]},
            {"id": "mistral", "name": "Mistral", "access": "api", "style": "openai", "status": "missing", "reason": "no key", "models": []},
            {"id": "local:spark", "name": "Local: spark", "access": "local", "style": "local", "status": "connected", "reason": "", "models": ["gpt-oss:120b", "tiny:3b"],
             "base_url": a.url + "/v1", "model_info": {}}]})
        self.write("config.json", {"providers": {"openai": {"base_url": a.url}}})
        Q.reset_state(self.home)
        models = S._load_models(self.home)
        by = {m["id"]: m for m in models["models"]}
        self.assertEqual(models["source"], "providers.json")
        self.assertEqual(by["anthropic/claude-opus-5-5"]["category"], "strong")   # from the suggested-model tables
        self.assertEqual(by["anthropic/claude-haiku-4-5"]["category"], "fast")
        self.assertEqual(by["openai/some-new-model"]["category"], "standard")   # unknown: the default
        self.assertEqual(by["local:spark/gpt-oss:120b"]["serves"], "standard")   # size heuristics
        self.assertIsNone(by["anthropic/claude-opus-5-5"]["G"])
        self.assertNotIn("mistral", "".join(by))
        self.launch(env={"OPENAI_API_KEY": OPENAI_KEY})
        st, _, j = self.chat(dict(MSG, model="openai/some-new-model"))
        self.assertEqual((st, j["choices"][0]["message"]["content"]), (200, "from a"))
        with fixed("debug", 2):   # unscored models are placed by their category: only opus counts as strong
            st, h, j = self.chat(dict(MSG, model="maestro/strong"))
        self.assertEqual(self.decision(h)["attempted"][0], "anthropic/claude-opus-5-5")   # (no anthropic key or server here, so it then fails over)

    def test_overrides_apply_to_the_fallback_and_to_models_json(self):
        self.write("models.json", {"version": 1, "models": [E("openai/x", "standard", 0.6), E("openai/y", "fast", 0.3)]})
        self.write("config.json", {"models_overrides": {"openai/x": {"enabled": False}, "openai/y": {"category": "strong"}}})
        by = {m["id"]: m for m in S._load_models(self.home)["models"]}
        self.assertFalse(by["openai/x"]["enabled"])
        self.assertEqual(by["openai/y"]["override"], {"category": "strong"})
        norm = S._norm_model(by["openai/y"], core.load_config(self.home))
        self.assertEqual(norm["cat_override"], "strong")


class Ui(ServerCase):
    def test_root_serves_the_single_file_ui_with_all_six_tabs(self):
        self.start([E("openai/x", "standard", 0.6)])
        st, h, raw = call(self.port, "GET", "/")
        html = raw.decode()
        self.assertEqual((st, h["content-type"]), (200, "text/html; charset=utf-8"))
        for tab in ("Overview", "Providers", "Models", "Routing", "Activity", "Try it"):
            self.assertIn('"%s"]' % tab, html)
        self.assertIn("prefers-color-scheme: dark", html)
        self.assertIn("width=device-width", html)
        self.assertIn("default-src 'none'", h["content-security-policy"])
        self.assertNotIn(self.token, html)   # the page carries no secret; the token arrives in the URL fragment

    def test_ui_local_panel_shows_what_is_served_and_folds_the_alternatives_away(self):
        with open(S.UI_FILE, encoding="utf-8") as f:
            html = f.read()
        for text in ("Could run locally", "Using: ", "Alternatives (would replace what you serve)", "Maestro never downloads or starts models",
                     "installed but not loaded, so not used", "routable !== false", "Stop the current server first"):
            self.assertIn(text, html)
        self.assertNotIn("start a local server", html)

    def test_ui_makes_no_external_requests(self):
        import re
        with open(S.UI_FILE, encoding="utf-8") as f:
            html = f.read()
        urls = set(re.findall(r"""(?:src|href|action)=["'](https?:[^"']+)""", html)) | set(re.findall(r"url\((https?:[^)]+)\)", html)) | set(re.findall(r"""@import\s+["']?(https?:[^"';)]+)""", html))
        self.assertEqual(urls, set())
        self.assertNotIn("<link rel=\"stylesheet\"", html)
        self.assertNotIn("fetch(\"http", html)

    def test_ui_script_parses(self):
        import re
        import shutil
        import subprocess
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        with open(S.UI_FILE, encoding="utf-8") as f:
            html = f.read()
        js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(js)
        self.addCleanup(os.remove, f.name)
        res = subprocess.run([node, "--check", f.name], capture_output=True, text=True, timeout=60)
        self.assertEqual(res.returncode, 0, res.stderr)


class Backlog(ServerCase):
    def test_a_burst_of_concurrent_requests_all_succeed(self):
        self.assertGreaterEqual(S.Server.request_queue_size, 128)
        a = self.up(name="a", custom=lambda req: (time.sleep(0.05), json_reply(completion(req["json"].get("model", "?"), "ok")))[1])
        self.start([E("openai/x", "standard", 0.6)], {"providers": {"openai": {"base_url": a.url}}})
        out = []

        def one():
            out.append(self.chat(MSG)[0])
        threads = [threading.Thread(target=one) for _ in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(sorted(out), [200] * 40)


class CliServe(unittest.TestCase):
    def test_serve_flushes_its_banner_when_stdout_is_redirected(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        class Buffered(io.StringIO):   # like a file or pipe: nothing is visible until flush()
            def __init__(self):
                io.StringIO.__init__(self)
                self.flushed = ""

            def flush(self):
                self.flushed = self.getvalue()

        out = Buffered()
        seen = []

        def serve_forever(self_, *a, **kw):
            seen.append(out.flushed)   # what a reader of the redirected stdout could see once the server is up

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MAESTRO_HOME": tmp, "HOME": tmp}, clear=False), \
                mock.patch.object(S.Server, "serve_forever", serve_forever), mock.patch.object(S.Q, "refresh", lambda *a, **k: None), \
                redirect_stdout(out):
            os.environ.pop("MAESTRO_API_KEY", None)
            rc = S.cmd_serve(argparse.Namespace(host="127.0.0.1", port=0, open=False, no_discover=True))
        self.assertEqual(rc, 0)
        self.assertIn("is serving on", seen[0])
        self.assertIn("admin token:", seen[0])   # the banner is complete once flushed; the token itself is not in it (next tests)

    def serve_banner(self, tty, open_=False, home_env="MAESTRO_HOME"):
        """(stdout text, urls handed to webbrowser.open, the admin token, the home) of `maestro-router serve` with stdout a TTY or not."""
        import argparse
        import io
        from contextlib import redirect_stdout

        class Out(io.StringIO):
            def isatty(self):
                return tty

        out, opened = Out(), []
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MAESTRO_HOME": tmp, "HOME": tmp}, clear=False), \
                mock.patch.object(S.Server, "serve_forever", lambda *a, **k: None), mock.patch.object(S.Q, "refresh", lambda *a, **k: None), \
                mock.patch("webbrowser.open", lambda url, *a, **k: opened.append(url)), redirect_stdout(out):
            os.environ.pop("MAESTRO_API_KEY", None)
            rc = S.cmd_serve(argparse.Namespace(host="127.0.0.1", port=0, open=open_, no_discover=True))
            with open(os.path.join(tmp, S.ADMIN_TOKEN_FILE)) as f:
                token = f.read().strip()
        self.assertEqual(rc, 0)
        return out.getvalue(), opened, token, tmp

    def test_the_banner_prints_the_token_url_only_on_a_terminal(self):
        text, opened, token, _ = self.serve_banner(tty=True)
        self.assertIn("/#token=" + token, text)
        self.assertEqual(opened, [])

    def test_the_banner_of_a_redirected_stdout_has_no_token_and_names_the_file(self):
        text, opened, token, home = self.serve_banner(tty=False)
        self.assertFalse(token in text)   # (not assertNotIn: a failure would print the token)
        self.assertNotIn("#token", text)
        self.assertIn("Web UI:", text)
        self.assertIn("admin token: cat " + os.path.join(home, S.ADMIN_TOKEN_FILE), text)   # the real home, not a hard-coded ~

    def test_open_still_hands_the_browser_the_token_url(self):
        text, opened, token, _ = self.serve_banner(tty=False, open_=True)
        self.assertFalse(token in text)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].endswith("/#token=" + token), opened[0])


    def test_serve_refuses_a_public_bind_without_a_key(self):
        import argparse
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MAESTRO_HOME": tmp, "HOME": tmp}, clear=False):
            os.environ.pop("MAESTRO_API_KEY", None)
            err = io.StringIO()
            with redirect_stderr(err):
                rc = S.cmd_serve(argparse.Namespace(host="0.0.0.0", port=0, open=False, no_discover=True))
        self.assertEqual(rc, 2)
        self.assertIn("MAESTRO_API_KEY", err.getvalue())

    def test_router_does_not_import_the_server(self):
        import subprocess
        import sys
        code = ("import sys; sys.path.insert(0, %r); import maestro; maestro.route('fix the failing test', harness='claude-code', "
                "inventory={'version': 2, 'generated_at': 9e12, 'harnesses': {}, 'api_keys': {}, 'local_endpoints': [], 'local_models': [], 'endpoints_sig': [], 'hardware': {}, 'warnings': []}, "
                "config=maestro.load_config(%r)); bad = [m for m in ('maestro.server', 'maestro.onboard', 'maestro.clients', 'maestro.providers', 'maestro.quota', 'maestro.learn', 'maestro.systemone') if m in sys.modules]; "
                "print(bad)") % (os.path.join(helpers.ROOT, "skills", "maestro", "scripts"), tempfile.gettempdir())
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ, MAESTRO_HOME=tempfile.gettempdir()))
        self.assertEqual(res.stdout.strip(), "[]", res.stderr)


class JevPick(ServerCase):
    def test_jev_picks_among_the_top_five_when_a_key_is_set(self):
        a, b = self.up(name="a", text="from a"), self.up(name="b", text="from b")
        self.start([E("openai/a", "strong", 0.7, (1.0, 2.0)), E("groq/b", "strong", 0.6, (5.0, 20.0))],
                   {"providers": {"openai": {"base_url": a.url}, "groq": {"base_url": b.url}}}, env_extra={"TYPESAFE_API_KEY": "jev-key"},
                   jev_fetch=lambda url, headers, body, timeout: (200, json.dumps({"answers": {
                       "model": {"choice": "groq/b", "confidence": 0.9, "probabilities": {}}, "job": {"choice": "plan", "confidence": 0.9},
                       "reasoning": {"score": 3}, "scope": {"score": 1}, "ambiguity": {"score": 1}, "risk": {"score": 1}}}).encode()))
        st, h, j = self.chat(MSG)
        d = self.decision(h)
        self.assertEqual((j["model"], d["job"]), ("groq/b", "plan"))
        self.assertIn("jev chose groq/b", d["reason"])
        self.assertEqual(self.rows()[0]["job"], "plan")


if __name__ == "__main__":
    unittest.main()
