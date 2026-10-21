"""Local and network models: parsing, ratings, specialists, eligibility, detect, run, scan."""
import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from helpers import cls, inventory, local, maestro as m


class FakeModelServer:
    """A tiny model server on 127.0.0.1: Ollama (/api/tags, /api/chat) or OpenAI-compatible
    (/v1/models, /v1/chat/completions). Records every request as (method, path, headers, json body).
    `loaded`: the Ollama models /api/ps reports (default: all of `models`). `states`: LM Studio's /api/v0/models, {name: "loaded" | "not-loaded"}."""

    def __init__(self, kind="ollama", models=("gpt-oss:120b",), sizes=None, require_key=None, loaded=None, states=None):
        self.kind, self.models, self.sizes, self.require_key, self.seen = kind, models, sizes or {}, require_key, []
        self.loaded, self.states = models if loaded is None else loaded, states
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _handle(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                outer.seen.append((method, self.path, dict(self.headers), body))
                if outer.require_key and self.headers.get("Authorization") != "Bearer " + outer.require_key:
                    return self._reply(401, {"error": "unauthorized"})
                if outer.kind == "ollama" and (method, self.path) == ("GET", "/api/tags"):
                    return self._reply(200, {"models": [{"name": x, "details": {"parameter_size": outer.sizes.get(x, "")}}
                                                        for x in outer.models]})
                if outer.kind == "ollama" and (method, self.path) == ("GET", "/api/ps"):
                    return self._reply(200, {"models": [{"name": x, "model": x} for x in outer.loaded]})
                if outer.kind == "openai" and outer.states is not None and (method, self.path) == ("GET", "/api/v0/models"):
                    return self._reply(200, {"data": [{"id": x, "state": st} for x, st in outer.states.items()]})
                if outer.kind == "ollama" and (method, self.path) == ("POST", "/api/chat"):
                    return self._reply(200, {"message": {"role": "assistant", "content": "ollama says: " + body["messages"][-1]["content"]}})
                if outer.kind == "openai" and (method, self.path) == ("GET", "/v1/models"):
                    return self._reply(200, {"data": [{"id": x} for x in outer.models]})
                if outer.kind == "openai" and (method, self.path) == ("POST", "/v1/chat/completions"):
                    return self._reply(200, {"choices": [{"message": {"content": "openai says: " + body["messages"][-1]["content"]}}]})
                self._reply(404, {"error": "not found"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Params(unittest.TestCase):
    def test_name_to_params(self):
        for name, want in (("llama3.1:70b", 70), ("gpt-oss:120b", 120), ("mixtral:8x7b", 56), ("qwen3:30b-a3b", 30),
                           ("qwen3-30b-a3b-instruct-2507", 30), ("llama3.2:3b-instruct-q4_K_M", 3),
                           ("qwen2.5-coder:7b", 7), ("meta-llama/Llama-3.1-8B-Instruct", 8),
                           ("tinyllama:1.1b", 1.1), ("gemma3:27b", 27), ("deepseek-r1:671b", 671)):
            self.assertEqual(m._params_b(name), want, name)
        for name in ("phi3:mini", "llama3", "qwen3:a3b", "gemma3n:e2b", ""):
            self.assertIsNone(m._params_b(name), name)

    def test_ollama_parameter_size_wins_and_handles_millions(self):
        self.assertEqual(m._params_b("anything", "70.6B"), 70.6)
        self.assertEqual(m._params_b("anything", "494.03M"), 0.49)
        self.assertEqual(m._params_b("qwen:32b", ""), 32)


class Serves(unittest.TestCase):
    def rec(self, name, params_b, endpoint="spark"):
        return {"id": "local:%s/%s" % (endpoint, name), "name": name, "endpoint_name": endpoint, "params_b": params_b}

    def test_defaults_from_parameter_count(self):
        for b, want in ((None, "local"), (3, "local"), (9.9, "local"), (10, "fast"), (32, "fast"), (39, "fast"),
                        (40, "standard"), (70, "standard"), (120, "standard"), (671, "standard")):
            self.assertEqual(m.rate_local(self.rec("x", b), {})[0], want, b)

    def test_never_strong_or_frontier_by_default(self):
        self.assertEqual(m.rate_local(self.rec("huge", 2000), {})[0], "standard")

    def test_config_override_by_name_endpoint_or_id(self):
        r = self.rec("gpt-oss:120b", 120)
        for key in ("gpt-oss:120b", "spark/gpt-oss:120b", "local:spark/gpt-oss:120b"):
            self.assertEqual(m.rate_local(r, {"local_models": {key: {"serves": "strong"}}})[0], "strong", key)
        self.assertEqual(m.rate_local(r, {"local_models": {"gpt-oss:120b": {"serves": "bogus"}}})[0], "standard")
        self.assertEqual(m.rate_local(self.rec("tiny", 1), {"local_models": {"tiny": {"serves": "frontier"}}})[0], "frontier")

    def test_discovered_serves_and_specialist_come_from_models_json_but_config_wins(self):
        big = self.rec("qwen3-coder:30b", 30)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"MAESTRO_HOME": tmp}):
            self.assertEqual(m.rate_local(self.rec("mystery:30b", 30), {}), ("fast", []))   # no models.json: the size heuristic
            entries = [{"id": "local:spark/mystery:30b", "serves": "strong", "G": 0.5, "specialist": False, "jobs": {}},
                       {"id": "local:spark/lifted:8b", "serves": "fast", "G": 0.4, "specialist": True, "jobs": {"code": 0.6, "ask": 0.4}},
                       {"id": "local:spark/bad:8b", "serves": "bogus"}, "junk"]
            with open(os.path.join(tmp, "models.json"), "w") as f:
                json.dump({"version": 1, "models": entries}, f)
            self.assertEqual(m.rate_local(self.rec("mystery:30b", 30), {}), ("strong", []))   # discover's scored view, not the size
            self.assertEqual(m.rate_local(self.rec("lifted:8b", 8), {}), ("fast", ["code"]))
            self.assertEqual(m.rate_local(self.rec("bad:8b", 8), {})[0], "local")            # an unusable entry is ignored
            self.assertEqual(m.rate_local(self.rec("other:70b", 70), {})[0], "standard")      # not in models.json
            self.assertEqual(m.rate_local(self.rec("mystery:30b", 30), {"local_models": {"mystery:30b": {"serves": "fast"}}})[0], "fast")
            self.assertEqual(m.rate_local(self.rec("mystery:30b", 30), {"models": {"local": {"standard": "spark/mystery:30b"}}})[0], "standard")
            self.assertEqual(m.rate_local(self.rec("lifted:8b", 8), {"specialists": {"lifted:8b": ["data"]}}), ("fast", ["data"]))
            self.assertEqual(m.rate_local(big, {})[1], ["code", "debug", "review"])   # the name heuristic still applies without the file entry

    def test_serves_reaches_the_inventory_and_can_be_overridden_at_route_time(self):
        rec = local("gpt-oss:120b", 120, endpoint="spark")
        inv = inventory(local_models=[rec])
        pol = {"local_models": {"gpt-oss:120b": {"serves": "strong"}}}
        s = m.select_seat("strong", cls("plan", 2), inv, "claude-code", pol, None)
        self.assertEqual(s["model"], "local:spark/gpt-oss:120b")  # rated at route time; no re-detect needed
        s = m.select_seat("strong", cls("plan", 2), inv, "claude-code", {}, None)
        self.assertEqual(s["model"], "opus")


class Specialists(unittest.TestCase):
    def test_name_heuristics(self):
        self.assertEqual(m._specialist_jobs("qwen2.5-coder:7b"), ["code", "debug", "review"])
        for name in ("devstral:24b", "starcoder2:15b", "codellama:13b", "deepseek-coder-v2"):
            self.assertEqual(m._specialist_jobs(name), ["code", "debug", "review"], name)
        self.assertEqual(m._specialist_jobs("deepseek-math:7b"), ["data"])
        self.assertEqual(m._specialist_jobs("sqlcoder:15b"), ["code", "debug", "review", "data"])
        self.assertEqual(m._specialist_jobs("llama3.1:70b"), [])

    def test_specialists_get_a_floor_of_fast_and_config_overrides(self):
        rec = {"id": "local:s/x", "name": "qwen2.5-coder:3b", "endpoint_name": "s", "params_b": 3}
        self.assertEqual(m.rate_local(rec, {}), ("fast", ["code", "debug", "review"]))
        self.assertEqual(m.rate_local(dict(rec, name="plain:3b"), {}), ("local", []))
        self.assertEqual(m.rate_local(dict(rec, name="plain:3b"), {"specialists": {"plain:3b": ["data", "nope"]}}), ("fast", ["data"]))
        pol = {"local_models": {"qwen2.5-coder:3b": {"specialist_for": ["review"], "serves": "standard"}}}
        self.assertEqual(m.rate_local(rec, pol), ("standard", ["review"]))
        self.assertEqual(m.rate_local(rec, {"local_models": {"qwen2.5-coder:3b": {"specialist_for": []}}})[1], [])

    def test_specialist_wins_for_its_job_and_reports_the_category(self):
        inv = inventory(local_models=[local("sqlcoder:15b", 15), local("gpt-oss:120b", 120, endpoint="spark")])
        s = m.select_seat("fast", cls("data", 0), inv, "claude-code", {}, None)  # both serve fast: the specialist wins
        self.assertEqual((s["model"], s["category"]), ("local:this-machine/sqlcoder:15b", "specialist"))
        self.assertEqual(s["fallback"], [{"seat": "claude-code", "model": "haiku", "category": "fast"}])

    def test_specialist_only_serves_up_to_its_rating(self):
        inv = inventory(local_models=[local("sqlcoder:15b", 15), local("gpt-oss:120b", 120, endpoint="spark")])
        s = m.select_seat("standard", cls("data", 1), inv, "claude-code", {}, None)  # 15B serves fast: too weak here
        self.assertEqual((s["model"], s["category"]), ("local:spark/gpt-oss:120b", "local"))
        pol = {"local_models": {"sqlcoder:15b": {"serves": "standard"}}}  # unless the user says otherwise
        s = m.select_seat("standard", cls("data", 1), inv, "claude-code", pol, None)
        self.assertEqual((s["model"], s["category"]), ("local:this-machine/sqlcoder:15b", "specialist"))
        s = m.select_seat("strong", cls("data", 2), inv, "claude-code", {}, None)
        self.assertEqual((s["kind"], s["model"]), ("subagent", "opus"))

    def test_specialist_is_used_only_for_its_jobs(self):
        inv = inventory(local_models=[local("sqlcoder:15b", 15)])
        s = m.select_seat("fast", cls("ask", 1), inv, "claude-code", {}, None)
        self.assertEqual((s["kind"], s["model"]), ("subagent", "haiku"))

    def test_specialists_config_can_name_a_cloud_model(self):
        inv = inventory(clis=("codex",))
        pol = {"harness_models": {"codex": {"fast": "codex-mini", "standard": "codex-code"}},
               "specialists": {"codex-code": ["code"]}}
        s = m.select_seat("fast", cls("code", 1), inv, "claude-code", pol, None)
        self.assertEqual((s["seat"], s["model"], s["category"]), ("codex", "codex-code", "specialist"))
        s = m.select_seat("fast", cls("ask", 1), inv, "claude-code", pol, None)  # not its job
        self.assertEqual((s["seat"], s["model"]), ("claude-code", "haiku"))


class Eligibility(unittest.TestCase):
    def names(self, c, inv, harness="claude-code", pol=None):
        return {x["name"] for x in m.build_candidates(c, inv, harness, pol or {})}

    def test_plain_local_models_are_chat_only(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark")])
        want = "local:spark/gpt-oss:120b"
        for job in ("ask", "summarize", "write", "plan", "data"):
            self.assertIn(want, self.names(cls(job, 1), inv), job)
        for job in sorted(m.TOOL_JOBS):
            self.assertNotIn(want, self.names(cls(job, 1), inv), job)
        self.assertNotIn(want, self.names(cls("write", 1, needs=("tools",)), inv))

    def test_tools_job_goes_local_only_through_a_native_harness_subagent(self):
        rec = local("gpt-oss:120b", 120, endpoint="spark")
        inv = inventory(local_models=[rec])
        pol = {"harness_models": {"opencode": {"standard": "ollama/gpt-oss:120b", "strong": "cloud/big-model"}}}
        c = cls("code", 1, needs=("tools",))
        s = m.select_seat("standard", c, inv, "opencode", pol, None)
        self.assertEqual(s["kind"], "subagent")
        self.assertEqual(s["delegate"], {"kind": "subagent", "harness": "opencode", "model": "ollama/gpt-oss:120b",
                                         "agent": "maestro-standard"})
        self.assertEqual(s["category"], "local")
        # ... and not from a harness that has no such configuration
        s = m.select_seat("standard", c, inv, "claude-code", {}, None)
        self.assertEqual((s["kind"], s["model"]), ("subagent", "sonnet"))
        s = m.select_seat("standard", c, inv, "opencode", {"harness_models": {"opencode": {"standard": "cloud/mid"}}}, None)
        self.assertEqual(s["model"], "cloud/mid")
        # prefer_local false turns the native local subagent off too
        s = m.select_seat("standard", c, inv, "opencode", dict(pol, prefer_local=False), None)
        self.assertNotEqual(s["category"], "local")

    def test_native_local_subagent_is_not_also_listed_as_a_plain_local_model(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark")])
        pol = {"harness_models": {"opencode": {"standard": "spark/gpt-oss:120b"}}}
        names = self.names(cls("write", 1), inv, "opencode", pol)
        self.assertIn("opencode:spark/gpt-oss:120b", names)
        self.assertNotIn("local:spark/gpt-oss:120b", names)

    def test_max_tier_caps_cloud_candidates_only(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark")])
        names = self.names(cls("write", 1), inv, pol={"max_tier": "fast"})
        self.assertIn("local:spark/gpt-oss:120b", names)
        self.assertNotIn("claude-code:opus", names)

    def test_jev_descriptions_for_local_and_specialist_models(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark"), local("sqlcoder:15b", 15)])
        desc = {c["name"]: c["description"] for c in m.build_candidates(cls("data", 1), inv, "claude-code", {})}
        self.assertEqual(desc["local:spark/gpt-oss:120b"],
                         "local model on spark (120B), free and private, serves up to standard-level work, no repo tools")
        self.assertIn("tuned for code, debug, review, data", desc["local:this-machine/sqlcoder:15b"])


class RouteWithLocalModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"MAESTRO_HOME": self.tmp.name, "HOME": self.tmp.name, "PATH": ""}
        self.inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark")])

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, text, **kw):
        kw.setdefault("harness", "claude-code")
        kw.setdefault("inventory", self.inv)
        return m.route(text, env=self.env, home=self.tmp.name, classifier="rules", **kw)

    def test_spark_120b_is_chosen_for_write_d1_over_sonnet(self):
        cfg = dict(m.load_config(self.tmp.name), classifier="rules")
        d = self.go("write a readme for the billing service", config=cfg)
        self.assertEqual((d["job"], d["difficulty"]), ("write", 1))
        self.assertEqual((d["category"], d["seat"], d["model"]), ("local", "local", "local:spark/gpt-oss:120b"))
        self.assertEqual(d["delegate"]["kind"], "local")
        self.assertIn("serves standard", d["reason"])
        self.assertEqual(d["fallback"][0], {"seat": "claude-code", "model": "sonnet", "category": "standard"})
        d = self.go("write a readme for the billing service", config=dict(cfg, prefer_local=False))
        self.assertEqual((d["category"], d["model"]), ("standard", "sonnet"))

    def test_debug_stays_with_a_claude_subagent(self):
        d = self.go("why's the checkout test flaky")
        self.assertEqual(d["job"], "debug")
        self.assertEqual((d["seat"], d["delegate"]["kind"]), ("claude-code", "subagent"))

    def test_hard_write_goes_to_the_cloud_when_the_local_model_is_too_weak(self):
        d = self.go("write the whole launch announcement", config=dict(m.load_config(self.tmp.name),
                                                                       ladder={"write": ["strong"] * 4}))
        self.assertEqual(d["model"], "opus")

    def test_stale_inventory_shape_or_changed_endpoints_trigger_a_redetect(self):
        path = os.path.join(self.tmp.name, "inventory.json")
        cfg = m.load_config(self.tmp.name)
        with mock.patch.object(m, "detect", wraps=m.detect) as det:
            old = dict(inventory(), version=1)
            with open(path, "w") as f:
                json.dump(old, f)
            m.ensure_inventory(self.tmp.name, self.env, cfg=cfg)
            self.assertEqual(det.call_count, 1)
            m.ensure_inventory(self.tmp.name, self.env, cfg=cfg)   # fresh now: served from the cache
            self.assertEqual(det.call_count, 1)
            cfg2 = dict(cfg, local_endpoints=["http://127.0.0.1:1"])
            m.ensure_inventory(self.tmp.name, self.env, cfg=cfg2)  # endpoint list changed
            self.assertEqual(det.call_count, 2)


class Detect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"PATH": "", "HOME": self.tmp.name, "MAESTRO_HOME": self.tmp.name}
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.close()
        self.tmp.cleanup()

    def serve(self, **kw):
        s = FakeModelServer(**kw)
        self.servers.append(s)
        return s

    def detect(self, cfg, **kw):
        return m.detect(home=self.tmp.name, env=self.env, cfg=dict(m.load_config(self.tmp.name), **cfg),
                        hw_probe=lambda: {"memory_gb": 512.0}, **kw)

    def test_endpoint_list_order_and_dedupe(self):
        eps = m.local_endpoints({"local_endpoints": ["http://spark.local:11434/", {"name": "gpu", "url": "localhost:11434"},
                                                     {"url": ""}, 5]}, {"OLLAMA_HOST": "0.0.0.0"})
        self.assertEqual([(e["name"], e["url"]) for e in eps[:3]],
                         [("spark.local", "http://spark.local:11434"), ("gpu", "http://127.0.0.1:11434"),
                          ("this-machine", "http://127.0.0.1:1234")])
        self.assertEqual(sorted(e["url"] for e in eps if e["builtin"]),
                         sorted("http://127.0.0.1:%d" % p for p in (1234, 30000, 8000, 8080)))  # 11434 came from config
        self.assertEqual(len({e["url"] for e in eps}), len(eps))
        eps = m.local_endpoints({}, {"OLLAMA_HOST": "gpu-box"})
        self.assertEqual(eps[0], {"name": "gpu-box", "url": "http://gpu-box:11434", "kind": "ollama",
                                  "api_key_env": None, "builtin": False})

    def test_detects_an_ollama_endpoint_from_config(self):
        srv = self.serve(models=("gpt-oss:120b", "llama3.2:3b", "nomic-embed-text"), sizes={"gpt-oss:120b": "116.8B", "llama3.2:3b": "3.2B"})
        t0 = time.perf_counter()
        inv = self.detect({"local_endpoints": [{"name": "spark", "url": srv.url}]})
        self.assertLess(time.perf_counter() - t0, 2.0)
        models = {x["name"]: x for x in inv["local_models"] if x["endpoint_name"] == "spark"}
        self.assertEqual(set(models), {"gpt-oss:120b", "llama3.2:3b"})  # embedding models are skipped
        big = models["gpt-oss:120b"]
        self.assertEqual((big["id"], big["kind"], big["params_b"], big["serves"], big["specialist_for"], big["on_this_machine"]),
                         ("local:spark/gpt-oss:120b", "ollama", 116.8, "standard", [], True))
        self.assertEqual(models["llama3.2:3b"]["serves"], "local")
        self.assertEqual(inv["version"], m.INVENTORY_VERSION)
        self.assertIn(srv.url, inv["endpoints_sig"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "inventory.json")))
        self.assertEqual(srv.seen[0][:2], ("GET", "/api/tags"))

    def test_kind_is_auto_detected_openai_after_ollama(self):
        srv = self.serve(kind="openai", models=("meta-llama/Llama-3.1-70B-Instruct",))
        inv = self.detect({"local_endpoints": [srv.url]})
        rec = next(x for x in inv["local_models"] if x["url"] == srv.url)
        self.assertEqual((rec["kind"], rec["params_b"], rec["serves"]), ("openai", 70, "standard"))
        self.assertEqual([p for _, p, _, _ in srv.seen], ["/api/tags", "/v1/models", "/api/v0/models"])

    def test_ollama_host_env_and_config_overrides_reach_the_inventory(self):
        srv = self.serve(models=("qwen2.5-coder:7b",))
        self.env["OLLAMA_HOST"] = srv.url.replace("http://", "")
        inv = self.detect({"local_models": {"qwen2.5-coder:7b": {"serves": "standard"}}})
        rec = next(x for x in inv["local_models"] if x["url"] == srv.url)
        self.assertEqual((rec["serves"], rec["specialist_for"]), ("standard", ["code", "debug", "review"]))

    def test_unreachable_endpoints_are_skipped_and_fast(self):
        t0 = time.perf_counter()
        inv = self.detect({"local_endpoints": ["http://127.0.0.1:9"]})
        self.assertLess(time.perf_counter() - t0, 2.0)
        self.assertEqual(inv["local_models"], [])
        self.assertEqual(inv["local_endpoints"], [])

    def test_the_suite_never_probes_this_machines_real_model_servers(self):
        self.assertFalse(m.PROBE_BUILTIN_PORTS)   # helpers.py
        self.assertEqual(os.environ.get("MAESTRO_NO_BUILTIN_PORTS"), "1")   # for subprocesses
        with mock.patch.object(m, "_local_fetch", side_effect=AssertionError("probed a real localhost port")):
            self.assertEqual(self.detect({})["local_endpoints"], [])

    def test_builtin_ports_can_be_skipped_and_only_they_are(self):
        srv = self.serve(models=("tiny:3b",))
        port = int(srv.url.rsplit(":", 1)[1])
        names = lambda inv: [x["name"] for x in inv["local_models"]]
        with mock.patch.object(m, "BUILTIN_PORTS", ((port, "ollama"),)):
            self.assertEqual(names(self.detect({})), [])   # off for the suite
            with mock.patch.object(m, "PROBE_BUILTIN_PORTS", True):
                self.assertEqual(names(self.detect({})), ["tiny:3b"])   # the production default
                self.env["MAESTRO_NO_BUILTIN_PORTS"] = "1"
                self.assertEqual(names(self.detect({})), [])
                self.assertEqual(names(self.detect({"local_endpoints": [srv.url]})), ["tiny:3b"])   # a configured endpoint still is
            ep = m.local_endpoints({}, {})[0]
            self.assertTrue(ep["builtin"])
            self.assertEqual(m.probe_endpoint(ep, m._local_fetch)["models"][0]["name"], "tiny:3b")   # an injected fetch is used

    def test_probes_run_in_parallel(self):
        calls = []

        def slow(ep):
            calls.append(ep["url"])
            time.sleep(0.2)
            return None
        t0 = time.perf_counter()
        self.detect({"local_endpoints": ["http://a:1", "http://b:1", "http://c:1"]}, probe=slow)
        self.assertEqual(len(calls), 8)
        self.assertLess(time.perf_counter() - t0, 0.6)  # 8 x 0.2 s sequentially would be 1.6 s

    def test_probe_endpoint_uses_the_key_env_but_the_inventory_never_stores_it(self):
        srv = self.serve(kind="openai", models=("m-8b",), require_key="sk-very-secret")
        self.env["SPARK_KEY"] = "sk-very-secret"
        cfg = {"local_endpoints": [{"name": "vllm", "url": srv.url, "kind": "openai", "api_key_env": "SPARK_KEY"}]}
        inv = self.detect(cfg)
        self.assertEqual([x["name"] for x in inv["local_models"] if x["endpoint_name"] == "vllm"], ["m-8b"])
        with open(os.path.join(self.tmp.name, "inventory.json")) as f:
            self.assertNotIn("sk-very-secret", f.read())
        del self.env["SPARK_KEY"]
        self.assertEqual([x for x in self.detect(cfg)["local_models"] if x["endpoint_name"] == "vllm"], [])  # 401 without it

    def test_memory_warning_only_for_models_on_this_machine(self):
        srv = self.serve(models=("big:70b",))
        cfg = {"local_endpoints": [{"name": "spark.example", "url": srv.url.replace("127.0.0.1", "localhost")}]}
        inv = m.detect(home=self.tmp.name, env=self.env, cfg=dict(m.load_config(self.tmp.name), **cfg),
                       hw_probe=lambda: {"memory_gb": 16.0})
        self.assertTrue(any("big:70b" in w for w in inv["warnings"]))  # localhost: this machine
        cfg = {"local_endpoints": [{"name": "spark", "url": "http://spark.example:11434"}]}
        inv = m.detect(home=self.tmp.name, env=self.env, cfg=dict(m.load_config(self.tmp.name), **cfg),
                       hw_probe=lambda: {"memory_gb": 16.0},
                       probe=lambda ep: {"kind": "ollama", "latency_ms": 1.0, "models": [{"name": "big:70b", "size": None}]}
                       if ep["name"] == "spark" else None)
        rec = next(x for x in inv["local_models"] if x["endpoint_name"] == "spark")
        self.assertFalse(rec["on_this_machine"])
        self.assertEqual(inv["warnings"], [])

    def test_detect_cli_prints_local_models(self):
        srv = self.serve(models=("gpt-oss:120b",), sizes={"gpt-oss:120b": "116.8B"})
        with open(os.path.join(self.tmp.name, "config.json"), "w") as f:
            json.dump({"local_endpoints": [{"name": "spark", "url": srv.url}]}, f)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": self.tmp.name, "PATH": ""}), contextlib.redirect_stdout(out):
            self.assertEqual(m.main(["detect"]), 0)
        self.assertIn("spark (ollama, %s): gpt-oss:120b [standard]" % srv.url, out.getvalue())


class Scan(unittest.TestCase):
    def test_scan_is_bounded_and_probes_the_own_subnet_only(self):
        seen = []

        def probe(ep):
            seen.append(ep["url"])
            if ep["url"] == "http://192.168.7.50:11434":
                return {"kind": "ollama", "latency_ms": 3.0, "models": [{"name": "gpt-oss:120b"}]}
            time.sleep(0.05)
            return None
        t0 = time.perf_counter()
        found = m.scan_network(env={}, probe=probe, local_ip="192.168.7.20")
        self.assertLess(time.perf_counter() - t0, 6.0)
        self.assertEqual(len(seen), 253 * 2)  # .1-.254 without our own address, ports 11434 and 8000
        self.assertTrue(all(u.startswith("http://192.168.7.") for u in seen))
        self.assertFalse(any(u.startswith("http://192.168.7.20:") for u in seen))
        self.assertEqual(found, [{"name": "192.168.7.50", "url": "http://192.168.7.50:11434", "kind": "ollama",
                                  "api_key_env": None, "builtin": False, "models": 1}])

    def test_scan_gives_up_at_the_budget(self):
        t0 = time.perf_counter()
        found = m.scan_network(env={}, probe=lambda ep: time.sleep(0.4), local_ip="10.0.0.5", budget=0.2, workers=4)
        self.assertLess(time.perf_counter() - t0, 1.5)
        self.assertEqual(found, [])

    def test_scan_respects_no_scan_and_ignores_loopback(self):
        with self.assertRaises(RuntimeError):
            m.scan_network(env={"MAESTRO_NO_SCAN": "1"}, probe=lambda ep: self.fail("probed"), local_ip="10.0.0.5")
        self.assertEqual(m.scan_network(env={}, probe=lambda ep: self.fail("probed"), local_ip="127.0.0.1"), [])

    def test_scan_never_runs_unless_asked(self):
        with mock.patch.object(m, "scan_network") as scan, tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"MAESTRO_HOME": tmp, "PATH": ""}), contextlib.redirect_stdout(io.StringIO()):
            m.detect(home=tmp, cfg=m.load_config(tmp), probe=lambda ep: None)
            m.main(["detect"])
            m.route("hello", harness="claude-code")
            scan.assert_not_called()

    def test_detect_scan_prints_and_save_writes_the_config(self):
        found = [{"name": "192.168.7.50", "url": "http://192.168.7.50:11434", "kind": "ollama",
                  "api_key_env": None, "builtin": False, "models": 2}]
        probe = lambda ep: ({"kind": "ollama", "latency_ms": 1.0, "models": [{"name": "gpt-oss:120b", "size": "116.8B"}]}
                            if ep["url"] == found[0]["url"] else None)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"prefer_local": False, "local_endpoints": ["http://old:11434"]}, f)
            env = {"MAESTRO_HOME": tmp, "PATH": ""}
            with mock.patch.dict(os.environ, env), mock.patch.object(m, "scan_network", return_value=found), \
                    mock.patch.object(m, "probe_endpoint", side_effect=lambda ep, *a, **k: probe(ep)):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(m.main(["detect", "--scan"]), 0)
                self.assertIn("scan: 1 endpoint(s) found", out.getvalue())
                self.assertIn("--scan --save", out.getvalue())
                self.assertIn("gpt-oss:120b", out.getvalue())  # shown for this run ...
                self.assertFalse(os.path.exists(os.path.join(tmp, "inventory.json")))  # ... but not cached
                with open(os.path.join(tmp, "config.json")) as f:
                    self.assertEqual(json.load(f)["local_endpoints"], ["http://old:11434"])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(m.main(["detect", "--scan", "--save"]), 0)
                with open(os.path.join(tmp, "config.json")) as f:
                    saved = json.load(f)
                self.assertEqual(saved["prefer_local"], False)  # the rest of the config is untouched
                self.assertEqual(saved["local_endpoints"][1], {"name": "192.168.7.50", "url": "http://192.168.7.50:11434", "kind": "ollama"})
                self.assertEqual(len(saved["local_endpoints"]), 2)
                m.save_endpoints(found, tmp)  # saving twice does not duplicate
                with open(os.path.join(tmp, "config.json")) as f:
                    self.assertEqual(len(json.load(f)["local_endpoints"]), 2)

    def test_cli_scan_refused_by_env(self):
        with mock.patch.dict(os.environ, {"MAESTRO_NO_SCAN": "1", "MAESTRO_HOME": "/nonexistent-maestro"}), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(m.main(["detect", "--scan"]), 2)
        self.assertIn("MAESTRO_NO_SCAN", err.getvalue())


class Run(unittest.TestCase):
    def setUp(self):
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.close()

    def serve(self, **kw):
        s = FakeModelServer(**kw)
        self.servers.append(s)
        return s

    def cfg(self, srv, **ep):
        return dict(m.load_config("/nonexistent"), local_endpoints=[dict({"name": "spark", "url": srv.url}, **ep)])

    def test_ollama_chat(self):
        srv = self.serve()
        out = m.run_model("local:spark/gpt-oss:120b", "say hi", "be brief", self.cfg(srv), {}, env={})
        self.assertEqual(out, "ollama says: say hi")
        method, path, headers, body = srv.seen[-1]
        self.assertEqual((method, path), ("POST", "/api/chat"))
        self.assertEqual(body, {"model": "gpt-oss:120b", "stream": False,
                                "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "say hi"}]})
        self.assertNotIn("Authorization", headers)

    def test_openai_compatible_chat_with_bearer_from_key_env(self):
        srv = self.serve(kind="openai", models=("llama-70b",), require_key="s3cret")
        cfg = self.cfg(srv, kind="openai", api_key_env="VLLM_KEY")
        out = m.run_model("local:spark/llama-70b", "hello", None, cfg, {}, env={"VLLM_KEY": "s3cret"})
        self.assertEqual(out, "openai says: hello")
        method, path, headers, body = srv.seen[-1]
        self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
        self.assertEqual(body, {"model": "llama-70b", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(headers["Authorization"], "Bearer s3cret")
        with self.assertRaises(RuntimeError) as cm:  # no key in the environment: the server says 401
            m.run_model("local:spark/llama-70b", "hello", None, cfg, {}, env={})
        self.assertIn("401", str(cm.exception))
        self.assertNotIn("s3cret", str(cm.exception))

    def test_kind_comes_from_the_inventory_or_is_probed(self):
        srv = self.serve(kind="openai", models=("m",))
        inv = {"local_models": [{"id": "local:box/m", "url": srv.url, "kind": "openai"}]}
        self.assertEqual(m.run_model("local:box/m", "x", None, dict(m.load_config("/nonexistent")), inv, env={}), "openai says: x")
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{"name": "box", "url": srv.url}])
        self.assertEqual(m.run_model("local:box/m", "y", None, cfg, {}, env={}), "openai says: y")  # probed: /api/tags, then /v1/models

    def test_errors_are_runtime_errors_with_no_secrets(self):
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{"name": "dead", "url": "http://127.0.0.1:9", "kind": "ollama"}])
        for model_id, needle in (("gpt-oss", "local:<endpoint>/<model>"), ("local:nowhere/x", "unknown endpoint"),
                                 ("local:dead/x", "failed")):
            with self.assertRaises(RuntimeError) as cm:
                m.run_model(model_id, "x", None, cfg, {}, env={})
            self.assertIn(needle, str(cm.exception))
        srv = self.serve(models=())
        with self.assertRaises(RuntimeError):
            m.run_model("local:spark/x", "x", None, self.cfg(srv, kind="openai"), {}, env={})  # 404: wrong kind

    @mock.patch.dict(os.environ, {"PATH": "", "MAESTRO_NO_BUILTIN_PORTS": "1"}, clear=True)
    def test_upstream_errors_redact_endpoint_key_before_shortening(self):
        key = "fake-local-key-secret"
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{
            "name": "s", "url": "http://example.invalid:1", "kind": "openai", "api_key_env": "LOCAL_KEY"}])
        bodies = [
            (b"invalid credential " + key.encode(), "invalid credential ***"),
            (json.dumps({"error": {"message": "invalid " + key}}).encode(), "invalid ***"),
            (b"x" * 190 + key.encode(), "x" * 160),
            (json.dumps({"message": "invalid " + key}).encode().replace(b"-", b"\\u002d"), "invalid ***"),
        ]
        for kind in ("openai", "ollama"):
            for body, expected in bodies:
                with self.subTest(kind=kind, body=body), self.assertRaises(RuntimeError) as err:
                    m.run_model("local:s/m", "hello", cfg=dict(cfg, local_endpoints=[dict(cfg["local_endpoints"][0], kind=kind)]),
                                inv={}, env={"LOCAL_KEY": key}, fetch=lambda *args: (401, body), allow_load=True)
                self.assertEqual(str(err.exception), "HTTP 401 from http://example.invalid:1: " + expected)
                self.assertNotIn(key, str(err.exception))
                self.assertNotIn(key[:10], str(err.exception))

    @mock.patch.dict(os.environ, {"PATH": "", "MAESTRO_NO_BUILTIN_PORTS": "1"}, clear=True)
    def test_structured_error_values_redact_json_encoded_keys(self):
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{
            "name": "s", "url": "http://example.invalid:1", "kind": "openai", "api_key_env": "LOCAL_KEY"}])
        for key in ('fake-local-"secret', "fake-local-é-secret", "fake-local-\\secret"):
            for message in (["rejected", key], {"credential": key}):
                with self.subTest(key=key, message=message):
                    body = json.dumps({"error": {"message": message}}).encode()
                    with self.assertRaises(RuntimeError) as err:
                        m.run_model("local:s/m", "hello", cfg=cfg, inv={}, env={"LOCAL_KEY": key},
                                    fetch=lambda *args: (401, body), allow_load=True)
                    self.assertIn("***", str(err.exception))
                    self.assertNotIn(key, str(err.exception))
                    self.assertNotIn(json.dumps(key)[1:-1], str(err.exception))

    @mock.patch.dict(os.environ, {"PATH": "", "MAESTRO_NO_BUILTIN_PORTS": "1"}, clear=True)
    def test_nested_escape_errors_redact_before_truncation(self):
        key = "fake-nested-key"
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{
            "name": "s", "url": "http://example.invalid:1", "kind": "openai", "api_key_env": "LOCAL_KEY"}])
        nested = json.dumps({"credential": key}).replace("-", r"\u002d")
        for depth in (1, 2, 3, 6):
            message = nested
            for _ in range(depth - 1):
                message = json.dumps({"message": message})
            body = json.dumps({"error": {"message": message}}).encode()
            with self.subTest(depth=depth):
                with self.assertRaises(RuntimeError) as err:
                    m.run_model("local:s/m", "hello", cfg=cfg, inv={}, env={"LOCAL_KEY": key},
                                fetch=lambda *args: (401, body), allow_load=True)
                output = str(err.exception)
                self.assertTrue(key not in output, "literal synthetic credential must be absent")
                self.assertTrue(key.replace("-", r"\u002d") not in output, "escaped synthetic credential must be absent")
                self.assertTrue("***" in output, "nested credential must be redacted or omitted")

    @mock.patch.dict(os.environ, {"PATH": "", "MAESTRO_NO_BUILTIN_PORTS": "1"}, clear=True)
    def test_oversized_error_is_omitted_without_partial_credential(self):
        from maestro import providers
        key = "fake-oversized-key"
        body = json.dumps({"error": {"message": "x" * 20000 + key}}).encode()
        result = providers.error_message(body, key)
        self.assertTrue(result == providers.OMITTED_ERROR, "oversized credential-bearing errors must be omitted")
        self.assertTrue(key not in result, "synthetic credential must be absent")

    @mock.patch.dict(os.environ, {"PATH": "", "MAESTRO_NO_BUILTIN_PORTS": "1"}, clear=True)
    def test_cli_run_never_prints_echoed_endpoint_key(self):
        key = "fake-cli-key-secret"
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "config.json"), "w") as f:
                json.dump({"local_allow_load": True, "local_endpoints": [{"name": "s", "url": "http://example.invalid:1",
                           "kind": "openai", "api_key_env": "LOCAL_KEY"}]}, f)
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, {"MAESTRO_HOME": home, "LOCAL_KEY": key, "PATH": ""}, clear=True), \
                    mock.patch.object(m, "_local_fetch", return_value=(401, ("rejected " + key).encode())), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(m.main(["run", "local:s/m", "hello"]), 1)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("rejected ***", err.getvalue())
            self.assertNotIn(key, err.getvalue())

    @mock.patch.dict(os.environ, {"PATH": "", "MAESTRO_NO_BUILTIN_PORTS": "1"}, clear=True)
    def test_injected_success_preserves_reply_and_authentication(self):
        key = "fake-success-key"
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{
            "name": "s", "url": "http://example.invalid:1", "kind": "openai", "api_key_env": "LOCAL_KEY"}])
        def fetch(url, headers, body, timeout):
            self.assertEqual(headers["Authorization"], "Bearer " + key)
            self.assertEqual(json.loads(body)["messages"], [{"role": "user", "content": "hello"}])
            return 200, json.dumps({"choices": [{"message": {"content": "normal reply"}}]}).encode()
        self.assertEqual(m.run_model("local:s/m", "hello", cfg=cfg, inv={}, env={"LOCAL_KEY": key},
                                     fetch=fetch, allow_load=True), "normal reply")

    def test_timeout_default_and_override(self):
        seen = []

        def fetch(url, headers, body, timeout):
            seen.append(timeout)
            return 200, json.dumps({"message": {"content": "ok"}}).encode()
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{"name": "s", "url": "http://s:1", "kind": "ollama"}])
        m.run_model("local:s/m", "x", None, cfg, {}, env={}, fetch=fetch, allow_load=True)   # allow_load: no live GET first
        m.run_model("local:s/m", "x", None, cfg, {}, env={}, fetch=fetch, timeout=5, allow_load=True)
        self.assertEqual(seen, [300.0, 5])

    def test_cli_run_reads_the_task_from_args_or_stdin_and_prints_only_the_reply(self):
        srv = self.serve()
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"local_endpoints": [{"name": "spark", "url": srv.url, "kind": "ollama"}]}, f)
            with mock.patch.dict(os.environ, {"MAESTRO_HOME": tmp}):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(m.main(["run", "local:spark/gpt-oss:120b", "say", "hi", "--system", "terse"]), 0)
                self.assertEqual(out.getvalue(), "ollama says: say hi\n")
                self.assertEqual(srv.seen[-1][3]["messages"][0], {"role": "system", "content": "terse"})
                out = io.StringIO()
                with mock.patch("sys.stdin", io.StringIO("from stdin")), contextlib.redirect_stdout(out):
                    self.assertEqual(m.main(["run", "local:spark/gpt-oss:120b"]), 0)
                self.assertEqual(out.getvalue(), "ollama says: from stdin\n")
                out = io.StringIO()
                with contextlib.redirect_stdout(out):  # a task that starts with a dash is still a task
                    self.assertEqual(m.main(["run", "local:spark/gpt-oss:120b", "-x is the flag"]), 0)
                self.assertEqual(out.getvalue(), "ollama says: -x is the flag\n")
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(m.main(["run", "local:nowhere/x", "hi"]), 1)
                self.assertIn("unknown endpoint", err.getvalue())

    def test_the_delegate_argv_round_trips_through_the_cli(self):
        srv = self.serve(models=("gpt-oss:120b",))
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"local_endpoints": [{"name": "spark", "url": srv.url}]}, f)
            inv = m.detect(home=tmp, env={"PATH": "", "HOME": tmp, "MAESTRO_HOME": tmp}, hw_probe=lambda: {},
                           cfg=m.load_config(tmp))
            d = m.route("write a readme for the billing service", harness="claude-code", inventory=inv,
                        config=dict(m.load_config(tmp), classifier="rules"), env={"MAESTRO_HOME": tmp}, home=tmp)
            self.assertEqual(d["delegate"]["kind"], "local")
            argv = [a.replace("{task}", "document the billing service") for a in d["delegate"]["argv"]]
            import subprocess
            res = subprocess.run(argv, capture_output=True, text=True, timeout=30,
                                 env=dict(os.environ, MAESTRO_HOME=tmp, PATH=os.environ.get("PATH", "")))
            self.assertEqual((res.returncode, res.stdout), (0, "ollama says: document the billing service\n"), res.stderr)


class Loaded(unittest.TestCase):
    """Installed is not served: Ollama loads a model on its first chat request and LM Studio JIT-loads what /v1/models lists,
    so Maestro only counts what /api/ps or the server's own state says is loaded, and only ever sends GET requests to find out."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"PATH": "", "HOME": self.tmp.name, "MAESTRO_HOME": self.tmp.name}
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.close()
        self.tmp.cleanup()

    def serve(self, **kw):
        s = FakeModelServer(**kw)
        self.servers.append(s)
        return s

    def flags(self, srv, **ep):
        out = m.probe_endpoint(dict({"url": srv.url, "kind": None, "api_key_env": None}, **ep), timeout=2.0)
        return out["kind"], {x["name"]: x["loaded"] for x in out["models"]}

    def test_ollama_api_ps_says_which_installed_models_are_loaded(self):
        srv = self.serve(models=("gpt-oss:120b", "llama3.2:3b"), loaded=("gpt-oss:120b",))
        self.assertEqual(self.flags(srv), ("ollama", {"gpt-oss:120b": True, "llama3.2:3b": False}))
        self.assertEqual([(a, b) for a, b, _, _ in srv.seen], [("GET", "/api/tags"), ("GET", "/api/ps")])   # probes only read

    def test_ollama_without_an_answer_from_api_ps_counts_nothing_as_loaded(self):
        def fetch(url, headers, body, timeout):
            return (200, json.dumps({"models": [{"name": "a:7b"}]}).encode()) if url.endswith("/api/tags") else (404, b"")
        out = m.probe_endpoint({"url": "http://h:11434", "kind": "ollama"}, fetch=fetch)
        self.assertEqual([(x["name"], x["loaded"]) for x in out["models"]], [("a:7b", False)])

    def test_lm_studio_state_from_api_v0_models(self):
        srv = self.serve(kind="openai", models=("qwen3-30b", "gemma-3-4b"), states={"qwen3-30b": "loaded", "gemma-3-4b": "not-loaded"})
        self.assertEqual(self.flags(srv), ("openai", {"qwen3-30b": True, "gemma-3-4b": False}))
        self.assertEqual({(a, b) for a, b, _, _ in srv.seen} - {("GET", "/api/tags")},
                         {("GET", "/v1/models"), ("GET", "/api/v0/models")})

    def test_vllm_style_servers_serve_what_v1_models_lists(self):
        srv = self.serve(kind="openai", models=("openai/gpt-oss-120b",))   # /api/v0/models is a 404
        self.assertEqual(self.flags(srv), ("openai", {"openai/gpt-oss-120b": True}))

    def test_detect_records_loaded_and_the_inventory_and_summary_show_it(self):
        srv = self.serve(models=("gpt-oss:120b", "llama3.2:3b"), loaded=("gpt-oss:120b",))
        inv = m.detect(home=self.tmp.name, env=self.env, hw_probe=lambda: {},
                       cfg=dict(m.load_config(self.tmp.name), local_endpoints=[{"name": "spark", "url": srv.url}]))
        got = {x["name"]: x["loaded"] for x in inv["local_models"] if x["endpoint_name"] == "spark"}
        self.assertEqual(got, {"gpt-oss:120b": True, "llama3.2:3b": False})
        text = m.summarize(inv)
        self.assertIn("llama3.2:3b [local, installed, not loaded: not used]", text)
        self.assertIn("gpt-oss:120b [standard]", text)
        self.assertTrue(all(method == "GET" for method, _, _, _ in srv.seen))

    def test_route_never_picks_a_model_that_is_not_loaded_unless_local_allow_load(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark", loaded=False)])
        cfg = dict(m.load_config(self.tmp.name), classifier="rules")
        kw = dict(harness="claude-code", inventory=inv, env=self.env, home=self.tmp.name, classifier="rules")
        d = m.route("write a readme for the billing service", config=cfg, **kw)
        self.assertNotEqual(d["seat"], "local")
        self.assertNotIn("local:spark/gpt-oss:120b", json.dumps(d))
        d = m.route("write a readme for the billing service", config=dict(cfg, local_allow_load=True), **kw)
        self.assertEqual(d["model"], "local:spark/gpt-oss:120b")
        inv["local_models"][0]["loaded"] = True
        self.assertEqual(m.route("write a readme for the billing service", config=cfg, **kw)["model"], "local:spark/gpt-oss:120b")
        old = inventory(local_models=[{k: v for k, v in local("gpt-oss:120b", 120, endpoint="spark").items() if k != "loaded"}])
        kw["inventory"] = old   # an inventory from before the field existed counts as loaded
        self.assertEqual(m.route("write a readme for the billing service", config=cfg, **kw)["model"], "local:spark/gpt-oss:120b")

    def test_models_table_leaves_out_a_model_that_is_not_loaded(self):
        inv = inventory(local_models=[local("gpt-oss:120b", 120, endpoint="spark", loaded=False)])
        rows, _ = m.models_rows(inv, m.load_config(self.tmp.name))
        self.assertEqual([r for r in rows if r[0] == "local"], [])

    def test_run_refuses_a_model_that_is_installed_but_not_loaded(self):
        srv = self.serve(models=("gpt-oss:120b", "tiny:3b"), loaded=("tiny:3b",))
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{"name": "spark", "url": srv.url}])
        with self.assertRaises(RuntimeError) as cm:
            m.run_model("local:spark/gpt-oss:120b", "hi", None, cfg, {}, env={})
        self.assertIn("installed on %s but not loaded" % srv.url, str(cm.exception))
        self.assertIn("--allow-load", str(cm.exception))
        self.assertNotIn("POST", [method for method, _, _, _ in srv.seen])   # nothing that could load it was sent
        self.assertEqual(m.run_model("local:spark/tiny:3b", "hi", None, cfg, {}, env={}), "ollama says: hi")   # loaded: fine
        self.assertEqual(m.run_model("local:spark/gpt-oss:120b", "hi", None, cfg, {}, env={}, allow_load=True), "ollama says: hi")
        self.assertEqual(m.run_model("local:spark/gpt-oss:120b", "hi", None, dict(cfg, local_allow_load=True), {}, env={}), "ollama says: hi")

    def test_the_live_state_wins_over_a_stale_inventory(self):
        srv = self.serve(models=("gpt-oss:120b",))   # loaded now, though the inventory saw it unloaded
        cfg = dict(m.load_config("/nonexistent"), local_endpoints=[{"name": "spark", "url": srv.url}])
        inv = {"local_models": [dict(local("gpt-oss:120b", 120, endpoint="spark", loaded=False), url=srv.url)]}
        self.assertEqual(m.run_model("local:spark/gpt-oss:120b", "hi", None, cfg, inv, env={}), "ollama says: hi")

    def test_cli_run_refuses_then_allows_with_the_flag(self):
        srv = self.serve(models=("gpt-oss:120b",), loaded=())
        with open(os.path.join(self.tmp.name, "config.json"), "w") as f:
            json.dump({"local_endpoints": [{"name": "spark", "url": srv.url, "kind": "ollama"}]}, f)
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": self.tmp.name}):
            with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(m.main(["run", "local:spark/gpt-oss:120b", "hi"]), 1)
            self.assertIn("not loaded", err.getvalue())
            self.assertEqual(out.getvalue(), "")
            self.assertEqual([x for x in srv.seen if x[0] == "POST"], [])
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(m.main(["run", "local:spark/gpt-oss:120b", "hi", "--allow-load"]), 0)
            self.assertEqual(out.getvalue(), "ollama says: hi\n")

    def test_local_allow_load_defaults_to_off(self):
        self.assertIs(m.DEFAULT_CONFIG["local_allow_load"], False)
        with open(os.path.join(self.tmp.name, "config.json"), "w") as f:
            json.dump({"local_allow_load": True}, f)
        self.assertIs(m.load_config(self.tmp.name)["local_allow_load"], True)


if __name__ == "__main__":
    unittest.main()


class UnsafeNames(unittest.TestCase):
    """A model server on the LAN cannot put shell syntax into the delegate argv an agent will run."""

    def test_probe_drops_model_names_with_shell_syntax(self):
        names = ["qwen2.5:7b-instruct", "openai/gpt-oss-120b", "hf.co/u/m:Q4_K_M", "bad; rm -rf ~", "$(curl evil)", "a`b`",
                 "-flag", "two words", "x" * 300]
        def fetch(url, headers, body, timeout):
            return 200, json.dumps({"data": [{"id": n} for n in names]}).encode()
        res = m.probe_endpoint({"url": "http://10.0.0.9:8000", "kind": "openai"}, fetch)
        self.assertEqual([m["name"] for m in res["models"]], names[:3])
        self.assertEqual(res["ignored"], len(names) - 3)

    def test_a_configured_endpoint_name_with_shell_syntax_falls_back_to_the_host(self):
        eps = m.local_endpoints({"local_endpoints": [{"name": "box; rm -rf ~", "url": "http://10.0.0.9:8000"}]},
                                   env={"MAESTRO_NO_BUILTIN_PORTS": "1"})
        self.assertEqual(eps[0]["name"], "10.0.0.9")


class UnsafeNamesStrict(unittest.TestCase):
    def test_a_trailing_newline_is_not_a_safe_name(self):
        for bad in ("model\n", "model\r", "model\n; rm -rf ~"):
            self.assertIsNone(m.SAFE_NAME.fullmatch(bad), repr(bad))
        def fetch(url, headers, body, timeout):
            return 200, json.dumps({"data": [{"id": "ok-model"}, {"id": "evil\n"}]}).encode()
        res = m.probe_endpoint({"url": "http://10.0.0.9:8000", "kind": "openai"}, fetch)
        self.assertEqual(([x["name"] for x in res["models"]], res["ignored"]), (["ok-model"], 1))

    def test_an_inventory_from_before_the_name_check_is_detected_again(self):
        import tempfile, time
        with tempfile.TemporaryDirectory() as home:
            env = {"MAESTRO_NO_BUILTIN_PORTS": "1"}
            sig = [e["url"] for e in m.local_endpoints({}, env)]   # the real signature: only the version can trigger detect
            stale = {"version": 2, "generated_at": time.time(), "endpoints_sig": sig, "local_models": [
                {"id": "local:box/evil\n", "name": "evil\n", "endpoint_name": "box"}]}
            m._write_json(os.path.join(home, "inventory.json"), stale)
            with mock.patch.object(m, "detect", return_value={"version": m.INVENTORY_VERSION, "fresh": True}) as det:
                inv = m.ensure_inventory(home=home, env=env, cfg={})
            self.assertTrue(det.called)
            self.assertTrue(inv.get("fresh"))

    def test_the_delegate_argv_id_is_rebuilt_from_the_checked_names(self):
        rec = dict(local("good", 70), id="local:this-machine/good;rm -rf ~")   # safe name and endpoint, tampered id
        inv = inventory(local_models=[rec])
        cands = [c for c in m._candidates(cls("ask", 0), inv, "unknown", {}) if c["kind"] == "local"]
        self.assertTrue(cands)
        for c in cands:
            self.assertNotIn("rm -rf", json.dumps(c))
            self.assertIn("local:this-machine/good", c["delegate"]["argv"])

    def test_seat_selection_skips_unsafe_names_in_any_inventory(self):
        inv = inventory(local_models=[local("good", 70), dict(local("bad", 70), name="bad;x", id="local:this-machine/bad;x")])
        cands = [c for c in m._candidates(cls("ask", 0), inv, "unknown", {}) if c["kind"] == "local"]
        self.assertTrue(any("good" in json.dumps(c) for c in cands))
        self.assertFalse(any("bad;x" in json.dumps(c) for c in cands))
