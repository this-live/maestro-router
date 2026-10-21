import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

import helpers
from helpers import maestro as core
from fake_http import FakeServer, json_reply

from maestro import providers as P
from maestro import quota as Q

NOW = 1_800_000_000.0


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.home = os.path.join(self.tmp, ".maestro")
        self.addCleanup(self._tmp.cleanup)
        Q.reset_state(self.home)
        self.addCleanup(Q.reset_state, self.home)


class Parsers(unittest.TestCase):
    def test_durations_and_timestamps(self):
        for text, secs in (("1s", 1), ("6m0s", 360), ("2m59.56s", 179.56), ("20ms", 0.02), ("1h2m3s", 3723), ("45", 45), ("0.5", 0.5)):
            self.assertAlmostEqual(Q.parse_duration(text), secs, msg=text)
        for bad in ("", "soon", "1x", None):
            self.assertIsNone(Q.parse_duration(bad))
        self.assertEqual(Q.parse_timestamp("2026-01-15T08:00:00Z"), datetime(2026, 1, 15, 8, tzinfo=timezone.utc).timestamp())
        self.assertEqual(Q.parse_timestamp("2026-01-15T08:00:00.5+00:00"), datetime(2026, 1, 15, 8, 0, 0, 500000, tzinfo=timezone.utc).timestamp())
        self.assertEqual(Q.parse_timestamp("2026-01-15T10:00:00+02:00"), datetime(2026, 1, 15, 8, tzinfo=timezone.utc).timestamp())
        self.assertIsNone(Q.parse_timestamp("2026-13-45T00:00:00Z"))
        self.assertEqual(Q.parse_timestamp(1_800_000_000), 1_800_000_000.0)
        self.assertEqual(Q.parse_timestamp(1_800_000_000_000), 1_800_000_000.0)  # milliseconds
        self.assertIsNone(Q.parse_timestamp("nope"))

    def test_retry_after_seconds_and_http_date(self):
        self.assertEqual(Q.parse_retry_after("12"), 12.0)
        self.assertEqual(Q.parse_retry_after("2.5"), 2.5)
        self.assertEqual(Q.parse_retry_after("Wed, 21 Oct 2015 07:28:30 GMT", now=1445412510 - 30), 30.0)
        self.assertIsNone(Q.parse_retry_after(None))
        self.assertIsNone(Q.parse_retry_after("later"))

    def test_openai_headers(self):
        w = Q.parse_openai_headers({
            "x-ratelimit-limit-requests": "5000", "x-ratelimit-remaining-requests": "4999", "x-ratelimit-reset-requests": "12ms",
            "x-ratelimit-limit-tokens": "800000", "x-ratelimit-remaining-tokens": "799000", "x-ratelimit-reset-tokens": "6m0s"}, now=NOW)
        self.assertEqual(set(w), {"requests", "tokens"})
        self.assertAlmostEqual(w["requests"]["fraction"], 4999 / 5000)
        self.assertAlmostEqual(w["requests"]["reset_at"], NOW + 0.012)
        self.assertAlmostEqual(w["tokens"]["reset_at"], NOW + 360)
        self.assertEqual(w["tokens"]["remaining"], 799000.0)

    def test_groq_and_cerebras_variants(self):
        groq = Q.parse_openai_headers({"x-ratelimit-limit-requests": "14400", "x-ratelimit-remaining-requests": "14370",
                                       "x-ratelimit-reset-requests": "2m59.56s", "x-ratelimit-limit-tokens": "18000",
                                       "x-ratelimit-remaining-tokens": "17997", "x-ratelimit-reset-tokens": "7.66s"}, now=NOW)
        self.assertAlmostEqual(groq["requests"]["reset_at"], NOW + 179.56)
        self.assertAlmostEqual(groq["tokens"]["reset_at"], NOW + 7.66)
        cer = Q.parse_openai_headers({"x-ratelimit-limit-requests-day": "14400", "x-ratelimit-remaining-requests-day": "14399",
                                      "x-ratelimit-reset-requests-day": "33011.38", "x-ratelimit-limit-tokens-minute": "60000",
                                      "x-ratelimit-remaining-tokens-minute": "59800", "x-ratelimit-reset-tokens-minute": "11.38"}, now=NOW)
        self.assertEqual(set(cer), {"requests-day", "tokens-minute"})
        self.assertAlmostEqual(cer["requests-day"]["reset_at"], NOW + 33011.38)
        self.assertAlmostEqual(cer["tokens-minute"]["fraction"], 59800 / 60000)

    def test_anthropic_headers(self):
        reset = iso(NOW + 90)
        h = {"anthropic-ratelimit-requests-limit": "50", "anthropic-ratelimit-requests-remaining": "48", "anthropic-ratelimit-requests-reset": reset,
             "anthropic-ratelimit-input-tokens-limit": "30000", "anthropic-ratelimit-input-tokens-remaining": "0",
             "anthropic-ratelimit-input-tokens-reset": reset, "anthropic-ratelimit-output-tokens-limit": "8000",
             "anthropic-ratelimit-output-tokens-remaining": "8000", "anthropic-ratelimit-output-tokens-reset": reset,
             "anthropic-ratelimit-tokens-limit": "38000", "anthropic-ratelimit-tokens-remaining": "8000", "anthropic-ratelimit-tokens-reset": reset}
        w = Q.parse_anthropic_headers(h)
        self.assertEqual(set(w), {"requests", "tokens", "input-tokens", "output-tokens"})
        self.assertAlmostEqual(w["requests"]["fraction"], 0.96)
        self.assertEqual(w["input-tokens"]["fraction"], 0.0)
        self.assertEqual(w["requests"]["reset_at"], NOW + 90)

    def test_parse_headers_picks_the_vendor_and_retry_after(self):
        w, ra = Q.parse_headers({"Retry-After": "7", "X-RateLimit-Limit-Requests": "10", "X-RateLimit-Remaining-Requests": "1",
                                 "X-RateLimit-Reset-Requests": "1s"}, now=NOW)
        self.assertEqual((ra, list(w)), (7.0, ["requests"]))
        self.assertEqual(Q.parse_headers({}), ({}, None))
        self.assertEqual(Q.parse_headers(None), ({}, None))


class Cooldowns(Tmp):
    def rec(self, status, headers=None, at=NOW, model="m", provider="groq"):
        return Q.record(provider, model, status, headers or {}, home=self.home, now=at)

    def test_429_with_retry_after_puts_the_model_on_cooldown_until_then(self):
        until = self.rec(429, {"retry-after": "30"})
        self.assertEqual(until, NOW + 30)
        ok, why = Q.is_available("groq", "m", self.home, now=NOW + 10)
        self.assertFalse(ok)
        self.assertIn("cooling down", why)
        self.assertTrue(Q.is_available("groq", "m", self.home, now=NOW + 31)[0])
        self.assertTrue(Q.is_available("groq", "other", self.home, now=NOW + 10)[0])  # per model
        self.assertEqual(Q.penalty("groq", "m", self.home, now=NOW + 10), 1.0)

    def test_default_cooldown_doubles_up_to_fifteen_minutes_and_success_resets(self):
        waits = [self.rec(503, at=NOW) - NOW for _ in range(7)]
        self.assertEqual(waits, [60, 120, 240, 480, 900, 900, 900])
        self.rec(200)
        self.assertEqual(self.rec(500) - NOW, 60)

    def test_429_uses_the_reset_of_an_emptied_window_when_there_is_no_retry_after(self):
        h = {"x-ratelimit-limit-requests": "10", "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "20s",
             "x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "900", "x-ratelimit-reset-tokens": "5s"}
        self.assertEqual(self.rec(429, h), NOW + 20)

    def test_connection_failures_count_and_2xx_updates_the_windows(self):
        self.assertEqual(self.rec(0) - NOW, 60)
        self.rec(200, {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "40", "x-ratelimit-reset-requests": "30s"})
        self.assertAlmostEqual(Q.headroom("groq", "m", self.home, now=NOW + 1), 0.4)
        self.assertTrue(Q.is_available("groq", "m", self.home, now=NOW + 1)[0])

    def test_auth_failures_put_the_whole_provider_on_cooldown_for_ten_minutes(self):
        self.assertEqual(self.rec(401), NOW + 600)
        for model in ("m", "other", None):
            ok, why = Q.is_available("groq", model, self.home, now=NOW + 599)
            self.assertFalse(ok, model)
            self.assertIn("auth failed", why)
        self.assertTrue(Q.is_available("groq", "m", self.home, now=NOW + 601)[0])
        self.assertTrue(Q.is_available("openai", "m", self.home, now=NOW + 10)[0])   # other providers are untouched
        self.assertEqual(self.rec(403, model="x") - NOW, 600)

    def test_no_answer_and_server_errors_cool_the_provider_but_a_429_stays_on_the_model(self):
        for status in (0, 502):
            Q.reset_state(self.home)
            self.rec(status)
            self.assertFalse(Q.is_available("groq", "other", self.home, now=NOW + 10)[0], status)
            self.assertTrue(Q.is_available("groq", "other", self.home, now=NOW + 61)[0], status)
        Q.reset_state(self.home)
        self.rec(429, {"retry-after": "30"})
        self.assertTrue(Q.is_available("groq", "other", self.home, now=NOW + 10)[0])
        self.assertFalse(Q.is_available("groq", "m", self.home, now=NOW + 10)[0])
        # ... unless the headers show an emptied per-organization window (Anthropic style)
        h = {"anthropic-ratelimit-requests-limit": "50", "anthropic-ratelimit-requests-remaining": "0",
             "anthropic-ratelimit-requests-reset": iso(NOW + 40)}
        self.rec(429, h, provider="anthropic")
        self.assertFalse(Q.is_available("anthropic", "other", self.home, now=NOW + 10)[0])
        # a success anywhere on the provider clears its cooldown
        Q.reset_state(self.home)
        self.rec(500)
        self.rec(200, at=NOW + 61)
        self.assertTrue(Q.is_available("groq", None, self.home, now=NOW + 62)[0])

    def test_state_is_persisted_and_reloaded(self):
        self.rec(429, {"retry-after": "100", "x-ratelimit-limit-requests": "10", "x-ratelimit-remaining-requests": "3", "x-ratelimit-reset-requests": "50s"})
        saved = json.loads(read(os.path.join(self.home, "quota.json")))
        self.assertEqual(saved["scopes"]["groq/m"]["cooldown_until"], NOW + 100)
        Q._STATE.clear()  # a new process
        self.assertFalse(Q.is_available("groq", "m", self.home, now=NOW + 10)[0])
        self.assertAlmostEqual(Q.headroom("groq", "m", self.home, now=NOW + 10), 0.3)


class Headroom(Tmp):
    def rec(self, remaining, limit=100, reset=60, model="m", provider="openai", at=NOW):
        Q.record(provider, model, 200, {"x-ratelimit-limit-requests": str(limit), "x-ratelimit-remaining-requests": str(remaining),
                                         "x-ratelimit-reset-requests": "%ds" % reset}, home=self.home, now=at)

    def test_unknown_is_neutral(self):
        self.assertIsNone(Q.headroom("openai", "m", self.home, now=NOW))
        self.assertEqual(Q.status("openai", "m", self.home, now=NOW)["state"], "unknown")
        self.assertEqual(Q.penalty("openai", "m", self.home, now=NOW), 0.0)
        self.assertTrue(Q.is_available("openai", "m", self.home, now=NOW)[0])

    def test_low_headroom_is_a_strong_penalty_but_still_eligible(self):
        self.rec(50)
        self.assertEqual((Q.status("openai", "m", self.home, now=NOW)["state"], Q.penalty("openai", "m", self.home, now=NOW)), ("ok", 0.0))
        self.rec(9)
        s = Q.status("openai", "m", self.home, now=NOW)
        self.assertEqual((s["state"], s["reset_at"]), ("low", NOW + 60))
        self.assertEqual(Q.penalty("openai", "m", self.home, now=NOW), 0.8)
        self.assertTrue(Q.is_available("openai", "m", self.home, now=NOW)[0])

    def test_an_empty_window_is_ineligible_until_it_resets(self):
        self.rec(0, reset=30)
        self.assertFalse(Q.is_available("openai", "m", self.home, now=NOW + 5)[0])
        self.assertEqual(Q.status("openai", "m", self.home, now=NOW + 5)["state"], "exhausted")
        self.assertTrue(Q.is_available("openai", "m", self.home, now=NOW + 31)[0])  # refilled
        self.assertIsNone(Q.headroom("openai", "m", self.home, now=NOW + 31))

    def test_the_worst_window_of_the_model_and_provider_wins(self):
        self.rec(80, model="m")
        Q.record("openai", None, 200, {"x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "50", "x-ratelimit-reset-tokens": "9s"},
                 home=self.home, now=NOW)
        self.assertAlmostEqual(Q.headroom("openai", "m", self.home, now=NOW), 0.05)
        self.assertAlmostEqual(Q.headroom("openai", "m", self.home, now=NOW + 10), 0.8)  # the token window expired


class Probes(Tmp):
    def test_openrouter_key_probe(self):
        body = {"data": {"label": "k", "usage": 4.0, "limit": 10.0, "limit_remaining": 0.5, "is_free_tier": False}}
        with FakeServer(lambda r: json_reply(body)) as srv:
            os.makedirs(self.home)
            write(os.path.join(self.home, "config.json"), json.dumps({"providers": {"openrouter": {"base_url": srv.url + "/api/v1"}}}))
            res = Q.probe_all({"OPENROUTER_API_KEY": "sk-or-1"}, self.home, now=NOW)
            self.assertEqual(res, {"openrouter": "ok"})
            self.assertEqual(srv.requests[0]["path"], "/api/v1/key")
            self.assertEqual(srv.requests[0]["headers"]["authorization"], "Bearer sk-or-1")
        self.assertAlmostEqual(Q.headroom("openrouter", "any/model", self.home, now=NOW), 0.05)
        self.assertEqual(Q.status("openrouter", None, self.home, now=NOW)["state"], "low")

    def test_openrouter_unlimited_key_stays_unknown_and_zero_left_is_exhausted(self):
        for data, want in (({"limit": None, "limit_remaining": None}, "unknown"), ({"limit": 5, "limit_remaining": 0}, "exhausted")):
            Q.reset_state(self.home)
            with FakeServer(lambda r: json_reply({"data": data})) as srv:
                spec = dict(P.BY_ID["openrouter"], base_url=srv.url)
                res = Q.probe_openrouter(spec, "k")
            sc = Q._scope(self.home, "openrouter")
            sc.update(windows=res["windows"], exhausted=res["exhausted"])
            self.assertEqual(Q.status("openrouter", None, self.home, now=NOW)["state"], want)

    def test_deepseek_balance_probe_marks_an_empty_account_exhausted(self):
        with FakeServer(lambda r: json_reply({"is_available": False, "balance_infos": [{"currency": "USD", "total_balance": "0.00"}]})) as srv:
            os.makedirs(self.home)
            write(os.path.join(self.home, "config.json"), json.dumps({"providers": {"deepseek": {"base_url": srv.url + "/v1"}}}))
            res = Q.probe_all({"DEEPSEEK_API_KEY": "sk-ds"}, self.home, now=NOW)
            self.assertEqual(res, {"deepseek": "ok"})
            self.assertEqual(srv.requests[0]["path"], "/user/balance")  # the origin, not under /v1
        ok, why = Q.is_available("deepseek", "deepseek-chat", self.home, now=NOW)
        self.assertFalse(ok)
        self.assertIn("out of balance", why)
        Q.reset_state(self.home)
        with FakeServer(lambda r: json_reply({"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "110.00"}]})) as srv:
            write(os.path.join(self.home, "config.json"), json.dumps({"providers": {"deepseek": {"base_url": srv.url}}}))
            Q.probe_all({"DEEPSEEK_API_KEY": "sk-ds"}, self.home, now=NOW)
        self.assertTrue(Q.is_available("deepseek", "deepseek-chat", self.home, now=NOW)[0])

    def test_probe_errors_are_reported_without_the_key_and_missing_keys_are_skipped(self):
        with FakeServer(lambda r: json_reply({"error": "no"}, 401)) as srv:
            os.makedirs(self.home)
            write(os.path.join(self.home, "config.json"), json.dumps({"providers": {"openrouter": {"base_url": srv.url}}}))
            res = Q.probe_all({"OPENROUTER_API_KEY": "sk-secret-key"}, self.home)
        self.assertIn("HTTP 401", res["openrouter"])
        self.assertNotIn("sk-secret-key", json.dumps(res))
        self.assertEqual(Q.probe_all({}, self.home), {})


def codex_event(ts, primary, secondary, plan="plus", wrap=True):
    rl = {"primary": primary, "secondary": secondary, "plan_type": plan}
    payload = {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1}}, "rate_limits": rl}
    return json.dumps({"timestamp": iso(ts), "type": "event_msg", "payload": payload} if wrap else dict(payload, timestamp=iso(ts)))


class Subscriptions(Tmp):
    def sessions(self, name, lines, day="2026/09/30"):
        path = os.path.join(self.tmp, ".codex", "sessions", day, name)
        write(path, "\n".join(lines) + "\n")
        return path

    def test_codex_resets_at_shape(self):
        self.sessions("rollout-2026-09-30T10-00-00-abc.jsonl", [
            json.dumps({"type": "session_meta", "payload": {}}),
            codex_event(NOW - 600, {"used_percent": 12.0, "window_minutes": 300, "resets_at": NOW + 3600},
                        {"used_percent": 40.0, "window_minutes": 10080, "resets_at": NOW + 86400}),
            codex_event(NOW - 60, {"used_percent": 31.5, "window_minutes": 300, "resets_at": NOW + 3000},
                        {"used_percent": 41.0, "window_minutes": 10080, "resets_at": NOW + 86000}, plan="pro"),
            json.dumps({"type": "event_msg", "payload": {"type": "agent_message"}})])
        lim = Q.codex_limits(self.tmp, now=NOW)
        self.assertEqual(lim["plan_type"], "pro")
        self.assertEqual((lim["primary"]["used_percent"], lim["secondary"]["used_percent"]), (31.5, 41.0))
        self.assertEqual(lim["primary"]["reset_at"], NOW + 3000)
        self.assertAlmostEqual(lim["primary"]["fraction"], 0.685)
        self.assertEqual(lim["primary"]["window_minutes"], 300)

    def test_codex_resets_in_seconds_shape_counts_from_the_event_time(self):
        self.sessions("rollout-b.jsonl", [codex_event(NOW - 1000, {"used_percent": 95.0, "window_minutes": 300, "resets_in_seconds": 4000},
                                                     {"used_percent": 20.0, "window_minutes": 10080, "resets_in_seconds": 500000}, wrap=False)])
        lim = Q.codex_limits(self.tmp, now=NOW)
        self.assertEqual(lim["primary"]["reset_at"], NOW - 1000 + 4000)
        self.assertEqual(lim["primary"]["used_percent"], 95.0)
        self.assertAlmostEqual(lim["primary"]["fraction"], 0.05)

    def test_an_expired_window_counts_as_zero_percent(self):
        self.sessions("rollout-c.jsonl", [codex_event(NOW - 20000, {"used_percent": 99.0, "window_minutes": 300, "resets_in_seconds": 3600},
                                                     {"used_percent": 60.0, "window_minutes": 10080, "resets_at": NOW + 1000})])
        lim = Q.codex_limits(self.tmp, now=NOW)
        self.assertEqual(lim["primary"]["used_percent"], 0.0)  # reset at NOW-16400
        self.assertEqual(lim["primary"]["fraction"], 1.0)
        self.assertEqual(lim["secondary"]["used_percent"], 60.0)
        self.sessions("rollout-d.jsonl", [codex_event(NOW - 20000, {"used_percent": 88.0, "window_minutes": 300}, {"used_percent": 1.0})], day="2026/10/01")
        lim = Q.codex_limits(self.tmp, now=NOW)  # no reset given: the window length after the event
        self.assertEqual(lim["primary"]["used_percent"], 0.0)

    def test_the_newest_file_wins_and_files_without_limits_are_skipped(self):
        self.sessions("rollout-old.jsonl", [codex_event(NOW - 5000, {"used_percent": 10.0, "resets_at": NOW + 100}, {"used_percent": 10.0, "resets_at": NOW + 100})], day="2026/09/29")
        self.sessions("rollout-new.jsonl", [json.dumps({"type": "session_meta"})], day="2026/09/30")
        self.assertEqual(Q.codex_limits(self.tmp, now=NOW)["primary"]["used_percent"], 10.0)  # falls back to the older file
        self.sessions("rollout-newer.jsonl", [codex_event(NOW - 5, {"used_percent": 77.0, "resets_at": NOW + 100}, {"used_percent": 5.0, "resets_at": NOW + 100})], day="2026/09/30")
        self.assertEqual(Q.codex_limits(self.tmp, now=NOW)["primary"]["used_percent"], 77.0)
        self.assertIsNone(Q.codex_limits(os.path.join(self.tmp, "nowhere"), now=NOW))

    def test_claude_statusline_cache(self):
        write(os.path.join(self.home, "claude_limits.json"), json.dumps({"updated_at": NOW - 10, "rate_limits": {
            "five_hour": {"used_percentage": 92.5, "resets_at": NOW + 1800}, "seven_day": {"used_percentage": 30, "resets_at": NOW - 5}}}))
        lim = Q.claude_limits(self.home, now=NOW)
        self.assertEqual(lim["five_hour"]["used_percent"], 92.5)
        self.assertEqual(lim["seven_day"]["used_percent"], 0.0)  # its reset has passed
        self.assertIsNone(Q.claude_limits(os.path.join(self.tmp, "none"), now=NOW))
        write(os.path.join(self.home, "claude_limits.json"), "not json")
        self.assertIsNone(Q.claude_limits(self.home, now=NOW))

    def test_refresh_feeds_headroom_and_availability(self):
        write(os.path.join(self.home, "claude_limits.json"), json.dumps({"rate_limits": {
            "five_hour": {"used_percentage": 93, "resets_at": NOW + 1800}, "seven_day": {"used_percentage": 10, "resets_at": NOW + 9000}}}))
        self.sessions("rollout-x.jsonl", [codex_event(NOW - 5, {"used_percent": 100.0, "resets_at": NOW + 600}, {"used_percent": 20.0, "resets_at": NOW + 9000})])
        found = Q.refresh_subscriptions(self.home, self.tmp, now=NOW)
        self.assertEqual(found["codex"]["primary"]["used_percent"], 100.0)
        self.assertAlmostEqual(Q.headroom("claude", "opus", self.home, now=NOW), 0.07)
        self.assertEqual((Q.status("claude", "opus", self.home, now=NOW)["state"], Q.penalty("claude", "opus", self.home, now=NOW)), ("low", 0.8))
        self.assertTrue(Q.is_available("claude", "opus", self.home, now=NOW)[0])
        self.assertFalse(Q.is_available("codex", "gpt-6", self.home, now=NOW)[0])  # 100% used
        self.assertTrue(Q.is_available("codex", "gpt-6", self.home, now=NOW + 601)[0])  # the window reset
        self.assertIsNone(Q.headroom("gemini", "x", self.home, now=NOW))  # no source: unknown
        self.assertEqual(Q.status("gemini", "x", self.home, now=NOW)["state"], "unknown")

    def test_refresh_without_any_source_is_quiet(self):
        self.assertEqual(Q.refresh_subscriptions(self.home, self.tmp, now=NOW), {"codex": None, "claude": None})
        self.assertIsNone(Q.headroom("codex", None, self.home, now=NOW))


class StatuslineInstall(Tmp):
    def test_install_writes_the_script_and_prints_the_snippet_without_touching_settings(self):
        settings = os.path.join(self.tmp, ".claude", "settings.json")
        write(settings, '{"keep": true}')
        with mock.patch.dict(os.environ, {"MAESTRO_HOME": self.home, "HOME": self.tmp}), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(core.main(["install", "--statusline"]), 0)
        script = os.path.join(self.home, "statusline.sh")
        self.assertTrue(os.access(script, os.X_OK))
        self.assertIn(script, out.getvalue())
        snippet = out.getvalue()[out.getvalue().index("{"):]
        self.assertEqual(json.loads(snippet), {"statusLine": {"type": "command", "command": script}})
        self.assertEqual(read(settings), '{"keep": true}')
        self.assertFalse(os.path.exists(os.path.join(self.tmp, ".claude", "agents")))  # --statusline installs nothing else

    def test_the_script_caches_rate_limits_and_prints_a_short_line(self):
        path, _ = Q.install_statusline(self.home)
        doc = {"model": {"id": "x"}, "cwd": "/secret/project", "rate_limits": {"five_hour": {"used_percentage": 42.4, "resets_at": NOW + 100},
                                                                                "seven_day": {"used_percentage": 7, "resets_at": NOW + 200}}}
        res = subprocess.run([path], input=json.dumps(doc), capture_output=True, text=True, timeout=30,
                             env=dict(os.environ, MAESTRO_HOME=self.home))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), "5h 42% 7d 7%")
        cached = json.loads(read(os.path.join(self.home, "claude_limits.json")))
        self.assertEqual(cached["rate_limits"]["five_hour"]["used_percentage"], 42.4)
        self.assertNotIn("secret", json.dumps(cached))  # only the rate limits are kept
        lim = Q.claude_limits(self.home, now=NOW)
        self.assertEqual(lim["five_hour"]["used_percent"], 42.4)
        res = subprocess.run([path], input="not json", capture_output=True, text=True, timeout=30, env=dict(os.environ, MAESTRO_HOME=self.home))
        self.assertEqual((res.returncode, res.stdout), (0, ""))
        res = subprocess.run([path], input='{"model": 1}', capture_output=True, text=True, timeout=30, env=dict(os.environ, MAESTRO_HOME=self.home))
        self.assertEqual((res.returncode, res.stdout), (0, ""))


if __name__ == "__main__":
    unittest.main()
