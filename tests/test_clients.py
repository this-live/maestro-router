import json
import os
import tempfile
import unittest

import helpers  # noqa: F401
from helpers import maestro as core
from fake_http import FakeServer, fake_cli, fake_spec, json_reply

from maestro import clients as C
from maestro import providers as P

KEY = "sk-test-SECRET-abcdef"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def sse(events):
    """Anthropic-shaped SSE bytes: `event:` + `data:` per event."""
    return [("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e))).encode() for e in events]


# recorded-shape fixtures of Anthropic Messages streaming
TEXT_STREAM = [
    {"type": "message_start", "message": {"id": "msg_01A", "type": "message", "role": "assistant", "model": "claude-x", "content": [],
                                          "stop_reason": None, "usage": {"input_tokens": 25, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 15}},
    {"type": "message_stop"},
]
TOOL_STREAM = [
    {"type": "message_start", "message": {"id": "msg_02B", "usage": {"input_tokens": 40, "output_tokens": 2}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm, secret reasoning"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCg=="}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Checking."}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ""}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}},
    {"type": "content_block_stop", "index": 2},
    {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_02", "name": "get_time", "input": {}}},
    {"type": "content_block_delta", "index": 3, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
    {"type": "content_block_stop", "index": 3},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 61}},
    {"type": "message_stop"},
]


def chunks(raw):
    """Decoded chunk objects and the DONE flag from concatenated SSE bytes."""
    objs, done = [], False
    for block in b"".join(raw).decode().split("\n\n"):
        block = block.strip()
        if block == "data: [DONE]":
            done = True
        elif block.startswith("data: "):
            objs.append(json.loads(block[6:]))
    return objs, done


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.env = {"HOME": self.tmp, "MAESTRO_HOME": self.tmp, "PATH": self.tmp, "FAKE_KEY": KEY}

    def spec(self, url, style="openai", **kw):
        return fake_spec(url, style, **kw)

    def call(self, srv, req, stream=False, style="openai", model="up-model", **kw):
        return C.complete(self.spec(srv.url, style), model, req, stream, env=self.env, home=self.tmp, **kw)


class OpenAIClient(Base):
    OK = {"id": "c1", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                                                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    def test_request_passes_through_with_the_upstream_model_and_key(self):
        req = {"model": "maestro/auto", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}], "tool_choice": "auto",
            "response_format": {"type": "json_object"}, "max_completion_tokens": 77, "temperature": 0.2, "seed": 5,
            "unknown_future_field": {"a": 1}, "stream_options": {"include_usage": True}}
        with FakeServer(lambda r: json_reply(self.OK, headers={"X-RateLimit-Remaining-Requests": "9"})) as srv:
            status, headers, body = self.call(srv, req)
            sent = srv.requests[0]
        self.assertEqual((status, body["choices"][0]["message"]["content"]), (200, "hi"))
        self.assertEqual(headers["x-ratelimit-remaining-requests"], "9")
        self.assertEqual((sent["method"], sent["path"]), ("POST", "/chat/completions"))
        self.assertEqual(sent["headers"]["authorization"], "Bearer " + KEY)
        js = sent["json"]
        self.assertEqual(js["model"], "up-model")
        self.assertIs(js["stream"], False)
        for k in ("messages", "tools", "tool_choice", "response_format", "max_completion_tokens", "temperature", "seed", "unknown_future_field"):
            self.assertEqual(js[k], req[k], k)
        self.assertNotIn("stream_options", js)  # only meaningful when streaming
        self.assertEqual(req["model"], "maestro/auto")  # the caller's dict is untouched

    def test_streaming_relays_sse_lines_and_notes_usage(self):
        lines = [b'data: {"choices":[{"delta":{"role":"assistant","content":"He"}}]}\n\n',
                 b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n',
                 b'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n',
                 b"data: [DONE]\n\n"]
        with FakeServer(lambda r: (200, {"Content-Type": "text/event-stream"}, iter(lines))) as srv:
            status, headers, body = self.call(srv, {"messages": [], "stream_options": {"include_usage": True}}, stream=True)
            self.assertEqual(status, 200)
            self.assertIsInstance(body, C.Stream)
            self.assertEqual(b"".join(body), b"".join(lines))
            self.assertEqual(body.usage["total_tokens"], 6)
            js = srv.requests[0]["json"]
        self.assertIs(js["stream"], True)
        self.assertEqual(js["stream_options"], {"include_usage": True})

    def test_developer_messages_become_system_except_for_openai_and_azure(self):
        req = {"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}]}
        with FakeServer(lambda r: json_reply(self.OK)) as srv:
            self.call(srv, req)
            self.assertEqual([m["role"] for m in srv.requests[0]["json"]["messages"]], ["system", "user"])
            self.assertEqual(srv.requests[0]["json"]["messages"][0]["content"], "be brief")
            self.assertEqual(req["messages"][0]["role"], "developer")   # the caller's dict is untouched
            for pid in ("openai", "azure", "azure-cognitive-services"):
                C.complete(self.spec(srv.url, id=pid), "m", req, env=self.env, home=self.tmp)
                self.assertEqual(srv.requests[-1]["json"]["messages"][0]["role"], "developer", pid)
            spec = {"id": "local:x", "name": "x", "style": "local", "base_url": srv.url + "/v1"}
            C.complete(spec, "qwen", req, env=self.env, home=self.tmp)
            self.assertEqual(srv.requests[-1]["json"]["messages"][0]["role"], "system")

    def test_a_stream_that_ends_without_done_or_finish_reason_raises(self):
        cut = [b'data: {"choices":[{"delta":{"content":"He"}}]}\n\n']
        with FakeServer(lambda r: (200, {"Content-Type": "text/event-stream"}, iter(cut))) as srv:
            body = self.call(srv, {"messages": []}, stream=True)[2]
            self.assertEqual(next(body).strip(), cut[0].strip())
            with self.assertRaises(C.UpstreamError):
                list(body)
        fin = [b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n']   # a finish_reason is enough, [DONE] may be missing
        with FakeServer(lambda r: (200, {"Content-Type": "text/event-stream"}, iter(fin))) as srv:
            body = self.call(srv, {"messages": []}, stream=True)[2]
            self.assertEqual(b"".join(body), fin[0])
            self.assertEqual(body.finish_reason, "stop")

    def test_reasoning_effort_is_dropped_and_retried_once_on_a_400_that_names_it(self):
        def handler(r):
            if "reasoning_effort" in r["json"]:
                return json_reply({"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}}, 400)
            return json_reply(self.OK)
        with FakeServer(handler) as srv:
            status, _, body = self.call(srv, {"messages": [], "reasoning_effort": "high"})
            self.assertEqual(status, 200)
            self.assertEqual(len(srv.requests), 2)
            self.assertNotIn("reasoning_effort", srv.requests[1]["json"])
            self.assertIn("reasoning_effort", srv.requests[0]["json"])

    def test_a_400_about_something_else_is_not_retried(self):
        with FakeServer(lambda r: json_reply({"error": {"message": "messages is required"}}, 400)) as srv:
            status, _, body = self.call(srv, {"reasoning_effort": "high"})
            self.assertEqual((status, len(srv.requests)), (400, 1))
        self.assertEqual(body["error"]["message"], "messages is required")

    def test_errors_are_normalized_and_never_echo_the_key(self):
        with FakeServer(lambda r: (429, {"Retry-After": "7"}, "slow down " + KEY)) as srv:
            status, headers, body = self.call(srv, {"messages": []})
        self.assertEqual((status, headers["retry-after"]), (429, "7"))
        self.assertIn("slow down", body["error"]["message"])
        self.assertNotIn(KEY, json.dumps(body))
        with FakeServer(lambda r: (200, {}, "not json")) as srv:
            self.assertEqual(self.call(srv, {"messages": []})[0], 502)

    def test_transport_failures_are_status_zero(self):
        status, headers, body = C.complete(self.spec("http://127.0.0.1:9"), "m", {"messages": []}, env=self.env, home=self.tmp)
        self.assertEqual((status, headers), (0, {}))
        self.assertEqual(body["error"]["type"], "connection_error")
        import time
        with FakeServer(lambda r: (time.sleep(1.5), json_reply(self.OK))[1]) as srv:
            t0 = time.time()
            self.assertEqual(self.call(srv, {"messages": []}, timeout=0.3)[0], 0)
            self.assertLess(time.time() - t0, 1.2)

    def test_missing_key_unknown_provider_and_uncallable(self):
        env = dict(self.env)
        del env["FAKE_KEY"]
        self.assertEqual(C.complete(self.spec("http://x"), "m", {}, env=env, home=self.tmp)[0], 401)
        self.assertEqual(C.complete("nope", "m", {}, env=env, home=self.tmp)[0], 404)
        self.assertEqual(C.complete("amazon-bedrock", "m", {}, env=dict(env, AWS_BEARER_TOKEN_BEDROCK="x"), home=self.tmp)[0], 501)

    def test_a_model_override_switches_the_wire_format_per_model(self):
        anth = {"id": "m", "content": [{"type": "text", "text": "from anthropic"}], "stop_reason": "end_turn", "usage": {}}
        with FakeServer(lambda r: json_reply(anth if r["path"] == "/messages" else self.OK)) as srv:
            spec = self.spec(srv.url, overrides={"claude-x": {"npm": "@ai-sdk/anthropic"}, "resp-only": {"shape": "responses"},
                                                 "elsewhere": {"api": srv.url + "/alt"}})
            req = {"messages": [{"role": "user", "content": "x"}]}
            call = lambda m: C.complete(spec, m, req, env=self.env, home=self.tmp)
            self.assertEqual(call("claude-x")[2]["choices"][0]["message"]["content"], "from anthropic")
            self.assertEqual(srv.requests[-1]["headers"]["x-api-key"], KEY)
            self.assertEqual(call("gpt-x")[2]["choices"][0]["message"]["content"], "hi")
            self.assertEqual(srv.requests[-1]["path"], "/chat/completions")
            self.assertEqual(call("elsewhere")[0], 200)
            self.assertEqual(srv.requests[-1]["path"], "/alt/chat/completions")
            status, _, body = call("resp-only")
            self.assertEqual((status, body["error"]["type"]), (501, "not_implemented"))
            self.assertIn("Responses", body["error"]["message"])

    def test_a_partly_configured_provider_is_a_401_naming_what_is_missing(self):
        env = dict(self.env, CLOUDFLARE_API_KEY="k")
        status, _, body = C.complete("cloudflare-workers-ai", "m", {}, env=env, home=self.tmp)
        self.assertEqual(status, 401)
        self.assertIn("CLOUDFLARE_ACCOUNT_ID", body["error"]["message"])

    def test_azure_style_sends_the_api_key_header_to_the_resource_url(self):
        with FakeServer(lambda r: json_reply(self.OK)) as srv:
            cfg = dict(core.load_config(self.tmp), providers={"azure": {"base_url": srv.url + "/openai/v1"}})
            env = dict(self.env, AZURE_API_KEY=KEY, AZURE_RESOURCE_NAME="r")
            self.assertEqual(C.complete("azure", "gpt-5", {"messages": []}, env=env, home=self.tmp, cfg=cfg)[0], 200)
            h = srv.requests[0]["headers"]
            self.assertEqual((h["api-key"], "authorization" in h, srv.requests[0]["path"]), (KEY, False, "/openai/v1/chat/completions"))

    def test_local_servers_get_an_optional_bearer_key_from_a_named_env_var(self):
        with FakeServer(lambda r: json_reply(self.OK)) as srv:
            spec = {"id": "local:x", "name": "x", "style": "local", "base_url": srv.url + "/v1", "key_env": "MY_LOCAL_KEY"}
            C.complete(spec, "qwen", {"messages": []}, env=dict(self.env, MY_LOCAL_KEY="lk"), home=self.tmp)
            C.complete(spec, "qwen", {"messages": []}, env=self.env, home=self.tmp)
            self.assertEqual(srv.requests[0]["headers"]["authorization"], "Bearer lk")
            self.assertNotIn("authorization", srv.requests[1]["headers"])
            self.assertEqual(srv.requests[0]["path"], "/v1/chat/completions")


class AnthropicRequest(unittest.TestCase):
    def test_system_user_assistant_and_defaults(self):
        body = C.to_anthropic_request({"messages": [
            {"role": "system", "content": "Be brief."}, {"role": "developer", "content": [{"type": "text", "text": "No lists."}]},
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "bye"}],
            "temperature": 0.3, "stop": "END"}, "claude-sonnet-x")
        self.assertEqual(body["system"], "Be brief.\n\nNo lists.")
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual(body["messages"][0]["content"], [{"type": "text", "text": "hi"}])
        self.assertEqual(body["max_tokens"], 16000)
        self.assertEqual((body["temperature"], body["stop_sequences"], body["model"]), (0.3, ["END"], "claude-sonnet-x"))
        for k in ("thinking", "tools", "output_config"):
            self.assertNotIn(k, body)
        self.assertEqual(C.to_anthropic_request({"messages": [], "max_tokens": 50}, "m")["max_tokens"], 50)
        self.assertEqual(C.to_anthropic_request({"messages": [], "max_completion_tokens": 60}, "m")["max_tokens"], 60)

    def test_tools_tool_calls_and_merged_tool_results(self):
        req = {"messages": [
            {"role": "user", "content": "weather in Paris and the time?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "noon"}]},
            {"role": "user", "content": "thanks"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
                      {"type": "function", "function": {"name": "get_time"}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
        body = C.to_anthropic_request(req, "m")
        self.assertEqual(body["tools"][0], {"name": "get_weather", "description": "d",
                                            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}})
        self.assertEqual(body["tools"][1]["input_schema"], {"type": "object", "properties": {}})
        self.assertEqual(body["tool_choice"], {"type": "tool", "name": "get_weather"})
        asst = body["messages"][1]["content"]
        self.assertEqual(asst[0], {"type": "text", "text": "Checking."})
        self.assertEqual(asst[1], {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}})
        self.assertEqual(asst[2]["input"], {})
        results = body["messages"][2]
        self.assertEqual(results["role"], "user")
        self.assertEqual(results["content"], [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"},
                                              {"type": "tool_result", "tool_use_id": "call_2", "content": "noon"}])  # one message
        self.assertEqual(body["messages"][3]["content"][0]["text"], "thanks")
        for tc, want in (("auto", {"type": "auto"}), ("required", {"type": "any"}), ("none", {"type": "none"})):
            self.assertEqual(C.to_anthropic_request(dict(req, tool_choice=tc), "m")["tool_choice"], want)

    def test_images_base64_and_url(self):
        body = C.to_anthropic_request({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]}, "m")
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[1], {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}})
        self.assertEqual(blocks[2], {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}})

    def test_reasoning_effort_becomes_output_config_never_thinking(self):
        body = C.to_anthropic_request({"messages": [], "reasoning_effort": "high"}, "claude-opus-x")
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertNotIn("thinking", body)
        self.assertEqual(C.to_anthropic_request({"messages": [], "reasoning_effort": "minimal"}, "claude-opus-x")["output_config"], {"effort": "low"})
        self.assertNotIn("output_config", C.to_anthropic_request({"messages": [], "reasoning_effort": "high"}, "claude-haiku-x"))
        self.assertNotIn("output_config", C.to_anthropic_request({"messages": [], "reasoning_effort": "bogus"}, "claude-opus-x"))


class AnthropicResponse(unittest.TestCase):
    def test_text_and_usage(self):
        out = C.from_anthropic_response({"id": "msg_1", "content": [{"type": "thinking", "thinking": "secret"},
                                                                    {"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}],
                                         "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 4}}, "claude-x")
        ch = out["choices"][0]
        self.assertEqual((out["object"], out["model"], out["id"]), ("chat.completion", "claude-x", "chatcmpl-msg_1"))
        self.assertEqual((ch["message"], ch["finish_reason"]), ({"role": "assistant", "content": "Hi there"}, "stop"))
        self.assertEqual(out["usage"], {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})
        self.assertNotIn("secret", json.dumps(out))

    def test_tool_use_becomes_tool_calls(self):
        out = C.from_anthropic_response({"content": [{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}}],
                                         "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1}}, "m")
        ch = out["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertIsNone(ch["message"]["content"])
        self.assertEqual(ch["message"]["tool_calls"], [{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}])

    def test_stop_reasons(self):
        for anth, oai in (("end_turn", "stop"), ("stop_sequence", "stop"), ("tool_use", "tool_calls"), ("max_tokens", "length"),
                          ("refusal", "content_filter"), (None, "stop")):
            out = C.from_anthropic_response({"content": [{"type": "text", "text": "x"}], "stop_reason": anth}, "m")
            self.assertEqual(out["choices"][0]["finish_reason"], oai, anth)

    def test_a_refusal_keeps_its_text(self):
        out = C.from_anthropic_response({"content": [{"type": "text", "text": "I can't help with that."}], "stop_reason": "refusal"}, "m")
        self.assertEqual(out["choices"][0]["finish_reason"], "content_filter")
        self.assertEqual(out["choices"][0]["message"]["content"], "I can't help with that.")


class AnthropicStreaming(unittest.TestCase):
    def translate(self, events, include_usage=False):
        st = C.Stream(lambda s: iter(()))
        raw = list(C.anthropic_stream_to_openai(iter(events), "claude-x", include_usage, st))
        return raw, st

    def test_text_deltas_role_finish_and_done(self):
        raw, st = self.translate(TEXT_STREAM)
        objs, done = chunks(raw)
        self.assertTrue(done)
        deltas = [c["choices"][0]["delta"] for c in objs]
        self.assertEqual(deltas[0], {"role": "assistant", "content": "Hel"})
        self.assertEqual(deltas[1], {"content": "lo"})
        self.assertEqual(objs[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(c["object"] == "chat.completion.chunk" and c["model"] == "claude-x" for c in objs))
        self.assertEqual(len({c["id"] for c in objs}), 1)
        self.assertEqual(st.usage, {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40})
        self.assertEqual(st.finish_reason, "stop")
        self.assertFalse(any(not c["choices"] for c in objs))  # no usage chunk unless asked

    def test_include_usage_adds_the_final_usage_chunk(self):
        objs, done = chunks(self.translate(TEXT_STREAM, include_usage=True)[0])
        self.assertEqual(objs[-1]["choices"], [])
        self.assertEqual(objs[-1]["usage"]["total_tokens"], 40)
        self.assertTrue(done)

    def test_tool_use_deltas_and_dropped_thinking(self):
        raw, st = self.translate(TOOL_STREAM)
        objs, done = chunks(raw)
        text = b"".join(raw).decode()
        self.assertNotIn("secret reasoning", text)
        self.assertNotIn("EqQBCg", text)
        deltas = [c["choices"][0]["delta"] for c in objs]
        self.assertEqual(deltas[0], {"role": "assistant", "content": "Checking."})
        calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
        first = calls[0]
        self.assertEqual((first["index"], first["id"], first["type"], first["function"]), (0, "toolu_01", "function", {"name": "get_weather", "arguments": ""}))
        args0 = "".join(c["function"]["arguments"] for c in calls if c["index"] == 0)
        self.assertEqual(json.loads(args0), {"city": "Paris"})
        second = next(c for c in calls if c.get("id") == "toolu_02")
        self.assertEqual((second["index"], second["function"]["name"]), (1, "get_time"))
        self.assertEqual(objs[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("id", next(c for c in calls if c["index"] == 0 and c is not first))  # id and name only on the first chunk
        self.assertEqual(st.usage["completion_tokens"], 61)
        self.assertTrue(done)

    def test_stop_reasons_in_the_stream(self):
        for anth, oai in (("max_tokens", "length"), ("refusal", "content_filter")):
            events = [dict(e) for e in TEXT_STREAM]
            events[-2] = {"type": "message_delta", "delta": {"stop_reason": anth}, "usage": {"output_tokens": 3}}
            objs, _ = chunks(self.translate(events)[0])
            self.assertEqual(objs[-1]["choices"][0]["finish_reason"], oai)

    def test_an_error_event_or_a_cut_stream_raises(self):
        err = TEXT_STREAM[:3] + [{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}]
        with self.assertRaises(C.UpstreamError) as cm:
            self.translate(err)
        self.assertIn("Overloaded", str(cm.exception))
        with self.assertRaises(C.UpstreamError):
            self.translate(TEXT_STREAM[:5])

    def test_an_error_before_any_content_emits_nothing_first(self):
        gen = C.anthropic_stream_to_openai(iter([TEXT_STREAM[0], {"type": "error", "error": {"message": "Overloaded"}}]), "m")
        with self.assertRaises(C.UpstreamError):
            next(gen)  # so a proxy can still fall back: no byte has been produced


class AnthropicClient(Base):
    MSG = {"id": "msg_9", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "pong"}],
           "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}

    def test_non_streaming_round_trip(self):
        hdrs = {"anthropic-ratelimit-requests-remaining": "49"}
        with FakeServer(lambda r: json_reply(self.MSG, headers=hdrs)) as srv:
            status, headers, body = self.call(srv, {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 9}, style="anthropic", model="claude-x")
            sent = srv.requests[0]
        self.assertEqual((sent["method"], sent["path"]), ("POST", "/messages"))
        self.assertEqual(sent["headers"]["x-api-key"], KEY)
        self.assertEqual(sent["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", sent["headers"])
        self.assertEqual((sent["json"]["model"], sent["json"]["max_tokens"], sent["json"]["messages"][0]["content"][0]["text"]), ("claude-x", 9, "ping"))
        self.assertNotIn("stream", sent["json"])
        self.assertEqual((status, body["choices"][0]["message"]["content"], body["usage"]["total_tokens"]), (200, "pong", 7))
        self.assertEqual(headers["anthropic-ratelimit-requests-remaining"], "49")

    def test_streaming_round_trip_over_http(self):
        with FakeServer(lambda r: (200, {"Content-Type": "text/event-stream"}, iter(sse(TOOL_STREAM)))) as srv:
            status, _, body = self.call(srv, {"messages": [{"role": "user", "content": "x"}], "stream_options": {"include_usage": True}},
                                        stream=True, style="anthropic", model="claude-x")
            raw = list(body)
            self.assertTrue(srv.requests[0]["json"]["stream"])
        objs, done = chunks(raw)
        self.assertTrue(done)
        self.assertEqual(status, 200)
        self.assertEqual(objs[-1]["usage"]["total_tokens"], 101)
        self.assertEqual(body.finish_reason, "tool_calls")

    def test_a_stream_cut_mid_way_raises_after_bytes(self):
        cut = sse(TEXT_STREAM)[:4]
        with FakeServer(lambda r: (200, {}, iter(cut))) as srv:
            _, _, body = self.call(srv, {"messages": []}, stream=True, style="anthropic")
            first = next(body)
            self.assertTrue(first.startswith(b"data: "))
            with self.assertRaises(C.UpstreamError):
                list(body)

    def test_errors_and_the_output_config_retry(self):
        def handler(r):
            if "output_config" in r["json"]:
                return json_reply({"type": "error", "error": {"type": "invalid_request_error", "message": "output_config: effort not supported"}}, 400)
            return json_reply(self.MSG)
        with FakeServer(handler) as srv:
            status, _, _ = self.call(srv, {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}, style="anthropic", model="claude-opus-x")
            self.assertEqual((status, len(srv.requests)), (200, 2))
            self.assertNotIn("output_config", srv.requests[1]["json"])
        with FakeServer(lambda r: json_reply({"type": "error", "error": {"type": "rate_limit_error", "message": "slow " + KEY}}, 429, {"retry-after": "3"})) as srv:
            status, headers, body = self.call(srv, {"messages": []}, style="anthropic")
        self.assertEqual((status, headers["retry-after"], body["error"]["type"]), (429, "3", "rate_limit_error"))
        self.assertNotIn(KEY, json.dumps(body))


GEMINI_HELP = 'case "$1" in --help) echo "--approval-mode --extensions --allowed-mcp-server-names"; exit 0;; esac; '


class CliClient(Base):
    def setUp(self):
        super().setUp()
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        self.env["PATH"] = self.bin + os.pathsep + "/usr/bin:/bin"
        self.log = os.path.join(self.tmp, "calls.log")

    def cli(self, provider, req, stream=False, model="m", **kw):
        return C.complete(provider, model, req, stream, env=self.env, home=self.tmp, **kw)

    def test_claude_gets_the_prompt_on_stdin_and_json_output_is_parsed(self):
        fake_cli(self.bin, "claude", 'echo "$@" > %s.args; cat > %s.stdin; echo \'{"type":"result","is_error":false,"result":"Sure thing","usage":{"input_tokens":11,"output_tokens":3}}\'' % (self.log, self.log))
        status, headers, body = self.cli("claude", {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hello"}]}, model="sonnet")
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "Sure thing")
        self.assertEqual(body["usage"], {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
        args = read(self.log + ".args").split()
        self.assertEqual(args[:3], ["-p", "--model", "sonnet"])
        self.assertIn("--strict-mcp-config", args)   # no MCP servers; `--tools ""` (all built-in tools off) is checked below
        self.assertEqual(args[-2:], ["--output-format", "json"])
        prompt = read(self.log + ".stdin")
        self.assertTrue(prompt.startswith("[System instructions]\nBe terse."))
        self.assertIn("[User]\nhello", prompt)

    def test_codex_and_gemini_read_the_prompt_from_stdin_never_argv(self):
        rec = 'for a in "$@"; do echo "$a"; done > %s.{0}; cat > %s.{0}.stdin; echo "{0} answer"' % (self.log, self.log)
        fake_cli(self.bin, "codex", rec.replace("{0}", "codex"))
        fake_cli(self.bin, "gemini", GEMINI_HELP + rec.replace("{0}", "gemini"))
        req = {"messages": [{"role": "user", "content": "what is 2+2 SECRET-PROMPT"}]}
        self.assertEqual(self.cli("codex", req, model="gpt-6")[2]["choices"][0]["message"]["content"], "codex answer")
        args = read(self.log + ".codex").split("\n")[:-1]
        self.assertEqual(args[:3], ["exec", "-m", "gpt-6"])
        self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
        for feature in ("shell_tool", "unified_exec", "view_image"):
            self.assertIn("features.%s=false" % feature, args)
        self.assertNotIn("SECRET-PROMPT", " ".join(args))
        self.assertIn("SECRET-PROMPT", read(self.log + ".codex.stdin"))
        body = self.cli("gemini", req, model="gem-x")[2]
        self.assertEqual(body["choices"][0]["message"]["content"], "gemini answer")
        args = read(self.log + ".gemini").split("\n")[:-1]
        self.assertEqual(args[:2], ["-m", "gem-x"])
        self.assertNotIn("--yolo", args)
        self.assertNotIn("SECRET-PROMPT", " ".join(args))
        self.assertIn("SECRET-PROMPT", read(self.log + ".gemini.stdin"))
        self.assertGreater(body["usage"]["total_tokens"], 0)

    def test_cli_argv_locks_tools_down(self):
        caps = {"--safe-mode", "--no-session-persistence", "--ignore-user-config", "--ephemeral", "--approval-mode", "--extensions",
                "--allowed-mcp-server-names"}
        argv, stdin = C._cli_argv("claude", "claude", "opus", "-rm -rf /", caps)
        self.assertEqual(argv[argv.index("--tools") + 1], "")   # every built-in tool off
        for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
            self.assertIn(flag, argv)
        self.assertEqual(stdin, "-rm -rf /")
        argv, stdin = C._cli_argv("codex", "codex", "gpt-6", "-rm -rf /", caps)
        for flag in ("--ignore-user-config", "--ephemeral", "--skip-git-repo-check"):
            self.assertIn(flag, argv)
        self.assertEqual((argv[argv.index("--sandbox") + 1], stdin), ("read-only", "-rm -rf /"))
        argv, stdin = C._cli_argv("gemini", "gemini", "gem", "-rm -rf /", caps)
        self.assertEqual(argv[argv.index("--approval-mode") + 1], "default")
        self.assertEqual(argv[argv.index("-e") + 1], "none")
        self.assertIn("--allowed-mcp-server-names", argv)
        self.assertEqual(stdin, "-rm -rf /")
        for cli in ("claude", "codex", "gemini"):
            self.assertNotIn("-rm -rf /", C._cli_argv(cli, cli, "m", "-rm -rf /", set())[0])
            self.assertNotIn("--yolo", C._cli_argv(cli, cli, "m", "x", caps)[0])

    def test_pi_seat_splits_provider_model_prompt_on_stdin(self):
        rec = 'for a in "$@"; do echo "$a"; done > %s.pi; cat > %s.pi.stdin; echo "pi answer"' % (self.log, self.log)
        fake_cli(self.bin, "pi", rec)
        req = {"messages": [{"role": "user", "content": "what is 2+2 SECRET-PROMPT"}]}
        body = self.cli("pi", req, model="xai/grok-4.7")[2]
        self.assertEqual(body["choices"][0]["message"]["content"], "pi answer")
        args = read(self.log + ".pi").split("\n")[:-1]
        self.assertEqual(args[args.index("--provider") + 1], "xai")
        self.assertEqual(args[args.index("--model") + 1], "grok-4.7")
        for flag in ("-p", "--no-tools", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-session"):
            self.assertIn(flag, args)
        self.assertNotIn("SECRET-PROMPT", " ".join(args))
        self.assertIn("SECRET-PROMPT", read(self.log + ".pi.stdin"))

    def test_pi_argv_locks_tools_down(self):
        argv, stdin = C._cli_argv("pi", "pi", "minimax/MiniMax-M3", "-rm -rf /")
        self.assertEqual(argv[argv.index("--provider") + 1], "minimax")
        self.assertEqual(argv[argv.index("--model") + 1], "MiniMax-M3")
        for flag in ("--no-tools", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-session"):
            self.assertIn(flag, argv)
        self.assertNotIn("-rm -rf /", argv)
        self.assertEqual(stdin, "-rm -rf /")

    def test_gemini_is_refused_when_its_restricting_flags_cannot_be_confirmed(self):
        req = {"messages": [{"role": "user", "content": "x"}]}
        fake_cli(self.bin, "gemini", 'case "$1" in --help) echo "usage: gemini -p"; exit 0;; esac; echo ran > %s.ran; echo answer'
                 % self.log)   # an older Gemini CLI: --help lists none of the restricting flags
        status, _, body = self.cli("gemini", req)
        self.assertEqual(status, 503)
        self.assertIn("restricted", body["error"]["message"])
        self.assertFalse(os.path.exists(self.log + ".ran"))
        C._CLI_CAPS.clear()
        fake_cli(self.bin, "gemini", 'case "$1" in --help) sleep 5;; esac; echo answer')   # the probe times out
        self.assertEqual(self.cli("gemini", req, timeout=0.5)[0], 503)

    def test_cli_seats_get_a_minimal_environment(self):
        fake_cli(self.bin, "codex", 'env > %s.env; echo ok' % self.log)
        env = dict(self.env, OPENAI_API_KEY="sk-openai-keep", ANTHROPIC_API_KEY="sk-ant-drop", GROQ_API_KEY="gsk-drop",
                   MAESTRO_API_KEY="mst-drop", CODEX_HOME="/tmp/codex-home", LANG="en_US.UTF-8")
        C.complete("codex", "m", {"messages": [{"role": "user", "content": "x"}]}, env=env, home=self.tmp)
        got = dict(l.split("=", 1) for l in read(self.log + ".env").splitlines() if "=" in l)
        for kept in ("PATH", "OPENAI_API_KEY", "CODEX_HOME", "LANG"):
            self.assertIn(kept, got)
        for dropped in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "MAESTRO_API_KEY"):
            self.assertNotIn(dropped, got)

    def test_tools_and_images_are_rejected_without_running_anything(self):
        fake_cli(self.bin, "claude", "echo ran > %s.ran; echo '{}'" % self.log)
        for req, word in (({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
                          ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]}, "images"),
                          ({"messages": [{"role": "assistant", "tool_calls": [{"id": "1"}]}]}, "tool"),
                          ({"messages": [{"role": "tool", "content": "r"}]}, "tool")):
            status, _, body = self.cli("claude", req)
            self.assertEqual(status, 400)
            self.assertIn(word, body["error"]["message"])
            self.assertEqual(body["error"]["code"], "cli_no_tools")
            self.assertEqual(C.cli_supports(req)[0], False)
        self.assertFalse(os.path.exists(self.log + ".ran"))
        self.assertEqual(C.cli_supports({"messages": [{"role": "user", "content": "plain"}]}), (True, ""))

    def test_faked_streaming_is_one_chunk_plus_done(self):
        fake_cli(self.bin, "gemini", GEMINI_HELP + "echo streamed")
        status, _, body = self.cli("gemini", {"messages": [{"role": "user", "content": "x"}], "stream_options": {"include_usage": True}}, stream=True)
        objs, done = chunks(list(body))
        self.assertTrue(done)
        self.assertEqual(objs[0]["choices"][0]["delta"], {"role": "assistant", "content": "streamed\n".strip()})
        self.assertEqual(objs[1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", objs[2])

    def test_failures_map_to_statuses(self):
        req = {"messages": [{"role": "user", "content": "x"}]}
        status = C.complete("codex", "m", req, env=dict(self.env, PATH=self.bin), home=self.tmp)[0]
        self.assertEqual(status, 503)  # not on PATH (only the empty bin dir: a host may have a real codex in /usr/bin)
        fake_cli(self.bin, "codex", "echo boom >&2; exit 3")
        status, _, body = self.cli("codex", req)
        self.assertEqual(status, 502)
        self.assertIn("boom", body["error"]["message"])
        fake_cli(self.bin, "gemini", GEMINI_HELP + "sleep 5")
        import time
        t0 = time.time()
        self.assertEqual(self.cli("gemini", req, timeout=0.5)[0], 504)
        self.assertLess(time.time() - t0, 4)
        fake_cli(self.bin, "claude", 'echo \'{"type":"result","is_error":true,"result":"not logged in"}\'')
        self.assertEqual(self.cli("claude", req)[0], 502)

    def test_it_runs_outside_the_working_directory(self):
        fake_cli(self.bin, "gemini", GEMINI_HELP + "pwd > %s.cwd; echo ok" % self.log)
        self.cli("gemini", {"messages": [{"role": "user", "content": "x"}]})
        self.assertNotEqual(os.path.realpath(read(self.log + ".cwd").strip()), os.path.realpath(os.getcwd()))

    def test_cli_text_counts_cache_tokens_in_prompt_usage(self):
        out = json.dumps({"result": "ok", "usage": {"input_tokens": 2, "cache_creation_input_tokens": 2889,
                                                    "cache_read_input_tokens": 649, "output_tokens": 4}})
        text, usage = C._cli_text("claude", out)
        self.assertEqual(text, "ok")
        self.assertEqual(usage["prompt_tokens"], 2 + 2889 + 649)
        self.assertEqual(usage["completion_tokens"], 4)

    def test_cli_env_keeps_launcher_vars_but_drops_other_providers_keys(self):
        env = {"PATH": "/usr/bin", "HOME": "/home/x", "MISE_SHELL": "bash", "__MISE_DIFF": "x",
               "ASDF_DIR": "/opt/asdf", "NVM_DIR": "/home/x/.nvm", "OPENAI_API_KEY": "sk-secret",
               "GEMINI_API_KEY": "g-secret", "MAESTRO_API_KEY": "m-secret", "ANTHROPIC_API_KEY": "a-key"}
        kept = C.cli_env("claude", env)
        for k in ("PATH", "HOME", "MISE_SHELL", "__MISE_DIFF", "ASDF_DIR", "NVM_DIR", "ANTHROPIC_API_KEY"):
            self.assertIn(k, kept)
        for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "MAESTRO_API_KEY"):
            self.assertNotIn(k, kept)

    def test_flatten_messages(self):
        self.assertEqual(C.flatten_messages([{"role": "user", "content": "just this"}]), "just this")
        text = C.flatten_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
                                   {"role": "user", "content": [{"type": "text", "text": "u2"}]}])
        self.assertEqual(text, "[System instructions]\nS\n\n[User]\nu1\n\n[Assistant]\na1\n\n[User]\nu2")


if __name__ == "__main__":
    unittest.main()
