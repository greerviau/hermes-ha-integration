from __future__ import annotations

import json
import unittest

from tests.test_support import FakeClientTimeout
from custom_components.hermes_conversation.api import (
    HermesApiClient,
    HermesAuthError,
    HermesConnectionError,
    HermesStreamSetupError,
)


class FakeResponse:
    def __init__(self, *, status=200, headers=None, json_data=None, text_data="", chunks=None):
        self.status = status
        self.headers = headers or {}
        self._json_data = json_data or {}
        self._text_data = text_data
        self.content = self
        self._chunks = [chunk.encode("utf-8") for chunk in (chunks or [])]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, headers=None, json=None, timeout=None, ssl=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
                "ssl": ssl,
            }
        )
        return self.responses.pop(0)

    def get(
        self,
        url,
        *,
        headers=None,
        timeout=None,
        ssl=None,
        allow_redirects=None,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "timeout": timeout,
                "ssl": ssl,
                "allow_redirects": allow_redirects,
            }
        )
        return self.responses.pop(0)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _health_response():
        return FakeResponse(
            json_data={
                "status": "ok",
                "platform": "hermes-agent",
                "version": "test",
            }
        )

    async def test_connection_probe_requires_authenticated_models_after_public_health(self):
        session = FakeSession(
            [self._health_response(), FakeResponse(status=401)]
        )
        client = HermesApiClient(
            session=session,
            host="agent.local",
            port=8443,
            api_key="wrong-key",
            profile="worker",
        )

        with self.assertRaises(HermesAuthError):
            await client.async_check_connection()

        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://agent.local:8443/profile/worker/v1/health",
                "https://agent.local:8443/profile/worker/v1/models",
            ],
        )
        self.assertFalse(session.calls[0]["allow_redirects"])
        self.assertFalse(session.calls[1]["allow_redirects"])
        self.assertEqual(session.calls[0]["headers"], {})
        self.assertEqual(
            session.calls[1]["headers"],
            {"Authorization": "Bearer wrong-key"},
        )

    async def test_native_probe_rejects_non_fail_closed_routing_before_credentials(self):
        for profile in ("worker", "default"):
            with self.subTest(profile=profile):
                session = FakeSession(
                    [
                        self._health_response(),
                        FakeResponse(json_data={"data": [{"id": "hermes-agent"}]}),
                    ]
                )
                client = HermesApiClient(
                    session=session,
                    host="agent.local",
                    port=8443,
                    api_key="must-not-leak",
                    profile=profile,
                    profile_route="native",
                )

                with self.assertRaises(HermesConnectionError):
                    await client.async_check_connection()

                self.assertEqual(
                    [call["url"] for call in session.calls],
                    ["https://agent.local:8443/p/hermes/v1/health"],
                )
                self.assertEqual(session.calls[0]["headers"], {})
                self.assertFalse(session.calls[0]["allow_redirects"])

    async def test_connection_probe_falls_back_to_models_when_legacy_health_is_missing(self):
        session = FakeSession(
            [
                FakeResponse(status=404),
                FakeResponse(
                    json_data={
                        "data": [
                            {"id": "hermes-agent", "owned_by": "hermes"}
                        ]
                    }
                ),
            ]
        )
        client = HermesApiClient(
            session=session,  # type: ignore[arg-type]
            host="agent.local",
            port=8443,
            api_key="legacy-key",
            profile="worker",
        )

        self.assertTrue(await client.async_check_connection())
        self.assertEqual(session.calls[0]["headers"], {})
        self.assertEqual(
            session.calls[1]["headers"],
            {"Authorization": "Bearer legacy-key"},
        )
        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://agent.local:8443/profile/worker/v1/health",
                "https://agent.local:8443/profile/worker/v1/models",
            ],
        )

    async def test_native_probe_rejects_ambiguous_legacy_health_before_credentials(self):
        session = FakeSession(
            [
                FakeResponse(status=404),
                FakeResponse(status=404),
                FakeResponse(
                    json_data={
                        "data": [
                            {"id": "hermes-agent", "owned_by": "hermes"}
                        ]
                    }
                ),
            ]
        )
        client = HermesApiClient(
            session=session,  # type: ignore[arg-type]
            host="agent.local",
            port=8443,
            api_key="must-not-leak",
            profile="worker",
            profile_route="native",
        )

        with self.assertRaises(HermesConnectionError):
            await client.async_check_connection()

        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://agent.local:8443/p/hermes/v1/health",
                "https://agent.local:8443/p/worker/v1/health",
            ],
        )
        self.assertTrue(all(call["headers"] == {} for call in session.calls))

    async def test_native_runtime_requests_reverify_routing_before_credentials(self):
        for operation in ("models", "non_streaming", "streaming"):
            with self.subTest(operation=operation):
                session = FakeSession([self._health_response()])
                client = HermesApiClient(
                    session=session,  # type: ignore[arg-type]
                    host="agent.local",
                    port=8443,
                    api_key="must-not-leak",
                    profile="worker",
                    profile_route="native",
                )

                with self.assertRaises(HermesConnectionError):
                    if operation == "models":
                        await client.async_get_models()
                    elif operation == "non_streaming":
                        await client.async_send_message(
                            [{"role": "user", "content": "hello"}]
                        )
                    else:
                        async for _ in client.async_stream_message(
                            [{"role": "user", "content": "hello"}]
                        ):
                            pass

                self.assertEqual(
                    [call["url"] for call in session.calls],
                    ["https://agent.local:8443/p/hermes/v1/health"],
                )
                self.assertEqual(session.calls[0]["headers"], {})
                self.assertFalse(session.calls[0]["allow_redirects"])

    async def test_legacy_fallback_rejects_generic_openai_models_response(self):
        session = FakeSession(
            [
                FakeResponse(status=404),
                FakeResponse(
                    json_data={
                        "data": [{"id": "generic-model", "owned_by": "other"}]
                    }
                ),
            ]
        )
        client = HermesApiClient(
            session,  # type: ignore[arg-type]
            "agent.local",
            8443,
            api_key="key",
        )

        with self.assertRaises(HermesConnectionError):
            await client.async_check_connection()

    async def test_connection_probe_rejects_non_hermes_health_body(self):
        session = FakeSession([FakeResponse(json_data={"status": "ok"})])
        client = HermesApiClient(session, "agent.local", 8443)

        with self.assertRaises(HermesConnectionError):
            await client.async_check_connection()

    async def test_connection_probe_rejects_health_redirect(self):
        session = FakeSession(
            [
                FakeResponse(
                    status=302,
                    json_data={"status": "ok", "platform": "hermes-agent"},
                )
            ]
        )
        client = HermesApiClient(session, "agent.local", 8443)

        with self.assertRaises(HermesConnectionError):
            await client.async_check_connection()

    def test_base_url_uses_root_for_blank_profile(self):
        client = HermesApiClient(
            session=FakeSession([]),
            host="agent.local",
            port=8443,
            profile="",
        )

        self.assertEqual(client.base_url, "https://agent.local:8443")

    def test_base_url_uses_root_for_blank_native_profile(self):
        client = HermesApiClient(
            session=FakeSession([]),
            host="agent.local",
            port=8443,
            profile="",
            profile_route="native",
        )

        self.assertEqual(client.base_url, "https://agent.local:8443")

    def test_base_url_normalizes_valid_profile(self):
        client = HermesApiClient(
            session=FakeSession([]),
            host="agent.local",
            port=8443,
            profile=" assistant_2 ",
        )

        self.assertEqual(
            client.base_url,
            "https://agent.local:8443/profile/assistant_2",
        )

    def test_base_url_uses_native_route_and_canonical_profile(self):
        client = HermesApiClient(
            session=FakeSession([]),
            host="agent.local",
            port=8443,
            profile=" Worker-Bot_2 ",
            profile_route="native",
        )

        self.assertEqual(
            client.base_url,
            "https://agent.local:8443/p/worker-bot_2",
        )

    def test_base_url_rejects_unknown_route_family(self):
        with self.assertRaises(ValueError):
            HermesApiClient(
                session=FakeSession([]),
                host="agent.local",
                port=8443,
                profile="worker",
                profile_route="/tenant",
            )

    def test_base_url_normalizes_dns_host_and_brackets_ipv6(self):
        dns_client = HermesApiClient(FakeSession([]), " AGENT.LOCAL. ", 8443)
        ipv6_client = HermesApiClient(FakeSession([]), "[2001:0db8::1]", 8443)

        self.assertEqual(dns_client.base_url, "https://agent.local:8443")
        self.assertEqual(ipv6_client.base_url, "https://[2001:db8::1]:8443")

    def test_base_url_rejects_non_host_input(self):
        for host in (
            "agent.local/path",
            "user@agent.local",
            "agent.local?query=1",
            "agent.local#fragment",
        ):
            with self.subTest(host=host), self.assertRaises(ValueError):
                HermesApiClient(FakeSession([]), host, 8443)

    def test_base_url_rejects_invalid_profile(self):
        with self.assertRaises(ValueError):
            HermesApiClient(
                session=FakeSession([]),
                host="agent.local",
                port=8443,
                profile="../assistant",
            )

    def test_base_url_rejects_reserved_native_profile(self):
        with self.assertRaises(ValueError):
            HermesApiClient(
                session=FakeSession([]),
                host="agent.local",
                port=8443,
                profile="hermes",
                profile_route="native",
            )

    async def test_profile_base_url_is_used_by_every_request_family(self):
        chunks = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}) + "\n",
            "data: [DONE]\n",
        ]
        session = FakeSession(
            [
                self._health_response(),
                FakeResponse(json_data={"data": [{"id": "hermes-agent"}]}),
                FakeResponse(json_data={"data": []}),
                FakeResponse(json_data={"choices": [{"message": {"content": "ok"}}]}),
                FakeResponse(chunks=chunks),
            ]
        )
        client = HermesApiClient(
            session=session,
            host="agent.local",
            port=8443,
            profile="worker",
        )

        self.assertTrue(await client.async_check_connection())
        await client.async_get_models()
        await client.async_send_message([{"role": "user", "content": "hello"}])
        self.assertEqual(
            [
                part
                async for part in client.async_stream_message(
                    [{"role": "user", "content": "hello"}]
                )
            ],
            ["ok"],
        )

        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://agent.local:8443/profile/worker/v1/health",
                "https://agent.local:8443/profile/worker/v1/models",
                "https://agent.local:8443/profile/worker/v1/models",
                "https://agent.local:8443/profile/worker/v1/chat/completions",
                "https://agent.local:8443/profile/worker/v1/chat/completions",
            ],
        )

    async def test_native_base_url_is_used_by_every_request_family(self):
        chunks = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}) + "\n",
            "data: [DONE]\n",
        ]
        session = FakeSession(
            [
                FakeResponse(status=404),
                self._health_response(),
                FakeResponse(json_data={"data": [{"id": "hermes-agent"}]}),
                FakeResponse(status=404),
                self._health_response(),
                FakeResponse(json_data={"data": []}),
                FakeResponse(status=404),
                self._health_response(),
                FakeResponse(json_data={"choices": [{"message": {"content": "ok"}}]}),
                FakeResponse(status=404),
                self._health_response(),
                FakeResponse(chunks=chunks),
            ]
        )
        client = HermesApiClient(
            session=session,
            host="agent.local",
            port=8443,
            api_key="profile-key",
            profile="Worker-Bot",
            profile_route="native",
        )

        self.assertTrue(await client.async_check_connection())
        await client.async_get_models()
        await client.async_send_message([{"role": "user", "content": "hello"}])
        self.assertEqual(
            [
                part
                async for part in client.async_stream_message(
                    [{"role": "user", "content": "hello"}]
                )
            ],
            ["ok"],
        )

        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://agent.local:8443/p/hermes/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/models",
                "https://agent.local:8443/p/hermes/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/models",
                "https://agent.local:8443/p/hermes/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/chat/completions",
                "https://agent.local:8443/p/hermes/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/health",
                "https://agent.local:8443/p/worker-bot/v1/chat/completions",
            ],
        )
        self.assertEqual(session.calls[0]["headers"], {})
        self.assertFalse(session.calls[0]["allow_redirects"])
        self.assertEqual(
            session.calls[2]["headers"],
            {"Authorization": "Bearer profile-key"},
        )
        self.assertFalse(session.calls[2]["allow_redirects"])
        for index in (0, 1, 3, 4, 6, 7, 9, 10):
            self.assertEqual(session.calls[index]["headers"], {})
            self.assertFalse(session.calls[index]["allow_redirects"])
        for index in (5, 8, 11):
            self.assertEqual(
                session.calls[index]["headers"],
                {"Authorization": "Bearer profile-key"},
            )

    async def test_health_non_success_status_raises_connection_error(self):
        client = HermesApiClient(
            session=FakeSession([FakeResponse(status=500)]),
            host="agent.local",
            port=8443,
        )

        with self.assertRaises(HermesConnectionError):
            await client.async_check_connection()

    async def test_health_unauthorized_statuses_are_connection_errors(self):
        for status in (401, 403):
            with self.subTest(status=status):
                client = HermesApiClient(
                    session=FakeSession([FakeResponse(status=status)]),
                    host="agent.local",
                    port=8443,
                )

                with self.assertRaises(HermesConnectionError):
                    await client.async_check_connection()

    async def test_non_streaming_preserves_session_header_and_model_timeout(self):
        session = FakeSession(
            [
                FakeResponse(
                    headers={"X-Hermes-Session-Id": "sess-2"},
                    json_data={"choices": [{"message": {"content": "hello"}}]},
                )
            ]
        )
        client = HermesApiClient(
            session=session,
            host="agent.local",
            port=8443,
            api_key="secret",
            model="custom-model",
            request_timeout=42,
        )

        result = await client.async_send_message(
            [{"role": "user", "content": "hi"}], session_id="sess-1"
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.session_id, "sess-2")
        self.assertEqual(client.last_session_id, "sess-2")
        self.assertEqual(session.calls[0]["headers"]["X-Hermes-Session-Id"], "sess-1")
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(session.calls[0]["json"]["model"], "custom-model")
        self.assertIsInstance(session.calls[0]["timeout"], FakeClientTimeout)
        self.assertEqual(session.calls[0]["timeout"].total, 42)

    async def test_streaming_preserves_returned_session_id(self):
        chunks = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Blue"}}]}) + "\n",
            "data: " + json.dumps({"choices": [{"delta": {"content": " sky"}}]}) + "\n",
            "data: [DONE]\n",
        ]
        session = FakeSession(
            [FakeResponse(headers={"X-Hermes-Session-Id": "sess-stream"}, chunks=chunks)]
        )
        client = HermesApiClient(
            session=session,
            host="agent.local",
            port=8443,
            request_timeout=12,
            stream_timeout=30,
        )

        parts = []
        async for part in client.async_stream_message(
            [{"role": "user", "content": "hi"}], session_id="old-session"
        ):
            parts.append(part)

        self.assertEqual("".join(parts), "Blue sky")
        self.assertEqual(client.last_session_id, "sess-stream")
        self.assertEqual(session.calls[0]["headers"]["X-Hermes-Session-Id"], "old-session")
        self.assertEqual(session.calls[0]["timeout"].total, 30)
        self.assertEqual(session.calls[0]["timeout"].sock_read, 12)

    async def test_streaming_ignores_custom_sse_events(self):
        chunks = [
            "event: hermes.tool.progress\n",
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": "do not speak"}}], "tool": "terminal"}
            )
            + "\n",
            "\n",
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "Safe answer"}}]})
            + "\n",
            "data: [DONE]\n",
        ]
        session = FakeSession([FakeResponse(chunks=chunks)])
        client = HermesApiClient(session=session, host="agent.local", port=8443)

        parts = []
        async for part in client.async_stream_message(
            [{"role": "user", "content": "hi"}]
        ):
            parts.append(part)

        self.assertEqual(parts, ["Safe answer"])

    async def test_streaming_ignores_tool_call_deltas(self):
        chunks = [
            "data: "
            + json.dumps({"choices": [{"delta": {"tool_calls": [{"id": "call-1"}]}}]})
            + "\n",
            "data: " + json.dumps({"choices": [{"delta": {"content": "Done"}}]}) + "\n",
            "data: [DONE]\n",
        ]
        session = FakeSession([FakeResponse(chunks=chunks)])
        client = HermesApiClient(session=session, host="agent.local", port=8443)

        parts = []
        async for part in client.async_stream_message(
            [{"role": "user", "content": "hi"}]
        ):
            parts.append(part)

        self.assertEqual(parts, ["Done"])

    async def test_streaming_rejected_status_raises_setup_error(self):
        session = FakeSession([FakeResponse(status=400, text_data="stream unsupported")])
        client = HermesApiClient(session=session, host="agent.local", port=8443)

        with self.assertRaises(HermesStreamSetupError):
            async for _part in client.async_stream_message(
                [{"role": "user", "content": "hi"}]
            ):
                pass


if __name__ == "__main__":
    unittest.main()
