"""Exercise real SDK serialization against a local HTTP server, without API charges."""

import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from providers import PROVIDERS, Client


@pytest.fixture
def endpoint():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path.endswith("/responses"):
                response = {"id": "resp_test", "object": "response", "created_at": 1,
                            "model": "test", "status": "completed", "usage": None,
                            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                                        "status": "completed", "content": [{"type": "output_text", "text": "E", "annotations": []}]}]}
            elif self.path.endswith("/messages"):
                response = {"id": "msg_test", "type": "message", "role": "assistant", "model": "test",
                            "content": [{"type": "text", "text": "E"}], "stop_reason": "end_turn",
                            "usage": {"input_tokens": 10, "output_tokens": 1}}
            elif "generateContent" in self.path:
                response = {"candidates": [{"content": {"role": "model", "parts": [
                    {"text": "reasoning", "thought": True}, {"text": "E"}]}, "finishReason": "STOP"}],
                            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1}}
            else:
                response = {"id": "chatcmpl_test", "object": "chat.completion", "created": 1,
                            "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "E", "reasoning_content": "reasoning"}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_sdk_image_requests(provider, endpoint, monkeypatch):
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(variable, "test-key")
    url, requests = endpoint
    client = Client(provider, "test", base_url=url, max_tokens=64, temperature=0)
    try:
        result = client.generate("Choose A-E.", b"image-bytes", "image/png")
    finally:
        client.close()
    assert result["text"] == "E"
    assert result["finish_reason"]
    assert len(requests) == 1
    path, body = requests[0]
    encoded = base64.b64encode(b"image-bytes").decode()
    if provider == "gemini":
        assert path == "/v1beta/models/test:generateContent"
        assert len(body["contents"]) == 1
        assert body["contents"][0]["parts"][0]["inlineData"] == {"data": encoded, "mime_type": "image/png"}
        assert body["generationConfig"]["maxOutputTokens"] == 64
        assert body["generationConfig"]["temperature"] == 0
    elif provider == "openai":
        assert path.endswith("/responses")
        assert len(body["input"]) == 1
        assert body["input"][0]["content"][0]["image_url"] == f"data:image/png;base64,{encoded}"
        assert body["max_output_tokens"] == 64
    elif provider == "anthropic":
        assert path.endswith("/messages")
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"][0]["source"] == {"type": "base64", "media_type": "image/png", "data": encoded}
        assert body["max_tokens"] == 64
    else:
        assert path.endswith("/chat/completions")
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"][0]["image_url"]["url"] == f"data:image/png;base64,{encoded}"
        assert body["max_tokens"] == 64


def test_vllm_request_options_and_default_sampling(endpoint):
    url, requests = endpoint
    options = {"chat_template_kwargs": {"enable_thinking": False}}
    client = Client("vllm", "test", base_url=url, request_options=options)
    try:
        client.generate("Choose A-E.", b"image", "image/png")
    finally:
        client.close()
    body = requests[0][1]
    assert body["chat_template_kwargs"] == options["chat_template_kwargs"]
    assert "temperature" not in body
