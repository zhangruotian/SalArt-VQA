"""Thin adapters to the official SDKs; one independent image question per call."""

import base64
import os

PROVIDERS = ("openai", "anthropic", "gemini", "kimi", "ollama", "vllm", "openai-compatible")


class Client:
    def __init__(self, provider, model, base_url=None, max_tokens=4096,
                 temperature=None, request_options=None):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.options = request_options or {}
        self.sampling = {} if temperature is None else {"temperature": temperature}
        connection = {"base_url": base_url} if base_url else {}
        if provider == "gemini":
            from google import genai
            from google.genai import errors, types
            self.types = types
            self.client = genai.Client(vertexai=False, api_key=os.environ["GEMINI_API_KEY"],
                                      http_options=types.HttpOptions(timeout=180_000, **connection))
            self.errors = (errors.APIError,)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(timeout=180, max_retries=2, **connection)
            self.errors = (anthropic.APIError,)
        else:
            import openai
            if provider == "kimi":
                connection = {"base_url": base_url or "https://api.moonshot.ai/v1",
                              "api_key": os.environ["MOONSHOT_API_KEY"]}
            elif provider in ("ollama", "vllm"):
                port = 11434 if provider == "ollama" else 8000
                connection = {"base_url": base_url or f"http://localhost:{port}/v1",
                              "api_key": os.environ.get(f"{provider.upper()}_API_KEY", "EMPTY")}
            elif provider == "openai-compatible":
                if not base_url:
                    raise ValueError("--base-url is required for openai-compatible")
                connection["api_key"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
            self.client = openai.OpenAI(timeout=180, max_retries=2, **connection)
            self.errors = (openai.APIError,)

    def generate(self, prompt, image, mime_type):
        encoded = base64.b64encode(image).decode("ascii")
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                contents=[self.types.Part.from_bytes(data=image, mime_type=mime_type), prompt],
                config=self.types.GenerateContentConfig(
                    **({"max_output_tokens": self.max_tokens, **self.sampling} | self.options)),
            )
            candidate = response.candidates[0] if response.candidates else None
            parts = candidate.content.parts if candidate and candidate.content else []
            return {"text": "".join(p.text for p in parts if p.text and not p.thought),
                    "finish_reason": str(candidate.finish_reason) if candidate else "blocked",
                    "usage": response.usage_metadata.model_dump(mode="json") if response.usage_metadata else None}
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}},
                    {"type": "text", "text": prompt},
                ]}], extra_body=self.sampling | self.options,
            )
            return {"text": "".join(block.text for block in response.content if block.type == "text"),
                    "finish_reason": response.stop_reason, "usage": response.usage.model_dump(mode="json")}
        image_url = f"data:{mime_type};base64,{encoded}"
        if self.provider == "openai":
            response = self.client.responses.create(
                model=self.model, max_output_tokens=self.max_tokens, **self.sampling,
                input=[{"role": "user", "content": [
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": prompt},
                ]}], extra_body=self.options,
            )
            return {"text": response.output_text, "finish_reason": response.status,
                    "usage": response.usage.model_dump(mode="json") if response.usage else None}
        response = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, **self.sampling,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]}], extra_body=self.options,
        )
        choice = response.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump(mode="json") if response.usage else None}

    def close(self):
        self.client.close()
