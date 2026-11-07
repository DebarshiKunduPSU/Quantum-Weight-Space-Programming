"""Model providers behind one interface: OpenAI, Anthropic, OpenAI-compatible endpoints, and a mock.

A model is named either by an alias from the config's ``models`` table or directly as
``provider:model_id`` (e.g. ``anthropic:claude-fable-5-1``).
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" | "assistant"
    parts: list  # str (text) or bytes (PNG image), in order

    @property
    def text(self) -> str:
        return "\n".join(p for p in self.parts if isinstance(p, str))


@dataclass
class LLMResponse:
    text: str
    model: str  # the model that actually answered, as reported by the API
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    extra: dict = field(default_factory=dict)


class FatalProviderError(RuntimeError):
    """Will fail the same way on every call (bad key, unknown model, rejected request): stop the run."""


def _b64(png: bytes) -> str:
    return base64.standard_b64encode(png).decode("ascii")


class OpenAIProvider:
    """OpenAI Responses API."""

    def __init__(self, model: str, opts: dict):
        from openai import OpenAI

        self.model, self.opts = model, opts
        self.client = OpenAI(
            api_key=os.environ.get(opts.get("api_key_env", "OPENAI_API_KEY")),
            base_url=opts.get("base_url"),
            timeout=opts.get("timeout_s", 1800),
        )

    def complete(self, system: str, messages: list[Message]) -> LLMResponse:
        import openai

        items = []
        for m in messages:
            if m.role == "assistant":
                items.append({"role": "assistant", "content": m.text})
                continue
            content = [
                {"type": "input_text", "text": p}
                if isinstance(p, str)
                else {"type": "input_image", "image_url": f"data:image/png;base64,{_b64(p)}"}
                for p in m.parts
            ]
            items.append({"role": "user", "content": content})
        kwargs = dict(model=self.model, instructions=system, input=items)
        if self.opts.get("max_output_tokens"):
            kwargs["max_output_tokens"] = self.opts["max_output_tokens"]
        if self.opts.get("effort"):
            kwargs["reasoning"] = {"effort": self.opts["effort"]}
        start = time.time()
        try:
            resp = self.client.responses.create(**kwargs)
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        ) as e:
            raise FatalProviderError(f"OpenAI rejected the request for model {self.model!r}: {e}") from e
        stop = resp.status or "completed"
        if resp.incomplete_details is not None and resp.incomplete_details.reason:
            stop = resp.incomplete_details.reason
        usage = resp.usage
        return LLMResponse(
            text=resp.output_text or "",
            model=resp.model,
            stop_reason=stop,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_s=time.time() - start,
        )


class OpenAIChatProvider:
    """Chat Completions API, for OpenAI-compatible servers (set base_url and api_key_env)."""

    def __init__(self, model: str, opts: dict):
        from openai import OpenAI

        self.model, self.opts = model, opts
        self.client = OpenAI(
            api_key=os.environ.get(opts.get("api_key_env", "OPENAI_API_KEY")),
            base_url=opts.get("base_url"),
            timeout=opts.get("timeout_s", 1800),
        )

    def complete(self, system: str, messages: list[Message]) -> LLMResponse:
        import openai

        chat = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "assistant":
                chat.append({"role": "assistant", "content": m.text})
                continue
            content = [
                {"type": "text", "text": p}
                if isinstance(p, str)
                else {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p)}"}}
                for p in m.parts
            ]
            chat.append({"role": "user", "content": content})
        kwargs = dict(model=self.model, messages=chat)
        if self.opts.get("max_output_tokens"):
            kwargs["max_tokens"] = self.opts["max_output_tokens"]
        if self.opts.get("effort"):
            kwargs["reasoning_effort"] = self.opts["effort"]
        start = time.time()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        ) as e:
            raise FatalProviderError(f"endpoint rejected the request for model {self.model!r}: {e}") from e
        choice, usage = resp.choices[0], resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            stop_reason=choice.finish_reason or "stop",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.time() - start,
        )


class AnthropicProvider:
    """Anthropic Messages API (streamed, so long thinking turns do not hit HTTP timeouts)."""

    def __init__(self, model: str, opts: dict):
        import anthropic

        self.model, self.opts = model, opts
        self.client = anthropic.Anthropic(timeout=opts.get("timeout_s", 1800))

    def complete(self, system: str, messages: list[Message]) -> LLMResponse:
        import anthropic

        msgs = []
        for m in messages:
            if m.role == "assistant":
                msgs.append({"role": "assistant", "content": m.text})
                continue
            content = [
                {"type": "text", "text": p}
                if isinstance(p, str)
                else {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(p)}}
                for p in m.parts
            ]
            msgs.append({"role": "user", "content": content})
        kwargs = dict(
            model=self.model,
            max_tokens=self.opts.get("max_output_tokens") or 32000,
            system=system,
            messages=msgs,
        )
        if self.opts.get("effort"):
            kwargs["output_config"] = {"effort": self.opts["effort"]}
        start = time.time()
        try:
            with self.client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as e:
            raise FatalProviderError(f"Anthropic rejected the request for model {self.model!r}: {e}") from e
        extra = {}
        if message.stop_reason == "refusal" and getattr(message, "stop_details", None):
            extra["refusal_category"] = message.stop_details.category
        return LLMResponse(
            text="".join(b.text for b in message.content if b.type == "text"),
            model=message.model,
            stop_reason=message.stop_reason or "end_turn",
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_s=time.time() - start,
            extra=extra,
        )


class MockProvider:
    """Offline stand-in for testing the pipeline: a fixed, data-blind circuit, no API call.

    ``mock:broken-first`` returns an invalid circuit first, to exercise the repair loop.
    """

    def __init__(self, model: str, opts: dict):
        self.model, self.class_labels = model, opts["class_labels"]

    def complete(self, system: str, messages: list[Message]) -> LLMResponse:
        m = re.search(r"with exactly (\d+) number", messages[0].text)
        if m:  # weights-only protocol: small nonzero angles
            k = int(m.group(1))
            weights = [round(0.1 * (i % 5 + 1), 2) for i in range(k)]
            return LLMResponse(f'```json\n{{"reasoning": "mock", "weights": {weights}}}\n```', f"mock:{self.model}", "end_turn")
        n = max(1, math.ceil(math.log2(len(self.class_labels))))
        pixels = [27, 36, 20, 43, 28, 35][:n]  # central pixels, one per readout qubit
        gates = [{"gate": "ry", "qubits": [q], "params": [f"pi * x[{pixels[q % len(pixels)]}]"]} for q in range(n)]
        if self.model == "broken-first" and len(messages) == 1:
            gates.append({"gate": "ry", "qubits": [n], "params": ["x[1] * x[2]"]})
        spec = {
            "reasoning": "mock circuit for pipeline testing",
            "n_qubits": n,
            "state_prep": {"type": "zero"},
            "gates": gates,
            "readout": {
                "type": "bitstrings",
                "qubits": list(range(n)),
                "classes": {str(c): [format(j, f"0{n}b")] for j, c in enumerate(self.class_labels)},
            },
        }
        return LLMResponse(f"```json\n{json.dumps(spec, indent=2)}\n```", f"mock:{self.model}", "end_turn")


PROVIDERS = {
    "openai": OpenAIProvider,
    "openai_chat": OpenAIChatProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def resolve_model(name: str, registry: dict) -> dict:
    """Alias or ``provider:model_id`` -> {"provider", "model", ...per-model options}."""
    if name in registry:
        return dict(registry[name])
    provider, sep, model = name.partition(":")
    if sep and provider in PROVIDERS and model:
        return {"provider": provider, "model": model}
    raise ValueError(
        f"unknown model {name!r}: use an alias from the config's 'models' table "
        f"({', '.join(registry) or 'none defined'}) or 'provider:model_id' with provider in {list(PROVIDERS)}"
    )


def make_provider(entry: dict, defaults: dict, class_labels: list[int]):
    """Per-model options override the config's ``llm`` defaults."""
    opts = {**defaults, **{k: v for k, v in entry.items() if k not in ("provider", "model")}}
    opts["class_labels"] = class_labels
    return PROVIDERS[entry["provider"]](entry["model"], opts)
