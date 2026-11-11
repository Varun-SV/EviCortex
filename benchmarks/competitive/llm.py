"""Provider-neutral chat-completion clients for the reader, judge, and extractors.

The benchmark never depends on one vendor. A client is selected with a spec string:

``claude-cli:<model>``
    Headless ``claude -p`` subprocess. Needs no API key when the Claude Code CLI is
    already signed in.
``anthropic:<model>``
    Anthropic Messages API over HTTPS. Reads ``ANTHROPIC_API_KEY``.
``openai:<model>``
    OpenAI Chat Completions. Reads ``OPENAI_API_KEY``.
``openai-compatible:<base_url>#<model>``
    Any server speaking the Chat Completions protocol (Ollama, vLLM, LM Studio,
    llama.cpp, Groq, Together, OpenRouter, Gemini's compatibility endpoint, ...).
    Reads ``OPENAI_COMPATIBLE_API_KEY`` when set.

Only the standard library is used. Responses are cached on disk by a digest of
provider identity, system prompt, user prompt, and output limit, so interrupted
runs resume without repeating paid calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Protocol
import urllib.error
import urllib.request


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_ms: float
    cached: bool = False


class LLMClient(Protocol):
    @property
    def identity(self) -> str: ...

    def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse: ...


class LLMError(RuntimeError):
    """A provider call failed after the configured retries."""


def _retry(call, *, attempts: int, base_delay: float):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - provider failures are heterogeneous
            last = error
            if attempt + 1 < attempts:
                time.sleep(base_delay * (2**attempt))
    raise LLMError(str(last)) from last


class ClaudeCLIClient:
    """Run one stateless completion through the Claude Code CLI.

    Tools, MCP servers, slash commands, and user/project settings are disabled so
    the call is a plain completion with a caller-supplied system prompt.
    """

    def __init__(self, model: str, *, timeout_seconds: float = 300.0, attempts: int = 4) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        self._cwd = tempfile.mkdtemp(prefix="evicortex-llm-")

    @property
    def identity(self) -> str:
        return f"claude-cli:{self.model}"

    def _once(self, system: str, prompt: str) -> LLMResponse:
        started = time.perf_counter()
        completed = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                self.model,
                "--output-format",
                "json",
                "--tools",
                "",
                "--system-prompt",
                system,
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--setting-sources",
                "",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=self._cwd,
            check=False,
        )
        latency = (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            raise LLMError(
                f"claude CLI exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout)[:400]}"
            )
        payload = json.loads(completed.stdout)
        if payload.get("is_error") or payload.get("subtype") != "success":
            raise LLMError(f"claude CLI error: {str(payload.get('result'))[:400]}")
        usage = payload.get("usage") or {}
        input_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        return LLMResponse(
            text=str(payload.get("result") or ""),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens") or 0),
            cost_usd=payload.get("total_cost_usd"),
            latency_ms=latency,
        )

    def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
        del max_tokens  # the CLI exposes no output cap; prompts request brevity
        return _retry(lambda: self._once(system, prompt), attempts=self.attempts, base_delay=4.0)


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"HTTP {error.code} from {url}: {detail}") from error


class AnthropicClient:
    def __init__(self, model: str, *, timeout_seconds: float = 300.0, attempts: int = 4) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        self._base = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com").rstrip("/")

    @property
    def identity(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

        def call() -> LLMResponse:
            started = time.perf_counter()
            payload = _post_json(
                f"{self._base}/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                self.timeout_seconds,
            )
            text = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            usage = payload.get("usage") or {}
            return LLMResponse(
                text=text,
                model=self.model,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cost_usd=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return _retry(call, attempts=self.attempts, base_delay=2.0)


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        label: str = "openai",
        timeout_seconds: float = 300.0,
        attempts: int = 4,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.label = label
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts

    @property
    def identity(self) -> str:
        if self.label == "openai":
            return f"openai:{self.model}"
        return f"openai-compatible:{self.base_url}#{self.model}"

    def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
        key = os.environ.get(self.api_key_env, "")
        headers = {"authorization": f"Bearer {key}"} if key else {}

        def call() -> LLMResponse:
            started = time.perf_counter()
            payload = _post_json(
                f"{self.base_url}/chat/completions",
                headers,
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                self.timeout_seconds,
            )
            usage = payload.get("usage") or {}
            return LLMResponse(
                text=str(payload["choices"][0]["message"].get("content") or ""),
                model=self.model,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cost_usd=None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return _retry(call, attempts=self.attempts, base_delay=2.0)


def make_client(spec: str) -> LLMClient:
    """Build a client from a ``provider:model`` spec (see the module docstring)."""

    provider, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"LLM spec must look like 'provider:model', got {spec!r}")
    if provider == "claude-cli":
        return ClaudeCLIClient(rest)
    if provider == "anthropic":
        return AnthropicClient(rest)
    if provider == "openai":
        return OpenAICompatibleClient(rest)
    if provider == "openai-compatible":
        base_url, _, model = rest.partition("#")
        if not model:
            raise ValueError("openai-compatible spec must be '<base_url>#<model>'")
        return OpenAICompatibleClient(
            model,
            base_url=base_url,
            api_key_env="OPENAI_COMPATIBLE_API_KEY",
            label="openai-compatible",
        )
    raise ValueError(f"unknown LLM provider {provider!r}")


class CachedLLM:
    """Disk-backed response cache with thread-safe appends and usage totals."""

    def __init__(self, client: LLMClient, cache_path: str | Path) -> None:
        self.client = client
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._in_flight: dict[str, threading.Event] = {}
        self._used: set[str] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self._cache[row["key"]] = row["response"]

    @property
    def identity(self) -> str:
        return self.client.identity

    def _key(self, system: str, prompt: str, max_tokens: int) -> str:
        material = json.dumps(
            [self.client.identity, system, prompt, max_tokens], ensure_ascii=False
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def complete(self, *, system: str, prompt: str, max_tokens: int = 512) -> LLMResponse:
        key = self._key(system, prompt, max_tokens)
        while True:
            with self._lock:
                self._used.add(key)
                hit = self._cache.get(key)
                if hit is not None:
                    return LLMResponse(**{**hit, "cached": True})
                pending = self._in_flight.get(key)
                if pending is None:
                    # This caller owns the provider call; identical concurrent
                    # requests wait for it instead of paying for a second answer.
                    self._in_flight[key] = threading.Event()
                    break
            pending.wait()
        try:
            response = self.client.complete(system=system, prompt=prompt, max_tokens=max_tokens)
            row = asdict(response)
            with self._lock:
                self._cache[key] = row
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"key": key, "response": row}, ensure_ascii=False) + "\n"
                    )
            return response
        finally:
            # On failure nothing is cached, so a waiter retries the call itself.
            with self._lock:
                self._in_flight.pop(key).set()

    def usage(self, *, used_only: bool = False) -> dict[str, float | int | None]:
        """Totals over the whole cache, or over the requests this instance served.

        ``used_only`` is the logical cost of the current run: it counts each
        distinct request once whether or not it was answered from the cache, and
        ignores entries left by other shards or earlier configurations.
        """

        with self._lock:
            rows = [
                row for key, row in self._cache.items() if not used_only or key in self._used
            ]
        return {
            "calls": len(rows),
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            # A provider that reports no price makes the total unknown, not zero.
            "cost_usd": (
                None
                if any(row["cost_usd"] is None for row in rows)
                else round(sum(float(row["cost_usd"]) for row in rows), 4)
            ),
        }
