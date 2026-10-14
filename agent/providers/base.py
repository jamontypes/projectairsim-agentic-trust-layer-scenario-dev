"""
Model-provider interface for LLM agents: the ONLY provider-specific seam.

The pilot loop (agent/llm_pilot.py) talks to a ChatSession; each provider
(Gemini now; Ollama or an OpenClaw runtime later) implements it. A session keeps
its own native conversation history, because some providers attach opaque data
to their turns (e.g. Gemini thought signatures) that must be sent back verbatim.

    session = provider.start(system=..., tools=[ToolSpec, ...])
    turn = await session.send_user("...")                 # -> ModelTurn
    turn = await session.send_tool_results([ToolResult])  # -> ModelTurn

A ModelTurn has text and/or tool calls. Tool calls are executed by the pilot
loop, never by the provider (no automatic function calling): every call must go
through the agent's own feed and action port, and every call is recorded.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict           # JSON schema (type "object")


@dataclass(frozen=True)
class ToolCall:
    call_id: str               # provider's id, or one the provider adapter made up
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    result: dict


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)     # token counts, if reported
    finish_reason: str | None = None


class ChatSession(Protocol):
    # Optional: "async def wait_turn() -> float" reserves the next request slot under the
    # provider's rate limit and returns the wait. The pilot calls it BEFORE starting its
    # per-call timeout, so quota pacing is never mistaken for a slow or failed model.

    async def send_user(self, text: str) -> ModelTurn: ...

    async def send_tool_results(self, results: list[ToolResult]) -> ModelTurn: ...


class ModelProvider(Protocol):
    name: str                  # e.g. "gemini"
    model: str                 # e.g. the model id

    def start(self, system: str, tools: list[ToolSpec]) -> ChatSession: ...


class RateLimiter:
    """Client-side request pacing for one API quota (share ONE limiter between all
    sessions/pilots that use the same key and model).

    Even spacing, not a burst window: calls are at least 60/rpm seconds apart. A
    sliding window would let a fast model burn the whole minute's quota in a few
    seconds and then stall for up to a minute, which would age every observation
    the model is deciding from. Even spacing bounds each wait at 60/rpm."""

    def __init__(self, rpm: float, clock=None, sleep=None):
        if not rpm > 0:
            raise ValueError("rpm must be > 0")
        self.rpm = rpm
        self.interval_s = 60.0 / rpm
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._next = None
        self._lock = None

    async def acquire(self) -> float:
        """Wait for this call's slot; returns how long it waited (seconds)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:              # concurrent pilots queue in order
            now = self._clock()
            wait = 0.0 if self._next is None else max(0.0, self._next - now)
            if wait > 0:
                await self._sleep(wait)
            self._next = max(now, self._next or now) + self.interval_s
            return wait
