"""Model providers.

Two different roles use a language model in this project:

1. the *agent policy* that is being traced  (AgentLLM: plan / read / ...)
2. the *judge* used by LLM-based attribution baselines (a plain text Completer)

Offline default: ``ScriptedPolicy`` (a deterministic, rule-based stand-in for an
LLM).  Results obtained with it say nothing about real LLM behaviour; they test
the tracing + extraction machinery.  For real runs plug in Ollama or any
OpenAI-compatible endpoint with ``make_completer`` / ``ChatPolicy``.

Standard library only (urllib), so nothing needs installing.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .world import ATTR_PATTERNS

_COMPANY_RE = re.compile(r"\b[A-Z][a-z]+ (?:Systems|Labs|Dynamics|Works|Networks|Robotics|Analytics|Logistics)\b")
_PLAN_STEP = ("lookup", "compute")


# ----------------------------------------------------------------------------
# Text completers (used by judges and by ChatPolicy)
# ----------------------------------------------------------------------------
class Completer:
    """Callable str -> str with token / latency accounting."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0

    def _complete(self, prompt: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, prompt: str) -> str:
        t0 = time.time()
        out = self._complete(prompt)
        self.seconds += time.time() - t0
        self.calls += 1
        self.prompt_tokens += max(1, len(prompt) // 4)
        self.completion_tokens += max(1, len(out) // 4)
        return out

    def usage(self) -> Dict[str, Any]:
        return {"model": self.name, "calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 2)}


class FakeCompleter(Completer):
    """Test double: ``fn(prompt) -> str``."""

    def __init__(self, fn: Callable[[str], str], name: str = "fake"):
        super().__init__(name)
        self.fn = fn

    def _complete(self, prompt: str) -> str:
        return self.fn(prompt)


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 300) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach model server at {url}: {e}. "
                           "Is Ollama running (`ollama serve`) / is the API key set?") from e


class OllamaCompleter(Completer):
    def __init__(self, model: str, host: Optional[str] = None, temperature: float = 0.0):
        super().__init__(f"ollama:{model}")
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.temperature = temperature

    def _complete(self, prompt: str) -> str:
        r = _post_json(f"{self.host}/api/generate",
                       {"model": self.model, "prompt": prompt, "stream": False,
                        "options": {"temperature": self.temperature, "num_ctx": 16384}},
                       {"Content-Type": "application/json"})
        return r.get("response", "")


class OpenAICompleter(Completer):
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, Groq, vLLM, LM Studio...)."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.0):
        super().__init__(f"openai:{model}")
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature

    def _complete(self, prompt: str) -> str:
        if not self.api_key and "api.openai.com" in self.base_url:
            raise RuntimeError("OPENAI_API_KEY is not set")
        r = _post_json(f"{self.base_url}/chat/completions",
                       {"model": self.model, "temperature": self.temperature,
                        "messages": [{"role": "user", "content": prompt}]},
                       {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        return r["choices"][0]["message"]["content"]


def make_completer(spec: str) -> Optional[Completer]:
    """'ollama:qwen2.5:7b' | 'openai:gpt-4o-mini' | 'none'."""
    if not spec or spec == "none":
        return None
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaCompleter(model or "qwen2.5:7b")
    if kind == "openai":
        return OpenAICompleter(model or "gpt-4o-mini")
    raise ValueError(f"unknown model spec {spec!r}; use ollama:<model>, openai:<model> or none")


# ----------------------------------------------------------------------------
# Agent policies
# ----------------------------------------------------------------------------
class ScriptedPolicy:
    """Deterministic rule-based stand-in for the agent's LLM."""

    name = "scripted"

    def plan(self, question: str) -> List[List[str]]:
        ents = _COMPANY_RE.findall(question)
        q = question.lower()
        if "combined" in q or "total" in q:
            return [["lookup", e, "revenue"] for e in ents] + [["compute", "sum", ""]]
        if q.startswith("which company"):
            return [["lookup", e, "revenue"] for e in ents] + [["compute", "argmax", ""]]
        if "city where" in q:
            return [["lookup", ents[0], "hq_city"], ["lookup", "<hq_city>", "population"]]
        return [["lookup", ents[0], "revenue"]]

    def read(self, text: str, entity: str, attr: str) -> Optional[str]:
        m = ATTR_PATTERNS[attr].search(text)
        return m.group(1) if m else None


class ChatPolicy(ScriptedPolicy):
    """Same interface, but plan() and read() are delegated to a real model.

    Falls back to the scripted rule when the model's reply cannot be parsed;
    ``fallbacks`` counts how often (report it - it is a reliability metric).
    """

    def __init__(self, completer: Completer):
        self.completer = completer
        self.name = f"chat[{completer.name}]"
        self.fallbacks = 0

    def plan(self, question: str) -> List[List[str]]:
        prompt = (
            "You are a planner. Break the question into steps. Reply with ONLY a JSON list. Each step is\n"
            '["lookup", <entity name>, <attribute>] with attribute in revenue|hq_city|population, or\n'
            '["compute", "sum"|"argmax", ""]. If a step needs the city found in an earlier step use the\n'
            'entity "<hq_city>".\n'
            f"Question: {question}\nJSON:")
        raw = self.completer(prompt)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            steps = json.loads(m.group(0)) if m else None
            if (isinstance(steps, list) and steps
                    and all(isinstance(s, list) and len(s) == 3 and s[0] in _PLAN_STEP for s in steps)):
                return [[str(x) for x in s] for s in steps]
        except json.JSONDecodeError:
            pass
        self.fallbacks += 1
        return super().plan(question)

    def read(self, text: str, entity: str, attr: str) -> Optional[str]:
        prompt = (f"Document:\n{text}\n\nWhat is the {attr.replace('_', ' ')} of {entity} according to the document? "
                  "Reply with only the value (digits only for numbers, one word for a city), or NONE.")
        raw = self.completer(prompt).strip()
        if raw.upper().startswith("NONE"):
            return None
        m = re.search(r"[A-Za-z]+" if attr == "hq_city" else r"\d[\d,]*", raw)
        if not m:
            self.fallbacks += 1
            return super().read(text, entity, attr)
        return m.group(0).replace(",", "")
