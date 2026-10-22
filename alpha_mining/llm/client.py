"""Chat clients for the two models, with thinking and web search switched off
explicitly and every request/response logged.

Both models default to thinking mode, so the switch is always sent, and a
response that still carries reasoning content raises instead of being used.
The logged request parameters and token counts are the evidence that thinking
and search were off.
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from alpha_mining.config import ROOT

LOG_DIR = ROOT / "logs" / "llm"
# Output cap. Thinking runs (supplementary S1/S2 only) get a larger cap because
# reasoning tokens count against it: at 4096 the reasoning exhausted the cap and
# left the answer empty (2026-10-18; see docs/design_freeze.md, addendum).
MAX_TOKENS = 4096
MAX_TOKENS_THINKING = 32768


@dataclass(frozen=True)
class ModelConfig:
    name: str             # short id used in run names
    model: str            # API model name
    key_envs: tuple       # first one that is set wins
    base_envs: tuple
    default_base: str
    thinking_off: dict
    thinking_on: dict


MODELS = {
    "deepseek": ModelConfig(
        "deepseek", "deepseek-flash", ("DEEPSEEK_API_KEY",), ("DEEPSEEK_API_BASE",),
        "https://api.deepseek.com",
        thinking_off={"thinking": {"type": "disabled"}},
        thinking_on={"thinking": {"type": "enabled"}},
    ),
    # Alibaba Cloud Model Studio, OpenAI-compatible mode.
    "qwen": ModelConfig(
        "qwen", "qwen3.8-flash", ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
        ("DASHSCOPE_API_BASE", "OPENAI_BASE_URL"), "https://maas.qianwenaiapi.com/compatible-mode/v1",
        thinking_off={"enable_thinking": False, "enable_search": False},
        thinking_on={"enable_thinking": True, "enable_search": False},
    ),
}


def _first_env(names) -> str | None:
    load_dotenv(ROOT / ".env")
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


@dataclass
class Reply:
    text: str
    reasoning: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    seconds: float
    raw: dict = field(repr=False, default_factory=dict)


class ChatClient:
    def __init__(self, model: str, thinking: bool, temperature: float, log_name: str):
        self.cfg = MODELS[model]
        key = _first_env(self.cfg.key_envs)
        if not key:
            raise RuntimeError(f"no API key found in {self.cfg.key_envs}")
        base = _first_env(self.cfg.base_envs) or self.cfg.default_base
        self.client = OpenAI(api_key=key, base_url=base, timeout=180, max_retries=3)
        self.base_url = base
        self.thinking = thinking
        self.temperature = temperature
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{log_name}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.served_model: str | None = None

    def chat(self, messages: list[dict], max_tokens: int | None = None, **log_meta) -> Reply:
        if max_tokens is None:
            max_tokens = MAX_TOKENS_THINKING if self.thinking else MAX_TOKENS
        extra = self.cfg.thinking_on if self.thinking else self.cfg.thinking_off
        params = {"model": self.cfg.model, "temperature": self.temperature,
                  "max_tokens": max_tokens, "extra_body": extra}
        t0 = time.time()
        r = self.client.chat.completions.create(messages=messages, **params)
        dt = time.time() - t0
        msg = r.choices[0].message
        extra_fields = msg.model_extra or {}
        reasoning = getattr(msg, "reasoning_content", None) or extra_fields.get("reasoning_content") or ""
        details = getattr(r.usage, "completion_tokens_details", None)
        reply = Reply(msg.content or "", reasoning, r.model, r.usage.prompt_tokens,
                      r.usage.completion_tokens, getattr(details, "reasoning_tokens", None), dt,
                      raw=r.model_dump())
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_url": self.base_url,
                                "request": {**params, "messages": messages}, "response": reply.raw,
                                "seconds": round(dt, 2), **log_meta}, ensure_ascii=False) + "\n")
        # The served model must stay the same for a whole run (aliases can move).
        if self.served_model is None:
            self.served_model = r.model
        elif r.model != self.served_model:
            raise RuntimeError(f"served model changed mid-run: {self.served_model} -> {r.model}")
        if not self.thinking and (reasoning or reply.reasoning_tokens):
            raise RuntimeError("thinking was requested off but the response contains reasoning")
        return reply
