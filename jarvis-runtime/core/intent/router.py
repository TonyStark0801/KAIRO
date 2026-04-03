"""Intent router — sends transcripts to Ollama and parses tool matches."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
_CONFIDENCE_THRESHOLD = 0.6
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class IntentResult:
    tool_name: str | None
    params: dict[str, Any]
    confidence: float
    raw_response: str


class IntentRouter:
    def __init__(self, host: str, port: int, model: str) -> None:
        self._host = host
        self._port = port
        self._model = model
        self._jinja = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import ollama
            client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            await client.list()
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to connect to Ollama")
            return False

    async def route(self, transcript: str, tool_metas: list, recent_commands: list[str]) -> IntentResult:
        if not self._healthy:
            return IntentResult(tool_name=None, params={}, confidence=0.0, raw_response="Intent routing unavailable")
        tools_json = json.dumps(
            [{"name": t.name, "description": t.description, "parameters": t.parameters_schema} for t in tool_metas], indent=2)
        system_template = self._jinja.get_template("system.j2")
        user_template = self._jinja.get_template("user.j2")
        system_prompt = system_template.render(tools_json=tools_json, recent_commands=recent_commands)
        user_prompt = user_template.render(transcript=transcript)
        try:
            import ollama
            client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            response = await client.chat(model=self._model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            raw = response["message"]["content"]
            return self._parse_response(raw)
        except Exception:
            logger.exception("Ollama chat failed")
            return IntentResult(tool_name=None, params={}, confidence=0.0, raw_response="LLM call failed")

    def _parse_response(self, raw: str) -> IntentResult:
        try:
            data = json.loads(raw)
            tool_name = data.get("tool")
            params = data.get("params", {})
            confidence = float(data.get("confidence", 0.0))
            if confidence < _CONFIDENCE_THRESHOLD:
                return IntentResult(tool_name=None, params=params, confidence=confidence, raw_response=raw)
            return IntentResult(tool_name=tool_name, params=params, confidence=confidence, raw_response=raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse LLM response: %s", raw[:200])
            return IntentResult(tool_name=None, params={}, confidence=0.0, raw_response=raw)
