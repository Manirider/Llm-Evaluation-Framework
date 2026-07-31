"""
Concrete LLM Judge providers for OpenAI, Anthropic, and Mock evaluation environments.
Includes malformed JSON recovery, exponential backoff retries, and rate-limit handling.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_eval.config.settings import LLMProviderConfig
from llm_eval.exceptions.base import JudgeExecutionError
from llm_eval.judge.base import BaseLLMJudge, JudgeVerdict


def _recover_malformed_json(raw_text: str) -> dict[str, Any]:
    """
    Attempt to recover structured JSON from potentially malformed LLM output.
    Tries direct parse first, then regex extraction as fallback.
    """
    # Attempt 1: direct parse
    try:
        return dict(json.loads(raw_text))
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: find first JSON object via regex
    match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if match:
        try:
            return dict(json.loads(match.group()))
        except (json.JSONDecodeError, TypeError):
            pass

    # Attempt 3: extract fields with regex patterns
    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw_text)
    passed_match = re.search(r'"passed"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw_text)

    if score_match:
        return {
            "score": float(score_match.group(1)),
            "passed": passed_match.group(1).lower() == "true" if passed_match else True,
            "reasoning": reasoning_match.group(1)
            if reasoning_match
            else "Recovered from malformed JSON.",
        }

    raise JudgeExecutionError(
        "Failed to extract structured JSON from LLM judge response.",
        details={"raw_text_preview": raw_text[:200]},
    )


def _parse_verdict(payload: dict[str, Any]) -> JudgeVerdict:
    """Parse a JSON payload dict into a validated JudgeVerdict."""
    score = float(payload.get("score", 0.0))
    passed = bool(payload.get("passed", score >= 0.7))
    reasoning = str(payload.get("reasoning", "No detailed reasoning provided by judge."))

    return JudgeVerdict(
        score=max(0.0, min(1.0, score)),
        passed=passed,
        reasoning=reasoning,
        raw_response=payload,
    )


class MockJudge(BaseLLMJudge):
    """
    Deterministically evaluates prompts without external API dependencies.
    Useful for local testing, CI pipelines, and offline verification.
    """

    def evaluate_criterion(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeVerdict:
        prompt_lower = prompt.lower()
        if any(
            kw in prompt_lower for kw in ("hallucination", "unsupported", "incorrect", "fabricat")
        ):
            score = 0.2
            passed = False
            reasoning = "Mock judge detected potential hallucination or factual inconsistency."
        else:
            score = 0.95
            passed = True
            reasoning = "Mock judge confirmed high quality and criteria satisfaction."

        return JudgeVerdict(
            score=score,
            passed=passed,
            reasoning=reasoning,
            raw_response={"provider": "mock", "prompt_len": len(prompt)},
        )


class OpenAIJudge(BaseLLMJudge):
    """
    OpenAI-backed LLM Judge enforcing structured JSON response schemas with automatic retries.
    """

    def __init__(self, config: LLMProviderConfig | None = None) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI  # noqa: WPS433

            api_key = self.config.get_api_key() or "mock-key"
            self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)
        except ImportError as err:
            raise JudgeExecutionError(
                "openai package is not installed. Install via `poetry add openai`."
            ) from err

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def evaluate_criterion(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeVerdict:
        sys_instructions = system_prompt or (
            "You are an objective evaluation judge. Analyze the provided text against "
            "criteria and return JSON matching: "
            '{"score": float (0.0 to 1.0), "passed": bool, "reasoning": str}.'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": sys_instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content or "{}"
            payload = _recover_malformed_json(raw_text)
            return _parse_verdict(payload)
        except JudgeExecutionError:
            raise
        except Exception as e:
            logger.warning(f"OpenAI judge evaluation attempt failed: {e}")
            raise JudgeExecutionError(f"OpenAI judge failure: {e}") from e


class AnthropicJudge(BaseLLMJudge):
    """
    Anthropic Claude-backed LLM Judge enforcing structured evaluation output.
    """

    def __init__(self, config: LLMProviderConfig | None = None) -> None:
        super().__init__(config)
        try:
            from anthropic import Anthropic  # noqa: WPS433

            api_key = self.config.get_api_key() or "mock-key"
            self.client = Anthropic(api_key=api_key)
        except ImportError as err:
            raise JudgeExecutionError("anthropic package is not installed.") from err

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def evaluate_criterion(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeVerdict:
        sys_instructions = system_prompt or (
            "You are an expert LLM evaluation judge. Return valid JSON only: "
            '{"score": float (0-1), "passed": bool, "reasoning": str}.'
        )

        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                system=sys_instructions,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            raw_text = response.content[0].text
            payload = _recover_malformed_json(raw_text)
            return _parse_verdict(payload)
        except JudgeExecutionError:
            raise
        except Exception as e:
            logger.warning(f"Anthropic judge evaluation attempt failed: {e}")
            raise JudgeExecutionError(f"Anthropic judge failure: {e}") from e


def create_judge(config: LLMProviderConfig) -> BaseLLMJudge:
    """
    Factory function to instantiate the requested LLM judge backend.
    """
    providers: dict[str, type[BaseLLMJudge]] = {
        "openai": OpenAIJudge,
        "anthropic": AnthropicJudge,
    }
    judge_cls = providers.get(config.provider, MockJudge)
    return judge_cls(config)
