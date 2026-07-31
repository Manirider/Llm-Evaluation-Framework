"""Extended tests for LLM Judge provider implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_eval.config.settings import LLMProviderConfig
from llm_eval.exceptions.base import JudgeExecutionError
from llm_eval.judge.providers import (
    AnthropicJudge,
    OpenAIJudge,
    _parse_verdict,
    _recover_malformed_json,
    create_judge,
)


class TestParseVerdict:
    def test_defaults(self) -> None:
        v = _parse_verdict({})
        assert v.score == 0.0
        assert v.passed is False

    def test_clamps_score(self) -> None:
        v = _parse_verdict({"score": 1.5, "passed": True, "reasoning": "x"})
        assert v.score == 1.0


class TestMalformedJsonEdgeCases:
    def test_nested_json_recovery_fails_gracefully(self) -> None:
        with pytest.raises(JudgeExecutionError):
            _recover_malformed_json("no score here at all")


class TestOpenAIJudge:
    @patch("openai.OpenAI")
    def test_successful_evaluation(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"score": 0.85, "passed": true, "reasoning": "Good"}'
                    )
                )
            ]
        )

        cfg = LLMProviderConfig(provider="openai", model_name="gpt-4o")
        judge = OpenAIJudge(cfg)
        verdict = judge.evaluate_criterion("Evaluate this response.")
        assert verdict.score == 0.85
        assert verdict.passed is True

    def test_import_error_raises(self) -> None:
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((JudgeExecutionError, TypeError, AttributeError)):
                OpenAIJudge(LLMProviderConfig(provider="openai"))

    @patch("openai.OpenAI")
    def test_api_failure_raises(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        cfg = LLMProviderConfig(provider="openai")
        judge = OpenAIJudge(cfg)
        with pytest.raises(JudgeExecutionError, match="OpenAI judge failure"):
            judge.evaluate_criterion("test")


class TestAnthropicJudge:
    @patch("anthropic.Anthropic")
    def test_successful_evaluation(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"score": 0.75, "passed": true, "reasoning": "Acceptable"}')]
        )

        cfg = LLMProviderConfig(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
        judge = AnthropicJudge(cfg)
        verdict = judge.evaluate_criterion("Evaluate this response.")
        assert verdict.score == 0.75

    @patch("anthropic.Anthropic")
    def test_api_failure_raises(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("rate limited")

        cfg = LLMProviderConfig(provider="anthropic")
        judge = AnthropicJudge(cfg)
        with pytest.raises(JudgeExecutionError, match="Anthropic judge failure"):
            judge.evaluate_criterion("test")


class TestCreateJudgeFactory:
    def test_openai_factory(self) -> None:
        with patch("openai.OpenAI"):
            cfg = LLMProviderConfig(provider="openai")
            judge = create_judge(cfg)
            assert isinstance(judge, OpenAIJudge)

    def test_anthropic_factory(self) -> None:
        with patch("anthropic.Anthropic"):
            cfg = LLMProviderConfig(provider="anthropic")
            judge = create_judge(cfg)
            assert isinstance(judge, AnthropicJudge)
