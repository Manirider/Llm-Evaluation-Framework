"""Tests for LLM Judge providers and factory."""

from __future__ import annotations

import pytest

from llm_eval.config.settings import LLMProviderConfig
from llm_eval.exceptions.base import JudgeExecutionError
from llm_eval.judge.base import JudgeVerdict
from llm_eval.judge.providers import (
    MockJudge,
    _recover_malformed_json,
    create_judge,
)


class TestMockJudge:
    def test_positive_verdict(self) -> None:
        judge = MockJudge()
        v = judge.evaluate_criterion("Evaluate factual correctness of this response.")
        assert isinstance(v, JudgeVerdict)
        assert v.score > 0.5
        assert v.passed is True

    def test_negative_keywords(self) -> None:
        judge = MockJudge()
        v = judge.evaluate_criterion("This contains hallucination and incorrect facts.")
        assert v.score < 0.5
        assert v.passed is False

    def test_raw_response(self) -> None:
        judge = MockJudge()
        v = judge.evaluate_criterion("Test")
        assert v.raw_response["provider"] == "mock"


class TestCreateJudge:
    def test_mock_factory(self) -> None:
        cfg = LLMProviderConfig(provider="mock")
        judge = create_judge(cfg)
        assert isinstance(judge, MockJudge)

    def test_unknown_provider_defaults_to_mock(self) -> None:
        cfg = LLMProviderConfig(provider="mock")
        judge = create_judge(cfg)
        assert isinstance(judge, MockJudge)


class TestMalformedJsonRecovery:
    def test_valid_json(self) -> None:
        result = _recover_malformed_json('{"score": 0.8, "passed": true, "reasoning": "Good"}')
        assert result["score"] == 0.8

    def test_json_in_text(self) -> None:
        text = 'Here is my evaluation: {"score": 0.7, "passed": true, "reasoning": "OK"} end.'
        result = _recover_malformed_json(text)
        assert result["score"] == 0.7

    def test_regex_fallback(self) -> None:
        text = 'The "score": 0.65, and "passed": false, with "reasoning": "Bad quality".'
        result = _recover_malformed_json(text)
        assert result["score"] == 0.65
        assert result["passed"] is False

    def test_unrecoverable_raises(self) -> None:
        with pytest.raises(JudgeExecutionError, match="Failed to extract"):
            _recover_malformed_json("completely invalid gibberish with no json whatsoever")
