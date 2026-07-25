"""Tests for the custom exception hierarchy."""

from __future__ import annotations

from llm_eval.exceptions.base import (
    ConfigurationError,
    DatasetValidationError,
    EmbeddingError,
    JudgeExecutionError,
    LLMEvalError,
    MetricExecutionError,
    PipelineExecutionError,
    ReportingError,
    VisualizationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self) -> None:
        exceptions = [
            ConfigurationError,
            DatasetValidationError,
            MetricExecutionError,
            JudgeExecutionError,
            EmbeddingError,
            PipelineExecutionError,
            ReportingError,
            VisualizationError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, LLMEvalError)
            assert issubclass(exc_cls, Exception)

    def test_message_property(self) -> None:
        e = LLMEvalError("Test message")
        assert e.message == "Test message"
        assert str(e) == "Test message"

    def test_details_in_str(self) -> None:
        e = LLMEvalError("Error", details={"key": "value"})
        assert "key" in str(e)
        assert "value" in str(e)

    def test_empty_details(self) -> None:
        e = LLMEvalError("Error")
        assert e.details == {}
        assert "Details" not in str(e)

    def test_catch_as_base(self) -> None:
        try:
            raise ConfigurationError("Config broken")
        except LLMEvalError as e:
            assert "Config broken" in str(e)
