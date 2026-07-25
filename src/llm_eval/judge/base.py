"""
LLM-as-a-Judge provider abstract base class and evaluation output models.
"""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from llm_eval.config.settings import LLMProviderConfig
from llm_eval.exceptions.base import JudgeExecutionError


class JudgeVerdict(BaseModel):
    """
    Structured output format returned by LLM-as-a-Judge evaluations.
    Chain-of-thought is excluded; only binary/continuous score and clear reasoning rationale are returned.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized quality score between 0.0 and 1.0")
    passed: bool = Field(..., description="Binary verdict against evaluation criteria")
    reasoning: str = Field(..., description="Concise justification for the score")
    raw_response: dict[str, Any] = Field(default_factory=dict, description="Raw provider response payload")


class BaseLLMJudge(ABC):
    """
    Abstract interface for LLM Judge providers (OpenAI, Anthropic, Mock).
    Handles prompt construction, structural JSON response parsing, and resilience retries.
    """

    def __init__(self, config: LLMProviderConfig | None = None) -> None:
        self.config = config or LLMProviderConfig()

    @abstractmethod
    def evaluate_criterion(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> JudgeVerdict:
        """
        Execute evaluation against a judge prompt and parse structured output into JudgeVerdict.
        """
        pass
