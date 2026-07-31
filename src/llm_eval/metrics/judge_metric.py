"""
LLM-as-a-Judge evaluation metric registered in the BaseMetric framework.

Wraps judge providers as a first-class pipeline metric, supporting configurable
evaluation criteria, bias mitigation prompts, and structured-output-only
responses (no chain-of-thought).
"""

from __future__ import annotations

from typing import Any

from llm_eval.config.settings import LLMProviderConfig
from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.judge.providers import MockJudge, create_judge
from llm_eval.schemas.evaluation import EvaluationSample

# Bias mitigation instructions — appended to every judge prompt.
_BIAS_MITIGATION = (
    "IMPORTANT INSTRUCTIONS FOR UNBIASED EVALUATION:\n"
    "- Evaluate ONLY the content quality against the stated criteria.\n"
    "- Do NOT favor longer or more verbose responses over concise ones.\n"
    "- Do NOT penalize or reward responses based on their position.\n"
    "- Do NOT inflate scores for responses that praise themselves.\n"
    "- Base your score strictly on factual accuracy, relevance, and completeness.\n"
)


@MetricRegistry.register("llm_judge")
class LLMJudgeMetric(BaseMetric):
    """
    Uses an LLM-as-a-Judge provider (OpenAI, Anthropic, or Mock) to evaluate
    the quality of ``actual_output`` against ``input_text`` and optional
    ``expected_output``.

    Returns structured scores only — chain-of-thought is never requested.
    """

    metric_name: str = "llm_judge"
    description: str = "LLM-as-a-Judge quality evaluation via external provider"

    def __init__(
        self,
        threshold: float | None = None,
        judge_config: LLMProviderConfig | None = None,
        criteria: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        if isinstance(judge_config, LLMProviderConfig):
            self.judge = create_judge(judge_config)
        elif isinstance(kwargs.get("judge_config"), LLMProviderConfig):
            self.judge = create_judge(kwargs["judge_config"])
        else:
            self.judge = MockJudge()

        self.criteria = criteria or (
            "Evaluate the response for factual accuracy, relevance to the question, "
            "completeness of the answer, and clarity of expression."
        )

    def _build_prompt(self, sample: EvaluationSample) -> str:
        """Construct the evaluation prompt with bias mitigation."""
        parts = [
            _BIAS_MITIGATION,
            f"EVALUATION CRITERIA: {self.criteria}\n",
            f"USER QUERY:\n{sample.input_text}\n",
            f"MODEL RESPONSE:\n{sample.actual_output}\n",
        ]
        if sample.expected_output:
            parts.append(f"REFERENCE ANSWER:\n{sample.expected_output}\n")
        if sample.retrieved_contexts:
            ctx_block = "\n---\n".join(sample.retrieved_contexts)
            parts.append(f"RETRIEVED CONTEXTS:\n{ctx_block}\n")

        parts.append(
            "Return ONLY a JSON object with keys: "
            '"score" (float 0.0-1.0), "passed" (bool), "reasoning" (str).'
        )
        return "\n".join(parts)

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        prompt = self._build_prompt(sample)
        verdict = self.judge.evaluate_criterion(prompt)

        details = {
            "judge_provider": type(self.judge).__name__,
            "raw_response": verdict.raw_response,
        }
        return verdict.score, verdict.reasoning, details
