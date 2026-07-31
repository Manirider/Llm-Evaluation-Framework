"""
BLEU and ROUGE-L classical metric implementations.
"""

from typing import Any

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.schemas.evaluation import EvaluationSample

# Ensure NLTK tokenizer models are pre-loaded
for _res in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res.split("/")[-1], quiet=True)


@MetricRegistry.register("bleu")
class BLEUMetric(BaseMetric):
    """
    Computes BLEU n-gram precision score between actual_output and expected_output.
    """

    metric_name = "bleu"
    description = "Bilingual Evaluation Understudy (BLEU) n-gram overlap metric"

    def __init__(self, threshold: float | None = None, n_grams: int = 4, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.n_grams = max(1, min(4, n_grams))
        self.smooth_fn = SmoothingFunction().method1

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.expected_output:
            return 0.0, "Skipped BLEU calculation: sample missing expected_output", {}

        reference_tokens = nltk.word_tokenize(sample.expected_output.lower())
        candidate_tokens = nltk.word_tokenize(sample.actual_output.lower())

        if not candidate_tokens or not reference_tokens:
            return 0.0, "Empty token list after tokenization", {}

        # Set n-gram weight tuple dynamically
        weights = tuple(1.0 / self.n_grams for _ in range(self.n_grams))
        score = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=weights,
            smoothing_function=self.smooth_fn,
        )

        reasoning = (
            f"BLEU-{self.n_grams} score computed across {len(candidate_tokens)} candidate tokens."
        )
        details = {
            "n_grams": self.n_grams,
            "candidate_token_count": len(candidate_tokens),
            "reference_token_count": len(reference_tokens),
        }
        return float(score), reasoning, details


@MetricRegistry.register("rouge_l")
class ROUGEMetric(BaseMetric):
    """
    Computes ROUGE-L Longest Common Subsequence F1 score.
    """

    metric_name = "rouge_l"
    description = "Recall-Oriented Understudy for Gisting Evaluation (ROUGE-L)"

    def __init__(self, threshold: float | None = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _compute(self, sample: EvaluationSample) -> tuple[float, str | None, dict[str, Any]]:
        if not sample.expected_output:
            return 0.0, "Skipped ROUGE-L calculation: sample missing expected_output", {}

        scores = self.scorer.score(sample.expected_output, sample.actual_output)
        rouge_l_stats = scores["rougeL"]

        f1_score = rouge_l_stats.fmeasure
        reasoning = (
            f"ROUGE-L F1 score: {f1_score:.4f} (Precision: {rouge_l_stats.precision:.4f}, "
            f"Recall: {rouge_l_stats.recall:.4f})"
        )
        details = {
            "precision": round(float(rouge_l_stats.precision), 4),
            "recall": round(float(rouge_l_stats.recall), 4),
            "f1": round(float(f1_score), 4),
        }
        return float(f1_score), reasoning, details
