"""
Configuration loader and settings validation using Pydantic v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_eval.exceptions.base import ConfigurationError


class LLMProviderConfig(BaseModel):
    """Configuration options for an LLM Provider (OpenAI, Anthropic, etc.)."""

    provider: Literal["openai", "anthropic", "mock"] = "openai"
    model_name: str = "gpt-4o"
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)

    def get_api_key(self) -> str | None:
        """Safely extract the plain-text API key value."""
        return self.api_key.get_secret_value() if self.api_key else None


class MetricConfig(BaseModel):
    """Individual metric settings and pass thresholds."""

    enabled: bool = True
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = Field(default=32, gt=0)
    normalize_embeddings: bool = True


class PipelineConfig(BaseModel):
    """Execution pipeline settings."""

    max_workers: int = Field(default=4, gt=0)
    fail_on_sample_error: bool = False
    cache_embeddings: bool = True


class ReportingConfig(BaseModel):
    """Reporting and visualization output options."""

    output_dir: Path = Path("eval_reports")
    formats: list[Literal["json", "markdown", "html", "csv"]] = [
        "json",
        "markdown",
        "html",
    ]
    generate_plots: bool = True


class EvaluationFrameworkConfig(BaseSettings):
    """
    Root configuration schema for the LLM Evaluation Framework.
    Supports environment variable overrides prefixed with ``LLM_EVAL_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_EVAL_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    project_name: str = "Enterprise LLM Evaluation"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path | None = None

    judge: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    metrics: dict[str, MetricConfig] = Field(
        default_factory=lambda: {
            "bleu": MetricConfig(enabled=True, threshold=0.3),
            "rouge_l": MetricConfig(enabled=True, threshold=0.4),
            "bert_score": MetricConfig(enabled=True, threshold=0.7),
            "embedding_similarity": MetricConfig(enabled=True, threshold=0.75),
            "faithfulness": MetricConfig(enabled=True, threshold=0.8),
            "context_relevancy": MetricConfig(enabled=True, threshold=0.7),
            "answer_relevancy": MetricConfig(enabled=True, threshold=0.7),
        }
    )

    @classmethod
    def load_from_file(cls, config_path: str | Path) -> EvaluationFrameworkConfig:
        """
        Load configuration from YAML or JSON file.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(content) or {}
            elif path.suffix == ".json":
                data = json.loads(content)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration format: {path.suffix}. Use YAML or JSON."
                )

            return cls(**data)
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise e
            raise ConfigurationError(
                f"Failed to parse configuration file '{path}': {e}"
            ) from e
