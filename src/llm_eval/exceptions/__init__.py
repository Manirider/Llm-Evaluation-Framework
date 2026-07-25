"""Custom exceptions for LLM Evaluation Framework."""

from .base import LLMError
from .config import ConfigurationError, ValidationError
from .metric import MetricError, MetricComputationError
from .judge import JudgeError, JudgeProviderError, JudgeSchemaError, JudgeRateLimitError
from .pipeline import PipelineError, PipelineStepError, PipelineRecoveryError
from .data import DataError, DataValidationError, DataLoadError
from .judge import JudgeError as JudgeErrorAlias
from .reporting import ReportingError, ReportGenerationError
from .visualization import VisualizationError, VisualizationExportError
from .config import ConfigurationError as ConfigError

__all__ = [
    "LLMError",
    "ConfigurationError",
    "ValidationError",
    "ConfigError",
    "MetricError",
    "MetricComputationError",
    "JudgeError",
    "JudgeProviderError",
    "JudgeSchemaError",
    "JudgeRateLimitError",
    "PipelineError",
    "PipelineStepError",
    "PipelineRecoveryError",
    "DataError",
    "DataValidationError",
    "DataLoadError",
    "ReportingError",
    "ReportGenerationError",
    "VisualizationError",
    "VisualizationExportError",
]