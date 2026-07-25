"""Tests for configuration loading, validation, and environment overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_eval.config.settings import EvaluationFrameworkConfig, LLMProviderConfig
from llm_eval.exceptions.base import ConfigurationError


class TestConfigDefaults:
    def test_default_project_name(self) -> None:
        cfg = EvaluationFrameworkConfig()
        assert cfg.project_name == "Enterprise LLM Evaluation"

    def test_default_metrics_present(self) -> None:
        cfg = EvaluationFrameworkConfig()
        assert "bleu" in cfg.metrics
        assert "rouge_l" in cfg.metrics
        assert "bert_score" in cfg.metrics

    def test_default_pipeline_workers(self) -> None:
        cfg = EvaluationFrameworkConfig()
        assert cfg.pipeline.max_workers == 4

    def test_default_judge_provider(self) -> None:
        cfg = EvaluationFrameworkConfig()
        assert cfg.judge.provider == "openai"

    def test_default_embedding_model(self) -> None:
        cfg = EvaluationFrameworkConfig()
        assert cfg.embeddings.model_name == "all-MiniLM-L6-v2"


class TestConfigFromYAML:
    def test_load_yaml(self, tmp_path: Path) -> None:
        data = {"project_name": "Test Project", "log_level": "DEBUG"}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(data))
        cfg = EvaluationFrameworkConfig.load_from_file(config_file)
        assert cfg.project_name == "Test Project"
        assert cfg.log_level == "DEBUG"

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"project_name": "YML Test"}))
        cfg = EvaluationFrameworkConfig.load_from_file(config_file)
        assert cfg.project_name == "YML Test"

    def test_load_json(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"project_name": "JSON Test"}))
        cfg = EvaluationFrameworkConfig.load_from_file(config_file)
        assert cfg.project_name == "JSON Test"


class TestConfigErrors:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            EvaluationFrameworkConfig.load_from_file("/nonexistent/file.yaml")

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.txt"
        config_file.write_text("hello")
        with pytest.raises(ConfigurationError, match="Unsupported"):
            EvaluationFrameworkConfig.load_from_file(config_file)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{invalid: yaml: content: [}")
        with pytest.raises(ConfigurationError):
            EvaluationFrameworkConfig.load_from_file(config_file)


class TestSecretStr:
    def test_api_key_hidden(self) -> None:
        cfg = LLMProviderConfig(api_key="sk-secret-key")
        assert "sk-secret-key" not in str(cfg)
        assert cfg.get_api_key() == "sk-secret-key"

    def test_api_key_none(self) -> None:
        cfg = LLMProviderConfig()
        assert cfg.get_api_key() is None
