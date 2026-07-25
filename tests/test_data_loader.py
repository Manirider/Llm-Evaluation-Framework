"""Tests for dataset loading (JSONL and CSV) with edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval.core.data_loader import DatasetLoader
from llm_eval.exceptions.base import DatasetValidationError


class TestJSONLLoader:
    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        data = [
            {"sample_id": "s1", "input_text": "Q1", "actual_output": "A1"},
            {"sample_id": "s2", "input_text": "Q2", "actual_output": "A2"},
        ]
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))
        samples = DatasetLoader.load(path)
        assert len(samples) == 2
        assert samples[0].sample_id == "s1"

    def test_auto_assigns_sample_id(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps({"input_text": "Q", "actual_output": "A"}))
        samples = DatasetLoader.load(path)
        assert samples[0].sample_id.startswith("sample_")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(DatasetValidationError, match="Invalid JSON"):
            DatasetLoader.load(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text("")
        with pytest.raises(DatasetValidationError, match="no valid"):
            DatasetLoader.load(path)

    def test_schema_violation_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps({"sample_id": "s1", "input_text": "", "actual_output": "A"}))
        with pytest.raises(DatasetValidationError):
            DatasetLoader.load(path)


class TestCSVLoader:
    def test_load_valid_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        path.write_text("input_text,actual_output\nQ1,A1\nQ2,A2\n")
        samples = DatasetLoader.load(path)
        assert len(samples) == 2

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        path.write_text("question,answer\nQ1,A1\n")
        with pytest.raises(DatasetValidationError, match="missing required"):
            DatasetLoader.load(path)


class TestLoaderEdgeCases:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(DatasetValidationError, match="not found"):
            DatasetLoader.load("/nonexistent/file.jsonl")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "test.txt"
        path.write_text("hello")
        with pytest.raises(DatasetValidationError, match="Unsupported"):
            DatasetLoader.load(path)
