"""Edge-case tests for dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm_eval.core.data_loader import DatasetLoader
from llm_eval.exceptions.base import DatasetValidationError


class TestDataLoaderEdgeCases:
    def test_unsupported_format(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(DatasetValidationError, match="Unsupported"):
            DatasetLoader.load(f)

    def test_empty_jsonl_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        with pytest.raises(DatasetValidationError, match="no valid"):
            DatasetLoader.load(f)

    def test_jsonl_blank_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "data.jsonl"
        f.write_text(
            "\n"
            + json.dumps({"sample_id": "s1", "input_text": "Q", "actual_output": "A"})
            + "\n\n"
        )
        samples = DatasetLoader.load(f)
        assert len(samples) == 1

    def test_jsonl_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.jsonl"
        f.write_text("{invalid json\n")
        with pytest.raises(DatasetValidationError, match="Invalid JSON"):
            DatasetLoader.load(f)

    def test_csv_missing_columns(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": [1]}).to_csv(f, index=False)
        with pytest.raises(DatasetValidationError, match="missing required"):
            DatasetLoader.load(f)

    def test_csv_string_contexts(self, tmp_path: Path) -> None:
        f = tmp_path / "ctx.csv"
        pd.DataFrame(
            {
                "sample_id": ["s1"],
                "input_text": ["Q"],
                "actual_output": ["A"],
                "retrieved_contexts": ['["ctx1", "ctx2"]'],
            }
        ).to_csv(f, index=False)
        samples = DatasetLoader.load(f)
        assert samples[0].retrieved_contexts == ["ctx1", "ctx2"]

    def test_csv_plain_string_context(self, tmp_path: Path) -> None:
        f = tmp_path / "ctx2.csv"
        pd.DataFrame(
            {
                "sample_id": ["s1"],
                "input_text": ["Q"],
                "actual_output": ["A"],
                "retrieved_contexts": ["single context string"],
            }
        ).to_csv(f, index=False)
        samples = DatasetLoader.load(f)
        assert samples[0].retrieved_contexts == ["single context string"]

    def test_csv_validation_error(self, tmp_path: Path) -> None:
        f = tmp_path / "invalid.csv"
        pd.DataFrame(
            {"sample_id": ["s1"], "input_text": ["  "], "actual_output": ["A"]}
        ).to_csv(f, index=False)
        with pytest.raises(DatasetValidationError):
            DatasetLoader.load(f)

    def test_file_not_found(self) -> None:
        with pytest.raises(DatasetValidationError, match="not found"):
            DatasetLoader.load("/nonexistent/path/data.jsonl")
