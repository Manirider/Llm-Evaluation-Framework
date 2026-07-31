"""
Evaluation dataset loaders for CSV and JSONL formats with strict schema validation.
"""

import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from llm_eval.exceptions.base import DatasetValidationError
from llm_eval.schemas.evaluation import EvaluationSample


class DatasetLoader:
    """
    Handles reading and validating evaluation dataset files (.jsonl and .csv).
    """

    @classmethod
    def load(cls, file_path: str | Path) -> list[EvaluationSample]:
        """
        Load samples from file based on extension.
        """
        path = Path(file_path)
        if not path.exists():
            raise DatasetValidationError(f"Dataset file not found at path: {path}")

        if path.suffix == ".jsonl":
            return cls._load_jsonl(path)
        elif path.suffix == ".csv":
            return cls._load_csv(path)
        else:
            raise DatasetValidationError(
                f"Unsupported file format '{path.suffix}'. Supported formats: .jsonl, .csv"
            )

    @classmethod
    def _load_jsonl(cls, path: Path) -> list[EvaluationSample]:
        samples: list[EvaluationSample] = []
        errors: list[str] = []

        with path.open(encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    raw_data = json.loads(line_str)
                    if "sample_id" not in raw_data:
                        raw_data["sample_id"] = f"sample_{line_idx}"
                    sample = EvaluationSample(**raw_data)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_idx}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_idx}: Schema Validation Failure - {e}")

        if errors:
            raise DatasetValidationError(
                f"Failed to load JSONL dataset '{path}'. Found {len(errors)} error(s):\n"
                + "\n".join(errors[:10])
                + (f"\n... and {len(errors) - 10} more" if len(errors) > 10 else "")
            )

        if not samples:
            raise DatasetValidationError(f"Dataset '{path}' contains no valid evaluation samples.")

        return samples

    @classmethod
    def _load_csv(cls, path: Path) -> list[EvaluationSample]:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse CSV file '{path}': {e}") from e

        required_cols = {"input_text", "actual_output"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise DatasetValidationError(f"CSV dataset missing required column(s): {missing_cols}")

        samples: list[EvaluationSample] = []
        errors: list[str] = []

        for row_idx, row in df.iterrows():
            record = row.to_dict()
            if "sample_id" not in record or pd.isna(record["sample_id"]):
                record["sample_id"] = f"sample_{row_idx + 1}"

            # Clean string conversions
            for key in ["input_text", "actual_output", "expected_output"]:
                if key in record and pd.isna(record[key]):
                    record[key] = None if key == "expected_output" else ""

            if "retrieved_contexts" in record and isinstance(record["retrieved_contexts"], str):
                try:
                    record["retrieved_contexts"] = json.loads(record["retrieved_contexts"])
                except Exception:
                    record["retrieved_contexts"] = [record["retrieved_contexts"]]

            try:
                sample = EvaluationSample(**record)
                samples.append(sample)
            except ValidationError as e:
                errors.append(f"Row {row_idx + 1}: {e}")

        if errors:
            raise DatasetValidationError(
                f"Failed to validate CSV dataset '{path}'. Errors:\n" + "\n".join(errors[:10])
            )

        return samples
