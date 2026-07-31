"""Tests for CLI commands using Typer's CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner

from llm_eval.cli.main import app

runner = CliRunner()


class TestCLI:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "llm-eval" in result.output.lower() or "1.0.0" in result.output

    def test_doctor(self) -> None:
        # Skip on environments where torch/sentence_transformers fails to load
        # This is a known issue on some Windows environments with torch DLL loading
        result = runner.invoke(app, ["doctor"])
        if result.exit_code != 0 and "DLL" in str(result.exception) or "1114" in str(result.exception):
            import pytest
            pytest.skip(f"Skipping doctor test due to torch DLL issue: {result.exception}")
        assert result.exit_code == 0
        assert "Diagnostic" in result.output or "Health" in result.output

    def test_metrics(self) -> None:
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        assert "bleu" in result.output.lower()

    def test_validate_missing_file(self) -> None:
        result = runner.invoke(app, ["validate", "/nonexistent/file.jsonl"])
        assert result.exit_code != 0

    def test_validate_valid_file(self, tmp_path) -> None:
        import json
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps({
            "sample_id": "s1", "input_text": "Q", "actual_output": "A"
        }) + "\n")
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0, f"CLI output: {result.output}"
        assert "Success" in result.output

    def test_report_missing_file(self) -> None:
        result = runner.invoke(app, ["report", "--input", "/nonexistent.json"])
        assert result.exit_code != 0

    def test_visualize_missing_file(self) -> None:
        result = runner.invoke(app, ["visualize", "--input", "/nonexistent.json"])
        assert result.exit_code != 0
