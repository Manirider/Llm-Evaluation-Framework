from typer.testing import CliRunner
from llm_eval.cli.main import app
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

runner = CliRunner()

# Test run command with options
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    json.dump({"sample_id": "s1", "input_text": "Q", "actual_output": "A", "expected_output": "A"}, f)
    f.write('\n')
    test_file = f.name

output_dir = tempfile.mkdtemp()

# Create a mock report
from llm_eval.schemas.evaluation import EvaluationRunReport, MetricStatistics

mock_report = EvaluationRunReport(
    run_id="test_run",
    dataset_size=1,
    configured_metrics=["bleu"],
    sample_results=[],
    metric_summary={
        "bleu": MetricStatistics(
            metric_name="bleu",
            count=1,
            mean=0.8,
            std_dev=0.0,
            variance=0.0,
            min=0.8,
            max=0.8,
            median=0.8,
            mode=0.8,
            p10=0.8,
            p25=0.8,
            p75=0.8,
            p90=0.8,
            skewness=0.0,
            kurtosis=0.0,
            ci_95_lower=0.8,
            ci_95_upper=0.8,
        )
    },
    execution_duration_seconds=0.1,
)

with patch("llm_eval.cli.main.EvaluationPipeline") as mock_pipeline_cls:
    mock_pipeline_cls.return_value.run_batch.return_value = mock_report
    result = runner.invoke(app, ['run', '--dataset', test_file, '--output-dir', output_dir, '--run-id', 'test_run'])
    print('Exit code:', result.exit_code)
    with open('test_output.txt', 'w', encoding='utf-8') as out:
        out.write(result.output)
    print('Output written')