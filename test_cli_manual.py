from typer.testing import CliRunner
from llm_eval.cli.main import app

runner = CliRunner()

# Test with strings
result = runner.invoke(app, ['run', '--dataset', 'test.jsonl', '--output-dir', 'out', '--run-id', 'test'])
print('Exit code:', result.exit_code)
with open('cli_output.txt', 'w', encoding='utf-8') as f:
    f.write(result.output)
print('Output written to cli_output.txt')