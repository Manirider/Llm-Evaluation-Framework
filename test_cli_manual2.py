from typer.testing import CliRunner
from llm_eval.cli.main import app
import json
import tempfile
from pathlib import Path

runner = CliRunner()

# Test validate with a path that has no spaces
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    json.dump({"sample_id": "s1", "input_text": "Q", "actual_output": "A"}, f)
    f.write('\n')
    test_file = f.name

result = runner.invoke(app, ['validate', test_file])
print('Exit code:', result.exit_code)
with open('test_output.txt', 'w', encoding='utf-8') as out:
    out.write(result.output)
print('Output written')