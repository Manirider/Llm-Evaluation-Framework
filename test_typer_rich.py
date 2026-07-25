import typer
from typer.testing import CliRunner

app = typer.Typer(rich_markup_mode="rich")

@app.command("run")
def run_cmd(
    dataset: str = typer.Option(..., "--dataset", "-d"),
    output_dir: str = typer.Option(..., "--output-dir", "-o"),
    run_id: str = typer.Option("eval_run", "--run-id"),
):
    print(f"dataset={dataset}, output_dir={output_dir}, run_id={run_id}")

if __name__ == "__main__":
    runner = CliRunner()
    result = runner.invoke(app, ['run', '--dataset', 'test.jsonl', '--output-dir', 'out', '--run-id', 'test'])
    with open('test_output.txt', 'w', encoding='utf-8') as f:
        f.write(f'Exit code: {result.exit_code}\n')
        f.write(f'Output: {result.output}\n')