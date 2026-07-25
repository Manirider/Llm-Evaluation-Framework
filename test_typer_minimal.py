import typer
from typer.testing import CliRunner

app = typer.Typer()

@app.command()
def hello(name: str = typer.Option("World", "--name", "-n")):
    print(f"Hello {name}")

if __name__ == "__main__":
    runner = CliRunner()
    # Try with option
    result = runner.invoke(app, ['--name', 'Test'])
    with open('typer_test_output.txt', 'w', encoding='utf-8') as f:
        f.write(f"Exit code: {result.exit_code}\n")
        f.write(f"Output: {result.output}\n")
    
    # Try positional
    result2 = runner.invoke(app, ['Test'])
    with open('typer_test_output2.txt', 'w', encoding='utf-8') as f:
        f.write(f"Exit code: {result2.exit_code}\n")
        f.write(f"Output: {result2.output}\n")