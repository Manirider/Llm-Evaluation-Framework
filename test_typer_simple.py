import typer
from typer.testing import CliRunner

# Without rich_markup_mode
app = typer.Typer()

@app.command()
def hello(name: str = typer.Option("World", "--name")):
    print(f"Hello {name}")

@app.command()
def goodbye(name: str = typer.Option("World", "--name")):
    print(f"Goodbye {name}")

if __name__ == "__main__":
    runner = CliRunner()
    
    result = runner.invoke(app, ['hello', '--name', 'Test'])
    with open('test_output.txt', 'w', encoding='utf-8') as f:
        f.write(f'Exit code: {result.exit_code}\n')
        f.write(f'Output: {result.output}\n')
    
    print("Done")