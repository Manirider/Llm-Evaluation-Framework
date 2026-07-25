import typer
from typer.testing import _get_command
from click.testing import CliRunner
from click import Command

app = typer.Typer(add_completion=False)

@app.callback()
def main():
    pass

@app.command()
def hello(name: str = typer.Option("World", "--name")):
    print(f"Hello {name}")

@app.command()
def goodbye():
    print("goodbye")

click_cmd = _get_command(app)
print('Type:', type(click_cmd).__name__)

if hasattr(click_cmd, 'commands'):
    hello_cmd = click_cmd.commands['hello']
    print('Hello params:', hello_cmd.params)
    runner = CliRunner()
    
    # Try invoke via group
    result = runner.invoke(click_cmd, ['hello', '--name', 'Test'])
    with open('test_output.txt', 'w', encoding='utf-8') as f:
        f.write(f'Exit code: {result.exit_code}\n')
        f.write(f'Output: {result.output}\n')
    
print('Done')