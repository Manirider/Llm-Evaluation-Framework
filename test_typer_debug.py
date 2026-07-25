import typer
from typer.testing import CliRunner

app = typer.Typer()

@app.command()
def hello(name: str = typer.Option("World", "--name")):
    print(f"Hello {name}")

@app.command()
def goodbye(name: str = typer.Option("World", "--name")):
    print(f"Goodbye {name}")

if __name__ == "__main__":
    runner = CliRunner()
    
    # Test 1: hello with option
    result = runner.invoke(app, ['hello', '--name', 'Test'])
    with open('test_output.txt', 'w', encoding='utf-8') as f:
        f.write(f'Test 1 Exit code: {result.exit_code}\n')
        f.write(f'Test 1 Output: {result.output}\n')
    
    # Test 2: hello without option
    result2 = runner.invoke(app, ['hello'])
    with open('test_output.txt', 'a', encoding='utf-8') as f:
        f.write(f'Test 2 Exit code: {result2.exit_code}\n')
        f.write(f'Test 2 Output: {result2.output}\n')
    
    # Test 3: Just 'run' subcommand
    result3 = runner.invoke(app, ['goodbye', '--name', 'Test'])
    with open('test_output.txt', 'a', encoding='utf-8') as f:
        f.write(f'Test 3 Exit code: {result3.exit_code}\n')
        f.write(f'Test 3 Output: {result3.output}\n')
    
    print("All done")