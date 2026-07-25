"""
Enterprise Typer CLI application for the LLM Evaluation Framework.

Commands:
    run        — Execute complete multi-metric evaluation pipeline.
    validate   — Validate dataset format and schema compliance.
    benchmark  — Run evaluation against built-in benchmark dataset.
    metrics    — List all registered evaluation metrics.
    report     — Generate reports from a saved JSON evaluation result.
    visualize  — Generate visualizations from a saved JSON evaluation result.
    version    — Print framework version information.
    doctor     — System diagnostic health check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from llm_eval import __version__
from llm_eval.config.settings import EvaluationFrameworkConfig
from llm_eval.core.base_metric import MetricRegistry
from llm_eval.core.data_loader import DatasetLoader
from llm_eval.pipeline.runner import EvaluationPipeline
from llm_eval.reporting.generator import ReportGenerator
from llm_eval.schemas.evaluation import EvaluationRunReport
from llm_eval.utils.logger import setup_logger
from llm_eval.visualization.engine import VisualAnalyticsEngine

app = typer.Typer(
    name="llm-eval",
    help="Enterprise Production LLM Evaluation Framework CLI",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ------------------------------------------------------------------
# Global callback
# ------------------------------------------------------------------


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Path to write structured log file"
    ),
) -> None:
    """Enterprise Production LLM Evaluation Framework."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level, log_file=log_file)


# ------------------------------------------------------------------
# version
# ------------------------------------------------------------------


@app.command("version")
def version_cmd() -> None:
    """Print framework version information."""
    console.print(
        Panel(
            f"[bold cyan]llm-eval[/bold cyan] v{__version__}\n"
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            title="LLM Evaluation Framework",
            border_style="blue",
        )
    )


# ------------------------------------------------------------------
# doctor
# ------------------------------------------------------------------


@app.command("doctor")
def doctor_cmd() -> None:
    """Perform diagnostic health check of installed dependencies and metric backends."""
    console.print("[bold blue]Running system diagnostic health checks…[/bold blue]\n")

    table = Table(title="LLM-Eval Diagnostic Health Status")
    table.add_column("Component", style="cyan", min_width=25)
    table.add_column("Status", style="green")

    table.add_row(
        "Python Version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    # Probe dependencies
    probes: list[tuple[str, str]] = [
        ("pydantic", "Pydantic v2"),
        ("sentence_transformers", "SentenceTransformers"),
        ("nltk", "NLTK"),
        ("rouge_score", "Rouge Score"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        ("openai", "OpenAI SDK"),
        ("anthropic", "Anthropic SDK"),
        ("tenacity", "Tenacity"),
        ("loguru", "Loguru"),
        ("jinja2", "Jinja2"),
    ]

    for module_name, label in probes:
        try:
            __import__(module_name)
            table.add_row(label, "✅ OK")
        except ImportError:
            table.add_row(label, "[red]❌ Not installed[/red]")

    # Import metrics & rag to trigger registration
    try:
        import llm_eval.metrics  # noqa: F401
        import llm_eval.rag  # noqa: F401
    except Exception:
        pass

    registered = MetricRegistry.list_available()
    table.add_row("Metric Registry", f"✅ {len(registered)} metrics registered")

    for m_name in registered:
        table.add_row(f"  └─ {m_name}", "registered")

    console.print(table)


# ------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------


@app.command("metrics")
def list_metrics_cmd() -> None:
    """List all available registered evaluation metrics."""
    # Trigger metric auto-registration
    try:
        import llm_eval.metrics  # noqa: F401
        import llm_eval.rag  # noqa: F401
    except Exception:
        pass

    table = Table(title="Registered Evaluation Metrics")
    table.add_column("Metric Name", style="bold magenta")
    table.add_column("Description", style="white")

    for name in MetricRegistry.list_available():
        metric_cls = MetricRegistry._registry[name]
        table.add_row(name, metric_cls.description)

    console.print(table)


# ------------------------------------------------------------------
# validate
# ------------------------------------------------------------------


@app.command("validate")
def validate_cmd(
    dataset: Path = typer.Argument(
        ..., help="Path to .jsonl or .csv dataset file"
    ),
) -> None:
    """Validate dataset structure and schema compliance."""
    console.print(f"Validating dataset file: [yellow]{dataset}[/yellow]…")
    try:
        samples = DatasetLoader.load(dataset)
        console.print(
            f"[bold green]Validation Success![/bold green] "
            f"Loaded {len(samples)} valid evaluation samples."
        )
    except Exception as e:
        console.print(f"[bold red]Dataset Validation Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------


@app.command("run")
def run_evaluation_cmd(
    dataset: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to evaluation dataset file (.jsonl or .csv)"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML/JSON configuration file"
    ),
    output_dir: Path = typer.Option(
        Path("eval_reports"), "--output-dir", "-o", help="Directory to save report artifacts"
    ),
    run_id: str = typer.Option("eval_run", "--run-id", help="Unique identifier for evaluation run"),
) -> None:
    """Execute complete LLM evaluation pipeline."""
    console.print(f"[bold green]Starting Evaluation Run:[/bold green] `{run_id}`")

    cfg = (
        EvaluationFrameworkConfig.load_from_file(config)
        if config and config.exists()
        else EvaluationFrameworkConfig()
    )
    cfg.reporting.output_dir = output_dir

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading dataset…", total=None)
            samples = DatasetLoader.load(dataset)
            progress.update(task, description=f"Loaded {len(samples)} samples")

            progress.update(task, description="Running evaluation pipeline…")
            pipeline = EvaluationPipeline(cfg)
            report = pipeline.run_batch(samples, run_id=run_id)

            progress.update(task, description="Generating reports…")
            report_gen = ReportGenerator(output_dir=output_dir)
            report_gen.generate_all(report)

            if cfg.reporting.generate_plots:
                progress.update(task, description="Generating visualizations…")
                viz_engine = VisualAnalyticsEngine(output_dir=output_dir / "visuals")
                viz_engine.generate_all_visuals(report)

            progress.update(task, description="Done ✓")

        console.print(f"\n[bold green]Evaluation Completed Successfully![/bold green]")
        console.print(f"Report artifacts saved to: [cyan]{output_dir.resolve()}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Evaluation Pipeline Failure:[/bold red] {e}")
        logger.exception("Pipeline execution error")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------
# benchmark
# ------------------------------------------------------------------


@app.command("benchmark")
def benchmark_cmd(
    output_dir: Path = typer.Option(
        Path("eval_reports/benchmark"), "--output-dir", "-o", help="Output directory"
    ),
) -> None:
    """Run built-in framework benchmark dataset."""
    benchmark_path = Path(__file__).parent.parent / "benchmark_dataset.jsonl"
    if not benchmark_path.exists():
        console.print(f"[bold red]Benchmark dataset not found at {benchmark_path}[/bold red]")
        raise typer.Exit(code=1)

    run_evaluation_cmd(
        dataset=benchmark_path, config=None, output_dir=output_dir, run_id="benchmark_run"
    )


# ------------------------------------------------------------------
# report (standalone)
# ------------------------------------------------------------------


@app.command("report")
def report_cmd(
    input_json: Path = typer.Option(
        ..., "--input", "-i", help="Path to a saved JSON evaluation report"
    ),
    output_dir: Path = typer.Option(
        Path("eval_reports"), "--output-dir", "-o", help="Output directory for generated reports"
    ),
) -> None:
    """Generate formatted reports from a previously saved JSON evaluation result."""
    if not input_json.exists():
        console.print(f"[bold red]Input file not found:[/bold red] {input_json}")
        raise typer.Exit(code=1)

    try:
        raw = json.loads(input_json.read_text(encoding="utf-8"))
        report = EvaluationRunReport(**raw)
        gen = ReportGenerator(output_dir=output_dir)
        generated = gen.generate_all(report)
        console.print(f"[bold green]Reports generated:[/bold green]")
        for fmt, path in generated.items():
            console.print(f"  {fmt}: [cyan]{path}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Report generation failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------
# visualize (standalone)
# ------------------------------------------------------------------


@app.command("visualize")
def visualize_cmd(
    input_json: Path = typer.Option(
        ..., "--input", "-i", help="Path to a saved JSON evaluation report"
    ),
    output_dir: Path = typer.Option(
        Path("eval_visuals"), "--output-dir", "-o", help="Output directory for charts"
    ),
) -> None:
    """Generate visualizations from a previously saved JSON evaluation result."""
    if not input_json.exists():
        console.print(f"[bold red]Input file not found:[/bold red] {input_json}")
        raise typer.Exit(code=1)

    try:
        raw = json.loads(input_json.read_text(encoding="utf-8"))
        report = EvaluationRunReport(**raw)
        engine = VisualAnalyticsEngine(output_dir=output_dir)
        visuals = engine.generate_all_visuals(report)
        console.print(f"[bold green]Visualizations generated:[/bold green]")
        for name, path in visuals.items():
            console.print(f"  {name}: [cyan]{path}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Visualization generation failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app()
