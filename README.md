# Enterprise LLM Evaluation Framework (`llm-eval`)

[![CI/CD Pipeline](https://github.com/enterprise/llm-evaluation-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/enterprise/llm-evaluation-framework/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: MyPy](https://img.shields.io/badge/typing-mypy-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)

A **production-grade, enterprise-ready** evaluation framework for Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) systems, Chatbots, AI Agents, and Tool-Calling systems. Built with Clean Architecture, Domain-Driven Design, and SOLID principles.


## ✨ Key Features

| Category | Capabilities |
|:---|:---|
| **Classical Metrics** | BLEU (configurable n-gram), ROUGE-L (precision/recall/F1) |
| **Semantic Metrics** | BERTScore (token-level F1), Embedding Cosine Similarity |
| **RAG Metrics** | Faithfulness, Context Relevancy, Answer Relevancy, Context Precision, Context Recall, Groundedness, Hallucination Detection |
| **LLM-as-a-Judge** | OpenAI, Anthropic, Mock providers with bias mitigation & structured output |
| **Reporting** | JSON, Markdown, HTML, CSV with executive summary, failure analysis & recommendations |
| **Visualization** | 9 publication-quality chart types (Radar, Box, Violin, Histogram, Heatmap, Correlation, KDE, Failure Breakdown, Metric Comparison) |
| **Statistics** | Mean, Median, Mode, Variance, Std Dev, Percentiles (P10/P25/P75/P90), Skewness, Kurtosis, 95% Confidence Intervals |
| **Infrastructure** | Docker, GitHub Actions CI/CD, Poetry, Pydantic v2, Rich CLI |

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI Layer (Typer + Rich)                   │
│   run │ validate │ benchmark │ metrics │ report │ visualize │ doctor│
├─────────────────────────────────────────────────────────────────────┤
│                       Pipeline Layer                                │
│            ThreadPoolExecutor ─ Error Isolation ─ Batching          │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│  Metrics     │  RAG Metrics │  LLM Judge   │  Embeddings           │
│  ├─ BLEU     │  ├─ Faith.   │  ├─ OpenAI   │  └─ SentenceTransf.  │
│  ├─ ROUGE-L  │  ├─ Ctx Rel. │  ├─ Anthrop. │     ├─ Caching       │
│  ├─ BERTSc.  │  ├─ Ans Rel. │  └─ Mock     │     ├─ Batching      │
│  └─ EmbSim   │  ├─ Ctx Pre. │              │     └─ GPU Support   │
│              │  ├─ Ctx Rec. │              │                       │
│              │  ├─ Ground.  │              │                       │
│              │  └─ Halluc.  │              │                       │
├──────────────┴──────────────┴──────────────┴───────────────────────┤
│                      Core Layer                                     │
│     BaseMetric (ABC) ─ MetricRegistry ─ DatasetLoader              │
├─────────────────────────────────────────────────────────────────────┤
│               Reporting + Visualization Layer                       │
│  Markdown │ HTML │ JSON │ CSV │ 9 Chart Types │ Executive Summary   │
├─────────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                              │
│  Pydantic v2 Schemas │ Config (YAML/JSON) │ Loguru │ Exceptions     │
└─────────────────────────────────────────────────────────────────────┘
```

### Folder Structure

```
llm-evaluation-framework/
├── .github/workflows/ci.yml          # GitHub Actions CI/CD pipeline
├── Dockerfile                         # Multi-stage Docker build
├── docker-compose.yml                 # Container orchestration
├── .dockerignore                      # Build context exclusions
├── pyproject.toml                     # Poetry project definition
├── config.example.yaml                # Annotated configuration example
├── README.md
├── src/
│   └── llm_eval/
│       ├── __init__.py                # Package metadata & version
│       ├── benchmark_dataset.jsonl    # 30-sample benchmark (8 difficulty tiers)
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py               # Typer CLI (8 commands)
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py            # Pydantic v2 configuration schemas
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base_metric.py         # BaseMetric ABC + MetricRegistry
│       │   └── data_loader.py         # JSONL/CSV dataset loader
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── service.py             # SentenceTransformer embedding service
│       ├── exceptions/
│       │   ├── __init__.py
│       │   └── base.py                # Domain exception hierarchy (8 types)
│       ├── judge/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseLLMJudge ABC + JudgeVerdict
│       │   └── providers.py           # OpenAI, Anthropic, Mock + JSON recovery
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── classical.py           # BLEU, ROUGE-L
│       │   ├── semantic.py            # BERTScore, Embedding Similarity
│       │   └── judge_metric.py        # LLM-as-a-Judge metric
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── runner.py              # ThreadPool evaluation pipeline
│       ├── rag/
│       │   ├── __init__.py
│       │   └── metrics.py             # 7 RAG metrics
│       ├── reporting/
│       │   ├── __init__.py
│       │   └── generator.py           # Multi-format report generator
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── evaluation.py          # Pydantic v2 domain models
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logger.py              # Loguru structured logging
│       │   └── stats.py               # Statistical computation
│       └── visualization/
│           ├── __init__.py
│           └── engine.py              # 9 chart-type visual analytics
└── tests/
    ├── conftest.py                    # Shared fixtures + mocked embeddings
    ├── test_cli.py
    ├── test_config.py
    ├── test_data_loader.py
    ├── test_embeddings.py
    ├── test_exceptions.py
    ├── test_judge.py
    ├── test_judge_metric.py
    ├── test_metrics_classical.py
    ├── test_metrics_semantic.py
    ├── test_pipeline.py
    ├── test_rag_metrics.py
    ├── test_reporting.py
    ├── test_stats.py
    └── test_visualization.py
```


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

```bash
# Clone the repository
git clone https://github.com/enterprise/llm-evaluation-framework.git
cd llm-evaluation-framework

# Install all dependencies
poetry install
```

### Run the Built-in Benchmark

```bash
poetry run llm-eval benchmark
```

This evaluates 30 benchmark samples across all enabled metrics and generates reports + visualizations in `eval_reports/benchmark/`.

### Run a Custom Evaluation

```bash
poetry run llm-eval run \
  --dataset my_dataset.jsonl \
  --config config.yaml \
  --output-dir eval_results \
  --run-id my_experiment_v1
```


## 🛠️ CLI Command Reference

| Command | Description | Example |
|:---|:---|:---|
| `llm-eval run` | Execute full evaluation pipeline | `llm-eval run -d data.jsonl -c config.yaml` |
| `llm-eval validate` | Validate dataset schema compliance | `llm-eval validate -d data.jsonl` |
| `llm-eval benchmark` | Run against built-in 30-sample benchmark | `llm-eval benchmark` |
| `llm-eval metrics` | List all registered evaluation metrics | `llm-eval metrics` |
| `llm-eval report` | Generate reports from saved JSON results | `llm-eval report -i results.json` |
| `llm-eval visualize` | Generate charts from saved JSON results | `llm-eval visualize -i results.json` |
| `llm-eval version` | Print version information | `llm-eval version` |
| `llm-eval doctor` | Diagnostic health check of dependencies | `llm-eval doctor` |

### Global Options

```bash
llm-eval --verbose          # Enable debug logging
llm-eval --log-file run.log # Write logs to file
```


## ⚙️ Configuration

The framework supports **YAML** and **JSON** configuration files with automatic defaults and environment variable overrides.

### Configuration File

Copy the example and customize:

```bash
cp config.example.yaml config.yaml
```

### Environment Variable Overrides

All settings can be overridden with `LLM_EVAL_` prefixed environment variables. Nested keys use double underscores:

```bash
export LLM_EVAL_JUDGE__PROVIDER=openai
export LLM_EVAL_JUDGE__API_KEY=sk-your-key-here
export LLM_EVAL_PIPELINE__MAX_WORKERS=8
export LLM_EVAL_LOG_LEVEL=DEBUG
```

### Configuration Schema

```yaml
project_name: "My Evaluation"
log_level: INFO                    # DEBUG | INFO | WARNING | ERROR

judge:
  provider: mock                   # openai | anthropic | mock
  model_name: gpt-4o
  api_key: null                    # Or use LLM_EVAL_JUDGE__API_KEY env var
  temperature: 0.0
  max_tokens: 1024
  max_retries: 3

embeddings:
  model_name: all-MiniLM-L6-v2
  device: cpu                      # cpu | cuda | mps
  batch_size: 32

pipeline:
  max_workers: 4
  fail_on_sample_error: false
  cache_embeddings: true

reporting:
  output_dir: eval_reports
  formats: [json, markdown, html, csv]
  generate_plots: true

metrics:
  bleu:
    enabled: true
    threshold: 0.3
    params: { n_grams: 4 }
  rouge_l:
    enabled: true
    threshold: 0.4
  bert_score:
    enabled: true
    threshold: 0.7
  faithfulness:
    enabled: true
    threshold: 0.8
  # ... see config.example.yaml for all options
```

## 📊 Metrics Reference

### Classical Metrics

| Metric | Description | Range | Key |
|:---|:---|:---|:---|
| **BLEU** | N-gram precision overlap between output and reference | 0.0 – 1.0 | `bleu` |
| **ROUGE-L** | Longest Common Subsequence F1 score | 0.0 – 1.0 | `rouge_l` |

### Semantic Metrics

| Metric | Description | Range | Key |
|:---|:---|:---|:---|
| **BERTScore** | Token-level semantic similarity F1 via dense embeddings | 0.0 – 1.0 | `bert_score` |
| **Embedding Similarity** | Cosine similarity between sentence embeddings | 0.0 – 1.0 | `embedding_similarity` |

### RAG Metrics

| Metric | Description | Range | Key |
|:---|:---|:---|:---|
| **Faithfulness** | Fraction of output sentences grounded in retrieved context | 0.0 – 1.0 | `faithfulness` |
| **Context Relevancy** | Average semantic similarity of contexts to the query | 0.0 – 1.0 | `context_relevancy` |
| **Answer Relevancy** | Semantic similarity of output to input query | 0.0 – 1.0 | `answer_relevancy` |
| **Context Precision** | Position-weighted precision of relevant contexts | 0.0 – 1.0 | `context_precision` |
| **Context Recall** | Coverage of ground-truth claims by contexts | 0.0 – 1.0 | `context_recall` |
| **Groundedness** | Per-claim traceability to individual context passages | 0.0 – 1.0 | `groundedness` |
| **Hallucination Score** | Inverse hallucination ratio (1.0 = no hallucination) | 0.0 – 1.0 | `hallucination_score` |

### LLM-as-a-Judge

| Metric | Description | Range | Key |
|:---|:---|:---|:---|
| **LLM Judge** | External LLM quality evaluation with bias mitigation | 0.0 – 1.0 | `llm_judge` |

## 📈 Visual Analytics & Reports

The framework generates publication-quality outputs automatically:

### Report Formats
- **HTML** — Dark-themed executive dashboard with pass/fail badges and metric tables
- **Markdown** — GitHub-compatible report with ranking, statistics, and recommendations
- **JSON** — Machine-readable full evaluation payload
- **CSV** — Flat per-sample per-metric export for spreadsheet analysis

### Report Sections
- Executive summary with quality tiers
- Metric ranking (best → worst)
- Pass/fail analysis with rates
- Failure analysis (worst samples per weakest metric)
- Best performing examples
- Actionable recommendations

### 9 Visualization Chart Types
1. **Radar Chart** — Global mean performance summary
2. **Box Plot** — Score distributions with outlier detection
3. **Violin Plot** — Combined density and box plot
4. **Histogram** — Score frequency distributions per metric
5. **KDE Distribution** — Kernel density estimation curves
6. **Correlation Heatmap** — Inter-metric score correlation matrix
7. **Sample × Metric Heatmap** — Per-sample score matrix
8. **Failure Breakdown** — Pass/fail counts by metric
9. **Metric Comparison** — Grouped bars of mean/median/P25/P75


## 📋 Dataset Format

### JSONL Format (recommended)

```jsonl
{"sample_id": "s1", "input_text": "What is AI?", "actual_output": "AI is artificial intelligence.", "expected_output": "Artificial Intelligence.", "retrieved_contexts": ["AI studies intelligent agents."]}
```

### CSV Format

```csv
input_text,actual_output,expected_output,retrieved_contexts
"What is AI?","AI is artificial intelligence.","Artificial Intelligence.","[\"AI studies intelligent agents.\"]"
```

### Required Fields

| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `sample_id` | string | Auto-generated if absent | Unique identifier |
| `input_text` | string | ✅ | Input prompt or query |
| `actual_output` | string | ✅ | Model-generated response |
| `expected_output` | string | Optional | Ground truth reference |
| `retrieved_contexts` | list[str] | Optional | RAG context passages |
| `tools_called` | list[dict] | Optional | Tool invocation logs |
| `metadata` | dict | Optional | Custom tags |


## 🐳 Docker Deployment

### Build and Run

```bash
docker-compose up --build
```

### Custom Evaluation via Docker

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/eval_reports:/app/eval_reports \
  -e LLM_EVAL_JUDGE__API_KEY=sk-your-key \
  llm-eval run --dataset /app/data/my_dataset.jsonl
```

### Features
- Multi-stage build for minimal image size
- `HEALTHCHECK` instruction
- Volume mounts for data and reports
- Environment variable configuration

## 🧪 Testing

### Run Full Test Suite

```bash
poetry run pytest tests/ -v --cov=src/llm_eval --cov-report=term-missing
```

### Test Architecture
- **14 test modules** covering all components
- **Mocked embedding service** (auto-use fixture — no model downloads)
- **Deterministic pseudo-embeddings** for reproducible results
- Tests cover: config, schemas, data loading, all 12 metrics, judge, pipeline, reporting, visualization, CLI, exceptions, embeddings, statistics

## 🔧 Developer Guide

### Adding a Custom Metric

1. Create a new file in `src/llm_eval/metrics/`:

```python
from llm_eval.core.base_metric import BaseMetric, MetricRegistry
from llm_eval.schemas.evaluation import EvaluationSample

@MetricRegistry.register("my_metric")
class MyMetric(BaseMetric):
    metric_name = "my_metric"
    description = "My custom evaluation metric"

    def _compute(self, sample: EvaluationSample):
        # Your metric logic here
        score = 0.85
        reasoning = "Computed via custom logic"
        details = {"custom_field": "value"}
        return score, reasoning, details
```

2. Import it in `src/llm_eval/metrics/__init__.py`
3. Add to config:

```yaml
metrics:
  my_metric:
    enabled: true
    threshold: 0.7
```

### Code Quality

```bash
# Lint
poetry run ruff check src/ tests/

# Format
poetry run ruff format src/ tests/

# Type check
poetry run mypy src/ --ignore-missing-imports

# Security scan
poetry run bandit -r src/ -ll
```

## 🤝 Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-metric`
3. Write code following existing patterns
4. Add tests in the appropriate `tests/test_*.py` file
5. Ensure all checks pass: `poetry run ruff check && poetry run pytest`
6. Submit a pull request

### Standards
- All code must be typed (MyPy strict)
- All public APIs must have docstrings
- All metrics must register via `@MetricRegistry.register()`
- All new features must have tests
- Follow SOLID, DRY, KISS principles

## ❓ FAQ & Troubleshooting

### Q: How do I use GPU for embeddings?

Set `device: cuda` in config or `LLM_EVAL_EMBEDDINGS__DEVICE=cuda`:

```yaml
embeddings:
  device: cuda    # or 'mps' for Apple Silicon
```

### Q: How do I use a real LLM judge instead of the mock?

```yaml
judge:
  provider: openai    # or 'anthropic'
  model_name: gpt-4o
  api_key: sk-your-key-here
```

Or via environment:

```bash
export LLM_EVAL_JUDGE__PROVIDER=openai
export LLM_EVAL_JUDGE__API_KEY=sk-your-key
```

### Q: My evaluation crashes on one bad sample. How do I continue?

Set `fail_on_sample_error: false` (default). The pipeline isolates failures per-sample and continues.

### Q: How do I evaluate only specific metrics?

Disable unwanted metrics in config:

```yaml
metrics:
  bleu:
    enabled: true
  rouge_l:
    enabled: false    # Disabled
```

### Q: NLTK data not found error?

The framework auto-downloads required NLTK data. If behind a firewall:

```python
import nltk
nltk.download('punkt', download_dir='/path/to/nltk_data')
```

### Q: How do I generate reports from a previous run?

```bash
llm-eval report --input eval_reports/report_eval_run.json --output-dir new_reports/
llm-eval visualize --input eval_reports/report_eval_run.json --output-dir new_charts/
```

## 📄 License

This project is available under the MIT License.

## Author

MANIKANTA SURYASAI
AIML ENGINEER | DEVELOPER
