# LLM Evaluation Framework - Architecture Documentation

## Overview

This document describes the architecture of the LLM Evaluation Framework - a production-ready, enterprise-grade framework for evaluating Large Language Models across multiple dimensions including text generation quality, RAG quality, and LLM-as-a-Judge evaluations.

## Architecture Principles

### Core Principles
1. **SOLID Principles** - All modules follow SOLID principles
2. **Plugin Architecture** - Metrics, judges, and reporters are plugins
3. **Configuration-Driven** - All behavior controlled via YAML/JSON config
4. **Plugin Architecture** - Metrics, judges, reporters as plugins
5. **Async-First** - All I/O operations are async-first
2. **Type Safety** - Full type hints with mypy/pyright compliance
3. **Observability** - Structured logging, metrics, tracing built-in
4. **Resilience** - Retries, circuit breakers, graceful degradation
5. **Testability** - Dependency injection, interfaces, 95%+ coverage target
6. **Security First** - No secrets in code, secure defaults, input validation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer (Typer)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Pipeline Orchestrator                             │
├──────────────────┬──────────────────┬──────────────────┬──────────────────┤
│   Data Loader    │  Metric Engine   │   LLM Judge      │   RAG Metrics    │
│   (Plugins)      │   (Plugins)      │   (Providers)    │   (Plugins)      │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│      Embedding Engine          │    LLM Judge        │   Reporting      │
│    (Embeddings + Cache)        │   (Providers)       │   (Plugins)      │
├────────────────────────────────┼─────────────────────┼──────────────────┤
│              Aggregation & Statistics Engine                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                    Reporting & Visualization Engine                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                    Configuration & Validation Layer (Pydantic)              │
├─────────────────────────────────────────────────────────────────────────────┤
│              Core Infrastructure (Logging, Config, Exceptions, Caching)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Architecture

### 1. Core Layer (`src/llm_eval/core/`)
```
core/
├── __init__.py
├── base.py              # Abstract base classes (ABC)
├── registry.py          # Plugin registry pattern
├── config.py            # Configuration management
├── context.py           # Execution context
├── events.py            # Event system for hooks
├── metrics.py           # Metric base classes
├── models.py            # Core data models
├── pipeline.py          # Pipeline orchestration
├── runner.py            # Pipeline runner
├── context.py           # Execution context
└── exceptions.py        # Custom exceptions
```

**Key Components:**
- `Metric` - Abstract base class for all metrics
- `MetricResult` - Standardized metric result
- `EvaluationContext` - Execution context with config, cache, logger
- `Pipeline` - Pipeline definition and execution
- `PipelineRunner` - Pipeline execution engine with retries, retries
- `MetricRegistry` - Plugin registry for metrics
- `JudgeRegistry` - Plugin registry for LLM judges
- `ReporterRegistry` - Plugin registry for reporters
- `EventBus` - Event-driven hooks (on_start, on_metric_complete, etc.)

### 2. Configuration Layer (`src/llm_eval/config/`)
```
config/
├── __init__.py
├── models.py           # Pydantic models for config
├── loader.py           # YAML/JSON/env loader with validation
├── schemas.py          # JSON schemas for validation
├── validators.py       # Custom validators
└── defaults.py         # Default configurations
```

**Configuration Hierarchy:**
1. Default values (code)
2. Config file (YAML/JSON)
3. Environment variables (prefix: `LLM_EVAL_`)
4. CLI arguments (highest priority)

### 3. CLI Layer (`src/llm_eval/cli/`)
```
cli/
├── __init__.py
├── main.py             # Main Typer app
├── commands/
│   ├── __init__.py
│   ├── evaluate.py     # evaluate command
│   ├── benchmark.py    # benchmark command
│   ├── report.py       # report generation
│   ├── visualize.py    # visualization
│   ├── benchmark.py    # benchmark dataset
│   ├── validate.py     # config validation
│   ├── list.py         # list metrics/judges/reporters
│   └── init.py         # init config
├── callbacks.py        # CLI callbacks
├── formatters.py       # Output formatters
└── callbacks.py        # Callbacks
```

**Commands:**
- `llm-eval evaluate` - Run evaluation pipeline
- `llm-eval benchmark` - Run benchmark dataset
- `llm-eval report` - Generate reports
- `llm-eval visualize` - Generate visualizations
- `llm-eval validate` - Validate config
- `llm-eval list` - List available metrics/judges/reporters
- `llm-eval init` - Initialize config
- `llm-eval validate-config` - Validate config file

### 4. Data Layer (`src/llm_eval/core/data.py`, `src/llm_eval/benchmarks/`)
```
data/
├── __init__.py
├── loader.py           # Dataset loader with validation
├── validators.py       # Data validators
├── schemas.py          # Dataset schemas
├── transforms.py       # Data transformations
└── samplers.py         # Sampling strategies

benchmarks/
├── __init__.py
├── registry.py         # Benchmark registry
├── base.py             # Base benchmark class
├── loader.py           # Benchmark loader
└── datasets/
    ├── __init__.py
    ├── benchmark_v1.py     # 25+ benchmark examples
    ├── hallucination.py    # Hallucination test cases
    ├── multihop.py         # Multi-hop reasoning
    ├── hallucination.py    # Hallucination detection
    ├── ambiguous.py        # Ambiguous questions
    ├── noisy_retrieval.py  # Noisy retrieval
    ├── long_context.py     # Long context
    ├── short_context.py    # Short context
    └── conflicting.py      # Conflicting contexts
```

### 5. Metrics Engine (`src/llm_eval/metrics/`)
```
metrics/
├── __init__.py
├── base.py              # Base metric classes
├── registry.py          # Metric registry
├── registry.py          # Metric registry
├── bleu/
│   ├── __init__.py
│   ├── metric.py        # BLEU implementation
│   └── config.py        # BLEU config
├── rouge/
│   ├── __init__.py
│   ├── metric.py        # ROUGE implementation
│   └── config.py
├── bertscore/
│   ├── __init__.py
│   ├── metric.py        # BERTScore with caching
│   ├── embeddings.py    # Embedding management
│   └── config.py
├── rag/
│   ├── __init__.py
│   ├── base.py          # Base RAG metric
│   ├── faithfulness.py  # Faithfulness metric
│   ├── context_relevancy.py
│   ├── answer_relevancy.py
│   ├── context_precision.py
│   ├── context_recall.py
│   ├── groundedness.py
│   ├── hallucination.py
│   ├── base_llm.py      # LLM-based RAG metrics
│   ├── base_embedding.py # Embedding-based RAG metrics
│   └── hybrid.py        # Hybrid approach
├── statistical/
│   ├── __init__.py
│   ├── aggregator.py    # Statistical aggregator
│   ├── statistics.py    # Statistical functions
│   └── confidence.py    # Confidence intervals
└── custom/
    ├── __init__.py
    └── custom.py        # Custom metric template
```

### 6. Embeddings Engine (`src/llm_eval/embeddings/`)
```
embeddings/
├── __init__.py
├── base.py              # Embedding provider abstraction
├── cache.py             # Embedding cache (disk/memory)
├── providers/
│   ├── __init__.py
│   ├── sentence_transformers.py
│   ├── openai.py
│   ├── cohere.py
│   └── huggingface.py
├── cache/
│   ├── __init__.py
│   ├── disk.py          # Disk cache (SQLite/parquet)
│   ├── memory.py        # In-memory LRU cache
│   └── hybrid.py        # Hybrid cache
├── batch.py             # Batch processing
└── pool.py              # Embedding pool for reuse
```

### 7. LLM Judge (`src/llm_eval/judge/`)
```
judge/
├── __init__.py
├── base.py              # Base judge abstraction
├── registry.py          # Judge registry
├── schemas.py           # Judge schemas (JSON schema)
├── providers/
│   ├── __init__.py
│   ├── base.py          # Base provider
│   ├── openai.py        # OpenAI provider
│   ├── anthropic.py     # Anthropic provider
│   ├── base.py          # Base provider class
│   └── custom.py        # Custom provider template
├── retry.py             # Retry logic with backoff
├── schema_enforcer.py   # JSON schema enforcement
├── bias_mitigation.py   # Bias mitigation prompts
├── parser.py            # Response parser
└── prompts/
    ├── __init__.py
    ├── faithfulness.py
    ├── relevancy.py
    ├── hallucination.py
    └── templates/       # Prompt templates
```

### 8. RAG Metrics (`src/llm_eval/rag/`)
```
rag/
├── __init__.py
├── base.py              # Base RAG metric
├── faithfulness.py      # Faithfulness (answer grounded in context)
├── context_relevancy.py # Context relevancy
├── answer_relevancy.py  # Answer relevancy
├── context_precision.py # Context precision
├── context_recall.py    # Context recall
├── groundedness.py      # Groundedness
├── hallucination.py     # Hallucination detection
├── context_utilization.py # Context utilization
├── noise_sensitivity.py # Noise sensitivity
├── base_llm.py          # LLM-based implementations
├── base_embedding.py    # Embedding-based implementations
├── hybrid.py            # Hybrid implementations
├── judge_based.py       # LLM Judge based
└── utils.py             # RAG utilities
```

### 9. Pipeline Orchestration (`src/llm_eval/pipeline/`)
```
pipeline/
├── __init__.py
├── base.py              # Pipeline base
├── runner.py            # Pipeline runner
├── steps/
│   ├── __init__.py
│   ├── load_data.py     # Data loading step
│   ├── compute_metrics.py # Metric computation
│   ├── run_judges.py    # LLM judge execution
│   ├── aggregate.py     # Aggregation step
│   ├── report.py        # Reporting step
│   └── visualize.py     # Visualization step
├── orchestrator.py      # Pipeline orchestrator
├── context.py           # Pipeline context
├── hooks.py             # Pipeline hooks
├── checkpoint.py        # Checkpointing
├── parallel.py          # Parallel execution
└── recovery.py          # Failure recovery
```

### 10. Aggregation & Statistics (`src/llm_eval/metrics/statistical/`)
```
statistical/
├── __init__.py
├── aggregator.py        # Result aggregation
├── statistics.py        # Statistical functions
│   ├── mean, median, mode
│   ├── variance, std_dev
│   ├── percentiles
│   ├── confidence_intervals
│   ├── min, max
│   ├── skewness, kurtosis
├── confidence.py        # Confidence intervals
├── bootstrap.py         # Bootstrap sampling
└── comparison.py        # Statistical comparison
```

### 11. Reporting (`src/llm_eval/reporting/`)
```
reporting/
├── __init__.py
├── base.py              # Base reporter
├── registry.py          # Reporter registry
├── markdown/
│   ├── __init__.py
│   └── reporter.py      # Markdown reporter
├── json/
│   ├── __init__.py
│   └── reporter.py      # JSON reporter
├── csv/
│   ├── __init__.py
│   └── reporter.py      # CSV reporter
├── html/
│   ├── __init__.py
│   └── reporter.py      # HTML reporter
├── formatters/
│   ├── __init__.py
│   ├── tables.py        # Table formatters
│   ├── summary.py       # Executive summary
│   ├── ranking.py       # Metric ranking
│   ├── failures.py      # Failure analysis
│   ├── examples.py      # Best/worst examples
│   └── recommendations.py # Recommendations
└── templates/
    ├── markdown.j2
    ├── html.j2
    └── css/
```

### 12. Visualization (`src/llm_eval/visualization/`)
```
visualization/
├── __init__.py
├── base.py              # Base visualizer
├── registry.py          # Visualizer registry
├── plots/
│   ├── __init__.py
│   ├── histogram.py     # Histogram
│   ├── radar.py         # Radar chart
│   ├── boxplot.py       # Box plot
│   ├── violin.py        # Violin plot
│   heatmap.py           # Heatmap
│   ├── correlation.py   # Correlation matrix
│   ├── distribution.py  # Distribution curves
│   ├── failures.py      # Failure breakdown
│   └── comparison.py    # Metric comparison
├── exporters/
│   ├── __init__.py
│   ├── matplotlib.py    # Matplotlib backend
│   ├── plotly.py        # Plotly backend
│   └── static.py        # Static export
├── themes/
│   ├── __init__.py
│   ├── publication.py   # Publication theme
│   ├── dark.py          # Dark theme
│   └── light.py         # Light theme
└── export.py            # Export utilities
```

### 13. Utilities (`src/llm_eval/utils/`)
```
utils/
├── __init__.py
├── logging.py           # Loguru setup
├── caching.py           # Caching utilities
├── retries.py           # Retry utilities
├── validation.py        # Validation utilities
├── serialization.py     # Serialization
├── timing.py            # Timing utilities
├── batching.py          # Batch processing
├── parallel.py          # Parallel execution
├── env.py               # Environment utilities
└── paths.py             # Path utilities
```

### 14. Exceptions (`src/llm_eval/exceptions/`)
```
exceptions/
├── __init__.py
├── base.py              # Base exception
├── config.py            # Config errors
├── validation.py        # Validation errors
├── metric.py            # Metric errors
├── judge.py             # Judge errors
├── pipeline.py          # Pipeline errors
├── data.py              # Data errors
├── judge.py             # Judge errors
├── reporting.py         # Reporting errors
└── visualization.py     # Visualization errors
```

### 15. Schemas (`src/llm_eval/schemas/`)
```
schemas/
├── __init__.py
├── evaluation.py        # Evaluation schemas
├── metric.py            # Metric schemas
├── judge.py             # Judge schemas
├── dataset.py           # Dataset schemas
├── report.py            # Report schemas
├── visualization.py     # Visualization schemas
└── config.py            # Config schemas
```

### 16. Models (`src/llm_eval/models/`)
```
models/
├── __init__.py
├── evaluation.py        # Evaluation models
├── metric.py            # Metric models
├── dataset.py           # Dataset models
├── judge.py             # Judge models
├── report.py            # Report models
└── visualization.py     # Visualization models
```

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Config    │────▶│  Pipeline   │────▶│  Data Loader │────▶│  Dataset    │
│   (YAML)    │     │  Runner     │     │  (Loader)    │     │  (Samples)  │
└─────────────┘     └─────────────┘     └──────────────┘     └──────┬──────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Reports    │◀───▶│  Reporting  │◀───▶│  Aggregation │◀───▶│  Metric     │
│  (MD/JSON/  │     │  Engine     │     │  & Stats     │     │  Engine     │
│   CSV/HTML) │     └─────────────┘     └──────────────┘     └──────┬──────┘
└─────────────┘                                                      │
                                                                      │
                                                                      ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Visualize   │◀───▶│ Visualize   │◀───▶│  LLM Judge   │◀───▶│  RAG Metrics│
│  Engine     │     │  Engine     │     │  Engine      │     │  & Metrics  │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
```

## Plugin Architecture

### Metric Plugin Interface
```python
class Metric(ABC):
    name: str
    version: str
    config_class: Type[BaseModel]
    
    @abstractmethod
    async def compute(self, 
                      predictions: List[str], 
                      references: List[str], 
                      context: EvaluationContext) -> MetricResult:
        ...
    
    @abstractmethod
    async def compute_batch(self, 
                            samples: List[Sample], 
                            context: EvaluationContext) -> List[MetricResult]:
        ...
```

### Judge Plugin Interface
```python
class JudgeProvider(ABC):
    name: str
    version: str
    supported_models: List[str]
    
    @abstractmethod
    async def judge(self, 
                    prompt: JudgePrompt, 
                    schema: Type[BaseModel]) -> BaseModel:
        ...
    
    @abstractmethod
    async def judge_batch(self, 
                          prompts: List[JudgePrompt], 
                          schema: Type[BaseModel]) -> List[BaseModel]:
        ...
```

### Reporter Plugin Interface
```python
class Reporter(ABC):
    name: str
    version: str
    output_format: str
    
    @abstractmethod
    async def generate(self, 
                       results: EvaluationResults, 
                       output_path: Path) -> Path:
        ...
```

## Configuration Schema

### Top-Level Config
```yaml
evaluation:
  name: "my-evaluation"
  description: "Evaluation description"
  version: "1.0.0"

data:
  source: "path/to/dataset.jsonl"
  format: "jsonl"  # jsonl, json, csv, huggingface
  split: "test"
  sampling:
    strategy: "random"
    size: 100
    seed: 42

metrics:
  - name: "bleu"
    config:
      n_gram: 4
      smooth: true
  - name: "rouge"
    config:
      rouge_types: ["rouge1", "rouge2", "rougeL"]
  - name: "bertscore"
    config:
      model: "microsoft/deberta-xlarge-mnli"
      batch_size: 32
      device: "cuda"
  - name: "faithfulness"
    config:
      provider: "openai"
      model: "gpt-4"

judges:
  - name: "openai"
    config:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"
      temperature: 0
      max_retries: 3
      timeout: 60

pipeline:
  parallel: true
  max_workers: 4
  checkpoint: true
  checkpoint_dir: ".checkpoints"

reporting:
  formats: ["markdown", "json", "html", "csv"]
  output_dir: "./reports"
  include_visualizations: true

visualization:
  enabled: true
  formats: ["png", "html"]
  theme: "publication"
  dpi: 300

logging:
  level: "INFO"
  format: "json"
  file: "logs/evaluation.log"
  rotation: "10 MB"
  retention: "7 days"

cache:
  enabled: true
  backend: "disk"  # memory, disk, hybrid
  path: ".cache/embeddings"
  ttl: "7d"

performance:
  parallel: true
  max_workers: 4
  batch_size: 32
  cache_embeddings: true
  device: "auto"  # auto, cpu, cuda, mps

benchmarks:
  - name: "benchmark_v1"
    split: "test"
    metrics: ["faithfulness", "context_relevancy"]
```

## Security Architecture

### Secrets Management
- No secrets in code or config files
- Environment variables with `LLM_EVAL_` prefix
- Support for secret managers (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault)
- `.env` file support (gitignored)

### Input Validation
- All inputs validated via Pydantic models
- JSON schema validation for LLM judge outputs
- Input sanitization for file paths
- SQL injection prevention for cache

### Secure Defaults
- No telemetry by default
- No telemetry without explicit opt-in
- Secure defaults for timeouts, retries
- No telemetry without explicit opt-in

## Observability

### Structured Logging (Loguru)
```python
logger = logger.bind(
    evaluation_id="eval-123",
    metric="faithfulness",
    sample_id="sample-123"
)
logger.info("Computing faithfulness", 
            sample_id="sample-123",
            duration_ms=150)
```

### Metrics Collection
- Prometheus metrics endpoint
- Custom metrics for pipeline monitoring
- Latency histograms
- Error counters

### Tracing
- OpenTelemetry compatible
- Span for each metric computation
- Span for each LLM call
- Span for each pipeline step

## Performance Optimization

### Embedding Caching
- Disk cache (SQLite/Parquet) with TTL
- In-memory LRU cache
- Hybrid cache (memory + disk)
- Cache warming strategies

### Parallel Execution
- Thread pool for I/O bound (LLM calls)
- Process pool for CPU bound (metrics)
- Configurable worker pools
- Backpressure handling

### Batch Processing
- Automatic batching for embeddings
- Automatic batching for LLM judge calls
- Configurable batch sizes
- Memory-aware batching

### Embedding Reuse
- Embedding pool for reuse
- LRU eviction
- Persistent cache across runs

## Error Handling & Resilience

### Retry Logic
- Exponential backoff with jitter
- Configurable max retries
- Retry on specific exceptions
- Circuit breaker pattern

### Graceful Degradation
- Continue on metric failure
- Skip failed samples with logging
- Partial results on failure
- Checkpoint/resume capability

### Error Types
- `ConfigurationError` - Config issues
- `ValidationError` - Data validation
- `MetricError` - Metric computation
- `JudgeError` - LLM judge errors
- `PipelineError` - Pipeline execution
- `DataError` - Data loading/validation
- `ReportingError` - Report generation
- `VisualizationError` - Visualization

## Testing Strategy

### Test Pyramid
```
         ┌─────────────┐
         │  E2E Tests  │  ← 5%
         ├─────────────┤
         │ Integration │  ← 15%
         ├─────────────┤
         │   Unit      │  ← 80%
         └─────────────┘
```

### Test Categories
- **Unit Tests** (80%) - Individual units, mocked dependencies
- **Integration Tests** (15%) - Component integration, real deps
- **E2E Tests** (5%) - Full pipeline, real APIs (optional)
- **Contract Tests** - Plugin interfaces
- **Property Tests** - Property-based testing for metrics
- **Benchmark Tests** - Performance regression
- **Visual Regression** - Visualization output

### Coverage Targets
- Overall: 95%+
- Core: 98%+
- Metrics: 95%+
- CLI: 90%+
- Visualization: 85%+

## Deployment Architecture

### Docker
```
┌─────────────────────────────────────┐
│         Runtime Image (slim)        │
├─────────────────────────────────────┤
│  Python 3.11+ slim                  │
│  ┌─────────────────────────────┐   │
│  │  llm-eval (installed)       │   │
│  │  + dependencies             │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Multi-Stage Build
1. **Builder** - Compile dependencies, run tests
2. **Runtime** - Minimal runtime with only runtime deps

### Kubernetes
- Deployment with HPA
- ConfigMap for config
- Secret for secrets
- PVC for cache
- Prometheus monitoring

## Development Workflow

### Pre-commit Hooks
- Black formatting
- Ruff linting
- MyPy type checking
- Pyright type checking
- Bandit security
- Pre-commit hooks

### CI/CD Pipeline
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Lint      │──▶│  Type Check │──▶│   Tests     │──▶│  Coverage   │
│  (Ruff)     │   │  (MyPy)     │   │  (Pytest)   │   │  (95%+)     │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Docker    │◀──│  Security   │◀──│  Build      │◀──│  Artifacts  │
│  Build      │   │  Scan       │   │  Wheels     │   │  Upload     │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

## Future Extensibility

### Planned Extensions
1. **Custom Metric SDK** - SDK for writing custom metrics
2. **Judge Provider SDK** - SDK for new LLM providers
3. **Reporter SDK** - SDK for custom reporters
4. **Visualizer SDK** - SDK for custom visualizations
5. **Benchmark SDK** - SDK for custom benchmarks
6. **Streaming Evaluation** - Streaming evaluation support
7. **Distributed Evaluation** - Distributed evaluation
8. **Active Learning** - Active learning integration
9. **Model Comparison** - A/B testing framework
10. **Guardrails Integration** - Guardrails integration

## Appendix: Directory Structure

```
llm-evaluation-framework/
├── docs/
│   ├── architecture/
│   │   └── ARCHITECTURE.md
│   ├── user-guide/
│   ├── developer-guide/
│   └── api/
├── src/
│   └── llm_eval/
│       ├── __init__.py
│       ├── cli/
│       ├── config/
│       ├── core/
│       ├── pipeline/
│       ├── metrics/
│       ├── embeddings/
│       ├── judge/
│       ├── rag/
│       ├── pipeline/
│       ├── reporting/
│       ├── visualization/
│       ├── utils/
│       ├── exceptions/
│       ├── schemas/
│       ├── models/
│       └── benchmarks/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── fixtures/
│   └── benchmarks/
├── benchmarks/
│   └── datasets/
├── configs/
│   ├── defaults.yaml
│   ├── examples/
│   └── schemas/
├── scripts/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── .github/
│   └── workflows/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── .ruff.toml
├── .mypy.ini
├── pyrightconfig.json
└── bandit.yaml
```