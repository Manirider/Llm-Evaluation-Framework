"""
Multi-format report generation engine supporting Markdown, HTML, JSON, and CSV.

Includes executive summary, metric ranking, failure analysis, best/worst
examples, and actionable recommendations.
"""

from __future__ import annotations

import csv
from pathlib import Path

from jinja2 import Template
from loguru import logger

from llm_eval.exceptions.base import ReportingError
from llm_eval.schemas.evaluation import (
    EvaluationRunReport,
    SampleEvaluationResult,
)

# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------


def _metric_quality_tier(mean: float) -> str:
    """Classify a metric mean score into a quality tier label."""
    if mean >= 0.9:
        return "Excellent"
    if mean >= 0.75:
        return "Good"
    if mean >= 0.6:
        return "Fair"
    if mean >= 0.4:
        return "Poor"
    return "Critical"


def _compute_pass_fail(report: EvaluationRunReport) -> dict[str, dict[str, int]]:
    """Compute pass/fail counts per metric across all samples."""
    counts: dict[str, dict[str, int]] = {}
    for res in report.sample_results:
        for m_name, m_res in res.metrics.items():
            if m_name not in counts:
                counts[m_name] = {"pass": 0, "fail": 0, "n/a": 0}
            if m_res.passed is True:
                counts[m_name]["pass"] += 1
            elif m_res.passed is False:
                counts[m_name]["fail"] += 1
            else:
                counts[m_name]["n/a"] += 1
    return counts


def _worst_n(
    report: EvaluationRunReport, metric_name: str, n: int = 3
) -> list[SampleEvaluationResult]:
    """Return the N lowest-scoring samples for a given metric."""
    scored = [
        (res, res.metrics[metric_name].score)
        for res in report.sample_results
        if metric_name in res.metrics
    ]
    scored.sort(key=lambda x: x[1])
    return [r for r, _ in scored[:n]]


def _best_n(
    report: EvaluationRunReport, metric_name: str, n: int = 3
) -> list[SampleEvaluationResult]:
    """Return the N highest-scoring samples for a given metric."""
    scored = [
        (res, res.metrics[metric_name].score)
        for res in report.sample_results
        if metric_name in res.metrics
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:n]]


def _generate_recommendations(
    report: EvaluationRunReport,
) -> list[str]:
    """Generate actionable recommendations based on metric statistics."""
    recs: list[str] = []
    for m_name, stats in report.metric_summary.items():
        tier = _metric_quality_tier(stats.mean)
        if tier in ("Critical", "Poor"):
            recs.append(
                f"⚠️  **{m_name}** mean={stats.mean:.4f} ({tier}): "
                f"Investigate root cause. Review worst-performing samples for patterns."
            )
        if stats.std_dev > 0.25:
            recs.append(
                f"📊 **{m_name}** has high variance (σ={stats.std_dev:.4f}): "
                f"Consider stratifying evaluation by difficulty or input type."
            )
        if stats.skewness < -1.0:
            recs.append(
                f"📉 **{m_name}** is left-skewed (skew={stats.skewness:.2f}): "
                f"A minority of samples are dragging scores down significantly."
            )
    if not recs:
        recs.append("✅ All metrics within acceptable ranges. No critical actions required.")
    return recs


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


HTML_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Evaluation Report - {{ report.run_id }}</title>
    <style>
        :root { --bg: #0f172a; --surface: #1e293b; --border: #334155; --text: #e2e8f0; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --yellow: #eab308; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
        .container { max-width: 1280px; margin: 0 auto; padding: 2rem; }
        h1 { font-size: 1.75rem; color: #f8fafc; border-bottom: 2px solid var(--accent); padding-bottom: 0.75rem; margin-bottom: 1.5rem; }
        h2 { font-size: 1.25rem; color: #cbd5e1; margin: 2rem 0 1rem; border-left: 4px solid var(--accent); padding-left: 0.75rem; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
        .stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
        table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.875rem; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: rgba(59,130,246,0.1); color: #93c5fd; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
        tr:hover { background: rgba(59,130,246,0.05); }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .badge-pass { background: rgba(34,197,94,0.15); color: var(--green); }
        .badge-fail { background: rgba(239,68,68,0.15); color: var(--red); }
        .badge-tier { background: rgba(59,130,246,0.15); color: var(--accent); }
        .rec { padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
        code { background: rgba(59,130,246,0.1); padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 LLM Evaluation Report</h1>

        <div class="card">
            <div class="grid">
                <div><div class="stat-label">Run ID</div><div class="stat-value" style="font-size:1rem">{{ report.run_id }}</div></div>
                <div><div class="stat-label">Timestamp</div><div class="stat-value" style="font-size:1rem">{{ report.timestamp.strftime('%Y-%m-%d %H:%M UTC') }}</div></div>
                <div><div class="stat-label">Dataset Size</div><div class="stat-value">{{ report.dataset_size }}</div></div>
                <div><div class="stat-label">Duration</div><div class="stat-value">{{ report.execution_duration_seconds }}s</div></div>
            </div>
        </div>

        <h2>📊 Metric Aggregates</h2>
        <div class="card">
            <table>
                <thead><tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>95% CI</th><th>Tier</th></tr></thead>
                <tbody>
                {% for name, stats in report.metric_summary.items() %}
                <tr>
                    <td><strong>{{ name }}</strong></td>
                    <td>{{ "%.4f"|format(stats.mean) }}</td>
                    <td>{{ "%.4f"|format(stats.median) }}</td>
                    <td>{{ "%.4f"|format(stats.std_dev) }}</td>
                    <td>{{ "%.4f"|format(stats.min) }}</td>
                    <td>{{ "%.4f"|format(stats.max) }}</td>
                    <td>[{{ "%.4f"|format(stats.ci_95_lower) }}, {{ "%.4f"|format(stats.ci_95_upper) }}]</td>
                    <td><span class="badge badge-tier">{{ tiers[name] }}</span></td>
                </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        <h2>✅ Pass / Fail Summary</h2>
        <div class="card">
            <table>
                <thead><tr><th>Metric</th><th>Pass</th><th>Fail</th><th>N/A</th><th>Pass Rate</th></tr></thead>
                <tbody>
                {% for m_name, c in pass_fail.items() %}
                <tr>
                    <td><strong>{{ m_name }}</strong></td>
                    <td><span class="badge badge-pass">{{ c.pass }}</span></td>
                    <td><span class="badge badge-fail">{{ c.fail }}</span></td>
                    <td>{{ c['n/a'] }}</td>
                    <td>{{ "%.1f"|format(c.pass / (c.pass + c.fail) * 100 if (c.pass + c.fail) > 0 else 0) }}%</td>
                </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        <h2>📋 Sample Results</h2>
        <div class="card" style="overflow-x:auto">
            <table>
                <thead><tr><th>Sample ID</th><th>Input</th><th>Output</th><th>Metric Scores</th></tr></thead>
                <tbody>
                {% for res in report.sample_results %}
                <tr>
                    <td><code>{{ res.sample_id }}</code></td>
                    <td>{{ res.sample.input_text[:80] }}{% if res.sample.input_text|length > 80 %}…{% endif %}</td>
                    <td>{{ res.sample.actual_output[:80] }}{% if res.sample.actual_output|length > 80 %}…{% endif %}</td>
                    <td>{% for m_name, m_val in res.metrics.items() %}<div><strong>{{ m_name }}:</strong> {{ "%.4f"|format(m_val.score) }}{% if m_val.passed is not none %} <span class="badge {{ 'badge-pass' if m_val.passed else 'badge-fail' }}">{{ '✓' if m_val.passed else '✗' }}</span>{% endif %}</div>{% endfor %}</td>
                </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        {% if recommendations %}
        <h2>💡 Recommendations</h2>
        <div class="card">{% for r in recommendations %}<div class="rec">{{ r }}</div>{% endfor %}</div>
        {% endif %}
    </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """
    Export evaluation run reports to Markdown, HTML, JSON, and CSV.
    Includes executive summary, metric ranking, failure analysis, best/worst
    examples, and actionable recommendations.
    """

    def __init__(self, output_dir: Path | str = "eval_reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self, report: EvaluationRunReport, base_name: str | None = None
    ) -> dict[str, Path]:
        """Generate reports in all supported file formats."""
        prefix = base_name or f"report_{report.run_id}"
        generated: dict[str, Path] = {}

        try:
            generated["json"] = self.to_json(report, self.output_dir / f"{prefix}.json")
            generated["markdown"] = self.to_markdown(report, self.output_dir / f"{prefix}.md")
            generated["html"] = self.to_html(report, self.output_dir / f"{prefix}.html")
            generated["csv"] = self.to_csv(report, self.output_dir / f"{prefix}.csv")
        except Exception as e:
            raise ReportingError(f"Report generation failed: {e}") from e

        logger.info(f"Generated evaluation reports in directory: {self.output_dir}")
        return generated

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, report: EvaluationRunReport, file_path: Path) -> Path:
        content = report.model_dump_json(indent=2)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def to_markdown(self, report: EvaluationRunReport, file_path: Path) -> Path:
        lines: list[str] = [
            f"# LLM Evaluation Report: `{report.run_id}`",
            "",
            "## Executive Summary",
            "",
            f"- **Timestamp**: {report.timestamp.isoformat()}",
            f"- **Dataset Size**: {report.dataset_size} samples",
            f"- **Metrics Evaluated**: {len(report.configured_metrics)}",
            f"- **Execution Time**: {report.execution_duration_seconds}s",
            "",
        ]

        # Metric ranking by mean score
        ranked = sorted(report.metric_summary.items(), key=lambda x: x[1].mean, reverse=True)
        lines.extend(
            [
                "## Metric Ranking (Best → Worst)",
                "",
                "| Rank | Metric | Mean | Tier |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for rank, (m_name, stats) in enumerate(ranked, 1):
            tier = _metric_quality_tier(stats.mean)
            lines.append(f"| {rank} | `{m_name}` | {stats.mean:.4f} | {tier} |")

        # Full statistics table
        lines.extend(
            [
                "",
                "## Detailed Metric Statistics",
                "",
                "| Metric | Mean | Median | Std Dev | Min | Max | P25 | P75 | 95% CI |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for m_name, stats in report.metric_summary.items():
            lines.append(
                f"| `{m_name}` | {stats.mean:.4f} | {stats.median:.4f} | "
                f"{stats.std_dev:.4f} | {stats.min:.4f} | {stats.max:.4f} | "
                f"{stats.p25:.4f} | {stats.p75:.4f} | "
                f"[{stats.ci_95_lower:.4f}, {stats.ci_95_upper:.4f}] |"
            )

        # Pass/Fail summary
        pass_fail = _compute_pass_fail(report)
        lines.extend(
            [
                "",
                "## Pass / Fail Analysis",
                "",
                "| Metric | Pass | Fail | Pass Rate |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for m_name, c in pass_fail.items():
            total = c["pass"] + c["fail"]
            rate = (c["pass"] / total * 100) if total > 0 else 0
            lines.append(f"| `{m_name}` | {c['pass']} | {c['fail']} | {rate:.1f}% |")

        # Failure analysis — worst examples per weakest metric
        if ranked:
            weakest_metric = ranked[-1][0]
            worst = _worst_n(report, weakest_metric, n=3)
            lines.extend(
                [
                    "",
                    f"## Failure Analysis — Worst Samples for `{weakest_metric}`",
                    "",
                ]
            )
            for sample_res in worst:
                m_res = sample_res.metrics.get(weakest_metric)
                score_str = f"{m_res.score:.4f}" if m_res else "N/A"
                lines.append(
                    f"- **{sample_res.sample_id}** (score={score_str}): "
                    f"`{sample_res.sample.input_text[:60]}…`"
                )
                if m_res and m_res.reasoning:
                    lines.append(f"  - Reasoning: {m_res.reasoning[:120]}")

            # Best examples for strongest metric
            strongest_metric = ranked[0][0]
            best = _best_n(report, strongest_metric, n=3)
            lines.extend(
                [
                    "",
                    f"## Best Samples for `{strongest_metric}`",
                    "",
                ]
            )
            for sample_res in best:
                m_res = sample_res.metrics.get(strongest_metric)
                score_str = f"{m_res.score:.4f}" if m_res else "N/A"
                lines.append(
                    f"- **{sample_res.sample_id}** (score={score_str}): "
                    f"`{sample_res.sample.input_text[:60]}…`"
                )

        # Recommendations
        recs = _generate_recommendations(report)
        lines.extend(["", "## Recommendations", ""])
        for rec in recs:
            lines.append(f"- {rec}")

        # Per-sample detail table
        lines.extend(
            [
                "",
                "## Per-Sample Results",
                "",
                "| Sample ID | Metric | Score | Passed | Reasoning |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for res in report.sample_results:
            for m_name, m_res in res.metrics.items():
                passed_str = (
                    "N/A" if m_res.passed is None else ("✅ Pass" if m_res.passed else "❌ Fail")
                )
                reasoning = (m_res.reasoning or "").replace("\n", " ")[:100]
                lines.append(
                    f"| `{res.sample_id}` | `{m_name}` | {m_res.score:.4f} | "
                    f"{passed_str} | {reasoning} |"
                )

        file_path.write_text("\n".join(lines), encoding="utf-8")
        return file_path

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def to_html(self, report: EvaluationRunReport, file_path: Path) -> Path:
        tiers = {
            m_name: _metric_quality_tier(stats.mean)
            for m_name, stats in report.metric_summary.items()
        }
        pass_fail = _compute_pass_fail(report)
        recs = _generate_recommendations(report)

        template = Template(HTML_REPORT_TEMPLATE)
        html_content = template.render(
            report=report,
            tiers=tiers,
            pass_fail=pass_fail,
            recommendations=recs,
        )
        file_path.write_text(html_content, encoding="utf-8")
        return file_path

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, report: EvaluationRunReport, file_path: Path) -> Path:
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_id",
                    "input_text",
                    "actual_output",
                    "expected_output",
                    "metric_name",
                    "score",
                    "passed",
                    "reasoning",
                ]
            )
            for res in report.sample_results:
                for m_name, m_res in res.metrics.items():
                    writer.writerow(
                        [
                            res.sample_id,
                            res.sample.input_text,
                            res.sample.actual_output,
                            res.sample.expected_output or "",
                            m_name,
                            m_res.score,
                            m_res.passed,
                            m_res.reasoning or "",
                        ]
                    )
        return file_path
