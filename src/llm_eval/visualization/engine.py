"""
Publication-quality visual analytics engine using Matplotlib, Seaborn, and Plotly.

Generates nine chart types:
1. Radar chart — mean scores across all metrics
2. Box plot — score distributions with outlier detection
3. Correlation heatmap — inter-metric correlation matrix
4. Distribution curves — KDE density plots per metric
5. Histogram — score frequency distributions
6. Violin plot — combined density + box plot
7. Heatmap — sample × metric score matrix
8. Failure breakdown — bar chart of pass/fail per metric
9. Metric comparison — grouped bar chart of mean, median, p25, p75
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from llm_eval.exceptions.base import VisualizationError
from llm_eval.schemas.evaluation import EvaluationRunReport

# Force non-interactive backend for server/CI environments
matplotlib.use("Agg")


class VisualAnalyticsEngine:
    """
    Generates publication-quality charts and visualizations for evaluation runs.
    """

    _DPI: int = 300
    _PALETTE: str = "Set2"

    def __init__(self, output_dir: Path | str = "eval_visuals") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette=self._PALETTE)
        plt.rcParams.update(
            {
                "font.sans-serif": "DejaVu Sans",
                "figure.dpi": self._DPI,
                "savefig.bbox": "tight",
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all_visuals(self, report: EvaluationRunReport) -> dict[str, Path]:
        """Generate the complete set of analytics figures."""
        visuals: dict[str, Path] = {}
        df = self._report_to_dataframe(report)

        if df.empty:
            logger.warning("Empty dataframe from report; skipping visual generation.")
            return visuals

        try:
            visuals["radar"] = self.plot_radar_chart(report, self.output_dir / "radar_chart.png")
            visuals["boxplot"] = self.plot_score_boxplot(df, self.output_dir / "score_boxplot.png")
            visuals["correlation"] = self.plot_correlation_heatmap(
                df, self.output_dir / "correlation_heatmap.png"
            )
            visuals["distributions"] = self.plot_metric_distributions(
                df, self.output_dir / "metric_distributions.png"
            )
            visuals["histogram"] = self.plot_histogram(df, self.output_dir / "score_histogram.png")
            visuals["violin"] = self.plot_violin(df, self.output_dir / "score_violin.png")
            visuals["heatmap"] = self.plot_sample_heatmap(
                df, self.output_dir / "sample_heatmap.png"
            )
            visuals["failure"] = self.plot_failure_breakdown(
                report, self.output_dir / "failure_breakdown.png"
            )
            visuals["comparison"] = self.plot_metric_comparison(
                report, self.output_dir / "metric_comparison.png"
            )
        except Exception as e:
            raise VisualizationError(f"Chart generation failed: {e}") from e

        logger.info(f"Generated {len(visuals)} visual analytics charts in: {self.output_dir}")
        return visuals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _report_to_dataframe(self, report: EvaluationRunReport) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for res in report.sample_results:
            row: dict[str, object] = {"sample_id": res.sample_id}
            for m_name, m_res in res.metrics.items():
                row[m_name] = m_res.score
            records.append(row)
        return pd.DataFrame(records)

    @staticmethod
    def _metric_cols(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c != "sample_id"]

    # ------------------------------------------------------------------
    # 1. Radar Chart
    # ------------------------------------------------------------------

    def plot_radar_chart(self, report: EvaluationRunReport, file_path: Path) -> Path:
        """Radar chart of mean scores across all metrics."""
        labels = list(report.metric_summary.keys())
        values = [s.mean for s in report.metric_summary.values()]

        if not labels:
            return file_path

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_loop = values + [values[0]]
        angles_loop = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        ax.plot(angles_loop, values_loop, color="#2563eb", linewidth=2.5)
        ax.fill(angles_loop, values_loop, color="#3b82f6", alpha=0.2)
        ax.scatter(angles, values, color="#1d4ed8", s=50, zorder=5)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, size=10, weight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"Metric Performance Summary ({report.run_id})",
            size=14,
            weight="bold",
            pad=20,
        )

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 2. Box Plot
    # ------------------------------------------------------------------

    def plot_score_boxplot(self, df: pd.DataFrame, file_path: Path) -> Path:
        """Box plot showing score distribution per metric."""
        metric_cols = self._metric_cols(df)
        if not metric_cols:
            return file_path

        fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.2), 6))
        sns.boxplot(data=df[metric_cols], ax=ax, palette=self._PALETTE, width=0.5)
        sns.stripplot(
            data=df[metric_cols],
            ax=ax,
            color="black",
            alpha=0.3,
            jitter=0.2,
            size=4,
        )
        ax.set_title("Metric Score Distributions & Outliers", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score (0.0 – 1.0)", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        plt.xticks(rotation=20, fontweight="bold", fontsize=9)

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 3. Correlation Heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(self, df: pd.DataFrame, file_path: Path) -> Path:
        """Correlation matrix between evaluation metrics."""
        metric_cols = self._metric_cols(df)
        if len(metric_cols) < 2:
            return file_path

        fig, ax = plt.subplots(figsize=(max(7, len(metric_cols)), max(5, len(metric_cols) * 0.8)))
        corr = df[metric_cols].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
            cbar=True,
            square=True,
            linewidths=0.5,
        )
        ax.set_title("Metric Score Correlation Matrix", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 4. Distribution Curves (KDE)
    # ------------------------------------------------------------------

    def plot_metric_distributions(self, df: pd.DataFrame, file_path: Path) -> Path:
        """KDE kernel density distribution curves for each metric."""
        metric_cols = self._metric_cols(df)
        if not metric_cols:
            return file_path

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in metric_cols:
            sns.kdeplot(df[col].dropna(), ax=ax, label=col, fill=True, alpha=0.15, linewidth=2)

        ax.set_title("Metric Density Distributions (KDE)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.legend(title="Metrics", fontsize=9)

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 5. Histogram
    # ------------------------------------------------------------------

    def plot_histogram(self, df: pd.DataFrame, file_path: Path) -> Path:
        """Histogram of score frequency distributions per metric."""
        metric_cols = self._metric_cols(df)
        if not metric_cols:
            return file_path

        n_cols = min(3, len(metric_cols))
        n_rows = (len(metric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for idx, col in enumerate(metric_cols):
            row, c = divmod(idx, n_cols)
            ax = axes[row][c]
            ax.hist(df[col].dropna(), bins=15, color="#3b82f6", edgecolor="#1e40af", alpha=0.8)
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1.0)

        # Hide unused subplots
        for idx in range(len(metric_cols), n_rows * n_cols):
            row, c = divmod(idx, n_cols)
            axes[row][c].set_visible(False)

        fig.suptitle("Score Frequency Histograms", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 6. Violin Plot
    # ------------------------------------------------------------------

    def plot_violin(self, df: pd.DataFrame, file_path: Path) -> Path:
        """Violin plot combining density estimation and box plot."""
        metric_cols = self._metric_cols(df)
        if not metric_cols:
            return file_path

        melted = df[metric_cols].melt(var_name="Metric", value_name="Score")

        fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.2), 6))
        sns.violinplot(
            data=melted,
            x="Metric",
            y="Score",
            ax=ax,
            palette=self._PALETTE,
            inner="box",
            linewidth=1.2,
        )
        ax.set_title("Metric Score Violin Plots", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score (0.0 – 1.0)", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        plt.xticks(rotation=20, fontweight="bold", fontsize=9)

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 7. Sample × Metric Heatmap
    # ------------------------------------------------------------------

    def plot_sample_heatmap(self, df: pd.DataFrame, file_path: Path) -> Path:
        """Heatmap of scores across samples (rows) and metrics (columns)."""
        metric_cols = self._metric_cols(df)
        if not metric_cols:
            return file_path

        heatmap_df = (
            df.set_index("sample_id")[metric_cols] if "sample_id" in df.columns else df[metric_cols]
        )

        fig_height = max(4, len(heatmap_df) * 0.35)
        fig_width = max(6, len(metric_cols) * 1.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd_r",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.3,
            cbar_kws={"label": "Score"},
        )
        ax.set_title("Sample × Metric Score Heatmap", fontsize=14, fontweight="bold")
        ax.set_ylabel("Sample")

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 8. Failure Breakdown
    # ------------------------------------------------------------------

    def plot_failure_breakdown(self, report: EvaluationRunReport, file_path: Path) -> Path:
        """Stacked bar chart showing pass/fail counts per metric."""
        metrics: list[str] = []
        passes: list[int] = []
        fails: list[int] = []

        for m_name in report.configured_metrics:
            p_count = 0
            f_count = 0
            for res in report.sample_results:
                m_res = res.metrics.get(m_name)
                if m_res and m_res.passed is True:
                    p_count += 1
                elif m_res and m_res.passed is False:
                    f_count += 1
            metrics.append(m_name)
            passes.append(p_count)
            fails.append(f_count)

        if not metrics:
            return file_path

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.0), 6))
        ax.bar(x - width / 2, passes, width, label="Pass", color="#22c55e", alpha=0.85)
        ax.bar(x + width / 2, fails, width, label="Fail", color="#ef4444", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, fontweight="bold", fontsize=9)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Pass / Fail Breakdown by Metric", fontsize=14, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path

    # ------------------------------------------------------------------
    # 9. Metric Comparison (grouped bar)
    # ------------------------------------------------------------------

    def plot_metric_comparison(self, report: EvaluationRunReport, file_path: Path) -> Path:
        """Grouped bar chart comparing mean, median, P25, P75 per metric."""
        if not report.metric_summary:
            return file_path

        metrics = list(report.metric_summary.keys())
        means = [s.mean for s in report.metric_summary.values()]
        medians = [s.median for s in report.metric_summary.values()]
        p25s = [s.p25 for s in report.metric_summary.values()]
        p75s = [s.p75 for s in report.metric_summary.values()]

        x = np.arange(len(metrics))
        width = 0.2

        fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), 6))
        ax.bar(x - 1.5 * width, means, width, label="Mean", color="#3b82f6")
        ax.bar(x - 0.5 * width, medians, width, label="Median", color="#8b5cf6")
        ax.bar(x + 0.5 * width, p25s, width, label="P25", color="#f59e0b")
        ax.bar(x + 1.5 * width, p75s, width, label="P75", color="#10b981")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, fontweight="bold", fontsize=9)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title("Metric Statistical Comparison", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(file_path, dpi=self._DPI)
        plt.close(fig)
        return file_path
