"""Integration tests for the LlamaSum visualization functionality."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from llamasum.benchmark import run_benchmark
from llamasum.visualize import (
    load_multiple_results,
    plot_document_metrics,
    plot_metrics_comparison,
    plot_rouge_breakdown,
)

# Skip all tests if visualization dependencies are not available
pytestmark = pytest.mark.skipif(
    not VISUALIZATION_AVAILABLE, reason="Visualization dependencies not available"
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return {
        "documents": [
            {
                "id": "doc1",
                "text": (
                    "LlamaSum is a text summarization library. It provides multiple "
                    "approaches for generating summaries. It can handle both single "
                    "documents and collections of related documents. The library "
                    "includes extractive and abstractive summarization methods."
                ),
                "reference_summary": "LlamaSum is a text summarization library with multiple approaches.",
            },
            {
                "id": "doc2",
                "text": (
                    "Visualization tools help interpret benchmark results. They provide "
                    "graphical representations of performance metrics. Charts and graphs "
                    "make it easier to compare different models and configurations."
                ),
                "reference_summary": "Visualization tools help interpret benchmark results through graphical representations.",
            },
        ]
    }


@pytest.fixture
def benchmark_results(sample_dataset):
    """Run benchmarks and return the results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset file
        dataset_path = os.path.join(tmpdir, "dataset.json")
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(sample_dataset, f)

        # Run benchmarks with different configurations
        results = []

        # Basic summarizer with position extraction
        result_path_1 = os.path.join(tmpdir, "results_basic_position.json")
        result1 = run_benchmark(
            dataset_path=dataset_path,
            output_path=result_path_1,
            summarizer_type="basic",
            extractive_method="position",
            extractive_ratio=0.3,
        )
        results.append(result1)

        # Basic summarizer with tfidf extraction
        result_path_2 = os.path.join(tmpdir, "results_basic_tfidf.json")
        result2 = run_benchmark(
            dataset_path=dataset_path,
            output_path=result_path_2,
            summarizer_type="basic",
            extractive_method="tfidf",
            extractive_ratio=0.3,
        )
        results.append(result2)

        return results


class TestVisualizationIntegration:
    """Integration tests for visualization functionality."""

    def test_e2e_visualization_workflow(self, benchmark_results):
        """Test the end-to-end visualization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create metrics comparison
            comparison_path = os.path.join(tmpdir, "comparison.png")
            fig1 = plot_metrics_comparison(
                benchmark_results, output_path=comparison_path
            )

            # Check that the file exists and is a valid image
            assert os.path.exists(comparison_path)
            assert os.path.getsize(comparison_path) > 0
            assert fig1 is not None

            # Create document metrics visualization
            doc_metrics_path = os.path.join(tmpdir, "doc_metrics.png")
            fig2 = plot_document_metrics(
                benchmark_results[0], output_path=doc_metrics_path
            )

            # Check that the file exists and is a valid image
            assert os.path.exists(doc_metrics_path)
            assert os.path.getsize(doc_metrics_path) > 0
            assert fig2 is not None

            # Create ROUGE breakdown visualization
            rouge_path = os.path.join(tmpdir, "rouge_breakdown.png")
            fig3 = plot_rouge_breakdown(benchmark_results[0], output_path=rouge_path)

            # Check that the file exists and is a valid image
            assert os.path.exists(rouge_path)
            assert os.path.getsize(rouge_path) > 0
            assert fig3 is not None

            # Clean up
            plt.close("all")

    def test_custom_advanced_visualization(self, benchmark_results):
        """Test creating custom advanced visualizations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create DataFrame for analysis
            df_results = pd.DataFrame(
                [
                    {
                        "model": r.get("summarizer_type", "unknown"),
                        "method": r.get("config", {}).get("extractive_method", ""),
                        **r.get("overall_metrics", {}),
                    }
                    for r in benchmark_results
                ]
            )

            # Create a custom visualization (performance comparison)
            plt.figure(figsize=(8, 6))
            metrics = ["avg_overall_score", "avg_rouge_1_f"]
            metrics = [m for m in metrics if m in df_results.columns]

            if metrics:
                ax = df_results.plot(
                    x="method", y=metrics, kind="bar", figsize=(8, 6), width=0.7
                )

                plt.title("Performance Comparison")
                plt.xlabel("Extraction Method")
                plt.ylabel("Score")
                plt.legend(title="Metric")
                plt.tight_layout()

                custom_viz_path = os.path.join(tmpdir, "custom_viz.png")
                plt.savefig(custom_viz_path)

                # Check that the file exists and is a valid image
                assert os.path.exists(custom_viz_path)
                assert os.path.getsize(custom_viz_path) > 0

            # Clean up
            plt.close("all")

    def test_save_and_load_results(self, benchmark_results):
        """Test saving and loading benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save results to files
            result_paths = []
            for i, result in enumerate(benchmark_results):
                result_path = os.path.join(tmpdir, f"result_{i}.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                result_paths.append(result_path)

            # Load results from files
            loaded_results = load_multiple_results(result_paths)

            # Check that the loaded results match the original results
            assert len(loaded_results) == len(benchmark_results)

            for i, (loaded, original) in enumerate(
                zip(loaded_results, benchmark_results)
            ):
                assert loaded["summarizer_type"] == original["summarizer_type"]
                assert (
                    loaded["config"]["extractive_method"]
                    == original["config"]["extractive_method"]
                )
                assert loaded["document_count"] == original["document_count"]

                # Check that metrics are the same
                for metric, value in original["overall_metrics"].items():
                    assert metric in loaded["overall_metrics"]
                    assert loaded["overall_metrics"][metric] == pytest.approx(
                        value, abs=1e-6
                    )

    def test_command_line_simulation(self, benchmark_results):
        """Simulate command-line usage by calling main function."""
        import sys

        from llamasum.visualize import main

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save results to files
            result_paths = []
            for i, result in enumerate(benchmark_results):
                result_path = os.path.join(tmpdir, f"result_{i}.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                result_paths.append(result_path)

            # Create output directory
            output_dir = os.path.join(tmpdir, "viz_output")
            os.makedirs(output_dir, exist_ok=True)

            # Prepare command-line arguments
            sys_argv_backup = sys.argv
            try:
                sys.argv = [
                    "llamasum-viz",
                    *result_paths,
                    "--output",
                    output_dir,
                    "--type",
                    "all",
                ]

                # Run main function
                exit_code = main()

                # Check that the function ran successfully
                assert exit_code == 0

                # Check that output files were created
                assert os.path.exists(os.path.join(output_dir, "comparison.png"))
                assert os.listdir(output_dir)  # Check that directory is not empty
            finally:
                # Restore original sys.argv
                sys.argv = sys_argv_backup
