"""Tests for the visualization module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

import llamasum
from llamasum.visualize import (
    create_comparison_dataframe,
    load_benchmark_results,
    load_multiple_results,
    plot_document_metrics,
    plot_metrics_comparison,
    plot_rouge_breakdown,
)

# Skip all tests if visualization dependencies are not available
pytestmark = pytest.mark.skipif(
    not VISUALIZATION_AVAILABLE, reason="Visualization dependencies not available"
)


class TestVisualizationFunctions:
    """Test visualization utility functions."""

    def test_load_benchmark_results(self):
        """Test loading benchmark results from JSON."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                {
                    "dataset": "test_dataset.json",
                    "summarizer_type": "basic",
                    "document_count": 2,
                    "overall_metrics": {
                        "avg_processing_time": 0.5,
                        "avg_compression_ratio": 0.3,
                    },
                },
                f,
            )
            temp_path = f.name

        try:
            # Load the results
            results = load_benchmark_results(temp_path)

            # Check the result
            assert results["dataset"] == "test_dataset.json"
            assert results["summarizer_type"] == "basic"
            assert results["document_count"] == 2
            assert results["overall_metrics"]["avg_processing_time"] == 0.5
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_load_multiple_results(self):
        """Test loading multiple benchmark results."""
        # Create temporary JSON files
        paths = []

        for i in range(2):
            with tempfile.NamedTemporaryFile(
                suffix=".json", mode="w", delete=False
            ) as f:
                json.dump(
                    {
                        "dataset": f"test_dataset_{i}.json",
                        "summarizer_type": "basic" if i == 0 else "hierarchical",
                        "document_count": i + 1,
                    },
                    f,
                )
                paths.append(f.name)

        try:
            # Load multiple results
            results = load_multiple_results(paths)

            # Check results
            assert len(results) == 2
            assert results[0]["dataset"] == "test_dataset_0.json"
            assert results[1]["summarizer_type"] == "hierarchical"
        finally:
            # Clean up
            for path in paths:
                os.unlink(path)

    def test_create_comparison_dataframe(self):
        """Test creating comparison DataFrame."""
        # Create sample results
        results = [
            {
                "dataset": "test_dataset.json",
                "summarizer_type": "basic",
                "document_count": 2,
                "config": {
                    "model_name": "model1",
                    "extractive_method": "tfidf",
                    "extractive_ratio": 0.3,
                },
                "overall_metrics": {"avg_processing_time": 0.5, "avg_rouge_1_f": 0.7},
            },
            {
                "dataset": "test_dataset.json",
                "summarizer_type": "hierarchical",
                "document_count": 2,
                "config": {"model_name": "model2", "extractive_method": "position"},
                "overall_metrics": {"avg_processing_time": 0.8, "avg_rouge_1_f": 0.6},
            },
        ]

        # Create DataFrame
        df = create_comparison_dataframe(results)

        # Check DataFrame
        assert df is not None
        assert len(df) == 2
        assert "display_name" in df.columns
        assert "avg_rouge_1_f" in df.columns
        assert df.iloc[0]["summarizer_type"] == "basic"
        assert df.iloc[1]["model_name"] == "model2"


@pytest.fixture
def sample_benchmark_results():
    """Create sample benchmark results for testing."""
    return {
        "dataset": "test_dataset.json",
        "summarizer_type": "basic",
        "config": {
            "model_name": "test_model",
            "extractive_method": "tfidf",
            "extractive_ratio": 0.3,
        },
        "document_count": 3,
        "total_time": 1.5,
        "document_results": {
            "doc1": {
                "summary": "This is a summary.",
                "processing_time": 0.5,
                "metrics": {
                    "overall_score": 0.8,
                    "compression_ratio": 0.3,
                    "density": 0.7,
                    "similarity_to_reference": 0.75,
                    "rouge": {
                        "rouge-1": {"p": 0.7, "r": 0.8, "f": 0.75},
                        "rouge-l": {"p": 0.6, "r": 0.7, "f": 0.65},
                    },
                },
            },
            "doc2": {
                "summary": "Another summary.",
                "processing_time": 0.4,
                "metrics": {
                    "overall_score": 0.7,
                    "compression_ratio": 0.25,
                    "density": 0.65,
                    "similarity_to_reference": 0.7,
                    "rouge": {
                        "rouge-1": {"p": 0.6, "r": 0.7, "f": 0.65},
                        "rouge-l": {"p": 0.5, "r": 0.6, "f": 0.55},
                    },
                },
            },
            "doc3": {
                "summary": "Third summary.",
                "processing_time": 0.6,
                "metrics": {
                    "overall_score": 0.75,
                    "compression_ratio": 0.35,
                    "density": 0.68,
                    "similarity_to_reference": 0.72,
                    "rouge": {
                        "rouge-1": {"p": 0.65, "r": 0.75, "f": 0.7},
                        "rouge-l": {"p": 0.55, "r": 0.65, "f": 0.6},
                    },
                },
            },
        },
        "overall_metrics": {
            "avg_processing_time": 0.5,
            "avg_compression_ratio": 0.3,
            "avg_overall_score": 0.75,
            "avg_rouge_1_f": 0.7,
            "avg_rouge_l_f": 0.6,
        },
    }


class TestVisualizationPlots:
    """Test visualization plot generation."""

    def test_plot_metrics_comparison(self, sample_benchmark_results):
        """Test plotting metrics comparison."""
        # Create a list of results
        results1 = sample_benchmark_results
        results2 = dict(results1)
        results2["summarizer_type"] = "hierarchical"
        results2["config"]["model_name"] = "other_model"
        results2["overall_metrics"]["avg_processing_time"] = 0.7
        results2["overall_metrics"]["avg_rouge_1_f"] = 0.65

        results_list = [results1, results2]

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            # Generate plot
            fig = plot_metrics_comparison(
                results_list,
                metrics=["avg_rouge_1_f", "avg_processing_time"],
                output_path=output_path,
            )

            # Check if plot was created
            assert fig is not None
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            # Clean up
            plt.close("all")
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_plot_document_metrics(self, sample_benchmark_results):
        """Test plotting document metrics."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            # Generate plot
            fig = plot_document_metrics(
                sample_benchmark_results,
                metrics=["overall_score", "compression_ratio"],
                output_path=output_path,
            )

            # Check if plot was created
            assert fig is not None
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            # Clean up
            plt.close("all")
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_plot_rouge_breakdown(self, sample_benchmark_results):
        """Test plotting ROUGE breakdown."""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            # Generate plot
            fig = plot_rouge_breakdown(
                sample_benchmark_results, output_path=output_path
            )

            # Check if plot was created
            assert fig is not None
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            # Clean up
            plt.close("all")
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_plots_with_missing_data(self):
        """Test plotting with incomplete data."""
        # Create minimal results without metrics
        minimal_results = {
            "dataset": "test_dataset.json",
            "summarizer_type": "basic",
            "document_results": {},
            "overall_metrics": {},
        }

        # These should handle the missing data gracefully without errors
        plot_metrics_comparison([minimal_results, minimal_results])
        plot_document_metrics(minimal_results)
        plot_rouge_breakdown(minimal_results)

        # Clean up
        plt.close("all")
