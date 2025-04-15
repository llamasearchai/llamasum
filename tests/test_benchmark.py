"""Tests for the benchmark module."""

import json
import os
import tempfile
from pathlib import Path

import pytest

import llamasum
from llamasum.benchmark import load_dataset, run_benchmark, save_results


class TestBenchmarkFunctions:
    """Test benchmark utility functions."""

    def test_load_dataset_json(self):
        """Test loading a JSON dataset."""
        # Create a temporary JSON dataset
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                {
                    "documents": [
                        {
                            "id": "1",
                            "text": "Sample text one.",
                            "reference_summary": "Summary one.",
                        },
                        {
                            "id": "2",
                            "text": "Sample text two.",
                            "reference_summary": "Summary two.",
                        },
                    ]
                },
                f,
            )
            temp_path = f.name

        try:
            # Load the dataset
            documents = load_dataset(temp_path)

            # Check the result
            assert len(documents) == 2
            assert documents[0]["id"] == "1"
            assert documents[0]["text"] == "Sample text one."
            assert documents[1]["reference_summary"] == "Summary two."
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_load_dataset_json_list(self):
        """Test loading a JSON dataset that's a direct list."""
        # Create a temporary JSON dataset
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                [
                    {
                        "id": "1",
                        "text": "Sample text one.",
                        "reference_summary": "Summary one.",
                    },
                    {
                        "id": "2",
                        "text": "Sample text two.",
                        "reference_summary": "Summary two.",
                    },
                ],
                f,
            )
            temp_path = f.name

        try:
            # Load the dataset
            documents = load_dataset(temp_path)

            # Check the result
            assert len(documents) == 2
            assert documents[0]["id"] == "1"
            assert documents[1]["id"] == "2"
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_save_results_json(self):
        """Test saving benchmark results to JSON."""
        results = {
            "dataset": "test_dataset.json",
            "summarizer_type": "basic",
            "document_results": {
                "1": {
                    "summary": "Test summary.",
                    "processing_time": 0.5,
                    "metrics": {"compression_ratio": 0.3, "overall_score": 0.8},
                }
            },
            "overall_metrics": {
                "avg_processing_time": 0.5,
                "avg_compression_ratio": 0.3,
                "avg_overall_score": 0.8,
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save results
            save_results(results, temp_path)

            # Load and verify
            with open(temp_path, "r") as f:
                loaded = json.load(f)

            assert loaded["summarizer_type"] == "basic"
            assert loaded["overall_metrics"]["avg_overall_score"] == 0.8
            assert loaded["document_results"]["1"]["summary"] == "Test summary."
        finally:
            # Clean up
            os.unlink(temp_path)


class TestBenchmarkExecution:
    """Test benchmark execution."""

    @pytest.fixture
    def sample_dataset_path(self):
        """Create a small sample dataset for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                {
                    "documents": [
                        {
                            "id": "test1",
                            "text": "This is a sample text for testing the benchmark functionality. "
                            "It needs to be reasonably long to test sentence extraction. "
                            "The benchmark should summarize this text effectively.",
                            "reference_summary": "Sample text for testing benchmark functionality.",
                        }
                    ]
                },
                f,
            )
            return f.name

    def test_run_benchmark_basic(self, sample_dataset_path):
        """Test running a benchmark with basic summarizer."""
        try:
            # Run a simple benchmark
            results = run_benchmark(
                dataset_path=sample_dataset_path,
                summarizer_type="basic",
                extractive_method="position",  # Simple method that doesn't need external deps
                extractive_ratio=0.5,
                max_length=50,
            )

            # Check results structure
            assert results["summarizer_type"] == "basic"
            assert "document_count" in results
            assert "document_results" in results
            assert "test1" in results["document_results"]
            assert "summary" in results["document_results"]["test1"]
            assert "metrics" in results["document_results"]["test1"]
            assert "overall_metrics" in results

            # Check metrics
            assert (
                "compression_ratio" in results["document_results"]["test1"]["metrics"]
            )
            assert "density" in results["document_results"]["test1"]["metrics"]
            assert "overall_score" in results["document_results"]["test1"]["metrics"]

            # Verify overall metrics
            assert "avg_compression_ratio" in results["overall_metrics"]
            assert "avg_overall_score" in results["overall_metrics"]
        finally:
            # Clean up
            os.unlink(sample_dataset_path)

    def test_benchmark_output_options(self, sample_dataset_path):
        """Test benchmark with different output options."""
        try:
            # Create output paths
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_file:
                json_path = json_file.name

            # Run benchmark with output path
            results = run_benchmark(
                dataset_path=sample_dataset_path,
                output_path=json_path,
                summarizer_type="basic",
                extractive_method="position",
            )

            # Verify file was created
            assert os.path.exists(json_path)

            # Verify content
            with open(json_path, "r") as f:
                saved_results = json.load(f)

            assert saved_results["summarizer_type"] == "basic"
            assert "document_results" in saved_results
            assert "overall_metrics" in saved_results
        finally:
            # Clean up
            os.unlink(sample_dataset_path)
            if os.path.exists(json_path):
                os.unlink(json_path)


class TestBenchmarkIntegration:
    """Integration tests for benchmark functionality with different summarizers."""

    @pytest.fixture
    def multi_document_dataset(self):
        """Create a dataset with multi-document examples."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                {
                    "documents": [
                        {
                            "id": "multi1",
                            "texts": [
                                "This is the first document about a topic.",
                                "This is the second document covering related information.",
                                "This is the third document with additional details.",
                            ],
                        }
                    ]
                },
                f,
            )
            return f.name

    def test_hierarchical_benchmark(self, sample_dataset_path):
        """Test benchmarking with hierarchical summarizer."""
        try:
            # Run benchmark with hierarchical summarizer
            results = run_benchmark(
                dataset_path=sample_dataset_path,
                summarizer_type="hierarchical",
                extractive_method="position",
                levels=2,
                include_ultra_short=True,
            )

            # Check results
            assert results["summarizer_type"] == "hierarchical"
            doc_results = results["document_results"]["test1"]

            # Check for additional data specific to hierarchical summarization
            assert "additional_data" in doc_results
            additional = doc_results["additional_data"]

            # Verify expected fields from hierarchical summarization
            expected_keys = ["level_2", "original"]
            if "ultra_short" in additional:  # This is optional and may not be present
                expected_keys.append("ultra_short")

            for key in expected_keys:
                assert key in additional, f"Expected key '{key}' in additional_data"
        finally:
            # Clean up
            os.unlink(sample_dataset_path)

    def test_multidoc_benchmark(self, multi_document_dataset):
        """Test benchmarking with multi-document summarizer."""
        try:
            # Run benchmark with multi-document summarizer
            results = run_benchmark(
                dataset_path=multi_document_dataset,
                summarizer_type="multidoc",
                extractive_method="position",
                strategy="concatenate",  # Simple strategy that works without dependencies
            )

            # Check results
            assert results["summarizer_type"] == "multidoc"
            assert "multi1" in results["document_results"]

            # Verify document was properly handled as multi-document
            assert "summary" in results["document_results"]["multi1"]

            # The concatenate strategy should produce a summary
            summary = results["document_results"]["multi1"]["summary"]
            assert len(summary) > 0
        finally:
            # Clean up
            os.unlink(multi_document_dataset)

    def test_benchmark_with_reference(self, sample_dataset_path):
        """Test benchmarking with reference summaries."""
        try:
            # Run benchmark
            results = run_benchmark(
                dataset_path=sample_dataset_path,
                summarizer_type="basic",
                extractive_method="position",
            )

            # Check reference-based metrics
            metrics = results["document_results"]["test1"]["metrics"]

            # ROUGE may not be available, but similarity should be
            assert "similarity_to_reference" in metrics

            # Check that similarity is reasonable (between 0 and 1)
            similarity = metrics["similarity_to_reference"]
            assert (
                0 <= similarity <= 1
            ), f"Similarity {similarity} should be between 0 and 1"
        finally:
            # Clean up
            os.unlink(sample_dataset_path)
