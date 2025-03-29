import os
import json
import tempfile
import unittest
from unittest import mock

import pytest

from llamasum.benchmark import run_benchmark
from llamasum.summarizer import LlamaSummarizer
from llamasum.report import (
    generate_report_html,
    generate_report_from_files,
    get_report_template,
    format_metric_value,
    create_summary_section,
    create_model_sections,
    main
)


class TestReportUtils(unittest.TestCase):
    """Test utility functions in the report module."""
    
    def test_get_report_template(self):
        """Test that the report template can be loaded."""
        template = get_report_template()
        self.assertIsInstance(template, str)
        self.assertIn("<!DOCTYPE html>", template)
        self.assertIn("<html", template)
        self.assertIn("{{title}}", template)
    
    def test_format_metric_value(self):
        """Test the metric value formatting."""
        # Test percentage formatting
        self.assertEqual(format_metric_value("rouge1", 0.75), "75.00%")
        self.assertEqual(format_metric_value("rouge2", 0.5), "50.00%")
        self.assertEqual(format_metric_value("rougeL", 0.333), "33.30%")
        
        # Test time formatting
        self.assertEqual(format_metric_value("time", 1.5), "1.50s")
        self.assertEqual(format_metric_value("processing_time", 120), "120.00s")
        
        # Test ratio formatting
        self.assertEqual(format_metric_value("compression_ratio", 0.25), "0.25")
        
        # Test default formatting
        self.assertEqual(format_metric_value("custom_metric", 0.123), "0.12")


class TestReportGeneration(unittest.TestCase):
    """Test the report generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a minimal sample result for testing
        self.sample_result = {
            "config": {
                "summarizer_type": "basic",
                "model_name": "test-model",
                "device": "cpu",
                "max_length": 100
            },
            "metrics": {
                "rouge1": 0.45,
                "rouge2": 0.22,
                "rougeL": 0.38,
                "compression_ratio": 0.3,
                "processing_time": 5.2
            },
            "documents": [
                {
                    "id": "doc1",
                    "title": "Test Document",
                    "text": "This is a test document with some content.",
                    "reference_summary": "Test document content.",
                    "generated_summary": "Test content.",
                    "metrics": {
                        "rouge1": 0.45,
                        "rouge2": 0.22,
                        "rougeL": 0.38
                    }
                }
            ]
        }
        
        # Save the sample result to a file
        self.result_file = os.path.join(self.temp_dir.name, "result.json")
        with open(self.result_file, 'w') as f:
            json.dump(self.sample_result, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_create_summary_section(self):
        """Test creation of the summary section."""
        results_list = [self.sample_result]
        html = create_summary_section(results_list)
        
        self.assertIsInstance(html, str)
        self.assertIn("Model Comparison", html)
        self.assertIn("test-model", html)
        self.assertIn("rouge1", html.lower())
    
    def test_create_model_sections(self):
        """Test creation of model detail sections."""
        results_list = [self.sample_result]
        html = create_model_sections(results_list)
        
        self.assertIsInstance(html, str)
        self.assertIn("test-model", html)
        self.assertIn("Configuration", html)
        self.assertIn("Document Results", html)
    
    def test_generate_report_html(self):
        """Test generating a complete HTML report."""
        output_path = os.path.join(self.temp_dir.name, "report.html")
        
        # Generate the report
        generate_report_html(
            results_list=[self.sample_result],
            output_path=output_path,
            report_title="Test Report"
        )
        
        # Check that the file exists and contains expected content
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("Test Report", content)
            self.assertIn("test-model", content)
            self.assertIn("rouge1", content.lower())
    
    def test_generate_report_from_files(self):
        """Test generating a report from result files."""
        output_path = os.path.join(self.temp_dir.name, "report_from_files.html")
        
        # Generate the report
        generate_report_from_files(
            file_paths=[self.result_file],
            output_path=output_path,
            report_title="Test Report From Files"
        )
        
        # Check that the file exists and contains expected content
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("Test Report From Files", content)
            self.assertIn("test-model", content)


class TestCommandLine(unittest.TestCase):
    """Test the command line interface for report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a minimal sample result for testing
        self.sample_result = {
            "config": {
                "summarizer_type": "basic",
                "model_name": "test-model"
            },
            "metrics": {
                "rouge1": 0.45,
                "rouge2": 0.22
            },
            "documents": []
        }
        
        # Save the sample result to files
        self.result_files = []
        for i in range(2):
            file_path = os.path.join(self.temp_dir.name, f"result{i}.json")
            with open(file_path, 'w') as f:
                json.dump(self.sample_result, f)
            self.result_files.append(file_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @mock.patch('argparse.ArgumentParser.parse_args')
    @mock.patch('webbrowser.open')
    def test_main_function(self, mock_webbrowser, mock_parse_args):
        """Test the main function with mocked arguments."""
        output_path = os.path.join(self.temp_dir.name, "cli_report.html")
        
        # Mock the argument parser
        mock_args = mock.MagicMock()
        mock_args.files = self.result_files
        mock_args.output = output_path
        mock_args.title = "CLI Test Report"
        mock_args.open = True
        mock_parse_args.return_value = mock_args
        
        # Run the main function
        main()
        
        # Check that the file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check that webbrowser.open was called
        mock_webbrowser.assert_called_once_with('file://' + os.path.abspath(output_path))


@pytest.mark.integration
class TestReportIntegration(unittest.TestCase):
    """Integration tests for the report module with benchmark."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a minimal sample dataset
        self.sample_dataset = [
            {
                "id": "test1",
                "title": "Test Document 1",
                "text": "This is a test document with some content for summarization.",
                "reference_summary": "Test document for summarization."
            },
            {
                "id": "test2",
                "title": "Test Document 2",
                "text": "Another test document with different content for the benchmark.",
                "reference_summary": "Another test for benchmarking."
            }
        ]
        
        # Save the sample dataset to a file
        self.dataset_file = os.path.join(self.temp_dir.name, "dataset.json")
        with open(self.dataset_file, 'w') as f:
            json.dump(self.sample_dataset, f)
        
        # Define the result file path
        self.result_file = os.path.join(self.temp_dir.name, "benchmark_result.json")
        self.report_file = os.path.join(self.temp_dir.name, "report.html")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_end_to_end_benchmark_to_report(self):
        """Test end-to-end flow from benchmark to report generation."""
        # Run a benchmark with a mock summarizer
        summarizer = LlamaSummarizer(model_name="dummy-model")
        
        # Monkey patch the summarize method to avoid loading a real model
        def mock_summarize(text, **kwargs):
            return f"Summary of: {text[:20]}..."
        
        summarizer.summarize = mock_summarize
        
        # Run the benchmark
        result = run_benchmark(
            dataset_path=self.dataset_file,
            output_path=self.result_file,
            summarizer=summarizer
        )
        
        # Check that the benchmark result was created
        self.assertTrue(os.path.exists(self.result_file))
        
        # Generate a report from the benchmark result
        generate_report_from_files(
            file_paths=[self.result_file],
            output_path=self.report_file,
            report_title="Benchmark Test Report"
        )
        
        # Check that the report was created
        self.assertTrue(os.path.exists(self.report_file))
        
        # Verify the report content
        with open(self.report_file, 'r') as f:
            content = f.read()
            self.assertIn("Benchmark Test Report", content)
            self.assertIn("dummy-model", content)
            self.assertIn("rouge", content.lower())


if __name__ == '__main__':
    unittest.main() 