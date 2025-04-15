#!/usr/bin/env python
"""
Example script to generate HTML reports from benchmark results.

This script demonstrates how to create comprehensive HTML reports
from benchmark results for easier analysis and sharing.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import webbrowser
from pathlib import Path

# Ensure llamasum package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llamasum.benchmark import run_benchmark
    from llamasum.report import generate_report_from_files, generate_report_html
except ImportError:
    print("Error: Could not import LlamaSum modules.")
    print("Make sure LlamaSum is installed or in your Python path.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    return {
        "documents": [
            {
                "id": "doc1",
                "title": "Text Summarization",
                "text": """
                Text summarization is the process of creating a concise and coherent version of 
                a longer document while preserving key information and the overall meaning.
                Summarization techniques can be broadly categorized into extractive and abstractive methods.
                Extractive summarization involves selecting important sentences or phrases from the original 
                text, while abstractive summarization involves generating new text that captures the 
                essence of the original content. Modern approaches often use deep learning and 
                transformer models to generate high-quality summaries.
                """.strip(),
                "reference_summary": "Text summarization creates concise versions of longer documents, using either extractive (selecting key sentences) or abstractive (generating new text) methods.",
            },
            {
                "id": "doc2",
                "title": "Data Visualization",
                "text": """
                Data visualization is the graphic representation of data to communicate information 
                clearly and efficiently. It involves the creation and study of the visual representation 
                of data, meaning information that has been abstracted in some schematic form to highlight 
                useful patterns and trends. Effective visualization helps users analyze and reason about 
                data, making complex data sets accessible and understandable. Common visualization 
                formats include charts, graphs, maps, and dashboards.
                """.strip(),
                "reference_summary": "Data visualization uses graphic representations to communicate information clearly, making complex data accessible through formats like charts, graphs, and maps.",
            },
        ]
    }


def run_demo_benchmarks(output_dir):
    """Run benchmarks with different configurations to generate sample results."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a sample dataset
    dataset = create_sample_dataset()
    dataset_path = os.path.join(output_dir, "sample_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Created sample dataset at {dataset_path}")

    # Define benchmark configurations
    configurations = [
        {
            "name": "basic_position",
            "type": "basic",
            "params": {"extractive_method": "position", "extractive_ratio": 0.3},
        },
        {
            "name": "basic_tfidf",
            "type": "basic",
            "params": {"extractive_method": "tfidf", "extractive_ratio": 0.3},
        },
    ]

    # Run benchmarks
    results = []
    for config in configurations:
        print(f"Running benchmark for {config['name']}...")
        output_path = os.path.join(output_dir, f"results_{config['name']}.json")

        result = run_benchmark(
            dataset_path=dataset_path,
            output_path=output_path,
            summarizer_type=config["type"],
            **config["params"],
        )

        results.append(result)
        print(f"Saved results to {output_path}")

    return results, [
        os.path.join(output_dir, f"results_{config['name']}.json")
        for config in configurations
    ]


def main():
    """Run the report generation example."""
    parser = argparse.ArgumentParser(
        description="Generate HTML reports from benchmark results"
    )

    parser.add_argument(
        "--results-dir",
        help="Directory containing benchmark results (if not provided, will run demo benchmarks)",
    )

    parser.add_argument(
        "--output-dir", default="./reports", help="Directory to save reports"
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated report in the default web browser",
    )

    args = parser.parse_args()

    # Check for matplotlib
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for reports")
        print("Install with: pip install matplotlib pandas")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get benchmark results
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return 1

        # Find all JSON files in the directory
        result_files = list(results_dir.glob("*.json"))
        if not result_files:
            print(f"Error: No JSON files found in {results_dir}")
            return 1

        print(f"Found {len(result_files)} result files in {results_dir}")

        # Option 1: Generate report from files
        report_path = output_dir / "benchmark_report.html"
        generate_report_from_files(
            file_paths=result_files,
            output_path=str(report_path),
            report_title=f"LlamaSum Benchmark Report - {len(result_files)} Configurations",
        )
    else:
        # Run demo benchmarks
        print("No results directory provided. Running demo benchmarks...")
        temp_dir = output_dir / "benchmark_data"
        results, result_files = run_demo_benchmarks(str(temp_dir))

        # Option 2: Generate report from results objects directly
        report_path = output_dir / "benchmark_report.html"
        generate_report_html(
            results_list=results,
            output_path=str(report_path),
            report_title="LlamaSum Demo Benchmark Report",
        )

    print(f"Report generated and saved to {report_path}")

    # Open in browser if requested
    if args.open and report_path.exists():
        print("Opening report in web browser...")
        webbrowser.open(f"file://{report_path.absolute()}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
