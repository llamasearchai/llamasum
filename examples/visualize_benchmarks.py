#!/usr/bin/env python
"""
Example script to visualize LlamaSum benchmark results.

This script demonstrates how to use the visualization module to create
visual representations of benchmark results for different summarization methods.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure llamasum package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llamasum.visualize import (
        load_multiple_results,
        plot_metrics_comparison,
        plot_document_metrics,
        plot_rouge_breakdown
    )
    from llamasum.benchmark import run_benchmark
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


def run_demo_benchmarks(output_dir):
    """Run simple benchmarks on sample data to generate visualization examples."""
    from llamasum.benchmark import load_dataset
    import json
    import tempfile
    
    # Create a simple test dataset
    sample_data = {
        "documents": [
            {
                "id": "doc1",
                "text": """
                LlamaSum is a powerful text summarization library. It provides multiple approaches
                to generate summaries from text documents. The library includes support for
                extractive summarization, abstractive summarization, and hybrid approaches.
                LlamaSum can handle both single documents and collections of related documents.
                It offers a flexible API and command-line interface for easy integration.
                """.strip(),
                "reference_summary": "LlamaSum is a text summarization library with multiple approaches and interfaces."
            },
            {
                "id": "doc2",
                "text": """
                Benchmark results help evaluate the performance of different summarization models.
                They provide metrics like ROUGE scores, processing time, and compression ratio.
                Comparing these metrics across models can identify the best approach for specific use cases.
                Visualizations make it easier to interpret complex benchmark results and identify patterns.
                """.strip(),
                "reference_summary": "Benchmarks evaluate summarization performance with metrics that can be visualized for easier interpretation."
            }
        ]
    }
    
    # Create a temporary dataset file
    dataset_path = os.path.join(output_dir, "demo_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    # Run benchmarks with different configurations
    results = []
    
    # Basic summarizer with different extractive methods
    for method in ["position", "tfidf"]:
        result = run_benchmark(
            dataset_path=dataset_path,
            output_path=os.path.join(output_dir, f"results_basic_{method}.json"),
            summarizer_type="basic",
            extractive_method=method,
            extractive_ratio=0.3
        )
        results.append(result)
    
    # Hierarchical summarizer
    result = run_benchmark(
        dataset_path=dataset_path,
        output_path=os.path.join(output_dir, "results_hierarchical.json"),
        summarizer_type="hierarchical",
        levels=2,
        include_ultra_short=True
    )
    results.append(result)
    
    return results


def main():
    """Run the visualization example."""
    parser = argparse.ArgumentParser(description="Visualize LlamaSum benchmark results")
    
    parser.add_argument(
        "--results-dir",
        help="Directory containing benchmark results JSON files (if not provided, will run demo benchmarks)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./visualizations",
        help="Directory to save output visualizations"
    )
    
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output file format"
    )
    
    args = parser.parse_args()
    
    # Check for matplotlib
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualizations")
        print("Install with: pip install matplotlib pandas seaborn")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Either load existing results or run demo benchmarks
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return 1
        
        result_files = list(results_dir.glob("*.json"))
        if not result_files:
            print(f"Error: No JSON files found in {results_dir}")
            return 1
        
        print(f"Loading {len(result_files)} benchmark result files...")
        results_list = load_multiple_results(result_files)
    else:
        print("No results directory provided. Running demo benchmarks...")
        results_list = run_demo_benchmarks(output_dir)
    
    if not results_list:
        print("Error: No valid benchmark results found")
        return 1
    
    print(f"Creating visualizations in {output_dir}...")
    
    # Metrics comparison
    if len(results_list) > 1:
        print("Generating metrics comparison...")
        comparison_path = output_dir / f"comparison.{args.format}"
        plot_metrics_comparison(
            results_list,
            figsize=(10, 6),
            output_path=str(comparison_path)
        )
    
    # Document metrics for each result
    for i, result in enumerate(results_list):
        summarizer_type = result.get("summarizer_type", f"result{i}")
        print(f"Generating document metrics for {summarizer_type}...")
        
        doc_metrics_path = output_dir / f"document_metrics_{summarizer_type}.{args.format}"
        plot_document_metrics(
            result,
            figsize=(10, 6),
            output_path=str(doc_metrics_path),
            max_docs=5
        )
    
    # ROUGE breakdown for results with reference summaries
    for i, result in enumerate(results_list):
        # Check if results contain ROUGE scores
        has_rouge = False
        for doc_data in result.get("document_results", {}).values():
            if "metrics" in doc_data and "rouge" in doc_data["metrics"]:
                has_rouge = True
                break
        
        if has_rouge:
            summarizer_type = result.get("summarizer_type", f"result{i}")
            print(f"Generating ROUGE breakdown for {summarizer_type}...")
            
            rouge_path = output_dir / f"rouge_breakdown_{summarizer_type}.{args.format}"
            plot_rouge_breakdown(
                result,
                figsize=(12, 5),
                output_path=str(rouge_path)
            )
    
    print(f"Visualizations created successfully in {output_dir}")
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sys.exit(main()) 