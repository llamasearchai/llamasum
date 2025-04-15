"""Benchmarking utilities for LlamaSum."""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from llamasum.config import HierarchicalConfig, MultiDocConfig, SummarizerConfig
from llamasum.evaluation import evaluate_summary
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.summarizer import LlamaSummarizer

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Load evaluation dataset from file.

    Args:
        dataset_path: Path to dataset file (JSON or CSV)

    Returns:
        List of document dictionaries
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load based on file extension
    if dataset_path.lower().endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if it's a list or object with items
        if isinstance(data, dict) and "documents" in data:
            return data["documents"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(
                "Invalid JSON format. Expected list or object with 'documents' field."
            )

    elif dataset_path.lower().endswith(".csv"):
        documents = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check for required fields
                if "text" not in row:
                    logger.warning("Row missing 'text' field, skipping")
                    continue
                documents.append(row)
        return documents

    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to file.

    Args:
        results: Benchmark results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if output_path.lower().endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    elif output_path.lower().endswith(".csv"):
        # Flatten the metrics for CSV
        flattened_results = []
        for doc_id, doc_results in results["document_results"].items():
            item = {"document_id": doc_id}
            item.update(doc_results["metrics"])
            # Handle nested metrics
            for metric, value in list(item.items()):
                if isinstance(value, dict):
                    for submetric, subvalue in value.items():
                        item[f"{metric}_{submetric}"] = subvalue
                    del item[metric]
            flattened_results.append(item)

        # Write CSV
        if flattened_results:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=flattened_results[0].keys())
                writer.writeheader()
                writer.writerows(flattened_results)
    else:
        logger.warning(f"Unsupported output format: {output_path}. Using JSON instead.")
        with open(f"{output_path}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def run_benchmark(
    dataset_path: str,
    output_path: Optional[str] = None,
    summarizer_type: str = "basic",
    model_name: Optional[str] = None,
    max_length: int = 150,
    min_length: int = 50,
    extractive_method: str = "tfidf",
    extractive_ratio: float = 0.3,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, Any]:
    """Run benchmark on dataset.

    Args:
        dataset_path: Path to dataset
        output_path: Path to save results (optional)
        summarizer_type: Type of summarizer (basic, hierarchical, multidoc)
        model_name: Name of model to use
        max_length: Maximum summary length
        min_length: Minimum summary length
        extractive_method: Method for extractive summarization
        extractive_ratio: Ratio for extractive summarization
        device: Device to use (cpu or cuda)
        **kwargs: Additional arguments for specific summarizer types

    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Loading dataset from {dataset_path}")
    documents = load_dataset(dataset_path)
    logger.info(f"Loaded {len(documents)} documents for benchmarking")

    # Set up base configuration
    config_args = {
        "model_name": model_name,
        "max_length": max_length,
        "min_length": min_length,
        "extraction_method": extractive_method,
        "extractive_ratio": extractive_ratio,
        "device": device,
    }

    # Initialize appropriate summarizer
    if summarizer_type == "hierarchical":
        config = HierarchicalConfig(**config_args)
        config.levels = kwargs.get("levels", 3)
        config.include_ultra_short = kwargs.get("include_ultra_short", True)
        summarizer = HierarchicalSummarizer(config)
    elif summarizer_type == "multidoc":
        config = MultiDocConfig(**config_args)
        config.strategy = kwargs.get("strategy", "extract_then_summarize")
        config.redundancy_threshold = kwargs.get("redundancy_threshold", 0.8)
        summarizer = MultiDocSummarizer(config)
    else:  # basic
        config = SummarizerConfig(**config_args)
        summarizer = LlamaSummarizer(config)

    # Prepare results structure
    results = {
        "dataset": dataset_path,
        "summarizer_type": summarizer_type,
        "config": config.__dict__,
        "document_count": len(documents),
        "start_time": time.time(),
        "document_results": {},
        "overall_metrics": {},
    }

    # Process each document
    for i, doc in enumerate(documents):
        doc_id = doc.get("id", str(i))
        logger.info(f"Processing document {i+1}/{len(documents)} (ID: {doc_id})")

        # Get text and reference if available
        text = doc.get("text", "")
        reference_summary = doc.get("reference_summary", None)

        if not text:
            logger.warning(f"Empty text for document {doc_id}, skipping")
            continue

        # Generate summary
        start_time = time.time()

        if summarizer_type == "hierarchical":
            summary_result = summarizer.summarize(text, **kwargs)
            # Use the first level summary for evaluation
            summary = summary_result.get("level_1", "")
        elif summarizer_type == "multidoc":
            # For multidoc, check if the document contains multiple texts
            if "texts" in doc and isinstance(doc["texts"], list):
                multiple_texts = doc["texts"]
                summary_result = summarizer.summarize(multiple_texts, **kwargs)
            else:
                # Treat as single document or split by delimiter if specified
                delimiter = kwargs.get("delimiter", "\n\n")
                if delimiter and delimiter in text:
                    multiple_texts = [
                        t.strip() for t in text.split(delimiter) if t.strip()
                    ]
                    summary_result = summarizer.summarize(multiple_texts, **kwargs)
                else:
                    # Fallback to basic summarization
                    summary_result = summarizer.summarize([text], **kwargs)

            # Extract the summary from the result
            if isinstance(summary_result, dict) and "summary" in summary_result:
                summary = summary_result["summary"]
            else:
                summary = str(summary_result)
        else:
            # Basic summarization
            summary = summarizer.summarize(text, **kwargs)

        processing_time = time.time() - start_time

        # Evaluate summary
        if reference_summary:
            metrics = evaluate_summary(summary, text, reference_summary)
        else:
            metrics = evaluate_summary(summary, text)

        # Store results for this document
        results["document_results"][doc_id] = {
            "summary": summary,
            "processing_time": processing_time,
            "metrics": metrics,
        }

        # If multidoc or hierarchical, store additional data
        if summarizer_type != "basic" and isinstance(summary_result, dict):
            results["document_results"][doc_id]["additional_data"] = {
                k: v
                for k, v in summary_result.items()
                if k not in ["summary", "level_1"]
            }

    # Add end time
    results["end_time"] = time.time()
    results["total_time"] = results["end_time"] - results["start_time"]

    # Calculate overall metrics
    overall_metrics = {
        "avg_processing_time": sum(
            d["processing_time"] for d in results["document_results"].values()
        )
        / len(results["document_results"]),
        "avg_compression_ratio": sum(
            d["metrics"]["compression_ratio"]
            for d in results["document_results"].values()
        )
        / len(results["document_results"]),
        "avg_overall_score": sum(
            d["metrics"]["overall_score"] for d in results["document_results"].values()
        )
        / len(results["document_results"]),
    }

    # Check if all have ROUGE scores
    rouge_available = all(
        "rouge" in d["metrics"] for d in results["document_results"].values()
    )
    if rouge_available:
        overall_metrics["avg_rouge_1_f"] = sum(
            d["metrics"]["rouge"]["rouge-1"]["f"]
            for d in results["document_results"].values()
        ) / len(results["document_results"])
        overall_metrics["avg_rouge_l_f"] = sum(
            d["metrics"]["rouge"]["rouge-l"]["f"]
            for d in results["document_results"].values()
        ) / len(results["document_results"])

    results["overall_metrics"] = overall_metrics

    # Save results if output path provided
    if output_path:
        save_results(results, output_path)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    """Command line entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="LlamaSum Benchmarking Tool")

    # Required arguments
    parser.add_argument("dataset", help="Path to dataset file (JSON or CSV)")

    # Output options
    parser.add_argument("-o", "--output", help="Path to save results (JSON or CSV)")

    # Summarizer options
    parser.add_argument(
        "--type",
        choices=["basic", "hierarchical", "multidoc"],
        default="basic",
        help="Type of summarizer to benchmark",
    )

    parser.add_argument("--model", help="Model name or path")

    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")

    # Summary parameters
    parser.add_argument(
        "--max-length", type=int, default=150, help="Maximum summary length"
    )

    parser.add_argument(
        "--min-length", type=int, default=50, help="Minimum summary length"
    )

    parser.add_argument(
        "--extractive-method",
        choices=["tfidf", "textrank", "position"],
        default="tfidf",
        help="Extractive summarization method",
    )

    parser.add_argument(
        "--extractive-ratio",
        type=float,
        default=0.3,
        help="Ratio for extractive summarization",
    )

    # Hierarchical options
    parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Number of levels for hierarchical summarization",
    )

    parser.add_argument(
        "--include-ultra-short",
        action="store_true",
        help="Include ultra-short summary in hierarchical results",
    )

    # Multidoc options
    parser.add_argument(
        "--strategy",
        choices=[
            "concatenate",
            "summarize_each",
            "extract_then_summarize",
            "cluster_then_summarize",
        ],
        default="extract_then_summarize",
        help="Strategy for multi-document summarization",
    )

    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.8,
        help="Threshold for redundancy removal in multi-document summarization",
    )

    parser.add_argument(
        "--delimiter",
        default="\n\n",
        help="Delimiter for splitting text into multiple documents",
    )

    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Extract arguments for the benchmark
    kwargs = {
        "levels": args.levels,
        "include_ultra_short": args.include_ultra_short,
        "strategy": args.strategy,
        "redundancy_threshold": args.redundancy_threshold,
        "delimiter": args.delimiter,
    }

    # Run benchmark
    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        summarizer_type=args.type,
        model_name=args.model,
        max_length=args.max_length,
        min_length=args.min_length,
        extractive_method=args.extractive_method,
        extractive_ratio=args.extractive_ratio,
        device=args.device,
        **kwargs,
    )


if __name__ == "__main__":
    main()
