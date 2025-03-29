#!/usr/bin/env python3
"""Example script demonstrating LlamaSum benchmarking."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure llamasum is in the path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from llamasum.benchmark import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run a benchmark demonstration."""
    parser = argparse.ArgumentParser(description="LlamaSum Benchmark Example")
    
    parser.add_argument(
        "--dataset",
        default=str(script_dir / "sample_dataset.json"),
        help="Path to dataset file (default: sample_dataset.json)"
    )
    
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path to save benchmark results (default: benchmark_results.json)"
    )
    
    parser.add_argument(
        "--type",
        choices=["basic", "hierarchical", "multidoc"],
        default="basic",
        help="Type of summarizer to benchmark (default: basic)"
    )
    
    parser.add_argument(
        "--model",
        default="sshleifer/distilbart-cnn-12-6",
        help="Model name (default: sshleifer/distilbart-cnn-12-6)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Maximum summary length (default: 150)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum summary length (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Print benchmark parameters
    logger.info("Running LlamaSum benchmark with the following parameters:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Summarizer type: {args.type}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Min length: {args.min_length}")
    
    # Additional parameters based on summarizer type
    kwargs = {}
    if args.type == "hierarchical":
        kwargs.update({
            "levels": 3,
            "include_ultra_short": True
        })
    elif args.type == "multidoc":
        kwargs.update({
            "strategy": "extract_then_summarize",
            "redundancy_threshold": 0.7
        })
    
    # Run benchmark
    try:
        results = run_benchmark(
            dataset_path=args.dataset,
            output_path=args.output,
            summarizer_type=args.type,
            model_name=args.model,
            max_length=args.max_length,
            min_length=args.min_length,
            device=args.device,
            **kwargs
        )
        
        # Print summary of results
        logger.info("\nBenchmark completed successfully!")
        logger.info(f"Processed {len(results['document_results'])} documents")
        logger.info(f"Total processing time: {results['total_time']:.2f} seconds")
        
        # Print overall metrics
        logger.info("\nOverall metrics:")
        for metric, value in results["overall_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 