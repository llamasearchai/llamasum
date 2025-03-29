"""Visualization utilities for LlamaSum benchmarking results."""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
from pathlib import Path

# Check if visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_benchmark_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load benchmark results from a JSON file.
    
    Args:
        file_path: Path to benchmark results JSON file
        
    Returns:
        Dictionary containing benchmark results
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


def load_multiple_results(file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
    """Load multiple benchmark results files.
    
    Args:
        file_paths: List of paths to benchmark results JSON files
        
    Returns:
        List of dictionaries containing benchmark results
    """
    results = []
    for path in file_paths:
        try:
            results.append(load_benchmark_results(path))
        except Exception as e:
            logger.warning(f"Error loading results from {path}: {e}")
    
    return results


def create_comparison_dataframe(results_list: List[Dict[str, Any]]) -> Optional['pd.DataFrame']:
    """Create a pandas DataFrame for comparison of multiple benchmark results.
    
    Args:
        results_list: List of benchmark results dictionaries
        
    Returns:
        DataFrame with comparison data or None if pandas is not available
    """
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not available. Install with: pip install pandas")
        return None
    
    if not results_list:
        return None
    
    # Extract key information from each result
    rows = []
    for result in results_list:
        # Get basic info
        row = {
            "dataset": result.get("dataset", "Unknown"),
            "summarizer_type": result.get("summarizer_type", "Unknown"),
            "document_count": result.get("document_count", 0),
            "total_time": result.get("total_time", 0),
        }
        
        # Get model and config info
        config = result.get("config", {})
        row["model_name"] = config.get("model_name", "Unknown")
        row["extractive_method"] = config.get("extractive_method", "Unknown")
        row["extractive_ratio"] = config.get("extractive_ratio", 0)
        row["max_length"] = config.get("max_length", 0)
        
        # Get overall metrics
        metrics = result.get("overall_metrics", {})
        for metric, value in metrics.items():
            row[metric] = value
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add a display name column for better labeling
    df["display_name"] = df.apply(
        lambda row: f"{row['summarizer_type']}_{row['model_name'].split('/')[-1]}",
        axis=1
    )
    
    return df


def plot_metrics_comparison(
    results_list: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None
) -> Optional['matplotlib.figure.Figure']:
    """Plot comparison of metrics across multiple benchmark results.
    
    Args:
        results_list: List of benchmark results dictionaries
        metrics: List of metrics to compare (defaults to standard set if None)
        figsize: Figure size (width, height) in inches
        output_path: Path to save the figure (if None, figure is displayed)
        
    Returns:
        Matplotlib figure or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None
    
    if not results_list:
        logger.warning("No results provided for comparison")
        return None
    
    # Create DataFrame for comparison
    df = create_comparison_dataframe(results_list)
    if df is None:
        return None
    
    # Default metrics to compare
    if metrics is None:
        available_metrics = [
            col for col in df.columns if col.startswith("avg_") and col != "avg_processing_time"
        ]
        # Add processing time separately as it's on a different scale
        if "avg_processing_time" in df.columns:
            available_metrics.append("avg_processing_time")
        metrics = available_metrics
    
    # Ensure all requested metrics exist
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        logger.warning("No valid metrics found for comparison")
        return None
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]  # Make it iterable for single metric
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Sort for the first metric only
        if i == 0:
            if metric == "avg_processing_time":
                # For processing time, lower is better
                df = df.sort_values(by=metric)
            else:
                # For other metrics, higher is better
                df = df.sort_values(by=metric, ascending=False)
        
        # Color based on summarizer type
        colors = {
            "basic": "blue",
            "hierarchical": "green",
            "multidoc": "orange"
        }
        bar_colors = [colors.get(t, "gray") for t in df["summarizer_type"]]
        
        # Create bar chart
        bars = ax.barh(df["display_name"], df[metric], color=bar_colors)
        ax.set_title(f"{metric.replace('avg_', '').replace('_', ' ').title()}")
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            if metric == "avg_processing_time":
                label = f"{width:.2f}s" 
            else:
                label = f"{width:.3f}"
            ax.text(width + 0.01 * ax.get_xlim()[1], 
                    bar.get_y() + bar.get_height()/2, 
                    label, 
                    va='center')
    
    # Add legend for summarizer types
    handles = [
        plt.Rectangle((0,0), 1, 1, color=colors.get(t))
        for t in ["basic", "hierarchical", "multidoc"]
    ]
    labels = ["Basic", "Hierarchical", "Multi-Document"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    
    # Add overall title
    fig.suptitle("Benchmark Metrics Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Figure saved to {output_path}")
    
    return fig


def plot_document_metrics(
    results: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    max_docs: int = 10
) -> Optional['matplotlib.figure.Figure']:
    """Plot metrics for each document in a benchmark result.
    
    Args:
        results: Benchmark results dictionary
        metrics: List of metrics to plot (defaults to standard set if None)
        figsize: Figure size (width, height) in inches
        output_path: Path to save the figure (if None, figure is displayed)
        max_docs: Maximum number of documents to display
        
    Returns:
        Matplotlib figure or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None
    
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not available. Install with: pip install pandas")
        return None
    
    # Get document results
    doc_results = results.get("document_results", {})
    if not doc_results:
        logger.warning("No document results found")
        return None
    
    # Create DataFrame with document metrics
    rows = []
    for doc_id, doc_data in doc_results.items():
        if "metrics" not in doc_data:
            continue
        
        row = {"document_id": doc_id, "processing_time": doc_data.get("processing_time", 0)}
        
        # Extract metrics
        metrics_data = doc_data["metrics"]
        for metric, value in metrics_data.items():
            if isinstance(value, dict):
                # Handle nested metrics like ROUGE
                for submetric, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        # Handle ROUGE's precision/recall/f1
                        for measure, measure_value in subvalue.items():
                            row[f"{metric}_{submetric}_{measure}"] = measure_value
                    else:
                        row[f"{metric}_{submetric}"] = subvalue
            else:
                row[metric] = value
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Limit to max_docs if needed
    if len(df) > max_docs:
        df = df.sort_values(by="overall_score", ascending=False).head(max_docs)
    
    # Default metrics to plot
    if metrics is None:
        metrics = ["overall_score", "compression_ratio", "density"]
        # Add ROUGE if available
        if any(col.startswith("rouge_") for col in df.columns):
            rouge_metrics = [col for col in df.columns if col.endswith("_f") and col.startswith("rouge_")]
            metrics.extend(rouge_metrics[:2])  # Add up to 2 ROUGE metrics
    
    # Ensure all requested metrics exist
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        logger.warning("No valid metrics found for plotting")
        return None
    
    # Create figure
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]  # Make it iterable for single metric
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_sorted = df.sort_values(by=metric, ascending=False)
        
        ax.barh(df_sorted["document_id"], df_sorted[metric])
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add data labels
        for j, v in enumerate(df_sorted[metric]):
            ax.text(v + 0.01 * ax.get_xlim()[1], j, f"{v:.3f}", va='center')
    
    # Add overall title
    fig.suptitle(f"Document Metrics for {results.get('summarizer_type', 'Unknown').title()} Summarizer", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Figure saved to {output_path}")
    
    return fig


def plot_rouge_breakdown(
    results: Dict[str, Any], 
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None,
    max_docs: int = 5
) -> Optional['matplotlib.figure.Figure']:
    """Plot ROUGE scores breakdown for benchmark results.
    
    Args:
        results: Benchmark results dictionary
        figsize: Figure size (width, height) in inches
        output_path: Path to save the figure (if None, figure is displayed)
        max_docs: Maximum number of documents to display
        
    Returns:
        Matplotlib figure or None if matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return None
    
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not available. Install with: pip install pandas")
        return None
    
    # Get document results
    doc_results = results.get("document_results", {})
    if not doc_results:
        logger.warning("No document results found")
        return None
    
    # Extract ROUGE scores
    rouge_data = []
    for doc_id, doc_data in doc_results.items():
        if "metrics" not in doc_data or "rouge" not in doc_data["metrics"]:
            continue
            
        rouge_scores = doc_data["metrics"]["rouge"]
        
        for rouge_type, scores in rouge_scores.items():
            if isinstance(scores, dict) and "f" in scores:
                rouge_data.append({
                    "document_id": doc_id,
                    "rouge_type": rouge_type,
                    "precision": scores.get("p", 0),
                    "recall": scores.get("r", 0),
                    "f1": scores.get("f", 0)
                })
    
    if not rouge_data:
        logger.warning("No ROUGE scores found in results")
        return None
        
    # Create DataFrame
    df = pd.DataFrame(rouge_data)
    
    # Get top documents by ROUGE-L F1 score
    top_docs = (df[df["rouge_type"] == "rouge-l"]
                .sort_values(by="f1", ascending=False)
                .head(max_docs)["document_id"].tolist())
    
    # Filter DataFrame to top documents and reshape for plotting
    df_filtered = df[df["document_id"].isin(top_docs)]
    
    # Create figure
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ROUGE-1 and ROUGE-L comparison
    for i, rouge_type in enumerate(["rouge-1", "rouge-l"]):
        ax = axes[i]
        df_rouge = df_filtered[df_filtered["rouge_type"] == rouge_type]
        
        # Sort by F1 score
        df_rouge = df_rouge.sort_values(by="f1", ascending=False)
        
        # Plot grouped bar chart
        x = np.arange(len(df_rouge))
        width = 0.25
        
        ax.bar(x - width, df_rouge["precision"], width, label="Precision")
        ax.bar(x, df_rouge["recall"], width, label="Recall")
        ax.bar(x + width, df_rouge["f1"], width, label="F1")
        
        # Add labels and title
        ax.set_xlabel("Document")
        ax.set_ylabel("Score")
        ax.set_title(f"{rouge_type.upper()} Scores")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Doc {i+1}" for i in range(len(df_rouge))], rotation=45)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add overall title
    fig.suptitle("ROUGE Scores Breakdown", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Figure saved to {output_path}")
    
    return fig


def main():
    """Command line entry point for visualization."""
    parser = argparse.ArgumentParser(description="Visualize LlamaSum benchmark results")
    
    # Main arguments
    parser.add_argument(
        "results_file",
        nargs="+",
        help="Path(s) to benchmark results JSON file(s)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Directory to save output visualizations"
    )
    
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg", "jpg"],
        default="png",
        help="Output file format"
    )
    
    # Visualization options
    parser.add_argument(
        "--type",
        choices=["comparison", "documents", "rouge", "all"],
        default="all",
        help="Type of visualization to generate"
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to visualize (comma-separated)"
    )
    
    parser.add_argument(
        "--figsize",
        default="12,8",
        help="Figure size in inches (width,height)"
    )
    
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help="Maximum number of documents to display"
    )
    
    args = parser.parse_args()
    
    # Check if visualization libraries are available
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available. Install with: pip install matplotlib")
        return 1
    
    if not PANDAS_AVAILABLE:
        logger.error("pandas not available. Install with: pip install pandas")
        return 1
    
    # Parse figure size
    try:
        figsize = tuple(map(int, args.figsize.split(",")))
        if len(figsize) != 2:
            figsize = (12, 8)
    except:
        figsize = (12, 8)
    
    # Parse metrics
    metrics = None
    if args.metrics:
        if isinstance(args.metrics, list):
            metrics = args.metrics
        else:
            metrics = args.metrics.split(",")
    
    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Load results
    results_list = load_multiple_results(args.results_file)
    if not results_list:
        logger.error("No valid benchmark results found")
        return 1
    
    # Generate visualizations
    if args.type in ["comparison", "all"] and len(results_list) > 1:
        output_path = None
        if args.output:
            output_path = os.path.join(args.output, f"comparison.{args.format}")
        
        plot_metrics_comparison(
            results_list,
            metrics=metrics,
            figsize=figsize,
            output_path=output_path
        )
    
    if args.type in ["documents", "all"]:
        for i, results in enumerate(results_list):
            output_path = None
            if args.output:
                summarizer_type = results.get("summarizer_type", f"result{i}")
                output_path = os.path.join(
                    args.output, 
                    f"document_metrics_{summarizer_type}.{args.format}"
                )
            
            plot_document_metrics(
                results,
                metrics=metrics,
                figsize=figsize,
                output_path=output_path,
                max_docs=args.max_docs
            )
    
    if args.type in ["rouge", "all"]:
        for i, results in enumerate(results_list):
            # Check if results contain ROUGE scores
            has_rouge = False
            for doc_data in results.get("document_results", {}).values():
                if "metrics" in doc_data and "rouge" in doc_data["metrics"]:
                    has_rouge = True
                    break
            
            if has_rouge:
                output_path = None
                if args.output:
                    summarizer_type = results.get("summarizer_type", f"result{i}")
                    output_path = os.path.join(
                        args.output, 
                        f"rouge_breakdown_{summarizer_type}.{args.format}"
                    )
                
                plot_rouge_breakdown(
                    results,
                    figsize=figsize,
                    output_path=output_path,
                    max_docs=args.max_docs
                )
    
    # If no output directory specified, show plots
    if not args.output:
        plt.show()
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    import sys
    sys.exit(main()) 