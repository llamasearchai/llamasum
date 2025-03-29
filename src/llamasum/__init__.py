"""LlamaSum: Advanced text summarization with hierarchical extractive-abstractive approaches."""

__version__ = "0.1.0"

from llamasum.summarizer import LlamaSummarizer
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.evaluation import evaluate_summary
from llamasum.benchmark import run_benchmark
from llamasum.visualize import plot_metrics_comparison, plot_document_metrics, plot_rouge_breakdown
from llamasum.report import generate_report_html, generate_report_from_files

__all__ = [
    "LlamaSummarizer", 
    "HierarchicalSummarizer", 
    "MultiDocSummarizer",
    "evaluate_summary",
    "run_benchmark",
    "plot_metrics_comparison",
    "plot_document_metrics",
    "plot_rouge_breakdown",
    "generate_report_html",
    "generate_report_from_files"
] 