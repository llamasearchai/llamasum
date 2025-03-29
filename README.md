# LlamaSum: Advanced Text Summarization

**✅ PROJECT STATUS: COMPLETE** - All planned features have been implemented, including core summarization, benchmarking, visualization, and reporting capabilities.

LlamaSum is a powerful text summarization library that combines extractive and abstractive approaches within a hierarchical framework to generate high-quality summaries for long documents and multi-document collections.

## Features

- **Multiple Summarization Approaches**: Extractive, abstractive, and hybrid summarization
- **Hierarchical Summarization**: Generate summaries at different levels of detail
- **Multi-Document Summarization**: Summarize collections of related documents
- **Hybrid Approaches**: Combine extractive and abstractive methods for better results
- **Controllable Generation**: Adjust parameters to control summary length and style
- **Multiple Interfaces**: Python API, command-line, web UI, and REST API
- **Comprehensive Evaluation**: Compare summaries using multiple metrics
- **Benchmarking Tools**: Compare and evaluate performance across different models and datasets
- **Visualization Tools**: Generate charts and graphs to analyze benchmark results

## Installation

```bash
pip install llamasum
```

For development or extra features:

```bash
# Development dependencies
pip install llamasum[dev]

# Extra features (Streamlit UI, API server, optimization)
pip install llamasum[extra]

# All dependencies
pip install llamasum[dev,extra]
```

## Quick Start

### Basic Usage

```python
from llamasum import LlamaSummarizer

# Initialize the summarizer
summarizer = LlamaSummarizer()

# Summarize a single document
text = """
Climate change is one of the most pressing challenges facing humanity today. Rising global temperatures, 
caused primarily by greenhouse gas emissions, are leading to more frequent and severe weather events, 
sea level rise, and disruptions to ecosystems worldwide. Addressing climate change requires a multi-faceted 
approach, including transitioning to renewable energy sources, improving energy efficiency, 
and developing sustainable transportation and agricultural systems. International cooperation, 
through agreements like the Paris Climate Accord, is essential for coordinating global efforts 
to reduce emissions and mitigate the impacts of climate change. Individuals can also contribute 
by reducing their carbon footprint through everyday choices.
"""

summary = summarizer.summarize(text, ratio=0.3)
print(summary)
```

### Hierarchical Summarization

```python
from llamasum import HierarchicalSummarizer

# Initialize hierarchical summarizer
summarizer = HierarchicalSummarizer()

# Create multi-level summary
result = summarizer.summarize(long_text, levels=3)

# Access different summary levels
print("Level 1 (detailed):", result["level_1"])
print("Level 2 (medium):", result["level_2"])
print("Level 3 (concise):", result["level_3"])
```

### Multi-Document Summarization

```python
from llamasum import MultiDocSummarizer

# Initialize multi-document summarizer
summarizer = MultiDocSummarizer()

# List of documents
documents = [doc1, doc2, doc3]

# Generate summary across all documents
summary = summarizer.summarize(documents)
print(summary)
```

### Benchmarking

You can benchmark and compare different summarization approaches:

```python
from llamasum.benchmark import run_benchmark
from llamasum.visualize import plot_metrics_comparison

# Run benchmarks with different configurations
results1 = run_benchmark(
    dataset_path="my_dataset.json",
    output_path="results_basic.json",
    summarizer_type="basic",
    extractive_method="tfidf"
)

results2 = run_benchmark(
    dataset_path="my_dataset.json",
    output_path="results_hierarchical.json",
    summarizer_type="hierarchical",
    levels=3
)

# Visualize comparison of results
plot_metrics_comparison([results1, results2], output_path="comparison.png")
```

Or using the command line:

```bash
# Run benchmark
llamasum-benchmark dataset.json --output results.json --type hierarchical

# Visualize results
llamasum-viz results.json --output visualizations/ --type all
```

### Visualization

LlamaSum provides tools to visualize benchmark results:

```python
from llamasum.visualize import (
    load_multiple_results,
    plot_metrics_comparison,
    plot_document_metrics,
    plot_rouge_breakdown
)

# Load benchmark results
results = load_multiple_results(["results1.json", "results2.json"])

# Compare metrics across different summarizers
plot_metrics_comparison(results, output_path="comparison.png")

# Visualize document-level metrics
plot_document_metrics(results[0], output_path="document_metrics.png")

# Visualize ROUGE scores
plot_rouge_breakdown(results[0], output_path="rouge_breakdown.png")
```

From the command line:

```bash
# Generate all visualizations
llamasum-viz results1.json results2.json --output viz_output/ --type all

# Compare specific metrics
llamasum-viz results1.json results2.json --metrics avg_rouge_l_f avg_processing_time
```

For advanced visualizations, use the provided example script:

```bash
# Run advanced visualizations with interactive features
python -m llamasum.examples.advanced_visualization --output-dir advanced_viz

# Use existing benchmark results
python -m llamasum.examples.advanced_visualization --results-dir results/ --output-dir advanced_viz
```

Advanced visualizations include radar charts, performance tradeoff plots, and document-specific analyses.

### HTML Reports

Generate comprehensive HTML reports from benchmark results:

```python
from llamasum.report import generate_report_from_files

# Generate HTML report from benchmark results
generate_report_from_files(
    file_paths=["results1.json", "results2.json"],
    output_path="benchmark_report.html",
    report_title="My Benchmark Report"
)
```

From the command line:

```bash
# Generate HTML report
llamasum-report results1.json results2.json -o report.html --title "LlamaSum Benchmark Results"
```

The reports include:
- Summary of benchmark configurations
- Interactive visualizations of metrics
- Detailed performance metrics for each model
- Document-level results with expandable sections
- Responsive design for easy sharing and viewing

For a complete example:

```bash
# Generate a report and open in browser
python -m llamasum.examples.generate_report --open
```

## Command Line Usage

LlamaSum provides a command-line interface for quick summarization:

```bash
# Basic summarization
llamasum summarize input.txt -o summary.txt

# Hierarchical summarization
llamasum summarize input.txt --hierarchical --levels 3 -o summary.json

# Multi-document summarization
llamasum summarize doc1.txt doc2.txt doc3.txt --multidoc -o summary.txt
```

## Web UI

Launch the web interface with:

```bash
llamasum ui
```

## API Server

Start the REST API server:

```bash
llamasum api --host localhost --port 8000
```

## License

MIT

## Documentation

Comprehensive documentation is available for all LlamaSum features:

- [Features Overview](docs/features_overview.md) - Complete list of all capabilities
- [Benchmarking Guide](docs/benchmarking.md) - How to evaluate and compare summarization approaches
- [Report Generation Guide](docs/report_generation.md) - Creating HTML reports from benchmark results
- [Project Completion Checklist](docs/project_completion.md) - Summary of implemented components
