# LlamaSum Benchmarking Guide

This guide explains how to use LlamaSum's benchmarking functionality to evaluate and compare different summarization approaches.

## Overview

The benchmarking module allows you to:

1. Evaluate summarization performance on custom datasets
2. Compare different summarization models and configurations
3. Generate standardized metrics for performance comparison
4. Save and analyze detailed results

## Dataset Format

LlamaSum supports benchmarking on datasets in JSON or CSV format.

### JSON Format

```json
{
  "name": "Sample Dataset",
  "description": "Description of the dataset",
  "documents": [
    {
      "id": "doc1",
      "title": "Document 1 Title",
      "text": "Full text of document 1...",
      "reference_summary": "Reference summary for document 1..."
    },
    {
      "id": "doc2",
      "title": "Document 2 Title",
      "text": "Full text of document 2...",
      "reference_summary": "Reference summary for document 2..."
    }
  ]
}
```

For multi-document examples, use the `texts` field instead of `text`:

```json
{
  "id": "multi1",
  "title": "Multi-Document Example",
  "texts": [
    "Text of first document...",
    "Text of second document...",
    "Text of third document..."
  ],
  "reference_summary": "Reference summary for all documents..."
}
```

### CSV Format

CSV files should include at least a `text` column and optionally `id` and `reference_summary` columns:

```
id,text,reference_summary
doc1,"Full text of document 1...","Reference summary for document 1..."
doc2,"Full text of document 2...","Reference summary for document 2..."
```

## Running Benchmarks

### Using the Command Line

The simplest way to run a benchmark is using the command-line interface:

```bash
llamasum-benchmark dataset.json --output results.json --type basic
```

#### Common Options

- `--type`: Type of summarizer (`basic`, `hierarchical`, or `multidoc`)
- `--model`: Model name or path (defaults to a small DistilBART model)
- `--max-length`: Maximum summary length
- `--min-length`: Minimum summary length
- `--extractive-method`: Method for extraction (`tfidf`, `textrank`, or `position`)
- `--extractive-ratio`: Ratio of text to extract (0.0-1.0)
- `--device`: Device to use (`cpu` or `cuda`)

#### Hierarchical Options

- `--levels`: Number of hierarchical levels (default: 3)
- `--include-ultra-short`: Include ultra-short summary

#### Multi-document Options

- `--strategy`: Strategy for multi-document summarization
- `--redundancy-threshold`: Threshold for redundancy removal
- `--delimiter`: Delimiter for splitting texts (when using single text field)

### Using the Python API

```python
from llamasum.benchmark import run_benchmark

# Basic summarization benchmark
results = run_benchmark(
    dataset_path="path/to/dataset.json",
    output_path="results.json",
    summarizer_type="basic",
    model_name="sshleifer/distilbart-cnn-12-6",
    max_length=150,
    min_length=50,
    extractive_method="tfidf",
    extractive_ratio=0.3,
    device="cpu"
)

# Hierarchical summarization benchmark
results = run_benchmark(
    dataset_path="path/to/dataset.json",
    output_path="results.json",
    summarizer_type="hierarchical",
    model_name="sshleifer/distilbart-cnn-12-6",
    levels=3,
    include_ultra_short=True
)

# Multi-document summarization benchmark
results = run_benchmark(
    dataset_path="path/to/dataset.json",
    output_path="results.json",
    summarizer_type="multidoc",
    strategy="extract_then_summarize",
    redundancy_threshold=0.7
)
```

## Understanding Results

Benchmark results include several metrics:

### Overall Metrics

- `avg_processing_time`: Average time to generate summaries
- `avg_compression_ratio`: Average ratio of summary length to original text length
- `avg_overall_score`: Average overall quality score (0-1)
- `avg_rouge_1_f`: Average ROUGE-1 F1 score (if available)
- `avg_rouge_l_f`: Average ROUGE-L F1 score (if available)

### Document-level Metrics

For each document, the following metrics are calculated:

- `compression_ratio`: Ratio of summary length to original text length
- `density`: Information density of the summary
- `readability`: Readability scores (Flesch Reading Ease, etc.)
- `similarity_to_original`: Similarity between summary and original text
- `similarity_to_reference`: Similarity between summary and reference (if available)
- `rouge`: ROUGE scores compared to reference (if available)
- `bleu`: BLEU score compared to reference (if available)
- `overall_score`: Weighted combination of metrics (0-1)

## Sample Results Structure

```json
{
  "dataset": "sample_dataset.json",
  "summarizer_type": "basic",
  "config": {
    "model_name": "sshleifer/distilbart-cnn-12-6",
    "max_length": 150,
    "extractive_method": "tfidf",
    "extractive_ratio": 0.3
  },
  "document_count": 3,
  "start_time": 1679012345.6789,
  "end_time": 1679012350.1234,
  "total_time": 4.4445,
  "document_results": {
    "doc1": {
      "summary": "Generated summary text...",
      "processing_time": 1.5,
      "metrics": {
        "compression_ratio": 0.25,
        "density": 0.68,
        "readability": {
          "flesch_reading_ease": 65.4,
          "automated_readability_index": 8.2
        },
        "similarity_to_original": 0.72,
        "similarity_to_reference": 0.81,
        "rouge": {
          "rouge-1": {"p": 0.75, "r": 0.78, "f": 0.76},
          "rouge-2": {"p": 0.52, "r": 0.55, "f": 0.53},
          "rouge-l": {"p": 0.70, "r": 0.73, "f": 0.71}
        },
        "bleu": 0.45,
        "overall_score": 0.78
      }
    },
    "doc2": {
      // Similar structure for other documents
    }
  },
  "overall_metrics": {
    "avg_processing_time": 1.8,
    "avg_compression_ratio": 0.27,
    "avg_rouge_1_f": 0.75,
    "avg_rouge_l_f": 0.70,
    "avg_overall_score": 0.76
  }
}
```

## Analyzing Results

To compare different models or configurations:

1. Run multiple benchmarks with different parameters
2. Compare the overall metrics (especially `avg_overall_score` and ROUGE scores)
3. Analyze per-document results to identify strengths and weaknesses

For domain-specific evaluations:

1. Create a dataset with documents from your domain
2. Include reference summaries if possible
3. Run benchmarks with different models/configurations
4. Select the approach that performs best on your data

## Visualizing Results

LlamaSum provides visualization tools to help interpret benchmark results:

### Command Line Visualization

```bash
# Generate all visualization types
llamasum-viz results1.json results2.json --output viz_dir/ --type all

# Generate only comparison charts
llamasum-viz results1.json results2.json --output viz_dir/ --type comparison

# Generate only document-level metrics
llamasum-viz results1.json --output viz_dir/ --type documents

# Generate ROUGE score breakdowns
llamasum-viz results1.json --output viz_dir/ --type rouge

# Specify output format
llamasum-viz results1.json --output viz_dir/ --format pdf

# Focus on specific metrics
llamasum-viz results1.json results2.json --metrics avg_rouge_l_f avg_overall_score
```

### Python API Visualization

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
plot_metrics_comparison(
    results, 
    metrics=["avg_rouge_l_f", "avg_processing_time", "avg_overall_score"],
    figsize=(12, 8),
    output_path="comparison.png"
)

# Visualize document-level metrics
plot_document_metrics(
    results[0],
    metrics=["overall_score", "compression_ratio"],
    max_docs=5,
    output_path="document_metrics.png"
)

# Visualize ROUGE scores
plot_rouge_breakdown(
    results[0],
    max_docs=3,
    output_path="rouge_breakdown.png"
)
```

### Visualization Types

1. **Metrics Comparison**: Bar charts comparing overall metrics across different summarizers or configurations
2. **Document Metrics**: Charts showing metrics for individual documents within a benchmark
3. **ROUGE Breakdown**: Detailed view of ROUGE metrics (precision, recall, F1) for top documents

### Example Visualizations

Here's what you can expect from the visualization tools:

- **Metrics Comparison**: Horizontal bar charts showing metrics like ROUGE scores, processing time, and overall scores across different summarization approaches.
- **Document Metrics**: Bar charts showing how specific documents performed on metrics like compression ratio, similarity, and readability.
- **ROUGE Breakdown**: Grouped bar charts showing precision, recall, and F1 scores for ROUGE-1 and ROUGE-L across top-performing documents.

### Advanced Visualizations

LlamaSum also provides advanced visualization capabilities for in-depth analysis:

```bash
# Run advanced visualizations
python -m llamasum.examples.advanced_visualization --output-dir advanced_viz
```

The advanced visualization script generates:

1. **Radar Charts**: Compare multiple metrics across different models in a normalized polar chart
2. **Processing Time Boxplots**: Analyze the distribution of processing times for each model
3. **Performance Tradeoffs**: Scatter plots showing speed vs. quality tradeoffs
4. **Document-specific Comparisons**: See how each model performs on different document types

To create custom advanced visualizations:

```python
from llamasum.visualize import load_multiple_results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load results
results = load_multiple_results(["results1.json", "results2.json"])

# Create a DataFrame for analysis
metrics_data = []
for r in results:
    metrics_data.append({
        "model": r.get("summarizer_type"),
        "method": r.get("config", {}).get("extractive_method", ""),
        **r.get("overall_metrics", {})
    })

df = pd.DataFrame(metrics_data)

# Create custom visualization
plt.figure(figsize=(10, 6))
sns.barplot(x="model", y="avg_rouge_l_f", hue="method", data=df)
plt.title("ROUGE-L F1 Score by Model and Method")
plt.tight_layout()
plt.savefig("custom_viz.png")
```

### Visualization Requirements

To use the visualization features, install the required dependencies:

```bash
# Install visualization dependencies
pip install llamasum[viz]

# Or install all optional dependencies
pip install llamasum[all]
```

## HTML Reports

LlamaSum can generate comprehensive HTML reports that combine visualizations, metrics, and document-level results into a single interactive document.

### Generating Reports

From the command line:

```bash
# Generate HTML report from benchmark results
llamasum-report results1.json results2.json -o benchmark_report.html --title "My Benchmark Report"
```

From Python:

```python
from llamasum.report import generate_report_from_files

# Generate HTML report from result files
generate_report_from_files(
    file_paths=["results1.json", "results2.json"],
    output_path="benchmark_report.html",
    report_title="My Benchmark Report"
)

# Or generate HTML report directly from result objects
from llamasum.report import generate_report_html

generate_report_html(
    results_list=[result1, result2],
    output_path="benchmark_report.html",
    report_title="My Benchmark Report"
)
```

### Report Features

The HTML reports include:

1. **Summary Section**: Overview of the benchmark configurations
2. **Model Comparison**: Visualizations comparing metrics across models
3. **Model Details**: Sections for each model with:
   - Configuration details (collapsible)
   - Performance metrics with formatted tables
   - Document metrics visualizations
   - ROUGE score breakdowns
   - Document-level results (collapsible)
4. **Interactive Elements**: Expandable sections and formatted data tables
5. **Responsive Design**: Works on desktop and mobile devices

### Running the Example

A complete example script is provided:

```bash
# Generate a report with demo data
python -m llamasum.examples.generate_report --output-dir ./reports

# Generate a report from existing results
python -m llamasum.examples.generate_report --results-dir ./benchmark_results --output-dir ./reports

# Generate and open in browser
python -m llamasum.examples.generate_report --open
```

### Sharing Reports

The generated HTML reports are self-contained (images are embedded as base64) and can be easily shared:

1. Email the HTML file to colleagues
2. Host on a web server or file-sharing service
3. Include in project documentation
4. Present during meetings or reviews

## Best Practices

1. **Use reference summaries** when possible for more accurate evaluation
2. **Balance dataset size** - larger datasets provide more reliable results but take longer to process
3. **Compare fairly** - use the same dataset and evaluation metrics when comparing different approaches
4. **Consider use case** - optimize for metrics that matter most for your application (speed vs. quality)
5. **Report all metrics** - don't cherry-pick favorable results
6. **Analyze failures** - examine documents where performance is poor

## Creating Custom Datasets

For the best evaluation:

1. Include diverse documents representing your use case
2. Provide high-quality reference summaries when possible
3. For multi-document datasets, group related documents using the `texts` field
4. Include documents of varying lengths to test scalability
5. Consider including domain-specific terminology if relevant

## Extended Example

```python
import json
from llamasum.benchmark import run_benchmark

# Run benchmarks with different models
models = [
    "sshleifer/distilbart-cnn-12-6",  # Small, fast model
    "facebook/bart-large-cnn",        # Larger, higher quality
    "t5-small"                        # Alternative architecture
]

results = {}

# Compare models on the same dataset
for model in models:
    model_name = model.split("/")[-1]
    print(f"Benchmarking {model_name}...")
    
    result = run_benchmark(
        dataset_path="my_dataset.json",
        output_path=f"results_{model_name}.json",
        summarizer_type="basic",
        model_name=model,
        max_length=150
    )
    
    # Store key metrics for comparison
    results[model_name] = {
        "avg_processing_time": result["overall_metrics"]["avg_processing_time"],
        "avg_rouge_l_f": result["overall_metrics"].get("avg_rouge_l_f", 0),
        "avg_overall_score": result["overall_metrics"]["avg_overall_score"]
    }

# Print comparison table
print("\nModel Comparison:")
print("-" * 80)
print(f"{'Model':<20} {'Proc. Time':<15} {'ROUGE-L':<15} {'Overall Score':<15}")
print("-" * 80)

for model, metrics in results.items():
    print(f"{model:<20} {metrics['avg_processing_time']:<15.2f} {metrics['avg_rouge_l_f']:<15.4f} {metrics['avg_overall_score']:<15.4f}")
``` 