# LlamaSum Report Generation Guide

This guide explains how to use LlamaSum's report generation module to create interactive HTML reports from benchmark results.

## Overview

The report generation module allows you to:

1. Create comprehensive HTML reports from benchmark results
2. Compare multiple summarization approaches in a single view
3. Visualize performance metrics with embedded charts
4. Share results with stakeholders in an accessible format
5. Generate reports via command line or Python API

## Installation Requirements

To use the report generation functionality, install the required dependencies:

```bash
# Install visualization and reporting dependencies
pip install llamasum[viz]

# Or install all optional dependencies
pip install llamasum[all]
```

These dependencies include:
- matplotlib
- pandas
- seaborn (for advanced visualizations)

## Basic Usage

### Command Line Interface

The simplest way to generate a report is using the command-line interface:

```bash
llamasum-report results1.json results2.json -o benchmark_report.html --title "My Benchmark Report"
```

#### Command Line Options

- `results_file`: One or more benchmark result JSON files (required)
- `-o, --output`: Path to save the HTML report (if omitted, report is only generated but not saved)
- `--title`: Title for the report (default: "LlamaSum Benchmark Report")
- `--open`: Open the generated report in the default web browser

### Python API

```python
# Generate a report from result files
from llamasum.report import generate_report_from_files

generate_report_from_files(
    file_paths=["results1.json", "results2.json"],
    output_path="benchmark_report.html",
    report_title="My Benchmark Report"
)

# Generate a report from result objects
from llamasum.report import generate_report_html

generate_report_html(
    results_list=[result1, result2],
    output_path="benchmark_report.html",
    report_title="My Benchmark Report"
)
```

## Report Structure

The generated HTML reports include:

### 1. Header Section
- Report title and generation timestamp
- Summary of the benchmarked configurations

### 2. Comparison Section
- Bar charts comparing key metrics across models:
  - ROUGE scores
  - Processing time
  - Compression ratio
  - Overall quality score

### 3. Model-Specific Sections
For each model/configuration:
- Configuration details (collapsible)
- Performance metrics in a formatted table
- Document metrics visualization
- ROUGE scores breakdown
- Document-level results (collapsible)

### 4. Interactive Elements
- Toggle buttons to show/hide detailed information
- Embedded visualizations
- Formatted tables for easier reading

## Example Workflow

A complete end-to-end workflow using the reporting feature:

### 1. Run Benchmarks
```python
from llamasum.benchmark import run_benchmark

# Run benchmarks with different configurations
result1 = run_benchmark(
    dataset_path="dataset.json",
    output_path="result_basic.json",
    summarizer_type="basic",
    extractive_method="tfidf"
)

result2 = run_benchmark(
    dataset_path="dataset.json",
    output_path="result_hierarchical.json",
    summarizer_type="hierarchical",
    levels=3
)
```

### 2. Generate Report
```python
from llamasum.report import generate_report_from_files
import webbrowser
import os

# Generate HTML report
report_path = "benchmark_report.html"
generate_report_from_files(
    file_paths=["result_basic.json", "result_hierarchical.json"],
    output_path=report_path,
    report_title="Basic vs. Hierarchical Summarization"
)

# Open in browser
webbrowser.open('file://' + os.path.abspath(report_path))
```

## Using the Example Script

LlamaSum provides a ready-to-use example script for report generation:

```bash
# Generate a report with demo data
python -m llamasum.examples.generate_report --output-dir ./reports

# Generate from existing results
python -m llamasum.examples.generate_report --results-dir ./benchmark_results --output-dir ./reports

# Generate and open in browser
python -m llamasum.examples.generate_report --open
```

The example script can be modified for your specific needs:

```python
import sys
from pathlib import Path
from llamasum.report import generate_report_from_files

# Define paths
results_dir = Path("./my_results")
output_path = Path("./my_report.html")

# Find all result files
result_files = list(results_dir.glob("*.json"))

# Generate report
if result_files:
    generate_report_from_files(
        file_paths=result_files,
        output_path=str(output_path),
        report_title=f"Summarization Benchmark Report - {len(result_files)} Models"
    )
    print(f"Report generated at {output_path}")
else:
    print("No result files found")
    sys.exit(1)
```

## Advanced Usage

### Customizing the Report

The report generation functions support customizations:

```python
from llamasum.report import generate_report_html

# Load results
results_list = [...]  # Your benchmark results

# Generate a customized report
generate_report_html(
    results_list=results_list,
    output_path="custom_report.html",
    report_title="Custom Analysis Report",
    # Advanced options could be added in future versions
)
```

### Report Sharing

The generated HTML reports are self-contained with embedded images (as base64) and can be easily shared:

1. Email the HTML file directly to stakeholders
2. Host on an internal web server
3. Include in project documentation
4. Present during team meetings

## Interpreting Reports

### Key Metrics to Look For

1. **ROUGE Scores**:
   - Higher scores indicate better alignment with reference summaries
   - ROUGE-1 measures unigram overlap
   - ROUGE-2 measures bigram overlap
   - ROUGE-L measures longest common subsequence

2. **Processing Time**:
   - Lower is better, but consider the trade-off with quality
   - Useful for applications with real-time requirements

3. **Compression Ratio**:
   - Measures how concise the summaries are
   - Lower values indicate more compressed summaries

4. **Overall Score**:
   - Combined metric that balances quality and efficiency
   - Useful for quick comparisons between models

### Document-Level Analysis

The document-level results section allows you to:
- Identify which types of documents perform well or poorly
- Analyze individual summaries and their metrics
- Spot patterns in performance across different document characteristics

## Integration with Other LlamaSum Features

The reporting module integrates seamlessly with other LlamaSum components:

```python
# Complete workflow example
from llamasum.benchmark import run_benchmark
from llamasum.visualize import plot_metrics_comparison
from llamasum.report import generate_report_html

# 1. Run benchmarks
results = []
for model in ["model1", "model2", "model3"]:
    result = run_benchmark(
        dataset_path="dataset.json",
        output_path=f"result_{model}.json",
        model_name=model
    )
    results.append(result)

# 2. Create individual visualizations
plot_metrics_comparison(results, output_path="comparison.png")

# 3. Generate comprehensive report
generate_report_html(
    results_list=results,
    output_path="full_report.html",
    report_title="Model Comparison Report"
)
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Error: "Visualization dependencies not available"
   - Solution: Install required packages with `pip install llamasum[viz]`

2. **Empty or Incomplete Reports**:
   - Check that the result files contain all expected metrics
   - Ensure the benchmark ran successfully
   - Verify that reference summaries were available for ROUGE scoring

3. **Visualization Errors**:
   - Update matplotlib and pandas to the latest versions
   - Check for conflicting packages that might affect visualization

### Getting Help

If you encounter issues with report generation:

1. Check the LlamaSum documentation
2. Look for error messages in the Python console
3. Run with logging enabled:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
from llamasum.report import generate_report_from_files
```

## Conclusion

The report generation module provides a powerful way to analyze and share benchmark results. By generating interactive HTML reports, you can easily compare different summarization approaches and make data-driven decisions about which models or configurations best suit your needs.

Use this feature to:
- Compare performance across different models
- Analyze strengths and weaknesses of each approach
- Share results with stakeholders in an accessible format
- Document your experiments and findings 