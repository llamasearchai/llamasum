"""Module for generating HTML reports from benchmark results."""

import argparse
import base64
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llamasum.visualize import (
    load_multiple_results,
    plot_document_metrics,
    plot_metrics_comparison,
    plot_rouge_breakdown,
)

logger = logging.getLogger(__name__)

# Check if visualization dependencies are available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def figure_to_base64(fig):
    """Convert a matplotlib figure to a base64 string."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    with tempfile.BytesIO() as buffer:
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image_png = buffer.getvalue()

    return base64.b64encode(image_png).decode("utf-8")


def generate_report_html(
    results_list: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    report_title: str = "LlamaSum Benchmark Report",
) -> str:
    """Generate an HTML report from benchmark results.

    Args:
        results_list: List of benchmark results dictionaries
        output_path: Path to save the HTML report (if None, report is returned as string)
        report_title: Title for the report

    Returns:
        HTML report as a string
    """
    if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
        logger.error(
            "Visualization dependencies not available. Install with: pip install llamasum[viz]"
        )
        return "<html><body><h1>Error: Required dependencies not available</h1></body></html>"

    if not results_list:
        logger.error("No valid benchmark results provided")
        return "<html><body><h1>Error: No valid benchmark results provided</h1></body></html>"

    # Generate figures
    comparison_fig = None
    if len(results_list) > 1:
        comparison_fig = plot_metrics_comparison(results_list)

    document_figs = []
    rouge_figs = []

    for result in results_list:
        # Document metrics
        doc_fig = plot_document_metrics(result)
        document_figs.append((result.get("summarizer_type", "unknown"), doc_fig))

        # Check if ROUGE scores are available
        has_rouge = False
        for doc_data in result.get("document_results", {}).values():
            if "metrics" in doc_data and "rouge" in doc_data["metrics"]:
                has_rouge = True
                break

        if has_rouge:
            rouge_fig = plot_rouge_breakdown(result)
            rouge_figs.append((result.get("summarizer_type", "unknown"), rouge_fig))

    # Convert figures to base64
    comparison_b64 = figure_to_base64(comparison_fig) if comparison_fig else None
    document_b64 = [(name, figure_to_base64(fig)) for name, fig in document_figs]
    rouge_b64 = [(name, figure_to_base64(fig)) for name, fig in rouge_figs]

    # Close all figures
    plt.close("all")

    # Create HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
        }}
        .summary-section {{
            margin-bottom: 30px;
        }}
        .viz-section {{
            margin-bottom: 40px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .model-section {{
            background-color: #f8f9fa;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .viz-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .viz-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .metrics-table {{
            margin-top: 20px;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #777;
        }}
        .toggle-button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }}
        .collapsible-content {{
            display: none;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{report_title}</h1>
            <p>Generated on {now}</p>
        </div>
        
        <div class="summary-section">
            <h2>Summary</h2>
            <p>This report compares {len(results_list)} different summarization configurations:</p>
            <ul>
"""

    # Add summary of each configuration
    for i, result in enumerate(results_list):
        summarizer_type = result.get("summarizer_type", "unknown")
        config = result.get("config", {})
        model_name = config.get("model_name", "default")

        html += f"                <li><strong>{summarizer_type.title()}</strong>"

        if "extractive_method" in config:
            html += f" with {config['extractive_method']} extraction"

        if "model_name" in config:
            html += f" using model {model_name}"

        html += "</li>\n"

    html += """            </ul>
        </div>
"""

    # Add comparison visualization if available
    if comparison_b64:
        html += (
            """        <div class="viz-section">
            <h2>Model Comparison</h2>
            <div class="viz-container">
                <img src="data:image/png;base64,"""
            + comparison_b64
            + """" alt="Metrics Comparison">
            </div>
        </div>
"""
        )

    # Add section for each model
    for i, result in enumerate(results_list):
        summarizer_type = result.get("summarizer_type", "unknown")
        config = result.get("config", {})

        html += f"""        <div class="model-section">
            <h2>{summarizer_type.title()} Summarizer</h2>
            <p>
                <button class="toggle-button" onclick="toggleContent('config-{i}')">Show/Hide Configuration</button>
            </p>
            <div id="config-{i}" class="collapsible-content">
                <h3>Configuration</h3>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
"""

        # Add configuration details
        for key, value in config.items():
            html += f"""                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
"""

        html += """                </table>
            </div>
            
            <h3>Performance Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
"""

        # Add metrics
        metrics = result.get("overall_metrics", {})
        for metric, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            html += f"""                <tr>
                    <td>{metric.replace('avg_', '').replace('_', ' ').title()}</td>
                    <td>{formatted_value}</td>
                </tr>
"""

        html += """            </table>
"""

        # Add document metrics visualization if available
        doc_viz = next(
            (b64 for name, b64 in document_b64 if name == summarizer_type), None
        )
        if doc_viz:
            html += f"""            <div class="viz-container">
                <h3>Document Metrics</h3>
                <img src="data:image/png;base64,{doc_viz}" alt="Document Metrics">
            </div>
"""

        # Add ROUGE breakdown visualization if available
        rouge_viz = next(
            (b64 for name, b64 in rouge_b64 if name == summarizer_type), None
        )
        if rouge_viz:
            html += f"""            <div class="viz-container">
                <h3>ROUGE Scores Breakdown</h3>
                <img src="data:image/png;base64,{rouge_viz}" alt="ROUGE Breakdown">
            </div>
"""

        # Add document results with toggle button
        html += f"""            <p>
                <button class="toggle-button" onclick="toggleContent('docs-{i}')">Show/Hide Document Results</button>
            </p>
            <div id="docs-{i}" class="collapsible-content">
                <h3>Document Results</h3>
"""

        # Add document results
        doc_results = result.get("document_results", {})
        for doc_id, doc_data in doc_results.items():
            html += f"""                <div class="document-result">
                    <h4>Document: {doc_id}</h4>
                    <p><strong>Summary:</strong> {doc_data.get('summary', 'N/A')}</p>
                    <p><strong>Processing Time:</strong> {doc_data.get('processing_time', 0):.2f}s</p>
                    
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
"""

            # Add document metrics
            doc_metrics = doc_data.get("metrics", {})
            for metric, value in doc_metrics.items():
                if metric == "rouge" or metric == "readability":
                    continue  # Skip complex nested metrics

                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)

                html += f"""                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td>{formatted_value}</td>
                        </tr>
"""

            html += """                    </table>
                </div>
"""

        html += """            </div>
        </div>
"""

    # Add footer and JavaScript
    html += """        <footer>
            <p>Generated by LlamaSum Benchmarking Tool</p>
        </footer>
    </div>

    <script>
        function toggleContent(id) {
            var content = document.getElementById(id);
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        }
    </script>
</body>
</html>
"""

    # Save or return the HTML
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Report saved to {output_path}")

    return html


def generate_report_from_files(
    file_paths: List[Union[str, Path]],
    output_path: Optional[str] = None,
    report_title: str = "LlamaSum Benchmark Report",
) -> str:
    """Generate an HTML report from benchmark result files.

    Args:
        file_paths: List of paths to benchmark result JSON files
        output_path: Path to save the HTML report
        report_title: Title for the report

    Returns:
        HTML report as a string
    """
    results_list = load_multiple_results(file_paths)
    return generate_report_html(results_list, output_path, report_title)


def main():
    """Command line entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate HTML reports from LlamaSum benchmark results"
    )

    parser.add_argument(
        "results_file", nargs="+", help="Path(s) to benchmark results JSON file(s)"
    )

    parser.add_argument("-o", "--output", help="Path to save the HTML report")

    parser.add_argument(
        "--title", default="LlamaSum Benchmark Report", help="Title for the report"
    )

    args = parser.parse_args()

    # Check if visualization libraries are available
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib not available. Install with: pip install matplotlib")
        return 1

    if not PANDAS_AVAILABLE:
        logger.error("pandas not available. Install with: pip install pandas")
        return 1

    # Generate report
    try:
        generate_report_from_files(
            file_paths=args.results_file,
            output_path=args.output,
            report_title=args.title,
        )

        if args.output:
            print(f"Report generated and saved to {args.output}")
        else:
            # If no output path is specified, print that we're only returning the HTML
            print("Report generated but not saved (no output path specified)")

        return 0
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    import sys

    sys.exit(main())
