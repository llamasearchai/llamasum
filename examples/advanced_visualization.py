#!/usr/bin/env python
"""
Advanced visualization example for LlamaSum benchmarking results.

This script demonstrates more sophisticated visualization techniques
for analyzing and comparing benchmarking results across different
summarization models and configurations.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure llamasum package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llamasum.benchmark import load_dataset, run_benchmark
    from llamasum.visualize import (
        load_multiple_results,
        plot_document_metrics,
        plot_metrics_comparison,
        plot_rouge_breakdown,
    )
except ImportError:
    print("Error: Could not import LlamaSum modules.")
    print("Make sure LlamaSum is installed or in your Python path.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization libraries not available.")
    print("Install required dependencies with: pip install llamasum[viz]")


def create_comparison_dataset():
    """Create a sample dataset for comparison visualization."""

    # Sample dataset with different types of content
    dataset = {
        "name": "LlamaSum Comparison Dataset",
        "description": "Dataset for comparing different summarization approaches",
        "documents": [
            # Technical content
            {
                "id": "tech1",
                "title": "Machine Learning Overview",
                "text": """
                Machine learning is a branch of artificial intelligence that focuses on developing systems 
                that can learn from and make decisions based on data. Machine learning algorithms build 
                mathematical models based on sample data, known as training data, to make predictions or 
                decisions without being explicitly programmed to do so. There are three main types of 
                machine learning: supervised learning, unsupervised learning, and reinforcement learning.
                
                In supervised learning, algorithms are trained using labeled examples, where the desired 
                output is known. The algorithm learns by comparing its actual output with the correct 
                outputs to find errors and modify the model accordingly. Common supervised learning 
                algorithms include linear regression, logistic regression, decision trees, random forests, 
                and neural networks.
                
                Unsupervised learning involves training algorithms on data without labeled responses. 
                Instead, the algorithm attempts to identify patterns and relationships in the data on its own. 
                Clustering is a common unsupervised learning technique, where algorithms group similar data 
                points together based on their features. K-means clustering, hierarchical clustering, 
                and DBSCAN are popular clustering algorithms.
                
                Reinforcement learning is a type of machine learning where an agent learns to make decisions 
                by performing actions in an environment to maximize some notion of cumulative reward. 
                The agent receives feedback in the form of rewards or penalties and adjusts its strategy 
                accordingly. Reinforcement learning has been successfully applied to various domains, 
                including game playing, robotics, and autonomous driving.
                """.strip(),
                "reference_summary": """
                Machine learning allows systems to learn from data and make decisions without explicit 
                programming. The three main types are supervised learning (using labeled data), 
                unsupervised learning (finding patterns without labels), and reinforcement learning 
                (learning through environment interaction and rewards).
                """.strip(),
            },
            # News content
            {
                "id": "news1",
                "title": "Climate Summit Results",
                "text": """
                The Global Climate Summit concluded yesterday with nations agreeing to significantly 
                reduce carbon emissions by 2030. The landmark agreement, signed by 195 countries, 
                commits developed nations to cutting emissions by 45% from 2010 levels, while 
                developing nations agreed to a 30% reduction over the same period.
                
                The summit also established a $100 billion annual fund to help vulnerable nations 
                adapt to climate change impacts and transition to renewable energy sources. 
                The fund will be financed primarily by developed countries and will begin 
                disbursements next year.
                
                "This agreement represents a turning point in our collective response to the climate 
                crisis," said UN Secretary-General Antonio Guterres. "For the first time, we have truly 
                global commitments that reflect the urgency of our situation."
                
                Environmental groups have generally praised the agreement but emphasized that 
                implementation will be crucial. "The targets are ambitious, which is what we need, 
                but now comes the hard part of actually meeting them," said Greenpeace spokesperson 
                Emma Rodriguez.
                
                The agreement also includes provisions for regular progress reviews every two years 
                and a mechanism for increasing targets if scientific assessments indicate greater 
                reductions are necessary to limit global warming to 1.5 degrees Celsius.
                """.strip(),
                "reference_summary": """
                195 countries agreed at the Global Climate Summit to reduce carbon emissions by 2030, 
                with developed nations committing to 45% cuts and developing nations to 30% cuts from 
                2010 levels. A $100 billion annual fund will help vulnerable nations adapt to climate 
                change and transition to renewable energy.
                """.strip(),
            },
            # Multi-document example
            {
                "id": "multi1",
                "title": "Space Exploration Updates",
                "texts": [
                    """
                    NASA's Artemis program aims to return humans to the Moon by 2025, establishing a 
                    sustainable presence on the lunar surface. The program will use the Space Launch 
                    System (SLS) rocket and Orion spacecraft to transport astronauts. Artemis also 
                    plans to land the first woman and first person of color on the Moon.
                    """.strip(),
                    """
                    SpaceX has made significant progress with its Starship program, designed for 
                    missions to the Moon and eventually Mars. The fully reusable spacecraft has 
                    completed several high-altitude test flights. SpaceX is collaborating with NASA 
                    to develop a lunar lander variant of Starship for the Artemis program.
                    """.strip(),
                    """
                    China's space program has expanded rapidly with the successful deployment of 
                    the Tiangong space station. The modular station now hosts rotating crews of 
                    taikonauts conducting scientific research. China has also landed rovers on the 
                    Moon and Mars, with plans for a crewed lunar mission by 2030.
                    """.strip(),
                ],
                "reference_summary": """
                Multiple space programs are advancing exploration: NASA's Artemis aims to return 
                humans to the Moon by 2025, SpaceX is developing the reusable Starship for lunar 
                and Mars missions, and China has established the Tiangong space station while 
                planning for a crewed lunar mission by 2030.
                """.strip(),
            },
        ],
    }

    return dataset


def run_comparative_benchmarks(output_dir):
    """Run benchmarks with different configurations for comparison."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    dataset = create_comparison_dataset()
    dataset_path = os.path.join(output_dir, "comparison_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # Define configurations to compare
    configs = [
        # Basic summarizer with different extraction methods
        {
            "name": "basic_position",
            "type": "basic",
            "extractive_method": "position",
            "extractive_ratio": 0.3,
        },
        {
            "name": "basic_tfidf",
            "type": "basic",
            "extractive_method": "tfidf",
            "extractive_ratio": 0.3,
        },
        # Hierarchical summarizer
        {"name": "hierarchical", "type": "hierarchical", "levels": 2},
        # Multi-document summarizer
        {"name": "multidoc", "type": "multidoc", "strategy": "cluster_then_summarize"},
    ]

    # Run benchmarks
    results = []
    for config in configs:
        print(f"Running benchmark for {config['name']}...")
        result_path = os.path.join(output_dir, f"results_{config['name']}.json")

        # Extract config for run_benchmark
        kwargs = {k: v for k, v in config.items() if k != "name"}

        # Run benchmark
        result = run_benchmark(
            dataset_path=dataset_path,
            output_path=result_path,
            summarizer_type=config["type"],
            **kwargs,
        )

        results.append(result)

    return results


def create_interactive_comparison(results_list, output_dir):
    """Create interactive comparison visualizations using advanced features."""
    if not VISUALIZATION_AVAILABLE:
        print("Cannot create interactive visualizations without required dependencies.")
        return

    # Set the style
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Create dataframes for analysis
    df_results = pd.DataFrame(
        [
            {
                "model": r.get("summarizer_type", "unknown"),
                "method": r.get("config", {}).get("extractive_method", ""),
                "documents": r.get("document_count", 0),
                "processing_time": r.get("total_time", 0),
                **r.get("overall_metrics", {}),
            }
            for r in results_list
        ]
    )

    # Create combined document metrics dataframe
    doc_rows = []
    for i, result in enumerate(results_list):
        model_name = f"{result.get('summarizer_type')}_{result.get('config', {}).get('extractive_method', '')}"

        for doc_id, doc_data in result.get("document_results", {}).items():
            if "metrics" in doc_data:
                doc_rows.append(
                    {
                        "model": model_name,
                        "document_id": doc_id,
                        "processing_time": doc_data.get("processing_time", 0),
                        **{
                            k: v
                            for k, v in doc_data["metrics"].items()
                            if not isinstance(v, dict)
                        },
                    }
                )

                # Extract ROUGE scores if available
                if "rouge" in doc_data["metrics"]:
                    for rouge_type, scores in doc_data["metrics"]["rouge"].items():
                        if isinstance(scores, dict):
                            for metric, value in scores.items():
                                doc_rows.append(
                                    {
                                        "model": model_name,
                                        "document_id": doc_id,
                                        "metric": f"{rouge_type}_{metric}",
                                        "value": value,
                                    }
                                )

    df_docs = pd.DataFrame(doc_rows)

    # 1. Create radar chart of metrics
    metrics = [
        "avg_overall_score",
        "avg_rouge_1_f",
        "avg_rouge_l_f",
        "avg_compression_ratio",
        "avg_similarity_to_reference",
    ]
    metrics = [m for m in metrics if m in df_results.columns]

    if metrics:
        # Normalize metrics for radar chart
        df_radar = df_results.copy()
        for metric in metrics:
            if metric in df_radar.columns:
                max_val = df_radar[metric].max()
                if max_val > 0:
                    df_radar[metric] = df_radar[metric] / max_val

        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # Number of metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Plot each model
        for i, row in df_radar.iterrows():
            values = [row[metric] if metric in row else 0 for metric in metrics]
            values += values[:1]  # Close the loop

            ax.plot(
                angles,
                values,
                linewidth=2,
                linestyle="solid",
                label=f"{row['model']}_{row['method']}",
            )
            ax.fill(angles, values, alpha=0.1)

        # Set labels and legend
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [m.replace("avg_", "").replace("_", " ").title() for m in metrics]
        )
        ax.set_title("Normalized Performance Metrics", size=15, y=1.1)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        # Save the figure
        plt.savefig(os.path.join(output_dir, "radar_chart.png"), bbox_inches="tight")
        plt.close()

    # 2. Document processing time comparison
    if "processing_time" in df_docs.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="model", y="processing_time", data=df_docs)
        plt.title("Document Processing Time by Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "processing_time_boxplot.png"))
        plt.close()

    # 3. Comparison by document type
    if "overall_score" in df_docs.columns and "document_id" in df_docs.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(x="document_id", y="overall_score", hue="model", data=df_docs)
        plt.title("Overall Score by Document and Model")
        plt.xlabel("Document")
        plt.ylabel("Overall Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "document_comparison.png"))
        plt.close()

    # 4. Create a performance vs. speed trade-off plot
    if all(
        col in df_results.columns
        for col in ["avg_processing_time", "avg_overall_score"]
    ):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="avg_processing_time",
            y="avg_overall_score",
            hue="model",
            size="documents",
            sizes=(100, 200),
            data=df_results,
        )
        plt.title("Performance vs. Speed Trade-off")
        plt.xlabel("Average Processing Time (s)")
        plt.ylabel("Average Overall Score")

        # Add annotations
        for i, row in df_results.iterrows():
            label = f"{row['model']}"
            if row.get("method"):
                label += f"_{row['method']}"
            plt.annotate(
                label,
                (row["avg_processing_time"], row["avg_overall_score"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_tradeoff.png"))
        plt.close()

    print(f"Advanced visualizations saved to {output_dir}")


def main():
    """Run the advanced visualization example."""
    parser = argparse.ArgumentParser(
        description="Advanced LlamaSum visualization example"
    )

    parser.add_argument(
        "--results-dir",
        help="Directory containing existing benchmark results (if not provided, will run benchmarks)",
    )

    parser.add_argument(
        "--output-dir",
        default="./advanced_viz",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    # Check for visualization dependencies
    if not VISUALIZATION_AVAILABLE:
        print(
            "Warning: Full visualization capabilities require additional dependencies."
        )
        print("Install with: pip install llamasum[viz]")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either load existing results or run benchmarks
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return 1

        # Load results
        result_files = list(results_dir.glob("results_*.json"))
        if not result_files:
            print(f"Error: No result files found in {results_dir}")
            return 1

        print(f"Loading {len(result_files)} benchmark results...")
        results_list = load_multiple_results(result_files)
    else:
        # Run benchmarks
        print("Running comparative benchmarks...")
        results_list = run_comparative_benchmarks(str(output_dir))

    if not results_list:
        print("Error: No valid benchmark results found")
        return 1

    print(f"Creating standard visualizations in {output_dir}...")

    # Standard visualizations from the visualize module
    if len(results_list) > 1:
        plot_metrics_comparison(
            results_list, output_path=str(output_dir / "metrics_comparison.png")
        )

    for result in results_list:
        summarizer_type = result.get("summarizer_type", "unknown")
        config = result.get("config", {})
        name = f"{summarizer_type}_{config.get('extractive_method', '')}"
        name = name.strip("_")

        plot_document_metrics(
            result, output_path=str(output_dir / f"document_metrics_{name}.png")
        )

        # Check if ROUGE scores are available
        has_rouge = False
        for doc_data in result.get("document_results", {}).values():
            if "metrics" in doc_data and "rouge" in doc_data["metrics"]:
                has_rouge = True
                break

        if has_rouge:
            plot_rouge_breakdown(
                result, output_path=str(output_dir / f"rouge_breakdown_{name}.png")
            )

    # Create advanced visualizations
    print("Creating advanced interactive visualizations...")
    create_interactive_comparison(results_list, str(output_dir))

    print(f"All visualizations saved to {output_dir}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
