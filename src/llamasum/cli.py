"""Command-line interface for LlamaSum."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llamasum.config import (
    HierarchicalConfig,
    MultiDocConfig,
    SummarizerConfig,
    load_config_from_env,
)
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.summarizer import LlamaSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """Read text from a file.

    Args:
        file_path: Path to the file

    Returns:
        Contents of the file as a string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def read_files(file_paths: List[str]) -> List[str]:
    """Read multiple files.

    Args:
        file_paths: List of paths to files

    Returns:
        List of file contents
    """
    return [read_file(file_path) for file_path in file_paths]


def write_output(text: str, output_file: Optional[str] = None):
    """Write text to output file or stdout.

    Args:
        text: Text to write
        output_file: Path to output file (None for stdout)
    """
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Output written to {output_file}")
        except Exception as e:
            logger.error(f"Error writing to {output_file}: {e}")
            print(text)  # Fallback to stdout
    else:
        print(text)


def write_json_output(data: Dict[str, Any], output_file: Optional[str] = None):
    """Write JSON data to output file or stdout.

    Args:
        data: Data to write
        output_file: Path to output file (None for stdout)
    """
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON output written to {output_file}")
        except Exception as e:
            logger.error(f"Error writing JSON to {output_file}: {e}")
            print(json.dumps(data, indent=2))  # Fallback to stdout
    else:
        print(json.dumps(data, indent=2))


def setup_summarizer(
    args,
) -> Union[LlamaSummarizer, HierarchicalSummarizer, MultiDocSummarizer]:
    """Set up the appropriate summarizer based on arguments.

    Args:
        args: Command-line arguments

    Returns:
        Configured summarizer
    """
    # Try to load config from environment or use defaults
    config = load_config_from_env()

    # Override with command-line args
    if args.model:
        config.model_name = args.model
    if args.device:
        config.device = args.device

    # Set max and min length
    config.max_length = args.max_length
    config.min_length = args.min_length

    # Determine which summarizer to use
    if args.hierarchical:
        hier_config = HierarchicalConfig.from_basic_config(config)
        hier_config.levels = args.levels
        hier_config.include_ultra_short = args.ultra_short
        return HierarchicalSummarizer(hier_config)
    elif args.multidoc:
        multidoc_config = MultiDocConfig.from_basic_config(config)
        multidoc_config.strategy = args.strategy
        multidoc_config.redundancy_threshold = args.redundancy
        return MultiDocSummarizer(multidoc_config)
    else:
        return LlamaSummarizer(config)


def process_input(args) -> Dict[str, Any]:
    """Process input and generate summary.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with summary results
    """
    # Get text from file, stdin, or argument
    text = ""
    documents = []

    if args.files:
        if len(args.files) == 1 and not args.multidoc:
            # Single file for basic or hierarchical
            text = read_file(args.files[0])
        else:
            # Multiple files for multidoc
            documents = read_files(args.files)
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        # Read from stdin if available
        text = sys.stdin.read()

    if not text and not documents:
        logger.error("No input text provided")
        return {"error": "No input text provided"}

    # Set up summarizer
    summarizer = setup_summarizer(args)

    # Generate summary
    try:
        if isinstance(summarizer, MultiDocSummarizer):
            # For multidoc, use either the provided documents or split text into documents
            if not documents and text:
                # Split text by double newline if no files provided
                documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]

            if args.queries:
                return summarizer.summarize_with_queries(
                    documents=documents,
                    queries=args.queries,
                    max_length=args.max_length,
                )
            else:
                return summarizer.summarize(
                    documents=documents,
                    max_length=args.max_length,
                    strategy=args.strategy,
                )
        elif isinstance(summarizer, HierarchicalSummarizer):
            if args.by_sections:
                return summarizer.summarize_by_sections(
                    text=text, max_length=args.max_length
                )
            else:
                return summarizer.summarize(
                    text=text, levels=args.levels, max_length=args.max_length
                )
        else:
            # Basic summarizer
            return {
                "summary": summarizer.summarize(
                    text=text, max_length=args.max_length, min_length=args.min_length
                )
            }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return {"error": str(e)}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LlamaSum: Advanced text summarization tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument("files", nargs="*", help="Input file(s) to summarize")
    input_group.add_argument(
        "-t", "--text", help="Text to summarize (alternative to file input)"
    )

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "-m",
        "--model",
        help="Model name or path (defaults to LLAMASUM_MODEL env var or distilbart-cnn)",
    )
    model_group.add_argument(
        "-d",
        "--device",
        help="Device to use (cpu, cuda, cuda:0, etc. defaults to LLAMASUM_DEVICE env var or cpu)",
    )

    # Summary options
    summary_group = parser.add_argument_group("Summary Options")
    summary_group.add_argument(
        "--max-length",
        type=int,
        default=150,
        help="Maximum length of the summary in words",
    )
    summary_group.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum length of the summary in words",
    )

    # Hierarchical options
    hierarchical_group = parser.add_argument_group("Hierarchical Options")
    hierarchical_group.add_argument(
        "--hierarchical", action="store_true", help="Enable hierarchical summarization"
    )
    hierarchical_group.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Number of levels for hierarchical summary",
    )
    hierarchical_group.add_argument(
        "--ultra-short",
        action="store_true",
        help="Include ultra-short (1-2 sentence) summary",
    )
    hierarchical_group.add_argument(
        "--by-sections", action="store_true", help="Summarize by detecting sections"
    )

    # Multi-document options
    multidoc_group = parser.add_argument_group("Multi-document Options")
    multidoc_group.add_argument(
        "--multidoc", action="store_true", help="Enable multi-document summarization"
    )
    multidoc_group.add_argument(
        "--strategy",
        choices=["extract_then_summarize", "iterative", "hierarchical"],
        default="extract_then_summarize",
        help="Strategy for multi-document summarization",
    )
    multidoc_group.add_argument(
        "--redundancy",
        type=float,
        default=0.8,
        help="Redundancy threshold (0-1) for removing similar content",
    )
    multidoc_group.add_argument(
        "--queries",
        nargs="+",
        help="Queries for focused summarization (for multi-document only)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output", help="Output file (default: print to stdout)"
    )
    output_group.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    output_group.add_argument(
        "--verbose", action="store_true", help="Print verbose output"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process input and generate summary
    result = process_input(args)

    # Write output
    if args.json or isinstance(result, dict) and len(result) > 1:
        # Always use JSON for complex results
        write_json_output(result, args.output)
    else:
        # Simple output for basic summarization
        if "error" in result:
            summary = f"Error: {result['error']}"
        elif "summary" in result:
            summary = result["summary"]
        else:
            # Fallback for hierarchical or multi-doc
            summary = json.dumps(result, indent=2)

        write_output(summary, args.output)


if __name__ == "__main__":
    main()
