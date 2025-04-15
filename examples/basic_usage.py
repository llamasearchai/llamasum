#!/usr/bin/env python3
"""Basic usage examples for LlamaSum."""

import os
import sys
from pathlib import Path

# Add parent directory to path to import llamasum
sys.path.append(str(Path(__file__).parent.parent))

from llamasum.config import HierarchicalConfig, MultiDocConfig, SummarizerConfig
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.summarizer import LlamaSummarizer


def basic_summarization():
    """Example of basic summarization."""
    print("=== Basic Summarization Example ===")

    # Sample text about climate change
    text = """Climate change is one of the most pressing challenges facing humanity today. 
    The Intergovernmental Panel on Climate Change (IPCC) has established that human activities 
    have warmed the planet by approximately 1.1°C since pre-industrial times. This warming is 
    primarily due to greenhouse gas emissions from burning fossil fuels, deforestation, and 
    industrial processes. The consequences of climate change include rising sea levels, more 
    frequent and severe weather events, disruptions to ecosystems, and threats to food security 
    and human health. Scientists warn that limiting global warming to 1.5°C above pre-industrial 
    levels is crucial to avoid the most catastrophic impacts. Achieving this goal requires rapid, 
    far-reaching transitions in energy, land, urban infrastructure, and industrial systems. 
    International cooperation, embodied in agreements like the Paris Climate Accord, aims to 
    coordinate global efforts to reduce emissions and build resilience against climate impacts. 
    However, current pledges and policies are insufficient to meet the 1.5°C target. 
    Addressing climate change demands immediate action across all sectors of society, 
    from government policy and business practices to individual lifestyle choices."""

    # Initialize summarizer with default configuration
    config = SummarizerConfig(
        model_name="sshleifer/distilbart-cnn-12-6",  # Using a smaller model for demonstration
        device="cpu",
        max_length=100,
    )
    summarizer = LlamaSummarizer(config)

    # Generate summary
    summary = summarizer.summarize(text)

    # Print results
    print("\nOriginal text:")
    print(text)
    print("\nSummary:")
    print(summary)
    print(f"\nCompression ratio: {len(summary.split()) / len(text.split()):.2f}")


def hierarchical_summarization():
    """Example of hierarchical summarization."""
    print("\n=== Hierarchical Summarization Example ===")

    # Load sample long text from file
    example_file = Path(__file__).parent / "sample_article.txt"

    # If the sample file doesn't exist, create it with placeholder text
    if not example_file.exists():
        # Short placeholder text about AI
        text = """# Artificial Intelligence Overview
        
        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
        programmed to think and learn like humans. The term was coined in 1956, and AI has evolved 
        significantly since then. Modern AI encompasses various approaches including machine learning, 
        neural networks, and natural language processing.
        
        ## Applications
        
        AI is applied in numerous domains including healthcare (for diagnosis and treatment planning), 
        finance (for fraud detection and algorithmic trading), transportation (for autonomous vehicles 
        and traffic management), and entertainment (for game playing and content recommendation).
        
        ## Challenges
        
        Despite its progress, AI faces challenges including ethical concerns about bias and fairness, 
        privacy issues, potential job displacement, and ensuring AI systems align with human values.
        
        ## Future Directions
        
        Future developments in AI may include more advanced general AI systems, improved human-AI 
        collaboration, neuromorphic computing, and better frameworks for AI governance and ethics."""

        # Write the placeholder text to file
        example_file.write_text(text)

    # Read the sample file
    text = example_file.read_text()

    # Initialize hierarchical summarizer
    config = HierarchicalConfig(
        model_name="sshleifer/distilbart-cnn-12-6",  # Using a smaller model for demonstration
        device="cpu",
        levels=3,  # Generate three levels of summaries
        include_ultra_short=True,
    )
    summarizer = HierarchicalSummarizer(config)

    # Generate hierarchical summary
    result = summarizer.summarize(text)

    # Print results
    print("\nGenerating hierarchical summary with 3 levels...\n")

    if "ultra_short" in result:
        print("Ultra-short summary:")
        print(result["ultra_short"])
        print()

    for i in range(1, 4):
        level_key = f"level_{i}"
        if level_key in result:
            print(f"Level {i} summary:")
            print(result[level_key])
            print()


def multi_document_summarization():
    """Example of multi-document summarization."""
    print("\n=== Multi-Document Summarization Example ===")

    # Sample documents about climate change from different perspectives
    documents = [
        """Climate change is the long-term alteration of temperature and weather patterns. 
        It is primarily caused by human activities, especially the burning of fossil fuels, 
        which releases greenhouse gases into the atmosphere. These gases trap heat from the 
        sun, causing global warming. The effects include rising sea levels, more intense 
        hurricanes, droughts, and heat waves. Climate scientists agree that immediate action 
        is necessary to reduce emissions and limit warming to manageable levels.""",
        """Addressing climate change requires both mitigation and adaptation. Mitigation involves 
        reducing greenhouse gas emissions through renewable energy, energy efficiency, and 
        changes in land use. Adaptation means adjusting to the current and expected effects 
        of climate change, such as building sea walls, developing drought-resistant crops, 
        and improving early warning systems for extreme weather events. International 
        cooperation is essential, as exemplified by the Paris Agreement.""",
        """The economic implications of climate change are significant. Without action, climate 
        change could reduce global GDP by up to 18% by 2050. However, transitioning to a 
        green economy also presents opportunities for innovation and job creation in sectors 
        like renewable energy and sustainable agriculture. A just transition ensures that 
        the benefits of climate action are shared fairly and the costs do not fall 
        disproportionately on vulnerable groups.""",
    ]

    # Initialize multi-document summarizer
    config = MultiDocConfig(
        model_name="sshleifer/distilbart-cnn-12-6",  # Using a smaller model for demonstration
        device="cpu",
        strategy="extract_then_summarize",  # Strategy for multi-doc summarization
        max_length=150,
    )
    summarizer = MultiDocSummarizer(config)

    # Generate multi-document summary
    result = summarizer.summarize(documents)

    # Print results
    print("\nInput documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(doc)

    print("\nMulti-document summary:")
    print(result["summary"])


def query_focused_summarization():
    """Example of query-focused multi-document summarization."""
    print("\n=== Query-Focused Summarization Example ===")

    # Same documents as in multi_document_summarization
    documents = [
        """Climate change is the long-term alteration of temperature and weather patterns. 
        It is primarily caused by human activities, especially the burning of fossil fuels, 
        which releases greenhouse gases into the atmosphere. These gases trap heat from the 
        sun, causing global warming. The effects include rising sea levels, more intense 
        hurricanes, droughts, and heat waves. Climate scientists agree that immediate action 
        is necessary to reduce emissions and limit warming to manageable levels.""",
        """Addressing climate change requires both mitigation and adaptation. Mitigation involves 
        reducing greenhouse gas emissions through renewable energy, energy efficiency, and 
        changes in land use. Adaptation means adjusting to the current and expected effects 
        of climate change, such as building sea walls, developing drought-resistant crops, 
        and improving early warning systems for extreme weather events. International 
        cooperation is essential, as exemplified by the Paris Agreement.""",
        """The economic implications of climate change are significant. Without action, climate 
        change could reduce global GDP by up to 18% by 2050. However, transitioning to a 
        green economy also presents opportunities for innovation and job creation in sectors 
        like renewable energy and sustainable agriculture. A just transition ensures that 
        the benefits of climate action are shared fairly and the costs do not fall 
        disproportionately on vulnerable groups.""",
    ]

    # Queries to focus the summary
    queries = ["economic impact of climate change", "solutions to climate change"]

    # Initialize multi-document summarizer
    config = MultiDocConfig(
        model_name="sshleifer/distilbart-cnn-12-6",  # Using a smaller model for demonstration
        device="cpu",
        max_length=100,
    )
    summarizer = MultiDocSummarizer(config)

    # Generate query-focused summary
    result = summarizer.summarize_with_queries(documents, queries)

    # Print results
    print("\nQueries:")
    for query in queries:
        print(f"- {query}")

    print("\nGeneral summary:")
    print(result["general_summary"])

    print("\nQuery-focused summaries:")
    for query, summary in result["query_summaries"].items():
        print(f"\nQuery: {query}")
        print(summary)


if __name__ == "__main__":
    # Run all examples
    try:
        basic_summarization()
        hierarchical_summarization()
        multi_document_summarization()
        query_focused_summarization()

        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()
