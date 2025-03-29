"""Web UI for LlamaSum using Streamlit."""

import logging
import sys
import os
import argparse
from typing import List, Dict, Any, Optional, Tuple
import time
import json

# Check if streamlit is installed
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Import LlamaSum modules
from llamasum.summarizer import LlamaSummarizer
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.config import (
    SummarizerConfig, 
    HierarchicalConfig, 
    MultiDocConfig,
    load_config_from_env
)
from llamasum.evaluation import evaluate_summary, calculate_compression_ratio

logger = logging.getLogger(__name__)


def run_streamlit_app():
    """Run the Streamlit web app for LlamaSum.
    
    This function sets up the Streamlit UI with various options for text summarization.
    """
    if not STREAMLIT_AVAILABLE:
        logger.error("Streamlit is not installed. Run: pip install streamlit")
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    # Configure Streamlit page
    st.set_page_config(
        page_title="LlamaSum Text Summarizer",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set up header
    st.title("📝 LlamaSum Text Summarizer")
    st.markdown(
        """
        A powerful text summarization tool that combines extractive and abstractive techniques.
        Upload documents, paste text, or provide URLs to generate high-quality summaries.
        """
    )
    
    # Create sidebar for options
    with st.sidebar:
        st.header("Configuration")
        
        # Summarization type
        summary_type = st.selectbox(
            "Summarization Type",
            options=["Basic", "Hierarchical", "Multi-Document"],
            index=0
        )
        
        # Model selection
        model_options = [
            "sshleifer/distilbart-cnn-12-6",  # Small, fast model
            "facebook/bart-large-cnn",        # High quality but slower
            "t5-small",                      # Smaller T5 model
            "t5-base"                        # Base T5 model
        ]
        
        # Check for custom models in environment
        env_model = os.environ.get("LLAMASUM_MODEL", "")
        if env_model and env_model not in model_options:
            model_options.insert(0, env_model)
        
        model_name = st.selectbox(
            "Model",
            options=model_options,
            index=0,
            help="Select the model to use for summarization"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            # Length control
            max_length = st.slider(
                "Maximum Summary Length",
                min_value=50,
                max_value=500,
                value=150,
                step=10,
                help="Maximum length of the generated summary in words"
            )
            
            min_length = st.slider(
                "Minimum Summary Length",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Minimum length of the generated summary in words"
            )
            
            # Extraction method
            extraction_method = st.selectbox(
                "Extraction Method",
                options=["tfidf", "textrank", "position"],
                index=0,
                help="Method used for extractive summarization"
            )
            
            extractive_ratio = st.slider(
                "Extractive Ratio",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Ratio of original text to extract"
            )
            
            # Hierarchical options (shown only for hierarchical summarization)
            if summary_type == "Hierarchical":
                levels = st.number_input(
                    "Number of Levels",
                    min_value=2,
                    max_value=5,
                    value=3,
                    help="Number of summary levels to generate"
                )
                
                include_ultra_short = st.checkbox(
                    "Include Ultra-Short Summary",
                    value=True,
                    help="Include a very concise 1-2 sentence summary"
                )
                
            # Multi-document options
            if summary_type == "Multi-Document":
                strategy = st.selectbox(
                    "Summarization Strategy",
                    options=[
                        "concatenate", 
                        "summarize_each", 
                        "extract_then_summarize",
                        "cluster_then_summarize"
                    ],
                    index=2,
                    help="Strategy for handling multiple documents"
                )
                
                similarity_threshold = st.slider(
                    "Redundancy Threshold",
                    min_value=0.3,
                    max_value=0.9,
                    value=0.7,
                    step=0.05,
                    help="Threshold for removing redundant content"
                )

        # Create a "Run" button for the sidebar that will be used to trigger summarization        
        run_button = st.sidebar.button("Generate Summary")
        
        # Show info about the library
        with st.expander("About LlamaSum"):
            st.markdown(
                """
                **LlamaSum** is an advanced text summarization library that combines extractive 
                and abstractive approaches for high-quality summaries. It supports hierarchical 
                and multi-document summarization.
                
                Features:
                - Basic, hierarchical, and multi-document summarization
                - Extractive methods: TF-IDF, TextRank, and position-based
                - Multiple pre-trained models for abstractive summarization
                - Summary evaluation metrics
                
                Visit [GitHub Repository](#) for more information.
                """
            )

    # Main content area
    if summary_type == "Basic":
        display_basic_summarization(model_name, max_length, min_length, extraction_method, extractive_ratio, run_button)
    elif summary_type == "Hierarchical":
        display_hierarchical_summarization(
            model_name, max_length, min_length, extraction_method, 
            extractive_ratio, levels, include_ultra_short, run_button
        )
    elif summary_type == "Multi-Document":
        display_multidoc_summarization(
            model_name, max_length, min_length, extraction_method, 
            extractive_ratio, strategy, similarity_threshold, run_button
        )


def display_basic_summarization(
    model_name: str,
    max_length: int,
    min_length: int,
    extraction_method: str,
    extractive_ratio: float,
    run_button: bool
):
    """Display UI for basic summarization.
    
    Args:
        model_name: The model to use
        max_length: Maximum summary length
        min_length: Minimum summary length
        extraction_method: Method to use for extraction
        extractive_ratio: Ratio of text to extract
        run_button: Whether the run button was clicked
    """
    # Text input area
    input_text = st.text_area(
        "Enter text to summarize",
        height=300,
        key="basic_input",
        help="Paste or type the text you want to summarize"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=["txt", "pdf", "docx"],
        help="Upload a document to summarize"
    )
    
    # Process file if uploaded
    if uploaded_file is not None:
        try:
            # For simplicity, just reading as text
            # In a real app, you would handle PDF and DOCX differently
            input_text = uploaded_file.read().decode("utf-8")
            st.success(f"File '{uploaded_file.name}' loaded successfully")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Process summarization if the run button was clicked and there's input
    if run_button and (input_text.strip() or uploaded_file):
        with st.spinner("Generating summary..."):
            try:
                # Create configuration
                config = SummarizerConfig(
                    model_name=model_name,
                    extractive_method=extraction_method,
                    extractive_ratio=extractive_ratio
                )
                
                # Initialize summarizer
                summarizer = LlamaSummarizer(config)
                
                # Generate summary
                start_time = time.time()
                summary = summarizer.summarize(
                    input_text,
                    max_length=max_length,
                    min_length=min_length
                )
                end_time = time.time()
                
                # Calculate metrics
                processing_time = end_time - start_time
                compression_ratio = calculate_compression_ratio(input_text, summary)
                word_count = len(summary.split())
                
                # Display summary
                st.subheader("Summary")
                st.write(summary)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Processing Time", f"{processing_time:.2f}s")
                col2.metric("Compression Ratio", f"{compression_ratio:.2%}")
                col3.metric("Word Count", word_count)
                
                # Show detailed evaluation
                with st.expander("Detailed Evaluation"):
                    metrics = evaluate_summary(summary, input_text)
                    st.json(metrics)
                
            except Exception as e:
                st.error(f"Error generating summary: {e}")
                logger.error(f"Error in basic summarization: {e}", exc_info=True)


def display_hierarchical_summarization(
    model_name: str,
    max_length: int,
    min_length: int,
    extraction_method: str,
    extractive_ratio: float,
    levels: int,
    include_ultra_short: bool,
    run_button: bool
):
    """Display UI for hierarchical summarization.
    
    Args:
        model_name: The model to use
        max_length: Maximum summary length
        min_length: Minimum summary length
        extraction_method: Method to use for extraction
        extractive_ratio: Ratio of text to extract
        levels: Number of hierarchy levels
        include_ultra_short: Whether to include ultra-short summary
        run_button: Whether the run button was clicked
    """
    # Text input area
    input_text = st.text_area(
        "Enter long text to summarize hierarchically",
        height=300,
        key="hierarchical_input",
        help="Paste or type the long text you want to summarize at multiple levels of detail"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=["txt", "pdf", "docx"],
        help="Upload a document to summarize hierarchically",
        key="hierarchical_uploader"
    )
    
    # Additional options
    section_based = st.checkbox(
        "Detect and summarize sections",
        value=True,
        help="Detect document sections and summarize each one separately"
    )
    
    # Process file if uploaded
    if uploaded_file is not None:
        try:
            input_text = uploaded_file.read().decode("utf-8")
            st.success(f"File '{uploaded_file.name}' loaded successfully")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Process summarization if the run button was clicked and there's input
    if run_button and (input_text.strip() or uploaded_file):
        with st.spinner("Generating hierarchical summary..."):
            try:
                # Create configuration
                config = HierarchicalConfig(
                    model_name=model_name,
                    extractive_method=extraction_method,
                    extractive_ratio=extractive_ratio,
                    levels=levels,
                    include_ultra_short=include_ultra_short,
                    max_length=max_length,
                    min_length=min_length
                )
                
                # Initialize summarizer
                summarizer = HierarchicalSummarizer(config)
                
                # Generate summary based on selected method
                start_time = time.time()
                if section_based:
                    result = summarizer.summarize_by_sections(input_text)
                    summary_type = "section-based"
                else:
                    result = summarizer.summarize(input_text, levels=levels)
                    summary_type = "standard"
                end_time = time.time()
                
                # Calculate metrics
                processing_time = end_time - start_time
                
                # Display results based on summary type
                if summary_type == "section-based":
                    st.subheader("Section-Based Summary")
                    
                    # Show overall summary
                    st.markdown("### Overall Summary")
                    st.write(result.get("overall_summary", ""))
                    
                    # Show individual section summaries
                    if "section_summaries" in result and result["section_summaries"]:
                        st.markdown("### Section Summaries")
                        for i, section_summary in enumerate(result["section_summaries"]):
                            with st.expander(f"Section {i+1} Summary"):
                                st.write(section_summary)
                else:
                    st.subheader("Hierarchical Summary")
                    
                    # Show ultra-short summary if available
                    if "ultra_short" in result:
                        st.markdown("### Ultra-Short Summary")
                        st.write(result["ultra_short"])
                    
                    # Show summaries for each level
                    for level in range(1, levels + 1):
                        level_key = f"level_{level}"
                        if level_key in result:
                            with st.expander(f"Level {level} Summary"):
                                st.write(result[level_key])
                
                # Display processing metrics
                st.metric("Processing Time", f"{processing_time:.2f}s")
                
            except Exception as e:
                st.error(f"Error generating hierarchical summary: {e}")
                logger.error(f"Error in hierarchical summarization: {e}", exc_info=True)


def display_multidoc_summarization(
    model_name: str,
    max_length: int,
    min_length: int,
    extraction_method: str,
    extractive_ratio: float,
    strategy: str,
    similarity_threshold: float,
    run_button: bool
):
    """Display UI for multi-document summarization.
    
    Args:
        model_name: The model to use
        max_length: Maximum summary length
        min_length: Minimum summary length
        extraction_method: Method to use for extraction
        extractive_ratio: Ratio of text to extract
        strategy: Strategy for multi-doc summarization
        similarity_threshold: Threshold for redundancy removal
        run_button: Whether the run button was clicked
    """
    # Multi-document input
    st.subheader("Input Documents")
    
    # Create a container for documents
    doc_container = st.container()
    
    # Initialize session state for documents if not exists
    if 'documents' not in st.session_state:
        st.session_state.documents = [""]  # Start with one empty document
    
    # Display document inputs in the container
    with doc_container:
        for i, doc in enumerate(st.session_state.documents):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.session_state.documents[i] = st.text_area(
                    f"Document {i+1}",
                    value=doc,
                    height=150,
                    key=f"doc_{i}"
                )
            with col2:
                if i > 0:  # Allow removing all but the first document
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.documents.pop(i)
                        st.experimental_rerun()
    
    # Add document button
    if st.button("Add Document"):
        st.session_state.documents.append("")
        st.experimental_rerun()
    
    # File uploader for multiple documents
    uploaded_files = st.file_uploader(
        "Or upload multiple files",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload multiple text documents to summarize together",
        key="multidoc_uploader"
    )
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            try:
                content = file.read().decode("utf-8")
                if content.strip():  # Only add non-empty documents
                    st.session_state.documents.append(content)
            except Exception as e:
                st.error(f"Error reading file {file.name}: {e}")
        
        # Clear the uploader after processing
        st.experimental_rerun()
    
    # Query-based summarization option
    enable_query = st.checkbox(
        "Enable Query-Focused Summarization",
        value=False,
        help="Focus the summary on specific queries or questions"
    )
    
    # Show query input if enabled
    queries = []
    if enable_query:
        query_text = st.text_area(
            "Enter queries (one per line)",
            height=100,
            help="Enter questions or topics to focus the summary on, one per line"
        )
        queries = [q.strip() for q in query_text.split("\n") if q.strip()]
    
    # Process summarization if the run button was clicked and there's input
    if run_button and any(doc.strip() for doc in st.session_state.documents):
        with st.spinner("Generating multi-document summary..."):
            try:
                # Filter out empty documents
                documents = [doc for doc in st.session_state.documents if doc.strip()]
                
                if not documents:
                    st.warning("Please enter at least one document")
                    return
                
                # Create configuration
                config = MultiDocConfig(
                    model_name=model_name,
                    extractive_method=extraction_method,
                    extractive_ratio=extractive_ratio,
                    max_length=max_length,
                    min_length=min_length,
                    multi_doc_strategy=strategy,
                    redundancy_threshold=similarity_threshold
                )
                
                # Initialize summarizer
                summarizer = MultiDocSummarizer(config)
                
                # Generate summary
                start_time = time.time()
                if enable_query and queries:
                    result = summarizer.query_focused_summarize(
                        documents,
                        queries=queries
                    )
                else:
                    result = summarizer.summarize(documents)
                end_time = time.time()
                
                # Calculate metrics
                processing_time = end_time - start_time
                
                # Display results
                st.subheader("Multi-Document Summary")
                
                # Show combined summary
                st.markdown("### Combined Summary")
                if isinstance(result, dict) and "summary" in result:
                    st.write(result["summary"])
                elif isinstance(result, str):
                    st.write(result)
                else:
                    st.write("Summary not available in expected format")
                
                # Show query-focused summaries if available
                if enable_query and queries and isinstance(result, dict) and "query_summaries" in result:
                    st.markdown("### Query-Focused Summaries")
                    for query, query_summary in result["query_summaries"].items():
                        with st.expander(f"Summary for: {query}"):
                            st.write(query_summary)
                
                # Display processing metrics
                st.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Show debug info in expander
                with st.expander("Debug Information"):
                    if isinstance(result, dict):
                        debug_info = {k: v for k, v in result.items() if k not in ["summary", "query_summaries"]}
                        st.json(debug_info)
                
            except Exception as e:
                st.error(f"Error generating multi-document summary: {e}")
                logger.error(f"Error in multi-document summarization: {e}", exc_info=True)


def run_web_ui(host: str = "localhost", port: int = 8501):
    """Run the web UI.
    
    Args:
        host: Host to run the web UI on
        port: Port to run the web UI on
    """
    if not STREAMLIT_AVAILABLE:
        logger.error("Streamlit is not installed. Run: pip install streamlit")
        print("Streamlit is not installed. Run: pip install streamlit")
        return
    
    import os
    import subprocess
    
    # Get the path to this script
    script_path = os.path.abspath(__file__)
    
    # Run streamlit with the appropriate command
    cmd = [
        "streamlit", "run", script_path,
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Error running Streamlit: {e}")
        print(f"Error running Streamlit: {e}")


if __name__ == "__main__":
    # This block is executed when the script is run directly
    parser = argparse.ArgumentParser(description="LlamaSum Web UI")
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host to run the web UI on"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the web UI on"
    )
    
    args = parser.parse_args()
    
    if STREAMLIT_AVAILABLE and len(sys.argv) > 1 and sys.argv[1] == "run":
        # Called by streamlit run
        run_streamlit_app()
    else:
        # Called directly
        run_web_ui(args.host, args.port) 