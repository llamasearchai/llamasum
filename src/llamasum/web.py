"""Web UI for LlamaSum using Streamlit."""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st

from llamasum.summarizer import LlamaSummarizer
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.multidoc import MultiDocSummarizer
from llamasum.config import SummarizerConfig, HierarchicalConfig, MultiDocConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="LlamaSum - Advanced Text Summarization",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_MODELS = [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "google/pegasus-xsum",
    "t5-small",
    "t5-base",
    "sshleifer/distilbart-cnn-12-6",
    "philschmid/bart-large-cnn-samsum",
]


def main():
    """Run the Streamlit web UI."""
    # Title and description
    st.title("LlamaSum: Advanced Text Summarization")
    st.markdown("""
    Summarize text documents using state-of-the-art AI models, 
    with support for basic, hierarchical, and multi-document summarization.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Select summarization mode
        mode = st.selectbox(
            "Summarization Mode",
            ["Basic", "Hierarchical", "Multi-Document"],
            index=0,
            help="Choose the summarization approach"
        )
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            DEFAULT_MODELS,
            index=0,
            help="Choose a pretrained model for summarization"
        )
        
        # Device selection
        device_options = ["cpu"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            device_options.append("cuda")
        if hasattr(st, "mac") and st.mac:
            device_options.append("mps")
            
        device = st.selectbox(
            "Device",
            device_options,
            index=0,
            help="Device to run the model on"
        )
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            # Common settings
            min_length = st.slider(
                "Minimum Length",
                min_value=10,
                max_value=100,
                value=30,
                help="Minimum length of the summary"
            )
            
            max_length = st.slider(
                "Maximum Length",
                min_value=50,
                max_value=1000,
                value=200,
                help="Maximum length of the summary"
            )
            
            # Mode-specific settings
            if mode == "Basic":
                extractive_ratio = st.slider(
                    "Extractive Ratio",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="Ratio of text to extract in the extractive phase"
                )
                
                basic_settings = {
                    "extractive_ratio": extractive_ratio
                }
                
            elif mode == "Hierarchical":
                levels = st.slider(
                    "Levels",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="Number of summary levels to generate"
                )
                
                include_ultra_short = st.checkbox(
                    "Include Ultra-Short Summary",
                    value=True,
                    help="Include an ultra-short (1-2 sentence) summary"
                )
                
                summarize_method = st.radio(
                    "Summarization Method",
                    ["Standard", "By Sections", "By Clustering"],
                    index=0,
                    help="Choose how to approach the summarization"
                )
                
                hierarchical_settings = {
                    "levels": levels,
                    "include_ultra_short": include_ultra_short,
                    "summarize_method": summarize_method
                }
                
            elif mode == "Multi-Document":
                strategy = st.radio(
                    "Strategy",
                    ["Iterative", "Extract Then Summarize", "Hierarchical"],
                    index=1,
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="Strategy for multi-document summarization"
                )
                
                use_queries = st.checkbox(
                    "Query-Focused Summarization",
                    value=False,
                    help="Focus summarization on specific queries/topics"
                )
                
                multidoc_settings = {
                    "strategy": strategy.lower().replace(" ", "_"),
                    "use_queries": use_queries
                }
        
        # Create "Run with Example" button
        if st.button("Load Example"):
            if mode == "Basic":
                st.session_state.text_input = get_example_text("basic")
            elif mode == "Hierarchical":
                st.session_state.text_input = get_example_text("hierarchical")
            elif mode == "Multi-Document":
                st.session_state.multi_docs = get_example_text("multidoc", multi=True)
    
    # Main content area
    if mode == "Multi-Document":
        render_multidoc_ui(
            model_name, 
            device, 
            min_length, 
            max_length, 
            multidoc_settings
        )
    else:
        # Single document UI
        if mode == "Basic":
            render_basic_ui(
                model_name, 
                device, 
                min_length, 
                max_length, 
                basic_settings
            )
        else:  # Hierarchical
            render_hierarchical_ui(
                model_name, 
                device, 
                min_length, 
                max_length, 
                hierarchical_settings
            )


def render_basic_ui(
    model_name: str, 
    device: str, 
    min_length: int, 
    max_length: int, 
    settings: Dict[str, Any]
):
    """Render UI for basic summarization.
    
    Args:
        model_name: Name of the model to use
        device: Device to run on
        min_length: Minimum summary length
        max_length: Maximum summary length
        settings: Additional settings
    """
    # Text input area
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
        
    text_input = st.text_area(
        "Input Text to Summarize",
        value=st.session_state.text_input,
        height=300,
        placeholder="Paste your text here..."
    )
    
    # File upload option
    uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
    
    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")
        st.session_state.text_input = text_input
        
    # Summarize button
    if st.button("Summarize"):
        if text_input:
            with st.spinner("Summarizing..."):
                try:
                    # Create config
                    config = SummarizerConfig(
                        model_name=model_name,
                        device=device,
                        min_length=min_length,
                        max_length=max_length,
                        extractive_ratio=settings["extractive_ratio"]
                    )
                    
                    # Initialize summarizer
                    summarizer = LlamaSummarizer(config)
                    
                    # Generate summary
                    summary = summarizer.summarize(text_input)
                    
                    # Display result
                    st.subheader("Summary")
                    st.markdown(f"**{summary}**")
                    
                    # Show stats
                    input_words = len(text_input.split())
                    summary_words = len(summary.split())
                    compression = round((1 - summary_words / input_words) * 100, 1)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Words", input_words)
                    col2.metric("Summary Words", summary_words)
                    col3.metric("Compression", f"{compression}%")
                    
                except Exception as e:
                    st.error(f"Error during summarization: {str(e)}")
                    logger.exception("Summarization error")
        else:
            st.warning("Please enter some text to summarize.")


def render_hierarchical_ui(
    model_name: str, 
    device: str, 
    min_length: int, 
    max_length: int, 
    settings: Dict[str, Any]
):
    """Render UI for hierarchical summarization.
    
    Args:
        model_name: Name of the model to use
        device: Device to run on
        min_length: Minimum summary length
        max_length: Maximum summary length
        settings: Additional settings
    """
    # Text input area
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
        
    text_input = st.text_area(
        "Input Text to Summarize",
        value=st.session_state.text_input,
        height=300,
        placeholder="Paste your text here for hierarchical summarization..."
    )
    
    # File upload option
    uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
    
    if uploaded_file is not None:
        text_input = uploaded_file.read().decode("utf-8")
        st.session_state.text_input = text_input
        
    # Summarize button
    if st.button("Generate Hierarchical Summary"):
        if text_input:
            with st.spinner("Generating hierarchical summary..."):
                try:
                    # Create config
                    config = HierarchicalConfig(
                        model_name=model_name,
                        device=device,
                        min_length=min_length,
                        max_length=max_length,
                        levels=settings["levels"],
                        include_ultra_short=settings["include_ultra_short"]
                    )
                    
                    # Initialize summarizer
                    summarizer = HierarchicalSummarizer(config)
                    
                    # Choose method based on selected approach
                    if settings["summarize_method"] == "By Sections":
                        result = summarizer.summarize_by_sections(text_input)
                    elif settings["summarize_method"] == "By Clustering":
                        result = summarizer.summarize_by_clustering(text_input)
                    else:
                        result = summarizer.summarize(text_input)
                    
                    # Display results
                    st.subheader("Hierarchical Summary")
                    
                    if "overall_summary" in result:
                        # Section-based or clustering result
                        st.markdown("### Overall Summary")
                        st.markdown(f"**{result['overall_summary']}**")
                        
                        if "section_summaries" in result:
                            st.markdown("### Section Summaries")
                            for i, summary in enumerate(result["section_summaries"]):
                                with st.expander(f"Section {i+1}"):
                                    st.write(summary)
                                    
                        if "cluster_summaries" in result:
                            st.markdown("### Cluster Summaries")
                            for i, summary in enumerate(result["cluster_summaries"]):
                                with st.expander(f"Cluster {i+1}"):
                                    st.write(summary)
                    else:
                        # Level-based result
                        if "ultra_short" in result:
                            st.markdown("### Ultra-Short Summary")
                            st.markdown(f"**{result['ultra_short']}**")
                        
                        # Display each level with expandable sections
                        for level in range(1, settings["levels"] + 1):
                            level_key = f"level_{level}"
                            if level_key in result:
                                level_name = f"Level {level}" if level > 1 else "Main Summary"
                                if level == 1:
                                    # Show the first level prominently
                                    st.markdown(f"### {level_name}")
                                    st.markdown(f"**{result[level_key]}**")
                                else:
                                    # Show other levels as expandable
                                    with st.expander(level_name):
                                        st.write(result[level_key])
                    
                except Exception as e:
                    st.error(f"Error during hierarchical summarization: {str(e)}")
                    logger.exception("Hierarchical summarization error")
        else:
            st.warning("Please enter some text to summarize.")


def render_multidoc_ui(
    model_name: str, 
    device: str, 
    min_length: int, 
    max_length: int, 
    settings: Dict[str, Any]
):
    """Render UI for multi-document summarization.
    
    Args:
        model_name: Name of the model to use
        device: Device to run on
        min_length: Minimum summary length
        max_length: Maximum summary length
        settings: Additional settings
    """
    # Initialize session state for documents
    if "multi_docs" not in st.session_state:
        st.session_state.multi_docs = ["", ""]
    
    st.markdown("### Input Documents")
    
    # Queries input (only if query-focused is selected)
    queries = []
    if settings["use_queries"]:
        query_input = st.text_input(
            "Enter queries (comma-separated)",
            placeholder="climate change, carbon emissions, renewable energy"
        )
        if query_input:
            queries = [q.strip() for q in query_input.split(",")]
    
    # Container for document inputs
    docs_container = st.container()
    
    # Add document button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("+ Add Document"):
            st.session_state.multi_docs.append("")
            
    with col2:
        # File upload for multiple documents
        uploaded_files = st.file_uploader(
            "Upload text files", 
            type=["txt"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Update the documents with uploaded files
            for i, file in enumerate(uploaded_files):
                if i >= len(st.session_state.multi_docs):
                    st.session_state.multi_docs.append("")
                st.session_state.multi_docs[i] = file.read().decode("utf-8")
    
    # Render document inputs
    with docs_container:
        updated_docs = []
        
        for i, doc in enumerate(st.session_state.multi_docs):
            col1, col2 = st.columns([20, 1])
            
            with col1:
                updated_doc = st.text_area(
                    f"Document {i+1}",
                    value=doc,
                    height=150,
                    key=f"doc_{i}"
                )
                updated_docs.append(updated_doc)
                
            with col2:
                if len(st.session_state.multi_docs) > 2:
                    if st.button("✕", key=f"remove_{i}"):
                        st.session_state.multi_docs.pop(i)
                        st.experimental_rerun()
                        
        # Update the documents in session state
        st.session_state.multi_docs = updated_docs
    
    # Summarize button
    if st.button("Summarize Documents"):
        # Filter out empty documents
        documents = [doc for doc in st.session_state.multi_docs if doc.strip()]
        
        if len(documents) > 1:
            with st.spinner("Summarizing multiple documents..."):
                try:
                    # Create config
                    config = MultiDocConfig(
                        model_name=model_name,
                        device=device,
                        min_length=min_length,
                        max_length=max_length,
                        strategy=settings["strategy"]
                    )
                    
                    # Initialize summarizer
                    summarizer = MultiDocSummarizer(config)
                    
                    # Generate summary
                    if settings["use_queries"] and queries:
                        result = summarizer.summarize_with_queries(
                            documents, 
                            queries
                        )
                    else:
                        result = summarizer.summarize(documents)
                    
                    # Display results
                    st.subheader("Multi-Document Summary")
                    
                    if "general_summary" in result:
                        # Query-based result
                        st.markdown("### General Summary")
                        st.markdown(f"**{result['general_summary']}**")
                        
                        if "query_summaries" in result:
                            st.markdown("### Query-Focused Summaries")
                            for query, summary in result["query_summaries"].items():
                                with st.expander(f"Query: {query}"):
                                    st.write(summary)
                    else:
                        # Standard result
                        st.markdown(f"**{result['summary']}**")
                        
                        # Show method info
                        st.caption(f"Method: {result.get('method', 'standard')} | "
                                  f"Documents: {result.get('num_documents', len(documents))}")
                        
                        # Show individual summaries if available
                        if "individual_summaries" in result:
                            st.markdown("### Individual Document Summaries")
                            for i, summary in enumerate(result["individual_summaries"]):
                                with st.expander(f"Document {i+1}"):
                                    st.write(summary)
                                    
                        # Show cluster summaries if available
                        if "cluster_summaries" in result:
                            st.markdown(f"### Cluster Summaries ({result.get('num_clusters', 0)} clusters)")
                            for i, summary in enumerate(result["cluster_summaries"]):
                                with st.expander(f"Cluster {i+1}"):
                                    st.write(summary)
                    
                except Exception as e:
                    st.error(f"Error during multi-document summarization: {str(e)}")
                    logger.exception("Multi-document summarization error")
        else:
            st.warning("Please add at least two documents with content to summarize.")


def get_example_text(example_type: str, multi: bool = False) -> Union[str, List[str]]:
    """Get example text for different summarization types.
    
    Args:
        example_type: Type of example ('basic', 'hierarchical', 'multidoc')
        multi: Whether to return multiple documents (for multidoc)
        
    Returns:
        Example text or list of texts
    """
    if example_type == "basic":
        return """Climate change is one of the most pressing challenges facing humanity today. 
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
        
    elif example_type == "hierarchical":
        return """# Artificial Intelligence: History, Applications, and Future Directions

## Introduction
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term was coined in 1956 by John McCarthy, who defined it as "the science and engineering of making intelligent machines." AI has evolved significantly since its inception and now encompasses a wide range of technologies and approaches.

## Historical Development
### Early Beginnings
The concept of artificial beings with intelligence dates back to ancient times, appearing in Greek myths, such as Talos of Crete and the talking bronze head of Albertus Magnus. However, the formal academic field of AI research began at a conference at Dartmouth College in 1956, where McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel Rochester brought together researchers interested in machine intelligence.

### AI Winter and Revival
The field experienced cycles of optimism followed by disappointment and funding cuts, known as "AI winters." The first occurred in the 1970s when early promise gave way to criticism and reduced funding. A second major winter happened in the late 1980s and early 1990s as expert systems failed to deliver on expectations.

### Modern Resurgence
The 21st century has seen a remarkable resurgence in AI research and applications, driven by:
- Increased computational power
- Availability of vast amounts of data
- Advances in algorithms, particularly in machine learning and neural networks
- Substantial investment from technology companies and governments

## Key AI Approaches

### Symbolic AI
Also known as "Good Old-Fashioned AI" (GOFAI), this approach uses symbolic representations of problems, logic, and search algorithms. It focuses on explicit knowledge representation and reasoning. Expert systems, which capture the knowledge of human experts in rule sets, exemplify this approach.

### Machine Learning
Machine learning enables computers to learn from data without being explicitly programmed. Major categories include:

#### Supervised Learning
Algorithms learn from labeled training data, making predictions or decisions without being explicitly programmed to perform the task. Examples include:
- Linear regression
- Support vector machines
- Decision trees
- Neural networks applied to structured data

#### Unsupervised Learning
These algorithms find patterns in unlabeled data, such as:
- Clustering algorithms like K-means
- Dimensionality reduction techniques
- Association rule learning

#### Reinforcement Learning
Algorithms learn optimal behaviors through trial-and-error interactions with an environment, receiving rewards or penalties. This approach has led to breakthroughs in:
- Game playing (AlphaGo, AlphaZero)
- Robotics
- Resource management
- Autonomous vehicles

### Deep Learning
A subset of machine learning based on artificial neural networks with multiple layers (hence "deep"). Key architectures include:
- Convolutional Neural Networks (CNNs) for image processing
- Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) for sequential data
- Transformers for natural language processing
- Generative Adversarial Networks (GANs) for content generation

## Current Applications

### Healthcare
AI applications in healthcare include:
- Disease diagnosis and prediction
- Drug discovery and development
- Personalized treatment plans
- Medical image analysis
- Hospital workflow optimization

### Financial Services
In finance, AI is used for:
- Algorithmic trading
- Fraud detection
- Risk assessment
- Customer service chatbots
- Loan approval processes

### Transportation
AI is revolutionizing transportation through:
- Autonomous vehicles
- Traffic prediction and management
- Ride-sharing optimization
- Logistics and supply chain management

### Manufacturing
In industry, AI enables:
- Predictive maintenance
- Quality control
- Supply chain optimization
- Generative design
- Collaborative robots (cobots)

### Entertainment and Media
AI applications in entertainment include:
- Content recommendation systems
- Content creation (music, art, writing)
- Video game character behavior
- Special effects
- Content moderation

## Ethical Considerations and Challenges

### Bias and Fairness
AI systems can perpetuate or amplify existing biases in training data. Addressing this requires:
- Diverse and representative training data
- Algorithmic fairness techniques
- Regular auditing for bias
- Inclusive development teams

### Privacy Concerns
AI often relies on vast amounts of personal data, raising concerns about:
- Data collection and consent
- Surveillance capabilities
- Data security
- Right to be forgotten

### Transparency and Explainability
Many advanced AI systems, particularly deep learning models, function as "black boxes" whose decision-making processes are opaque. Explainable AI (XAI) is an emerging field addressing this challenge.

### Job Displacement
Automation driven by AI may displace certain jobs, requiring:
- Workforce retraining
- Education system adaptation
- Social safety net considerations
- New economic models

### Safety and Control
Ensuring AI systems behave as intended and can be controlled raises questions about:
- Verification and validation methods
- Alignment with human values
- Security against adversarial attacks
- Recovery from failures

## Future Directions

### Artificial General Intelligence
Most current AI systems are narrow, excelling at specific tasks. Artificial General Intelligence (AGI) would possess human-like general intelligence, able to perform any intellectual task that a human can do.

### AI and Scientific Discovery
AI is increasingly being used to accelerate scientific discovery in fields such as:
- Drug discovery
- Materials science
- Climate modeling
- Particle physics
- Astronomy

### Human-AI Collaboration
Rather than replacing humans, many AI systems will augment human capabilities through:
- Decision support systems
- Creativity enhancement
- Cognitive extension
- Physical assistance

### Neuromorphic Computing
New computing architectures inspired by the human brain may provide more efficient platforms for AI:
- Spiking neural networks
- Brain-inspired hardware
- Analog computing approaches
- Quantum machine learning

### Global Governance and Standards
As AI becomes more powerful and pervasive, international cooperation on governance becomes essential, focusing on:
- Ethical standards
- Safety protocols
- Preventing harmful applications
- Ensuring equitable access and benefits

## Conclusion
Artificial intelligence has evolved from a theoretical concept to a transformative technology affecting virtually every aspect of modern society. While significant challenges remain, AI continues to advance rapidly, promising further breakthroughs in the coming decades. Balancing innovation with ethical considerations will be crucial to ensuring that AI development benefits humanity while minimizing potential harms. As we move forward, collaboration between technologists, policymakers, ethicists, and the public will be essential to shape AI's trajectory in positive directions."""
        
    elif example_type == "multidoc" and multi:
        return [
            """Climate change is the long-term alteration of temperature and typical weather patterns in a place. Climate change could refer to a particular location or the planet as a whole. Climate change may cause weather patterns to be less predictable. These unexpected weather patterns can make it difficult to maintain and grow crops in regions that rely on farming because expected temperature and rainfall levels can no longer be relied on. Climate change has also been connected with other damaging weather events such as more frequent and more intense hurricanes, floods, downpours, and winter storms.

In polar regions, the warming global temperatures associated with climate change have meant ice sheets and glaciers are melting at an accelerated rate from season to season. This contributes to sea levels rising in different regions of the planet. Together with expanding ocean waters due to rising temperatures, the resulting rise in sea level has begun to damage coastlines as a result of increased flooding and erosion.

The cause of current climate change is largely human activity, like burning fossil fuels, like natural gas, oil, and coal. Burning these materials releases what are called greenhouse gases into Earth's atmosphere. There, these gases trap heat from the sun's rays inside the atmosphere causing Earth's average temperature to rise. This rise in the planet's temperature is called global warming. The warming of the planet impacts local and regional climates. Throughout Earth's history, climate has continually changed. When occurring naturally, this is a slow process that has taken place over hundreds and thousands of years. The human influenced climate change that is occurring now is occurring at a much faster rate.""",
            
            """Addressing climate change requires both mitigation – reducing climate change by reducing greenhouse gas emissions and enhancing sinks – and adaptation – adjusting to actual or expected effects of climate change.

Mitigation may involve reductions in emissions through using renewable energy like solar, wind, and hydropower instead of burning fossil fuels. Energy efficiency improvements in buildings, transportation, and industry can also significantly reduce emissions. Agriculture and land use changes are other areas where mitigation is possible, including through reduced deforestation, improved forest management, and more sustainable farming practices.

Carbon capture, utilization, and storage (CCUS) technologies are being developed to remove carbon dioxide directly from emissions sources or from the atmosphere. These include engineered solutions like direct air capture as well as nature-based solutions such as reforestation and soil carbon sequestration.

Adaptation strategies help prepare for and reduce the impacts of climate change. These include building seawalls to protect against rising sea levels, developing drought-resistant crops, improving water management systems, creating early warning systems for extreme weather events, and designing infrastructure to withstand changing conditions.

Climate policies and international cooperation are essential for effective action. The Paris Agreement, adopted in 2015, aims to limit global warming to well below 2°C above pre-industrial levels, preferably to 1.5°C. Countries submit nationally determined contributions (NDCs) outlining their efforts to reduce emissions and adapt to climate impacts.""",
            
            """The economic impacts of climate change are complex and far-reaching. The costs of inaction are increasingly recognized as more expensive than the costs of reducing emissions. The Stern Review, published in 2006, estimated that without action, the overall costs of climate change would be equivalent to losing at least 5% of global GDP each year, now and forever, while the costs of reducing emissions could be limited to around 1% of global GDP each year.

Climate change affects economic productivity through multiple channels. Rising temperatures can reduce labor productivity, particularly in sectors with high exposure to outdoor temperatures such as agriculture and construction. Extreme weather events can damage infrastructure and disrupt supply chains. Changes in precipitation patterns can affect water availability for agriculture, industry, and human consumption.

Some sectors face particular challenges. Agriculture is highly sensitive to climate conditions, with changing temperatures and precipitation patterns affecting crop yields. Tourism may be affected as climate change alters the attractiveness of destinations. Insurance faces increasing claims from climate-related disasters.

However, addressing climate change also creates economic opportunities. The transition to a low-carbon economy is driving innovation and creating jobs in renewable energy, energy efficiency, and other clean technologies. According to the International Renewable Energy Agency, the renewable energy sector employed 11.5 million people globally in 2019, a number that continues to grow.

A just transition is important to ensure that the benefits of climate action are shared fairly and the costs do not fall disproportionately on vulnerable groups. This includes support for workers and communities dependent on carbon-intensive industries to develop new skills and economic opportunities."""
        ]
    
    # Default single document for multi-doc
    return """Climate change is the long-term alteration of temperature and typical weather patterns in a place. Climate change could refer to a particular location or the planet as a whole. Climate change may cause weather patterns to be less predictable. These unexpected weather patterns can make it difficult to maintain and grow crops in regions that rely on farming because expected temperature and rainfall levels can no longer be relied on."""


if __name__ == "__main__":
    main() 