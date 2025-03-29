"""Hierarchical summarization module for LlamaSum."""

import logging
from typing import List, Dict, Optional, Any, Union
import math

from llamasum.summarizer import LlamaSummarizer
from llamasum.config import HierarchicalConfig, SummarizerConfig
from llamasum.preprocessing import split_into_sentences, detect_sections, remove_redundant_sentences

logger = logging.getLogger(__name__)

# Try to import sklearn for clustering
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available, using basic clustering for hierarchical summarization")
    SKLEARN_AVAILABLE = False


class HierarchicalSummarizer:
    """Summarizer that creates multi-level summaries of different granularity."""
    
    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """Initialize hierarchical summarizer.
        
        Args:
            config: Configuration for hierarchical summarization
        """
        self.config = config or HierarchicalConfig()
        
        # Initialize base summarizer
        self.summarizer = LlamaSummarizer(config=self.config)
    
    def summarize(
        self,
        text: str,
        levels: Optional[int] = None,
        **kwargs
    ) -> Dict[str, str]:
        """Create hierarchical summary with multiple granularity levels.
        
        Args:
            text: Text to summarize
            levels: Number of summary levels to generate
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with summaries at different levels
        """
        if not text:
            return {"original": "", "error": "Empty text provided"}
        
        # Initialize result with the original text
        result = {"original": text}
        
        # If levels not specified, use the config value
        if levels is None:
            levels = self.config.levels
        
        # Determine appropriate number of levels based on text length
        sentences = split_into_sentences(text)
        if len(sentences) < 10:
            # Short text, reduce number of levels
            levels = min(2, levels)
        
        # Create summary for each level
        previous_summary = text
        for level in range(1, levels + 1):
            # Adjust max_length based on level
            max_length = kwargs.get('max_length', self.config.max_length)
            level_max_length = max(50, int(max_length / math.sqrt(level)))
            
            # Create summary for this level
            summary = self.summarizer.summarize(
                previous_summary,
                max_length=level_max_length,
                **kwargs
            )
            
            # Add to result
            result[f"level_{level}"] = summary
            
            # Use this summary as input for next level (more concise)
            previous_summary = summary
        
        # Add ultra-short summary if requested
        if self.config.include_ultra_short:
            result["ultra_short"] = self._generate_ultra_short(previous_summary)
        
        return result
    
    def summarize_by_sections(
        self,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Summarize text by first detecting and summarizing sections.
        
        Args:
            text: Text to summarize
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with overall summary and section summaries
        """
        if not text:
            return {"error": "Empty text provided"}
        
        # Detect sections in the text
        sections = detect_sections(text, min_section_length=self.config.min_section_length)
        
        if not sections:
            # If no sections detected, summarize as a whole
            summary = self.summarizer.summarize(text, **kwargs)
            return {
                "overall_summary": summary,
                "section_summaries": [],
                "num_sections": 0
            }
        
        # Summarize each section
        section_summaries = []
        for section in sections:
            # Adjust max length based on section length relative to total
            section_ratio = len(section) / len(text)
            section_max_length = kwargs.get('max_length', self.config.max_length)
            section_max_length = max(50, int(section_max_length * section_ratio))
            
            # Create summary for this section
            summary = self.summarizer.summarize(
                section,
                max_length=section_max_length,
                **kwargs
            )
            section_summaries.append(summary)
        
        # Remove redundancies between section summaries
        unique_summaries = remove_redundant_sentences(
            section_summaries,
            similarity_threshold=self.config.redundancy_threshold
        )
        
        # Create overall summary from section summaries
        combined_summary = " ".join(unique_summaries)
        overall_max_length = kwargs.get('max_length', self.config.max_length)
        overall_summary = self.summarizer.summarize(
            combined_summary,
            max_length=overall_max_length,
            **kwargs
        )
        
        return {
            "overall_summary": overall_summary,
            "section_summaries": section_summaries,
            "num_sections": len(sections)
        }
    
    def summarize_by_clustering(
        self,
        text: str,
        num_clusters: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Summarize text by clustering similar content.
        
        Args:
            text: Text to summarize
            num_clusters: Number of clusters to create
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with overall summary and cluster summaries
        """
        if not text:
            return {"error": "Empty text provided"}
        
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, falling back to section-based summarization")
            return self.summarize_by_sections(text, **kwargs)
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        if len(paragraphs) < 3:
            # Too few paragraphs for meaningful clustering
            return self.summarize(text, **kwargs)
        
        # Determine number of clusters
        if num_clusters is None:
            # Adaptive clustering: more clusters for longer texts
            num_clusters = min(
                self.config.max_clusters,
                max(2, int(math.sqrt(len(paragraphs)) / 2))
            )
        
        try:
            # Create TF-IDF vectors for paragraphs
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
            
            # Cluster paragraphs
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=42
            )
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Group paragraphs by cluster
            cluster_contents = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_contents:
                    cluster_contents[cluster_id] = []
                cluster_contents[cluster_id].append(paragraphs[i])
            
            # Summarize each cluster
            cluster_summaries = []
            for cluster_id, contents in cluster_contents.items():
                # Skip empty clusters
                if not contents:
                    continue
                
                # Combine cluster contents
                cluster_text = "\n\n".join(contents)
                
                # Determine max length for this cluster
                cluster_ratio = len(cluster_text) / len(text)
                cluster_max_length = kwargs.get('max_length', self.config.max_length)
                cluster_max_length = max(50, int(cluster_max_length * cluster_ratio))
                
                # Summarize cluster
                summary = self.summarizer.summarize(
                    cluster_text,
                    max_length=cluster_max_length,
                    **kwargs
                )
                cluster_summaries.append(summary)
            
            # Remove redundancies between cluster summaries
            unique_summaries = remove_redundant_sentences(
                cluster_summaries,
                similarity_threshold=self.config.redundancy_threshold
            )
            
            # Create overall summary from cluster summaries
            combined_summary = " ".join(unique_summaries)
            overall_max_length = kwargs.get('max_length', self.config.max_length)
            overall_summary = self.summarizer.summarize(
                combined_summary,
                max_length=overall_max_length,
                **kwargs
            )
            
            return {
                "overall_summary": overall_summary,
                "cluster_summaries": cluster_summaries,
                "num_clusters": len(cluster_contents)
            }
            
        except Exception as e:
            logger.error(f"Error in clustering-based summarization: {e}")
            # Fall back to section-based summarization
            return self.summarize_by_sections(text, **kwargs)
    
    def _generate_ultra_short(self, text: str) -> str:
        """Generate an ultra-short (1-2 sentence) summary.
        
        Args:
            text: Text to summarize
            
        Returns:
            Ultra-short summary
        """
        # If text is already very short, just return it
        word_count = len(text.split())
        if word_count <= 30:
            return text
        
        # Otherwise, create an ultra-short summary
        return self.summarizer.summarize(
            text,
            max_length=20,
            min_length=10
        ) 