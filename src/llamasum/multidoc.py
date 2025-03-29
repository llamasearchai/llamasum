"""Multi-document summarization module for LlamaSum."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import math

import numpy as np

from llamasum.summarizer import LlamaSummarizer
from llamasum.hierarchical import HierarchicalSummarizer
from llamasum.config import MultiDocConfig, SummarizerConfig
from llamasum.preprocessing import split_into_sentences, remove_redundant_sentences

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available, using basic similarity for multi-document summarization")
    SKLEARN_AVAILABLE = False


class MultiDocSummarizer:
    """Summarizer for multiple documents."""
    
    def __init__(self, config: Optional[MultiDocConfig] = None):
        """Initialize multi-document summarizer.
        
        Args:
            config: Configuration for multi-document summarization
        """
        self.config = config or MultiDocConfig()
        
        # Initialize base summarizer
        self.summarizer = LlamaSummarizer(config=self.config)
        
        # Initialize hierarchical summarizer (used for some strategies)
        self.hierarchical_summarizer = None
    
    def summarize(
        self, 
        documents: List[str],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Summarize multiple documents.
        
        Args:
            documents: List of documents to summarize
            max_length: Maximum length of the summary
            **kwargs: Additional parameters to pass to summarizers
            
        Returns:
            Dictionary with summary and metadata
        """
        if not documents:
            return {"error": "No documents provided"}
        
        # If only one document, use the regular summarizer
        if len(documents) == 1:
            summary = self.summarizer.summarize(
                documents[0],
                max_length=max_length or self.config.max_length,
                **kwargs
            )
            return {"summary": summary, "strategy": "single_document"}
        
        # Determine strategy to use
        strategy = kwargs.get("strategy", self.config.strategy)
        
        # Set max_length if not provided
        if max_length is None:
            max_length = self.config.max_length
        
        # Apply selected strategy
        if strategy == "iterative":
            return self._summarize_iterative(documents, max_length, **kwargs)
        elif strategy == "hierarchical":
            return self._summarize_hierarchical(documents, max_length, **kwargs)
        else:  # default: extract_then_summarize
            return self._summarize_extract_then_summarize(documents, max_length, **kwargs)
    
    def _summarize_iterative(
        self,
        documents: List[str],
        max_length: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Summarize documents iteratively (single-document summaries first, then combine).
        
        Args:
            documents: List of documents
            max_length: Maximum length of the final summary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with summary and metadata
        """
        # Step 1: Summarize each document individually
        individual_summaries = []
        for doc in documents:
            # Skip very short documents
            if len(doc.split()) < 20:
                individual_summaries.append(doc)
                continue
                
            # Create individual summary with proportional length
            doc_max_length = max(50, int(max_length / 2 / len(documents)))
            summary = self.summarizer.summarize(
                doc,
                max_length=doc_max_length,
                **kwargs
            )
            individual_summaries.append(summary)
        
        # Step 2: Remove redundancies between summaries
        if len(individual_summaries) > 1:
            individual_summaries = remove_redundant_sentences(
                individual_summaries,
                similarity_threshold=self.config.redundancy_threshold
            )
        
        # Step 3: Combine individual summaries
        combined_text = " ".join(individual_summaries)
        
        # Step 4: Create final summary from combined text
        final_summary = self.summarizer.summarize(
            combined_text,
            max_length=max_length,
            **kwargs
        )
        
        return {
            "summary": final_summary,
            "individual_summaries": individual_summaries,
            "strategy": "iterative",
            "num_docs": len(documents)
        }
    
    def _summarize_extract_then_summarize(
        self,
        documents: List[str],
        max_length: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract important sentences from all documents, then summarize them.
        
        Args:
            documents: List of documents
            max_length: Maximum length of the final summary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with summary and metadata
        """
        # Step 1: Split all documents into sentences
        all_sentences = []
        doc_boundaries = []
        current_position = 0
        
        for doc in documents:
            sentences = split_into_sentences(doc)
            if sentences:
                all_sentences.extend(sentences)
                current_position += len(sentences)
                doc_boundaries.append(current_position)
        
        if not all_sentences:
            return {"error": "No valid sentences found in documents"}
        
        # Step 2: Extract important sentences
        max_sentences = min(len(all_sentences), max(20, int(len(all_sentences) * 0.2)))
        selected_indices = self._extract_important_sentences(all_sentences, max_sentences)
        
        # Step 3: Order selected sentences by their original document and position
        selected_sentences = [all_sentences[i] for i in selected_indices]
        
        # If using document ordering
        if self.config.use_temporal_ordering and len(documents) > 1:
            # Create a list of (sentence, doc_idx, sent_idx) tuples
            sent_with_position = []
            for i in selected_indices:
                # Find which document this sentence belongs to
                doc_idx = 0
                while doc_idx < len(doc_boundaries) and i >= doc_boundaries[doc_idx]:
                    doc_idx += 1
                
                # Calculate sentence index within the document
                prev_boundary = 0 if doc_idx == 0 else doc_boundaries[doc_idx - 1]
                sent_idx = i - prev_boundary
                
                sent_with_position.append((all_sentences[i], doc_idx, sent_idx))
            
            # Sort by document index first, then by sentence index
            sent_with_position.sort(key=lambda x: (x[1], x[2]))
            
            # Extract only the sentences
            selected_sentences = [s[0] for s in sent_with_position]
        
        # Step 4: Combine selected sentences
        extracted_text = " ".join(selected_sentences)
        
        # Step 5: Generate abstractive summary from extracted content
        final_summary = self.summarizer.summarize(
            extracted_text,
            max_length=max_length,
            **kwargs
        )
        
        return {
            "summary": final_summary,
            "extracted_sentences": selected_sentences,
            "strategy": "extract_then_summarize",
            "num_docs": len(documents)
        }
    
    def _summarize_hierarchical(
        self,
        documents: List[str],
        max_length: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Use hierarchical approach for multi-document summarization.
        
        Args:
            documents: List of documents
            max_length: Maximum length of the final summary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with summary and metadata
        """
        # Initialize hierarchical summarizer if not already done
        if self.hierarchical_summarizer is None:
            self.hierarchical_summarizer = HierarchicalSummarizer(self.config)
        
        # Strategy 1: Cluster similar documents and summarize clusters
        if SKLEARN_AVAILABLE and len(documents) > 3:
            # Cluster similar documents
            document_clusters = self._cluster_documents(documents)
            
            # Summarize each cluster
            cluster_summaries = []
            for cluster in document_clusters:
                if not cluster:
                    continue
                
                if len(cluster) == 1:
                    # Single document in cluster
                    cluster_summary = self.summarizer.summarize(
                        cluster[0],
                        max_length=int(max_length / len(document_clusters)),
                        **kwargs
                    )
                else:
                    # Multiple documents in cluster
                    combined = " ".join(cluster)
                    cluster_summary = self.hierarchical_summarizer.summarize(
                        combined,
                        levels=2,
                        **kwargs
                    )["level_2"]
                
                cluster_summaries.append(cluster_summary)
            
            # Combine cluster summaries
            combined_summary = " ".join(cluster_summaries)
            
            # Generate final summary
            final_summary = self.summarizer.summarize(
                combined_summary,
                max_length=max_length,
                **kwargs
            )
            
            return {
                "summary": final_summary,
                "cluster_summaries": cluster_summaries,
                "num_clusters": len(document_clusters),
                "strategy": "hierarchical_clustering",
                "num_docs": len(documents)
            }
        
        # Strategy 2 (fallback): Treat the combined documents as sections
        else:
            combined_text = "\n\n".join(documents)
            result = self.hierarchical_summarizer.summarize_by_sections(
                combined_text,
                **kwargs
            )
            
            # Extract the overall summary
            final_summary = result.get("overall_summary", "")
            
            # Ensure it meets the max length requirement
            if len(final_summary.split()) > max_length:
                final_summary = self.summarizer.summarize(
                    final_summary,
                    max_length=max_length,
                    **kwargs
                )
            
            return {
                "summary": final_summary,
                "section_summaries": result.get("section_summaries", []),
                "strategy": "hierarchical_sections",
                "num_docs": len(documents)
            }
    
    def _extract_important_sentences(
        self,
        sentences: List[str],
        num_to_extract: int
    ) -> List[int]:
        """Extract important sentences from a list of sentences.
        
        Args:
            sentences: List of sentences
            num_to_extract: Number of sentences to extract
            
        Returns:
            List of indices of important sentences
        """
        if not SKLEARN_AVAILABLE or len(sentences) <= num_to_extract:
            # If sklearn is not available or we need all sentences,
            # just return all indices in order
            return list(range(len(sentences)))
        
        # Create TF-IDF vectors for sentences
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate sentence scores based on centrality
            centrality_scores = similarity_matrix.sum(axis=1)
            
            # Select top sentences by centrality score
            top_indices = centrality_scores.argsort()[-num_to_extract:][::-1]
            
            # Convert to python list and sort by position
            return sorted(top_indices.tolist())
            
        except Exception as e:
            logger.warning(f"Error during sentence extraction: {e}")
            # Fallback to regular spacing
            indices = list(range(0, len(sentences), max(1, len(sentences) // num_to_extract)))
            return indices[:num_to_extract]
    
    def _cluster_documents(
        self,
        documents: List[str]
    ) -> List[List[str]]:
        """Cluster similar documents together.
        
        Args:
            documents: List of documents
            
        Returns:
            List of document clusters
        """
        if not SKLEARN_AVAILABLE or len(documents) <= 2:
            # If sklearn is not available or we have few documents,
            # just return all documents as a single cluster
            return [documents]
        
        try:
            # Create document vectors
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Determine number of clusters
            num_clusters = min(
                self.config.num_clusters,
                max(2, len(documents) // 2)
            )
            
            # Cluster documents
            from sklearn.cluster import KMeans
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=42
            )
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Group documents by cluster
            document_clusters = [[] for _ in range(num_clusters)]
            for i, cluster_id in enumerate(clusters):
                document_clusters[cluster_id].append(documents[i])
            
            # Remove empty clusters
            document_clusters = [c for c in document_clusters if c]
            
            return document_clusters
            
        except Exception as e:
            logger.warning(f"Error during document clustering: {e}")
            # Fallback to no clustering
            return [documents]
    
    def summarize_with_queries(
        self,
        documents: List[str],
        queries: List[str],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate query-focused summaries for multiple documents.
        
        Args:
            documents: List of documents to summarize
            queries: List of queries to focus summaries
            max_length: Maximum length of summaries
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with general summary and query-focused summaries
        """
        if not documents:
            return {"error": "No documents provided"}
        
        if not queries:
            # If no queries provided, fall back to regular summarization
            return self.summarize(documents, max_length, **kwargs)
        
        # Set max_length if not provided
        if max_length is None:
            max_length = self.config.max_length
        
        # Step 1: Generate general summary
        general_result = self.summarize(
            documents,
            max_length=max_length,
            **kwargs
        )
        general_summary = general_result.get("summary", "")
        
        # Step 2: Generate query-focused summaries
        query_summaries = {}
        for query in queries:
            # Extract query-relevant content from documents
            query_content = self._extract_query_relevant_content(documents, query)
            
            # Summarize the query-relevant content
            query_summary = self.summarizer.summarize(
                query_content,
                max_length=max_length,
                **kwargs
            )
            
            query_summaries[query] = query_summary
        
        return {
            "general_summary": general_summary,
            "query_summaries": query_summaries,
            "num_queries": len(queries),
            "num_docs": len(documents)
        }
    
    def _extract_query_relevant_content(
        self,
        documents: List[str],
        query: str
    ) -> str:
        """Extract content relevant to a specific query.
        
        Args:
            documents: List of documents
            query: Query to focus on
            
        Returns:
            Extracted content relevant to the query
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: just combine all documents
            return " ".join(documents)
        
        try:
            # Split documents into sentences
            all_sentences = []
            for doc in documents:
                sentences = split_into_sentences(doc)
                all_sentences.extend(sentences)
            
            if not all_sentences:
                return " ".join(documents)
            
            # Create vectors for sentences and query
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            
            # Add query to the sentences to ensure its terms are in the vocabulary
            all_texts = all_sentences + [query]
            vectorizer.fit(all_texts)
            
            # Transform sentences and query
            sentence_vectors = vectorizer.transform(all_sentences)
            query_vector = vectorizer.transform([query])
            
            # Calculate similarity to query
            similarities = cosine_similarity(sentence_vectors, query_vector)
            
            # Select sentences with highest similarity to query
            num_to_select = min(len(all_sentences), max(20, int(len(all_sentences) * 0.3)))
            top_indices = similarities.flatten().argsort()[-num_to_select:][::-1]
            
            # Extract and combine selected sentences
            selected_sentences = [all_sentences[i] for i in sorted(top_indices)]
            return " ".join(selected_sentences)
            
        except Exception as e:
            logger.warning(f"Error during query-focused extraction: {e}")
            # Fallback to combining all documents
            return " ".join(documents) 