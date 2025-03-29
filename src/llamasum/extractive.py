"""Extractive summarization methods for LlamaSum."""

import logging
import math
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict, Counter

from llamasum.preprocessing import split_into_sentences

logger = logging.getLogger(__name__)

# Try to import sklearn for more advanced extractive methods
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available, using basic extractive methods")
    SKLEARN_AVAILABLE = False


def extract_key_sentences(
    sentences: List[str],
    extraction_ratio: float = 0.3,
    method: str = "tfidf",
    min_sentences: int = 3,
    max_sentences: int = 20
) -> List[str]:
    """Extract the most important sentences from a list of sentences.
    
    Args:
        sentences: List of sentences
        extraction_ratio: Ratio of sentences to extract (0.0-1.0)
        method: Extraction method (tfidf, textrank, positional)
        min_sentences: Minimum number of sentences to extract
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        List of extracted key sentences
    """
    if not sentences:
        return []
    
    # Calculate the number of sentences to extract
    num_sentences = max(
        min_sentences,
        min(max_sentences, int(len(sentences) * extraction_ratio))
    )
    
    # Ensure we don't extract more sentences than we have
    num_sentences = min(num_sentences, len(sentences))
    
    # Short-circuit if we need all sentences
    if num_sentences >= len(sentences):
        return sentences
    
    # Choose the extraction method
    if method == "textrank" and SKLEARN_AVAILABLE:
        return extract_by_textrank(sentences, num_sentences)
    elif method == "positional":
        return extract_by_position(sentences, num_sentences)
    else:  # default to tfidf
        if SKLEARN_AVAILABLE:
            return extract_by_tfidf(sentences, num_sentences)
        else:
            # Fallback to a simple method if sklearn is not available
            return extract_by_position(sentences, num_sentences)


def extract_by_tfidf(sentences: List[str], num_sentences: int) -> List[str]:
    """Extract sentences using TF-IDF scoring.
    
    Args:
        sentences: List of sentences
        num_sentences: Number of sentences to extract
        
    Returns:
        List of extracted sentences
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available, falling back to positional extraction")
        return extract_by_position(sentences, num_sentences)
    
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
        # Generate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Get feature names (words)
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
        
        # Calculate sentence scores based on TF-IDF values
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = sum(tfidf_matrix[i, j] for j in range(len(feature_names)))
            sentence_scores.append((i, score))
        
        # Sort sentences by score in descending order
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        top_indices = [idx for idx, _ in sentence_scores[:num_sentences]]
        
        # Sort indices by original position to maintain document flow
        top_indices.sort()
        
        # Return sentences in their original order
        return [sentences[i] for i in top_indices]
    
    except Exception as e:
        logger.error(f"Error in TF-IDF extraction: {e}")
        return extract_by_position(sentences, num_sentences)


def extract_by_textrank(sentences: List[str], num_sentences: int) -> List[str]:
    """Extract sentences using a TextRank-like algorithm.
    
    Args:
        sentences: List of sentences
        num_sentences: Number of sentences to extract
        
    Returns:
        List of extracted sentences
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available, falling back to positional extraction")
        return extract_by_position(sentences, num_sentences)
    
    try:
        # Create sentence vectors
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        sentence_vectors = vectorizer.fit_transform(sentences)
        
        # Create similarity matrix (graph)
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Set diagonal to 0 to avoid self-loops
        np.fill_diagonal(similarity_matrix, 0)
        
        # Calculate score for each sentence (sum of similarities)
        scores = np.sum(similarity_matrix, axis=1)
        
        # Get top sentences
        top_indices = scores.argsort()[-num_sentences:][::-1]
        
        # Sort indices by position in the document
        top_indices.sort()
        
        # Return sentences in their original order
        return [sentences[i] for i in top_indices]
    
    except Exception as e:
        logger.error(f"Error in TextRank extraction: {e}")
        return extract_by_position(sentences, num_sentences)


def extract_by_position(sentences: List[str], num_sentences: int) -> List[str]:
    """Extract sentences based on their position in the document.
    
    Args:
        sentences: List of sentences
        num_sentences: Number of sentences to extract
        
    Returns:
        List of extracted sentences
    """
    if not sentences:
        return []
    
    # For very short texts, just return all sentences
    if len(sentences) <= num_sentences:
        return sentences
    
    # For longer texts, prioritize beginning, middle, and end
    positions = []
    
    # Always include the first sentence
    positions.append(0)
    
    # Always include the last sentence (if more than 3 sentences)
    if len(sentences) > 3:
        positions.append(len(sentences) - 1)
    
    # For the rest, distribute evenly throughout the text
    remaining = num_sentences - len(positions)
    if remaining > 0:
        # Calculate step size for even distribution
        step = max(1, len(sentences) // (remaining + 1))
        
        # Get positions at regular intervals
        for i in range(step, len(sentences) - 1, step):
            positions.append(i)
            if len(positions) >= num_sentences:
                break
        
        # If we still have positions to fill (due to rounding issues)
        while len(positions) < num_sentences and len(positions) < len(sentences):
            # Add midpoint if not already in positions
            midpoint = len(sentences) // 2
            if midpoint not in positions and midpoint > 0 and midpoint < len(sentences) - 1:
                positions.append(midpoint)
            else:
                # Add sentences near the beginning
                for i in range(1, len(sentences) - 1):
                    if i not in positions:
                        positions.append(i)
                        break
            
            # Break if we've added enough positions
            if len(positions) >= num_sentences:
                break
    
    # Sort positions to maintain order
    positions.sort()
    
    # Get sentences at those positions
    return [sentences[i] for i in positions]


def extract_keywords(
    text: str,
    num_keywords: int = 10,
    min_freq: int = 2
) -> List[str]:
    """Extract key terms from text.
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        min_freq: Minimum frequency for a keyword
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Use sklearn's TF-IDF if available
    if SKLEARN_AVAILABLE:
        try:
            # Split text into sentences
            sentences = split_into_sentences(text)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=min_freq / len(sentences) if len(sentences) > 0 else 0.1
            )
            
            # Generate TF-IDF matrix
            vectorizer.fit_transform(sentences)
            
            # Get feature names (words)
            if hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
            else:
                feature_names = vectorizer.get_feature_names()
                
            # Get the idf values
            idfs = vectorizer.idf_
            
            # Create a dictionary of word: idf
            word_idf = dict(zip(feature_names, idfs))
            
            # Sort words by IDF (ascending, as lower IDF means more frequent)
            sorted_words = sorted(word_idf.items(), key=lambda x: x[1])
            
            # Return top keywords
            return [word for word, _ in sorted_words[:num_keywords]]
            
        except Exception as e:
            logger.warning(f"Error in TF-IDF keyword extraction: {e}")
            # Fall back to simple frequency
    
    # Fallback: simple word frequency
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words
    words = text.split()
    
    # Remove common stopwords
    stopwords = {
        'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'of',
        'if', 'this', 'that', 'these', 'those', 'it', 'its', 'with', 'as',
        'from', 'have', 'has', 'had', 'not', 'no', 'what', 'when', 'where',
        'which', 'who', 'whom', 'why', 'how'
    }
    
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Remove words that appear less than min_freq times
    word_counts = {word: count for word, count in word_counts.items() if count >= min_freq}
    
    # Sort by frequency (descending)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [word for word, _ in sorted_words[:num_keywords]] 