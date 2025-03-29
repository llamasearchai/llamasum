"""Text preprocessing functions for LlamaSum."""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional, Any

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """Clean and normalize input text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep punctuation)
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]{}]', '', text)
    
    # Trim whitespace
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    try:
        if NLTK_AVAILABLE:
            # Use NLTK's sentence tokenizer
            return sent_tokenize(text)
        else:
            # Simple fallback with regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning(f"Error splitting sentences: {e}")
        # Very simple fallback
        return [s.strip() + "." for s in text.split(".") if s.strip()]


def get_word_count(text: str) -> int:
    """Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of words
    """
    if not text:
        return 0
    
    # Split by whitespace and count non-empty items
    return len([w for w in text.split() if w.strip()])


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts (0-1).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity (intersection / union)
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union


def remove_redundant_sentences(
    sentences: List[str],
    similarity_threshold: float = 0.8
) -> List[str]:
    """Remove redundant sentences based on similarity.
    
    Args:
        sentences: List of sentences
        similarity_threshold: Threshold to consider sentences as redundant
        
    Returns:
        List of non-redundant sentences
    """
    if not sentences:
        return []
    
    if len(sentences) <= 1:
        return sentences
    
    unique_sentences = []
    
    for sentence in sentences:
        # Check if sentence is similar to any already-selected sentence
        is_redundant = False
        for unique_sentence in unique_sentences:
            similarity = calculate_similarity(sentence, unique_sentence)
            if similarity > similarity_threshold:
                is_redundant = True
                break
        
        # If not redundant, add to unique sentences
        if not is_redundant:
            unique_sentences.append(sentence)
    
    return unique_sentences


def detect_sections(
    text: str,
    min_section_length: int = 100
) -> List[str]:
    """Split text into logical sections based on formatting and content.
    
    Args:
        text: Input text
        min_section_length: Minimum characters for a valid section
        
    Returns:
        List of section texts
    """
    if not text or len(text) < min_section_length:
        return [text] if text else []
    
    # Method 1: Split by double newlines (paragraphs)
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    
    # Method 2: Look for section headings
    heading_pattern = r'(?:^|\n)#+\s+(.+?)(?:\n|$)'
    headings = re.finditer(heading_pattern, text)
    heading_positions = [(m.start(), m.group(0)) for m in headings]
    
    if heading_positions:
        # Use headings to split text
        section_texts = []
        for i, (pos, heading) in enumerate(heading_positions):
            # Get section start
            start = pos
            
            # Get section end (next heading or end of text)
            if i < len(heading_positions) - 1:
                end = heading_positions[i + 1][0]
            else:
                end = len(text)
            
            # Extract section content
            section = text[start:end].strip()
            if section and len(section) >= min_section_length:
                section_texts.append(section)
        
        # If we found valid sections using headings, use those
        if section_texts:
            sections = section_texts
    
    # Ensure sections meet minimum length
    valid_sections = []
    current_section = ""
    
    for section in sections:
        if len(section) >= min_section_length:
            # If we have accumulated text, add it to the current section
            if current_section:
                section = current_section + "\n\n" + section
                current_section = ""
            valid_sections.append(section)
        else:
            # Accumulate short sections
            if current_section:
                current_section += "\n\n" + section
            else:
                current_section = section
    
    # Add any remaining accumulated text
    if current_section and len(current_section) >= min_section_length:
        valid_sections.append(current_section)
    elif current_section and valid_sections:
        # Append to the last section if we have one
        valid_sections[-1] += "\n\n" + current_section
    elif current_section:
        # Just keep the short text as its own section
        valid_sections.append(current_section)
    
    # Ensure we have at least one section
    if not valid_sections and text:
        valid_sections = [text]
    
    return valid_sections 