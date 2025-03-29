"""Evaluation metrics for summarization quality."""

import logging
from typing import List, Dict, Union, Optional, Tuple
import re
import math

# Try to import libraries for ROUGE and other metrics
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def calculate_rouge(summary: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Calculate ROUGE scores between generated summary and reference summary.
    
    Args:
        summary: Generated summary
        reference: Reference/ground truth summary
        
    Returns:
        Dictionary with ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        logger.warning("ROUGE not available. Install with: pip install rouge")
        return {"error": "ROUGE not available"}
    
    if not summary or not reference:
        return {"error": "Empty summary or reference"}
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(summary, reference)[0]
        return scores
    except Exception as e:
        logger.error(f"Error calculating ROUGE: {e}")
        return {"error": str(e)}


def calculate_bleu(summary: str, reference: str) -> float:
    """Calculate BLEU score between generated summary and reference summary.
    
    Args:
        summary: Generated summary
        reference: Reference/ground truth summary
        
    Returns:
        BLEU score (0-1)
    """
    if not NLTK_AVAILABLE:
        logger.warning("NLTK not available. Install with: pip install nltk")
        return -1
    
    if not summary or not reference:
        return 0.0
    
    try:
        # Tokenize sentences
        summary_tokens = nltk.word_tokenize(summary.lower())
        reference_tokens = nltk.word_tokenize(reference.lower())
        
        # Apply smoothing for short sentences
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU with smoothing
        return sentence_bleu([reference_tokens], summary_tokens, smoothing_function=smoothing)
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        return 0.0


def calculate_cosine_similarity(summary: str, reference: str) -> float:
    """Calculate cosine similarity between summary and reference.
    
    Args:
        summary: Generated summary
        reference: Reference text or summary
        
    Returns:
        Cosine similarity score (0-1)
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available. Install with: pip install scikit-learn")
        return -1
    
    if not summary or not reference:
        return 0.0
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer().fit([summary, reference])
        vectors = vectorizer.transform([summary, reference])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def calculate_compression_ratio(original: str, summary: str) -> float:
    """Calculate compression ratio between original text and summary.
    
    Args:
        original: Original text
        summary: Generated summary
        
    Returns:
        Compression ratio (summary length / original length)
    """
    if not original or not summary:
        return 0.0
    
    # Count words in both texts
    original_words = len(original.split())
    summary_words = len(summary.split())
    
    # Avoid division by zero
    if original_words == 0:
        return 0.0
    
    return summary_words / original_words


def calculate_density(summary: str) -> float:
    """Calculate information density of the summary.
    
    Args:
        summary: Generated summary
        
    Returns:
        Information density score (higher is better)
    """
    if not summary:
        return 0.0
    
    # Count words and unique content words
    words = summary.lower().split()
    
    # Skip stop words if NLTK is available
    if NLTK_AVAILABLE:
        try:
            from nltk.corpus import stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                try:
                    nltk.download('stopwords', quiet=True)
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = set()
            
            content_words = [w for w in words if w not in stop_words and len(w) > 2]
        except Exception:
            # Fall back to simple filtering
            content_words = [w for w in words if len(w) > 2]
    else:
        # Simple heuristic: words longer than 2 chars
        content_words = [w for w in words if len(w) > 2]
    
    unique_content_words = set(content_words)
    
    # Calculate density
    if len(words) == 0:
        return 0.0
    
    # Density is the ratio of unique content words to total words
    return len(unique_content_words) / len(words)


def calculate_readability(text: str) -> Dict[str, float]:
    """Calculate readability metrics for the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with readability scores
    """
    if not text:
        return {"flesch_reading_ease": 0.0, "automated_readability_index": 0.0}
    
    # Count sentences, words, characters, syllables
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    word_count = len(words)
    char_count = sum(len(word) for word in words)
    
    if word_count == 0 or len(sentences) == 0:
        return {"flesch_reading_ease": 0.0, "automated_readability_index": 0.0}
    
    # Calculate Flesch Reading Ease
    # Higher scores = easier to read (60-70 is ideal)
    syllable_count = _estimate_syllables(text)
    flesch = 206.835 - 1.015 * (word_count / len(sentences)) - 84.6 * (syllable_count / word_count)
    
    # Clamp to reasonable range
    flesch = max(0.0, min(100.0, flesch))
    
    # Calculate Automated Readability Index
    # Lower is easier to read (around 7-8 is general public level)
    ari = 4.71 * (char_count / word_count) + 0.5 * (word_count / len(sentences)) - 21.43
    
    return {
        "flesch_reading_ease": flesch,
        "automated_readability_index": ari
    }


def evaluate_summary(
    summary: str,
    original_text: str,
    reference_summary: Optional[str] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Calculate comprehensive evaluation metrics for a summary.
    
    Args:
        summary: Generated summary
        original_text: Original text
        reference_summary: Reference/ground truth summary (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {}
    
    # Basic metrics
    results["compression_ratio"] = calculate_compression_ratio(original_text, summary)
    results["density"] = calculate_density(summary)
    results["readability"] = calculate_readability(summary)
    
    # Compare to original text
    results["similarity_to_original"] = calculate_cosine_similarity(summary, original_text)
    
    # Compare to reference if available
    if reference_summary:
        if ROUGE_AVAILABLE:
            results["rouge"] = calculate_rouge(summary, reference_summary)
        
        results["bleu"] = calculate_bleu(summary, reference_summary)
        results["similarity_to_reference"] = calculate_cosine_similarity(summary, reference_summary)
    
    # Calculate overall score (weighted combination)
    overall_score = _calculate_overall_score(results)
    results["overall_score"] = overall_score
    
    return results


def _estimate_syllables(text: str) -> int:
    """Estimate the number of syllables in text using a basic heuristic.
    
    Args:
        text: Text to analyze
        
    Returns:
        Estimated syllable count
    """
    # Basic syllable counting heuristic
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    
    count = 0
    for word in words:
        word_count = 0
        # Count vowel groups as syllables
        vowel_groups = re.findall(r'[aeiouy]+', word)
        word_count = len(vowel_groups)
        
        # Adjust for common patterns
        if word.endswith('e') and word_count > 1:
            word_count -= 1
        if word.endswith('le') and word_count > 1 and word[-3] not in 'aeiouy':
            word_count += 1
        if word_count == 0:  # Ensure at least one syllable
            word_count = 1
        
        count += word_count
    
    return count


def _calculate_overall_score(metrics: Dict[str, Union[float, Dict[str, float]]]) -> float:
    """Calculate an overall quality score from multiple metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        
    Returns:
        Overall quality score (0-1)
    """
    score = 0.0
    components = 0
    
    # Add density score (weighted by 0.2)
    if "density" in metrics and metrics["density"] > 0:
        score += 0.2 * min(1.0, metrics["density"] * 2)  # Scale appropriately
        components += 0.2
    
    # Add readability score (weighted by 0.2)
    if "readability" in metrics:
        readability = metrics["readability"]
        if "flesch_reading_ease" in readability:
            # Convert Flesch score to 0-1 scale (higher is better)
            # 100 is very easy, 0 is very difficult, aim for 60-70
            flesch_norm = min(1.0, max(0.0, readability["flesch_reading_ease"] / 100))
            score += 0.2 * flesch_norm
            components += 0.2
    
    # Add ROUGE score if available (weighted by 0.3)
    if "rouge" in metrics and isinstance(metrics["rouge"], dict):
        rouge = metrics["rouge"]
        if "rouge-l" in rouge and "f" in rouge["rouge-l"]:
            score += 0.3 * rouge["rouge-l"]["f"]
            components += 0.3
    
    # Add BLEU score if available (weighted by 0.1)
    if "bleu" in metrics and metrics["bleu"] > 0:
        score += 0.1 * metrics["bleu"]
        components += 0.1
    
    # Add similarity score if available (weighted by 0.2)
    if "similarity_to_reference" in metrics and metrics["similarity_to_reference"] > 0:
        score += 0.2 * metrics["similarity_to_reference"]
        components += 0.2
    elif "similarity_to_original" in metrics and metrics["similarity_to_original"] > 0:
        # If reference not available, use similarity to original but weigh less
        # We want some similarity but not too much (which could indicate copying)
        similarity = metrics["similarity_to_original"]
        # Penalize both very low and very high similarity
        adjusted_similarity = 1.0 - 2.0 * abs(0.5 - similarity)
        score += 0.2 * max(0, adjusted_similarity)
        components += 0.2
    
    # Normalize based on available components
    if components > 0:
        score = score / components
    
    return score 