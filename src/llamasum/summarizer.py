"""Core summarizer module for LlamaSum."""

import logging
from typing import List, Dict, Optional, Any, Union

from llamasum.config import SummarizerConfig
from llamasum.preprocessing import preprocess_text, split_into_sentences
from llamasum.extractive import extract_key_sentences

logger = logging.getLogger(__name__)

# Optional imports for the transformers library
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers and/or torch not available, using mock summarizer")
    TRANSFORMERS_AVAILABLE = False


class LlamaSummarizer:
    """Core summarizer that combines extractive and abstractive methods."""
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        """Initialize summarizer with configuration.
        
        Args:
            config: Configuration for the summarizer
        """
        self.config = config or SummarizerConfig()
        
        # Initialize model and tokenizer if transformers is available
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading model: {self.config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
                self.model.to(self.config.device)
                logger.info(f"Model loaded on {self.config.device}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # Create placeholder objects for testing
                self.tokenizer = None
                self.model = None
        else:
            # Create placeholder objects for testing
            self.tokenizer = None
            self.model = None
    
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Summarize the provided text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            min_length: Minimum length of summary in words
            **kwargs: Additional arguments for the summarization process
            
        Returns:
            Generated summary
        """
        if not text:
            return ""
        
        # Set default values from config if not provided
        max_length = max_length or self.config.max_length
        min_length = min_length or self.config.min_length
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Split into sentences
        sentences = split_into_sentences(processed_text)
        
        # Apply extractive summarization for long texts
        # to reduce input size and focus on important content
        extractive_ratio = kwargs.get('extractive_ratio', self.config.extractive_ratio)
        if len(sentences) > self.config.extractive_threshold:
            # Extract key sentences
            extraction_method = kwargs.get('extraction_method', self.config.extraction_method)
            key_sentences = extract_key_sentences(
                sentences, 
                extraction_ratio=extractive_ratio,
                method=extraction_method
            )
            
            # Combine sentences back into text
            extracted_text = " ".join(key_sentences)
        else:
            # For short texts, skip extractive phase
            extracted_text = processed_text
        
        # Apply abstractive summarization if transformers is available
        if TRANSFORMERS_AVAILABLE and self.model and self.tokenizer:
            return self._generate_abstractive_summary(
                extracted_text,
                max_length=max_length,
                min_length=min_length,
                **kwargs
            )
        else:
            # Fallback for when transformers is not available
            logger.warning("Transformers not available, returning extractive summary")
            return " ".join(extract_key_sentences(
                sentences,
                extraction_ratio=min(0.3, max_length / len(sentences) if len(sentences) > 0 else 0.3)
            ))
    
    def _generate_abstractive_summary(
        self,
        text: str,
        max_length: int,
        min_length: int,
        **kwargs
    ) -> str:
        """Generate an abstractive summary using the transformer model.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            min_length: Minimum length of summary in words
            **kwargs: Additional arguments for the generation process
            
        Returns:
            Generated summary
        """
        try:
            # Encode text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            )
            
            # Move inputs to the right device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": self.config.model_max_length,
                "min_length": min_length,
                "num_beams": self.config.num_beams,
                "do_sample": self.config.do_sample,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "length_penalty": self.config.length_penalty,
                "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
                "early_stopping": True
            }
            
            # Override with any provided kwargs
            gen_kwargs.update({k: v for k, v in kwargs.items() 
                              if k in gen_kwargs})
            
            # Generate summary
            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode summary
            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return extractive summary as fallback
            return " ".join(extract_key_sentences(
                split_into_sentences(text),
                extraction_ratio=min(0.3, max_length / len(text.split()) if text else 0.3)
            ))
    
    def summarize_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Summarize multiple texts in batch mode.
        
        Args:
            texts: List of texts to summarize
            batch_size: Number of texts to process at once
            **kwargs: Additional arguments for summarization
            
        Returns:
            List of summaries corresponding to input texts
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Process each text in the batch
            batch_results = [self.summarize(text, **kwargs) for text in batch]
            results.extend(batch_results)
        
        return results 