"""Tests for the LlamaSummarizer class."""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import llamasum
sys.path.append(str(Path(__file__).parent.parent))

from llamasum.summarizer import LlamaSummarizer
from llamasum.config import SummarizerConfig


class TestLlamaSummarizer(unittest.TestCase):
    """Test cases for the LlamaSummarizer class."""
    
    @patch("llamasum.summarizer.AutoTokenizer")
    @patch("llamasum.summarizer.AutoModelForSeq2SeqLM")
    def setUp(self, mock_model_class, mock_tokenizer_class):
        """Set up test fixtures for each test."""
        # Create mocks
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        
        # Create test config
        self.config = SummarizerConfig(
            model_name="test/model",
            device="cpu",
            max_length=100,
            min_length=20
        )
        
        # Create summarizer
        self.summarizer = LlamaSummarizer(self.config)
    
    def test_init_loads_model(self):
        """Test that the summarizer loads the model during initialization."""
        # The model should be set to the mock model
        self.assertEqual(self.summarizer.model, self.mock_model)
        self.assertEqual(self.summarizer.tokenizer, self.mock_tokenizer)
        
        # The model should be moved to the specified device
        self.mock_model.to.assert_called_once_with(self.config.device)
    
    def test_summarize_empty_text(self):
        """Test that summarize handles empty text appropriately."""
        result = self.summarizer.summarize("")
        self.assertEqual(result, "")
    
    @patch("llamasum.summarizer.split_into_sentences")
    @patch("llamasum.summarizer.preprocess_text")
    def test_summarize_short_text(self, mock_preprocess, mock_split):
        """Test summarization of very short text."""
        # Set up mocks
        mock_preprocess.return_value = "Short text."
        mock_split.return_value = ["Short text."]
        
        # Mock the tokenizer and model
        self.mock_tokenizer.return_tensors = "pt"
        self.mock_tokenizer.model_max_length = 512
        self.mock_tokenizer.decode.return_value = "Summary of short text."
        
        # Test
        result = self.summarizer.summarize("Short text.")
        
        # The model's generate method should be called
        self.mock_model.generate.assert_called_once()
        
        # The result should be the decoded output
        self.assertEqual(result, "Summary of short text.")
    
    @patch("llamasum.summarizer.extract_key_sentences")
    @patch("llamasum.summarizer.split_into_sentences")
    @patch("llamasum.summarizer.preprocess_text")
    def test_summarize_long_text(self, mock_preprocess, mock_split, mock_extract):
        """Test summarization of longer text with extractive phase."""
        # Set up mocks
        mock_preprocess.return_value = "This is a longer text. It has multiple sentences. We need to extract key ones."
        mock_split.return_value = [
            "This is a longer text.",
            "It has multiple sentences.",
            "We need to extract key ones.",
            "This is another sentence."
        ]
        mock_extract.return_value = [
            "This is a longer text.",
            "We need to extract key ones."
        ]
        
        # Mock the tokenizer and model
        self.mock_tokenizer.return_tensors = "pt"
        self.mock_tokenizer.model_max_length = 512
        self.mock_tokenizer.decode.return_value = "Summary of longer text."
        
        # Test
        result = self.summarizer.summarize("This is a longer text. It has multiple sentences.")
        
        # The extractive phase should be called
        mock_extract.assert_called_once()
        
        # The model's generate method should be called
        self.mock_model.generate.assert_called_once()
        
        # The result should be the decoded output
        self.assertEqual(result, "Summary of longer text.")
    
    def test_batch_summarization(self):
        """Test batch summarization."""
        # Mock the summarize method
        self.summarizer.summarize = MagicMock(side_effect=["Summary 1", "Summary 2", "Summary 3"])
        
        # Test
        results = self.summarizer.summarize_batch(["Text 1", "Text 2", "Text 3"], batch_size=2)
        
        # The summarize method should be called for each text
        self.assertEqual(self.summarizer.summarize.call_count, 3)
        
        # The results should be a list of the summaries
        self.assertEqual(results, ["Summary 1", "Summary 2", "Summary 3"])


if __name__ == "__main__":
    unittest.main() 