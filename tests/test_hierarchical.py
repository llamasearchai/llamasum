"""Tests for the HierarchicalSummarizer class."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path so we can import llamasum
sys.path.append(str(Path(__file__).parent.parent))

from llamasum.config import HierarchicalConfig
from llamasum.hierarchical import HierarchicalSummarizer


class TestHierarchicalSummarizer(unittest.TestCase):
    """Test cases for the HierarchicalSummarizer class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a mock for the base summarizer
        patcher = patch("llamasum.hierarchical.LlamaSummarizer")
        self.mock_summarizer_class = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_summarizer = MagicMock()
        self.mock_summarizer_class.return_value = self.mock_summarizer

        # Create test config
        self.config = HierarchicalConfig(
            model_name="test/model", device="cpu", levels=3, include_ultra_short=True
        )

        # Create hierarchical summarizer
        self.hierarchical_summarizer = HierarchicalSummarizer(self.config)

    def test_init_creates_base_summarizer(self):
        """Test that the hierarchical summarizer creates a base summarizer during initialization."""
        # The summarizer should be initialized with the config
        self.mock_summarizer_class.assert_called_once_with(config=self.config)

        # The summarizer should be set to the mock summarizer
        self.assertEqual(self.hierarchical_summarizer.summarizer, self.mock_summarizer)

    @patch("llamasum.hierarchical.split_into_sentences")
    def test_summarize_with_three_levels(self, mock_split):
        """Test hierarchical summarization with three levels."""
        # Set up mocks
        mock_split.return_value = [
            "Sentence 1.",
            "Sentence 2.",
            "Sentence 3.",
            "Sentence 4.",
            "Sentence 5.",
            "Sentence 6.",
            "Sentence 7.",
            "Sentence 8.",
            "Sentence 9.",
            "Sentence 10.",
            "Sentence 11.",
            "Sentence 12.",
        ]

        # Set up the summarize method to return different summaries for each level
        self.mock_summarizer.summarize.side_effect = [
            "Level 1 summary - most detailed.",
            "Level 2 summary - medium detail.",
            "Level 3 summary - least detailed.",
            "Ultra-short summary.",
        ]

        # Test
        result = self.hierarchical_summarizer.summarize(
            "Sample text with multiple sentences."
        )

        # The summarize method should be called for each level plus ultra-short
        self.assertEqual(self.mock_summarizer.summarize.call_count, 4)

        # The result should have entries for each level plus the original text
        self.assertEqual(result["original"], "Sample text with multiple sentences.")
        self.assertEqual(result["level_1"], "Level 1 summary - most detailed.")
        self.assertEqual(result["level_2"], "Level 2 summary - medium detail.")
        self.assertEqual(result["level_3"], "Level 3 summary - least detailed.")
        self.assertEqual(result["ultra_short"], "Ultra-short summary.")

    @patch("llamasum.hierarchical.split_into_sentences")
    def test_summarize_short_text_reduces_levels(self, mock_split):
        """Test that very short text results in fewer levels."""
        # Set up mocks for a short text
        mock_split.return_value = ["Sentence 1.", "Sentence 2.", "Sentence 3."]

        # Set up the summarize method to return different summaries
        self.mock_summarizer.summarize.side_effect = [
            "Level 1 summary.",
            "Level 2 summary.",
            "Ultra-short summary.",
        ]

        # Test
        result = self.hierarchical_summarizer.summarize("Short sample text.")

        # For very short text (< 10 sentences), we should reduce to 2 levels
        # So summarize should be called 3 times (2 levels + ultra-short)
        self.assertEqual(self.mock_summarizer.summarize.call_count, 3)

        # The result should have entries for two levels plus the original text
        self.assertEqual(result["original"], "Short sample text.")
        self.assertEqual(result["level_1"], "Level 1 summary.")
        self.assertEqual(result["level_2"], "Level 2 summary.")
        self.assertEqual(result["ultra_short"], "Ultra-short summary.")

    @patch("llamasum.hierarchical.HierarchicalSummarizer._split_into_sections")
    def test_summarize_by_sections(self, mock_split_sections):
        """Test section-based summarization."""
        # Set up mocks
        mock_split_sections.return_value = [
            "Section 1 with some content.",
            "Section 2 with more content.",
            "Section 3 with final content.",
        ]

        # Set up the summarize method to return different summaries
        self.mock_summarizer.summarize.side_effect = [
            "Summary of section 1.",
            "Summary of section 2.",
            "Summary of section 3.",
            "Overall summary of all sections.",
        ]

        # Mock remove_redundant_sentences to return the same list
        with patch(
            "llamasum.hierarchical.remove_redundant_sentences",
            side_effect=lambda x, **kwargs: x,
        ):
            # Test
            result = self.hierarchical_summarizer.summarize_by_sections(
                "Sample text with multiple sections."
            )

            # The summarize method should be called for each section plus the overall summary
            self.assertEqual(self.mock_summarizer.summarize.call_count, 4)

            # The result should have section summaries and overall summary
            self.assertEqual(
                result["overall_summary"], "Overall summary of all sections."
            )
            self.assertEqual(len(result["section_summaries"]), 3)
            self.assertEqual(result["section_summaries"][0], "Summary of section 1.")
            self.assertEqual(result["section_summaries"][1], "Summary of section 2.")
            self.assertEqual(result["section_summaries"][2], "Summary of section 3.")

    def test_generate_ultra_short(self):
        """Test generation of ultra-short summary."""
        # Set up the summarize method to return a summary
        self.mock_summarizer.summarize.return_value = "Ultra-short version."

        # Test with text that's short enough to not need further summarization
        short_result = self.hierarchical_summarizer._generate_ultra_short("Short text.")

        # For very short text, it should just return the input
        self.assertEqual(short_result, "Short text.")

        # Test with longer text that needs summarization
        longer_text = (
            "This is a much longer text that contains more than thirty words. "
            "It should be summarized to create an ultra-short version that "
            "captures just the essential information in one or two sentences."
        )

        longer_result = self.hierarchical_summarizer._generate_ultra_short(longer_text)

        # For longer text, it should call the summarizer
        self.mock_summarizer.summarize.assert_called_once()

        # The result should be the summarized version
        self.assertEqual(longer_result, "Ultra-short version.")


if __name__ == "__main__":
    unittest.main()
