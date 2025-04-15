"""Tests for the MultiDocSummarizer class."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path so we can import llamasum
sys.path.append(str(Path(__file__).parent.parent))

from llamasum.config import MultiDocConfig
from llamasum.multidoc import MultiDocSummarizer


class TestMultiDocSummarizer(unittest.TestCase):
    """Test cases for the MultiDocSummarizer class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a mock for the base summarizer
        patcher = patch("llamasum.multidoc.LlamaSummarizer")
        self.mock_summarizer_class = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_summarizer = MagicMock()
        self.mock_summarizer_class.return_value = self.mock_summarizer

        # Create test config
        self.config = MultiDocConfig(
            model_name="test/model",
            device="cpu",
            strategy="extract_then_summarize",
            max_length=150,
        )

        # Create multi-document summarizer
        self.multidoc_summarizer = MultiDocSummarizer(self.config)

    def test_init_creates_base_summarizer(self):
        """Test that the multi-document summarizer creates a base summarizer during initialization."""
        # The summarizer should be initialized with the config
        self.mock_summarizer_class.assert_called_once_with(config=self.config)

        # The summarizer should be set to the mock summarizer
        self.assertEqual(self.multidoc_summarizer.summarizer, self.mock_summarizer)

    def test_summarize_single_document(self):
        """Test summarization of a single document."""
        # Set up the summarize method to return a summary
        self.mock_summarizer.summarize.return_value = "Summary of single document."

        # Test
        result = self.multidoc_summarizer.summarize(["Single document text."])

        # For a single document, it should use the base summarizer
        self.mock_summarizer.summarize.assert_called_once_with(
            "Single document text.", max_length=self.config.max_length
        )

        # The result should be a dictionary with the summary
        self.assertEqual(result["summary"], "Summary of single document.")
        self.assertEqual(result["strategy"], "single_document")

    @patch("llamasum.multidoc.MultiDocSummarizer._summarize_extract_then_summarize")
    def test_summarize_multiple_documents_with_default_strategy(
        self, mock_extract_summarize
    ):
        """Test summarization of multiple documents with default strategy."""
        # Set up the mock to return a result
        mock_extract_summarize.return_value = {
            "summary": "Summary of multiple documents.",
            "strategy": "extract_then_summarize",
            "num_docs": 3,
        }

        # Test
        documents = ["Document 1.", "Document 2.", "Document 3."]
        result = self.multidoc_summarizer.summarize(documents)

        # It should use the extract_then_summarize strategy
        mock_extract_summarize.assert_called_once_with(
            documents, self.config.max_length
        )

        # The result should be the output of the strategy method
        self.assertEqual(result["summary"], "Summary of multiple documents.")
        self.assertEqual(result["strategy"], "extract_then_summarize")
        self.assertEqual(result["num_docs"], 3)

    @patch("llamasum.multidoc.MultiDocSummarizer._summarize_iterative")
    def test_summarize_with_iterative_strategy(self, mock_iterative):
        """Test summarization with iterative strategy."""
        # Set up the mock to return a result
        mock_iterative.return_value = {
            "summary": "Iterative summary of documents.",
            "strategy": "iterative",
            "num_docs": 3,
        }

        # Test with iterative strategy
        documents = ["Document 1.", "Document 2.", "Document 3."]
        result = self.multidoc_summarizer.summarize(documents, strategy="iterative")

        # It should use the iterative strategy
        mock_iterative.assert_called_once_with(documents, self.config.max_length)

        # The result should be the output of the strategy method
        self.assertEqual(result["summary"], "Iterative summary of documents.")
        self.assertEqual(result["strategy"], "iterative")

    @patch("llamasum.multidoc.split_into_sentences")
    @patch("llamasum.multidoc.MultiDocSummarizer._extract_important_sentences")
    def test_extract_then_summarize_strategy(self, mock_extract, mock_split):
        """Test the extract_then_summarize strategy."""
        # Set up mocks
        mock_split.side_effect = [
            ["Doc1 Sentence 1.", "Doc1 Sentence 2."],
            ["Doc2 Sentence 1.", "Doc2 Sentence 2."],
            ["Doc3 Sentence 1.", "Doc3 Sentence 2."],
        ]

        mock_extract.return_value = [0, 2, 3]

        # Set up the summarize method to return a summary
        self.mock_summarizer.summarize.return_value = (
            "Summary using extract-then-summarize."
        )

        # Test
        documents = ["Document 1.", "Document 2.", "Document 3."]

        # For testing extract_then_summarize, we need to patch the combined split sentences list
        # and ensure it matches what would be created by the side effect above
        all_sentences = [
            "Doc1 Sentence 1.",
            "Doc1 Sentence 2.",
            "Doc2 Sentence 1.",
            "Doc2 Sentence 2.",
            "Doc3 Sentence 1.",
            "Doc3 Sentence 2.",
        ]

        with patch("llamasum.multidoc.split_into_sentences") as mock_split:
            # First call gets all sentences from first doc, and so on
            mock_split.side_effect = [
                ["Doc1 Sentence 1.", "Doc1 Sentence 2."],
                ["Doc2 Sentence 1.", "Doc2 Sentence 2."],
                ["Doc3 Sentence 1.", "Doc3 Sentence 2."],
            ]

            with patch(
                "llamasum.multidoc.MultiDocSummarizer._extract_important_sentences"
            ) as mock_extract:
                # Return indices of important sentences
                mock_extract.return_value = [0, 2, 4]

                result = self.multidoc_summarizer._summarize_extract_then_summarize(
                    documents, max_length=150
                )

        # The result should be a dictionary with the summary
        self.assertEqual(result["strategy"], "extract_then_summarize")
        self.assertEqual(result["num_docs"], 3)

    def test_summarize_with_queries(self):
        """Test query-focused summarization."""
        # Set up mocks
        self.multidoc_summarizer.summarize = MagicMock(
            return_value={
                "summary": "General summary.",
                "strategy": "extract_then_summarize",
                "num_docs": 3,
            }
        )

        self.multidoc_summarizer._extract_query_relevant_content = MagicMock(
            side_effect=["Content for query 1.", "Content for query 2."]
        )

        self.mock_summarizer.summarize.side_effect = [
            "Summary for query 1.",
            "Summary for query 2.",
        ]

        # Test
        documents = ["Document 1.", "Document 2.", "Document 3."]
        queries = ["Query 1", "Query 2"]

        result = self.multidoc_summarizer.summarize_with_queries(
            documents, queries, max_length=100
        )

        # It should first get a general summary
        self.multidoc_summarizer.summarize.assert_called_once_with(
            documents, max_length=100
        )

        # It should extract content for each query
        self.assertEqual(
            self.multidoc_summarizer._extract_query_relevant_content.call_count, 2
        )

        # It should summarize the content for each query
        self.assertEqual(self.mock_summarizer.summarize.call_count, 2)

        # The result should include general and query-focused summaries
        self.assertEqual(result["general_summary"], "General summary.")
        self.assertEqual(result["query_summaries"]["Query 1"], "Summary for query 1.")
        self.assertEqual(result["query_summaries"]["Query 2"], "Summary for query 2.")
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["num_docs"], 3)


if __name__ == "__main__":
    unittest.main()
