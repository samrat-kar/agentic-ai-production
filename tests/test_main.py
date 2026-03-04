"""Tests for the CLI entry point (src.main)."""

import sys
from unittest.mock import patch, MagicMock

import pytest


class TestMainCLI:
    @patch("src.main.build_crew")
    def test_default_question(self, mock_build_crew, capsys):
        """When no args are passed, the default question is used."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "mock result"
        mock_build_crew.return_value = mock_crew

        with patch.object(sys, "argv", ["main"]):
            from src.main import main
            main()

        mock_crew.kickoff.assert_called_once()
        call_kwargs = mock_crew.kickoff.call_args
        assert "Explain RAG" in call_kwargs[1]["inputs"]["question"]

        captured = capsys.readouterr()
        assert "mock result" in captured.out

    @patch("src.main.build_crew")
    def test_custom_question(self, mock_build_crew):
        """CLI arguments are joined into the question string."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "ok"
        mock_build_crew.return_value = mock_crew

        with patch.object(sys, "argv", ["main", "What", "is", "AI?"]):
            from src.main import main
            main()

        call_kwargs = mock_crew.kickoff.call_args
        assert call_kwargs[1]["inputs"]["question"] == "What is AI?"
