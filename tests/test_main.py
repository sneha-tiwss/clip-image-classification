"""Unit tests for the root command-line entry point."""

from unittest.mock import patch

import main


def test_main_delegates_to_cli_main() -> None:
    with patch("main.cli_main", return_value=0) as cli_main:
        exit_code = main.main()

    cli_main.assert_called_once_with()
    assert exit_code == 0


def test_main_preserves_cli_exit_code() -> None:
    with patch("main.cli_main", return_value=7) as cli_main:
        exit_code = main.main()

    cli_main.assert_called_once_with()
    assert exit_code == 7
