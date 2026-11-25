"""Root entry point for the CLIP image-classification command-line interface."""

from clip_classifier.cli import main as cli_main


def main() -> int:
    """Launch the CLI and return its process exit code."""
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
