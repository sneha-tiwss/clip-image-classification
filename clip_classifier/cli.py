"""Command-line interface for zero-shot CLIP image classification."""

import argparse
from collections.abc import Sequence

from clip_classifier.classifier import CLIPImageClassifier
from clip_classifier.classifier import DEFAULT_MODEL_NAME
from clip_classifier.text_utils import DEFAULT_PROMPT_TEMPLATE


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser without performing classification."""
    parser = argparse.ArgumentParser(
        description="Classify an image using OpenAI CLIP and candidate labels."
    )
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="One or more candidate labels",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="CLIP model name")
    parser.add_argument("--device", default="cpu", help="PyTorch device, such as cpu or cuda")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Text prompt template containing {label}",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    return build_parser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI using the existing classifier implementation."""
    args = parse_args(argv)

    try:
        classifier = CLIPImageClassifier(
            model_name=args.model,
            device=args.device,
        )
        ranked_results = classifier.classify_image(
            args.image,
            args.labels,
            args.prompt_template,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    for rank, result in enumerate(ranked_results, start=1):
        print(f"{rank}. {result['label']}: {result['score']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
