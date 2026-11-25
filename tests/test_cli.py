"""Unit tests for the CLIP command-line argument interface."""

from unittest.mock import MagicMock, patch

from clip_classifier.cli import build_parser, main, parse_args
from clip_classifier.classifier import DEFAULT_MODEL_NAME
from clip_classifier.text_utils import DEFAULT_PROMPT_TEMPLATE


def test_parser_requires_image_and_labels() -> None:
    parser = build_parser()

    arguments = parser.parse_args(["--image", "photo.jpg", "--labels", "cat", "dog"])

    assert arguments.image == "photo.jpg"
    assert arguments.labels == ["cat", "dog"]


def test_parser_defaults() -> None:
    arguments = parse_args(["--image", "photo.jpg", "--labels", "cat", "dog"])

    assert arguments.model == DEFAULT_MODEL_NAME
    assert arguments.device == "cpu"
    assert arguments.prompt_template == DEFAULT_PROMPT_TEMPLATE


def test_parser_accepts_custom_arguments() -> None:
    arguments = parse_args([
        "--image",
        "photo.jpg",
        "--labels",
        "cat",
        "dog",
        "--model",
        "example/custom-model",
        "--device",
        "cuda:1",
        "--prompt-template",
        "A clear photo of {label}",
    ])

    assert arguments.image == "photo.jpg"
    assert arguments.labels == ["cat", "dog"]
    assert arguments.model == "example/custom-model"
    assert arguments.device == "cuda:1"
    assert arguments.prompt_template == "A clear photo of {label}"


@patch("clip_classifier.cli.CLIPImageClassifier")
def test_main_uses_existing_classifier(mock_classifier: MagicMock) -> None:
    mock_instance = mock_classifier.return_value
    mock_instance.classify_image.return_value = [
        {"label": "dog", "score": 0.7},
        {"label": "cat", "score": 0.3},
    ]

    exit_code = main([
        "--image",
        "photo.jpg",
        "--labels",
        "cat",
        "dog",
    ])

    assert exit_code == 0
    mock_classifier.assert_called_once_with(
        model_name=DEFAULT_MODEL_NAME,
        device="cpu",
    )
    mock_instance.classify_image.assert_called_once_with(
        "photo.jpg",
        ["cat", "dog"],
        DEFAULT_PROMPT_TEMPLATE,
    )


@patch("clip_classifier.cli.CLIPImageClassifier")
def test_main_returns_error_when_classification_fails(
    mock_classifier: MagicMock,
) -> None:
    mock_classifier.return_value.classify_image.side_effect = RuntimeError("failed")

    exit_code = main([
        "--image",
        "photo.jpg",
        "--labels",
        "cat",
    ])

    assert exit_code == 1
