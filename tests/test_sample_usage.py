"""Focused tests for the sample usage example."""

from unittest.mock import MagicMock, patch

from examples.sample_usage import CANDIDATE_LABELS, run_example


@patch("examples.sample_usage.CLIPImageClassifier")
def test_run_example_uses_public_classifier_api(mock_classifier: MagicMock) -> None:
    ranked_results = [
        {"label": "dog", "score": 0.6},
        {"label": "cat", "score": 0.3},
        {"label": "bird", "score": 0.1},
    ]
    mock_classifier.return_value.classify_image.return_value = ranked_results

    result = run_example("photo.jpg")

    mock_classifier.assert_called_once_with()
    mock_classifier.return_value.classify_image.assert_called_once_with(
        "photo.jpg",
        CANDIDATE_LABELS,
    )
    assert result is ranked_results


@patch("examples.sample_usage.CLIPImageClassifier")
def test_run_example_uses_placeholder_path_by_default(mock_classifier: MagicMock) -> None:
    mock_classifier.return_value.classify_image.return_value = []

    run_example()

    mock_classifier.return_value.classify_image.assert_called_once_with(
        "path/to/your/local/image.jpg",
        CANDIDATE_LABELS,
    )
