"""Unit tests for candidate-label validation and CLIP prompt preparation."""

import pytest

from clip_classifier.text_utils import (
    DEFAULT_PROMPT_TEMPLATE,
    LabelValidationError,
    PromptTemplateError,
    prepare_text_prompts,
    validate_candidate_labels,
)


def test_validate_candidate_labels_returns_original_strings() -> None:
    labels = ["Cat", "red sports car", "  dog  "]

    result = validate_candidate_labels(labels)

    assert result == labels


def test_validate_candidate_labels_accepts_tuple() -> None:
    result = validate_candidate_labels(("cat", "dog"))

    assert result == ["cat", "dog"]


@pytest.mark.parametrize("labels", [[], (), set()])
def test_validate_candidate_labels_rejects_empty_collection(labels: object) -> None:
    with pytest.raises(LabelValidationError, match="must not be empty"):
        validate_candidate_labels(labels)


@pytest.mark.parametrize("labels", [None, "cat", b"cat", 42, {"cat": 1}])
def test_validate_candidate_labels_rejects_invalid_types(labels: object) -> None:
    with pytest.raises(LabelValidationError):
        validate_candidate_labels(labels)


@pytest.mark.parametrize("labels", [["cat", ""], ["   "]])
def test_validate_candidate_labels_rejects_empty_or_whitespace(labels: object) -> None:
    with pytest.raises(LabelValidationError, match="empty or whitespace-only"):
        validate_candidate_labels(labels)


def test_validate_candidate_labels_rejects_non_string_collection_item() -> None:
    with pytest.raises(LabelValidationError, match=r"index 1.*must be a string"):
        validate_candidate_labels(["cat", 3])


def test_prepare_text_prompts_uses_default_template() -> None:
    prompts = prepare_text_prompts(["cat", "red sports car"])

    assert prompts == [
        "a photo of a cat",
        "a photo of a red sports car",
    ]
    assert DEFAULT_PROMPT_TEMPLATE == "a photo of a {label}"


def test_prepare_text_prompts_uses_custom_template() -> None:
    prompts = prepare_text_prompts(
        ["cat", "dog"],
        prompt_template="A clear image showing: {label}",
    )

    assert prompts == [
        "A clear image showing: cat",
        "A clear image showing: dog",
    ]


def test_prepare_text_prompts_preserves_label_text() -> None:
    prompts = prepare_text_prompts(["  dog  "])

    assert prompts == ["a photo of a   dog  "]


@pytest.mark.parametrize("template", ["a photo", "image of a category"])
def test_prepare_text_prompts_rejects_missing_placeholder(template: str) -> None:
    with pytest.raises(PromptTemplateError, match=r"\{label\}"):
        prepare_text_prompts(["cat"], template)


def test_prepare_text_prompts_rejects_unsupported_placeholder() -> None:
    with pytest.raises(PromptTemplateError, match="only contain the \\{label\\}"):
        prepare_text_prompts(["cat"], "{label} photographed by {camera}")


@pytest.mark.parametrize("template", ["A photo of {label", "A photo of {label!}"])
def test_prepare_text_prompts_rejects_malformed_template(template: str) -> None:
    with pytest.raises(PromptTemplateError):
        prepare_text_prompts(["cat"], template)


@pytest.mark.parametrize("template", [None, 42, ["{label}"]])
def test_prepare_text_prompts_rejects_non_string_template(template: object) -> None:
    with pytest.raises(PromptTemplateError, match="must be a string"):
        prepare_text_prompts(["cat"], template)  # type: ignore[arg-type]


def test_prepare_text_prompts_validates_labels() -> None:
    with pytest.raises(LabelValidationError):
        prepare_text_prompts([])
