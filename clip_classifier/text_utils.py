"""Utilities for validating candidate labels and preparing CLIP text prompts."""

from collections.abc import Collection, Mapping, Sequence
from string import Formatter


DEFAULT_PROMPT_TEMPLATE = "a photo of a {label}"


class LabelValidationError(ValueError):
    """Raised when candidate labels are missing or invalid."""


class PromptTemplateError(ValueError):
    """Raised when a prompt template is invalid."""


def validate_candidate_labels(labels: object) -> list[str]:
    """Validate candidate labels and return them without modifying their text.

    Args:
        labels: A non-empty collection of strings.

    Returns:
        A list containing the original label strings in collection order.

    Raises:
        LabelValidationError: If labels is empty, is not a collection of
            strings, or contains an empty or whitespace-only string.
    """
    if isinstance(labels, (str, bytes, Mapping)) or not isinstance(labels, Collection):
        raise LabelValidationError(
            "Candidate labels must be a non-empty collection of strings"
        )

    if len(labels) == 0:
        raise LabelValidationError("Candidate labels must not be empty")

    validated: list[str] = []
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise LabelValidationError(
                f"Candidate label at index {index} must be a string"
            )
        if not label.strip():
            raise LabelValidationError(
                f"Candidate label at index {index} must not be empty or whitespace-only"
            )
        validated.append(label)

    return validated


def prepare_text_prompts(
    labels: Sequence[str] | Collection[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> list[str]:
    """Prepare CLIP text prompts while preserving each original label.

    Args:
        labels: A validated or validation-ready collection of candidate labels.
        prompt_template: Template containing exactly the ``{label}`` placeholder.

    Returns:
        Text prompts in the same order as the candidate labels.

    Raises:
        LabelValidationError: If candidate labels are invalid.
        PromptTemplateError: If prompt_template is not a string or does not
            contain the ``{label}`` placeholder.
    """
    if not isinstance(prompt_template, str):
        raise PromptTemplateError("Prompt template must be a string")
    if "{label}" not in prompt_template:
        raise PromptTemplateError("Prompt template must contain the {label} placeholder")

    try:
        parsed_fields = [
            field_name
            for _, field_name, _, _ in Formatter().parse(prompt_template)
            if field_name is not None
        ]
    except ValueError as exc:
        raise PromptTemplateError("Prompt template is malformed") from exc

    if any(field_name != "label" for field_name in parsed_fields):
        raise PromptTemplateError(
            "Prompt template may only contain the {label} placeholder"
        )

    validated_labels = validate_candidate_labels(labels)

    try:
        return [prompt_template.format(label=label) for label in validated_labels]
    except (KeyError, ValueError, IndexError) as exc:
        raise PromptTemplateError("Prompt template could not be formatted") from exc
