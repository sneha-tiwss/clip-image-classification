"""CLIP model initialization for zero-shot image classification."""

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

from clip_classifier.image_utils import load_rgb_image
from clip_classifier.text_utils import (
    DEFAULT_PROMPT_TEMPLATE,
    LabelValidationError,
    prepare_text_prompts,
    validate_candidate_labels,
)


DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


class CLIPInitializationError(RuntimeError):
    """Raised when CLIP resources cannot be initialized."""


class CLIPProcessingError(RuntimeError):
    """Raised when an image or text input cannot be processed by CLIP."""


class CLIPSimilarityError(RuntimeError):
    """Raised when cosine similarity cannot be calculated from embeddings."""


class CLIPRankingError(RuntimeError):
    """Raised when similarity scores cannot be ranked against labels."""


class CLIPImageClassifier:
    """Provide Hugging Face CLIP model initialization and classification.

    Supports image and text embedding generation, cosine similarity calculation,
    softmax-relative label ranking, and single-image zero-shot classification.

    Args:
        model_name: Hugging Face model identifier or local model path.
        device: PyTorch device name. CPU is the default. CUDA may be requested
            explicitly as ``"cuda"`` or ``"cuda:<index>"``.

    Raises:
        CLIPInitializationError: If the requested device is unavailable or
            model and processor resources cannot be loaded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise CLIPInitializationError("Model name must be a non-empty string")

        self.model_name = model_name
        self.device = self._resolve_device("cpu" if device is None else device)

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
        except Exception as exc:
            raise CLIPInitializationError(
                f"Could not load CLIP model '{self.model_name}'"
            ) from exc

        try:
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise CLIPInitializationError(
                f"Could not initialize CLIP model on device '{self.device}'"
            ) from exc

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve and validate a CPU or CUDA device request."""
        if not isinstance(device, str) or not device.strip():
            raise CLIPInitializationError("Device must be a non-empty string")

        normalized = device.strip().lower()

        if normalized == "cpu":
            return torch.device("cpu")

        if normalized == "cuda" or normalized.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise CLIPInitializationError(
                    f"Requested CUDA device '{device}' is unavailable"
                )

            if normalized == "cuda":
                return torch.device("cuda")

            index_text = normalized.split(":", maxsplit=1)[1]
            if not index_text.isdigit():
                raise CLIPInitializationError(
                    f"Invalid CUDA device specification: '{device}'"
                )

            index = int(index_text)
            device_count = torch.cuda.device_count()
            if index >= device_count:
                raise CLIPInitializationError(
                    f"Requested CUDA device '{device}' is unavailable; "
                    f"available device count is {device_count}"
                )

            return torch.device("cuda", index)

        raise CLIPInitializationError(
            f"Unsupported device '{device}'. Use 'cpu', 'cuda', or 'cuda:<index>'"
        )

    def encode_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load an RGB image and return its CLIP image embedding.

        Args:
            image_path: Local path to an image file.

        Returns:
            A tensor containing the image embedding.

        Raises:
            CLIPProcessingError: If processor or model inference fails.
        """
        image = load_rgb_image(image_path)

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = self._move_inputs_to_device(inputs)

            with torch.inference_mode():
                output = self.model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
            return self._extract_feature_tensor(output)
        except Exception as exc:
            raise CLIPProcessingError(
                f"Could not encode image '{image_path}'"
            ) from exc

    def encode_text(
        self,
        labels: Sequence[str] | Collection[str],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> torch.Tensor:
        """Prepare text prompts and return their CLIP text embeddings.

        Args:
            labels: Non-empty collection of candidate label strings.
            prompt_template: Template containing the ``{label}`` placeholder.

        Returns:
            A tensor containing the text embeddings.

        Raises:
            LabelValidationError or PromptTemplateError: If text inputs or the
                prompt template are invalid.
            CLIPProcessingError: If processor or model inference fails.
        """
        prompts = prepare_text_prompts(labels, prompt_template)

        try:
            inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
            inputs = self._move_inputs_to_device(inputs)

            with torch.inference_mode():
                output = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask")
                    if hasattr(inputs, "get")
                    else None,
                )
            return self._extract_feature_tensor(output)
        except Exception as exc:
            raise CLIPProcessingError("Could not encode text prompts") from exc

    def _move_inputs_to_device(self, inputs: object) -> object:
        """Move processor outputs to the configured device."""
        try:
            return inputs.to(self.device)  # type: ignore[attr-defined]
        except AttributeError:
            if not isinstance(inputs, dict):
                raise CLIPProcessingError(
                    "Processor returned unsupported input type"
                )

            moved = {
                name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
            return moved

    def calculate_cosine_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize embedding tensors and calculate their cosine similarities.

        One-dimensional tensors are treated as a single embedding. Zero-vector
        rows are normalized safely to zeros, preventing NaN similarity values.

        Args:
            image_embeddings: Image embedding tensor with shape ``(D,)`` or
                ``(N, D)``.
            text_embeddings: Text embedding tensor with shape ``(D,)`` or
                ``(M, D)``.

        Returns:
            Similarity matrix with shape ``(N, M)``.

        Raises:
            CLIPSimilarityError: If inputs are invalid or embedding dimensions
                are incompatible.
        """
        if not isinstance(image_embeddings, torch.Tensor):
            raise CLIPSimilarityError("Image embeddings must be a PyTorch tensor")
        if not isinstance(text_embeddings, torch.Tensor):
            raise CLIPSimilarityError("Text embeddings must be a PyTorch tensor")
        if image_embeddings.is_complex() or text_embeddings.is_complex():
            raise CLIPSimilarityError("Embedding tensors must contain real numbers")
        if not image_embeddings.is_floating_point() or not text_embeddings.is_floating_point():
            raise CLIPSimilarityError("Embedding tensors must use a floating-point dtype")

        if image_embeddings.ndim not in (1, 2):
            raise CLIPSimilarityError("Image embeddings must be one- or two-dimensional")
        if text_embeddings.ndim not in (1, 2):
            raise CLIPSimilarityError("Text embeddings must be one- or two-dimensional")

        image = image_embeddings.unsqueeze(0) if image_embeddings.ndim == 1 else image_embeddings
        text = text_embeddings.unsqueeze(0) if text_embeddings.ndim == 1 else text_embeddings

        if image.shape[1] == 0 or text.shape[1] == 0:
            raise CLIPSimilarityError("Embedding dimension must be greater than zero")
        if image.shape[1] != text.shape[1]:
            raise CLIPSimilarityError(
                f"Embedding dimensions do not match: image={image.shape[1]}, "
                f"text={text.shape[1]}"
            )
        if image.shape[0] == 0 or text.shape[0] == 0:
            raise CLIPSimilarityError("Embedding tensors must contain at least one row")

        try:
            text = text.to(device=image.device, dtype=image.dtype)
            normalized_image = F.normalize(image, p=2, dim=1, eps=1e-12)
            normalized_text = F.normalize(text, p=2, dim=1, eps=1e-12)
            return normalized_image @ normalized_text.transpose(0, 1)
        except Exception as exc:
            raise CLIPSimilarityError("Could not calculate cosine similarity") from exc

    def rank_labels(
        self,
        similarity_scores: torch.Tensor,
        labels: object,
    ) -> list[dict[str, object]]:
        """Apply softmax and rank candidate labels by descending score.

        The returned scores are softmax-normalized relative values over the
        supplied candidates, not calibrated confidence probabilities.

        Args:
            similarity_scores: One-dimensional scores, or a two-dimensional
                tensor containing exactly one image row.
            labels: Non-empty collection of candidate label strings.

        Returns:
            A list of dictionaries containing ``label`` and ``score`` entries,
            ordered from highest to lowest softmax score.

        Raises:
            CLIPRankingError: If tensor, label, dimension, or count validation
                fails.
        """
        if not isinstance(similarity_scores, torch.Tensor):
            raise CLIPRankingError("Similarity scores must be a PyTorch tensor")
        if not similarity_scores.is_floating_point():
            raise CLIPRankingError("Similarity scores must use a floating-point dtype")
        if similarity_scores.is_complex():
            raise CLIPRankingError("Similarity scores must contain real numbers")

        if similarity_scores.ndim == 1:
            scores = similarity_scores
        elif similarity_scores.ndim == 2 and similarity_scores.shape[0] == 1:
            scores = similarity_scores[0]
        else:
            raise CLIPRankingError(
                "Similarity scores must be one-dimensional or have shape (1, N)"
            )

        try:
            validated_labels = validate_candidate_labels(labels)
        except LabelValidationError as exc:
            raise CLIPRankingError(str(exc)) from exc

        if scores.numel() != len(validated_labels):
            raise CLIPRankingError(
                f"Similarity score count ({scores.numel()}) does not match "
                f"candidate label count ({len(validated_labels)})"
            )

        if not torch.isfinite(scores).all():
            raise CLIPRankingError("Similarity scores must contain finite values")

        try:
            probabilities = torch.softmax(scores.detach().cpu(), dim=0)
            ranked_indices = sorted(
                range(len(validated_labels)),
                key=lambda index: (-float(probabilities[index].item()), index),
            )
            return [
                {
                    "label": validated_labels[index],
                    "score": float(probabilities[index].item()),
                }
                for index in ranked_indices
            ]
        except Exception as exc:
            raise CLIPRankingError("Could not calculate ranked label scores") from exc

    def classify_image(
        self,
        image_path: Union[str, Path],
        candidate_labels: Sequence[str] | Collection[str],
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> list[dict[str, object]]:
        """Classify one image against candidate labels.

        Args:
            image_path: Local path to the image.
            candidate_labels: Non-empty collection of candidate label strings.
            prompt_template: Template containing the ``{label}`` placeholder.

        Returns:
            Ranked dictionaries containing ``label`` and ``score`` entries.
        """
        image_embeddings = self.encode_image(image_path)
        text_embeddings = self.encode_text(candidate_labels, prompt_template)
        similarity_scores = self.calculate_cosine_similarity(
            image_embeddings,
            text_embeddings,
        )
        return self.rank_labels(similarity_scores, candidate_labels)

    def _extract_feature_tensor(self, output: object) -> torch.Tensor:
        """Extract a tensor from direct or container-style feature output."""
        if isinstance(output, torch.Tensor):
            return output

        pooler_output = getattr(output, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output

        if isinstance(output, dict):
            pooler_output = output.get("pooler_output")
            if isinstance(pooler_output, torch.Tensor):
                return pooler_output

        raise CLIPProcessingError("Unsupported model output format")
