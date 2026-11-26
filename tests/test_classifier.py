"""Unit tests for CLIP model initialization."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from clip_classifier.classifier import (
    DEFAULT_MODEL_NAME,
    CLIPImageClassifier,
    CLIPInitializationError,
    CLIPProcessingError,
    CLIPRankingError,
    CLIPSimilarityError,
)


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_default_model_selection(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    mock_processor.from_pretrained.assert_called_once_with(DEFAULT_MODEL_NAME)
    mock_clip_model.from_pretrained.assert_called_once_with(DEFAULT_MODEL_NAME)
    assert classifier.model_name == DEFAULT_MODEL_NAME


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_custom_model_selection(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    custom_name = "example/custom-clip-model"

    classifier = CLIPImageClassifier(model_name=custom_name)

    mock_processor.from_pretrained.assert_called_once_with(custom_name)
    mock_clip_model.from_pretrained.assert_called_once_with(custom_name)
    assert classifier.model_name == custom_name


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_model_is_set_to_evaluation_mode(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    mock_model = mock_clip_model.from_pretrained.return_value

    CLIPImageClassifier()

    mock_model.eval.assert_called_once_with()


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_default_device_is_cpu(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    mock_model = mock_clip_model.from_pretrained.return_value

    classifier = CLIPImageClassifier()

    assert classifier.device == torch.device("cpu")
    mock_model.to.assert_called_once_with(torch.device("cpu"))


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
@patch("clip_classifier.classifier.torch.cuda.device_count", return_value=2)
@patch("clip_classifier.classifier.torch.cuda.is_available", return_value=True)
def test_cuda_device_selection(
    mock_cuda_available: MagicMock,
    mock_device_count: MagicMock,
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    mock_model = mock_clip_model.from_pretrained.return_value

    classifier = CLIPImageClassifier(device="cuda:1")

    assert classifier.device == torch.device("cuda", 1)
    mock_model.to.assert_called_once_with(torch.device("cuda", 1))


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
@patch("clip_classifier.classifier.torch.cuda.is_available", return_value=False)
def test_unavailable_cuda_raises_clear_error(
    mock_cuda_available: MagicMock,
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    with pytest.raises(CLIPInitializationError, match=r"CUDA.*unavailable"):
        CLIPImageClassifier(device="cuda")

    mock_processor.from_pretrained.assert_not_called()
    mock_clip_model.from_pretrained.assert_not_called()


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
@patch("clip_classifier.classifier.torch.cuda.device_count", return_value=1)
@patch("clip_classifier.classifier.torch.cuda.is_available", return_value=True)
def test_out_of_range_cuda_index_raises_clear_error(
    mock_cuda_available: MagicMock,
    mock_device_count: MagicMock,
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    with pytest.raises(CLIPInitializationError, match=r"CUDA device.*unavailable"):
        CLIPImageClassifier(device="cuda:3")


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_invalid_model_name_is_rejected(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    with pytest.raises(CLIPInitializationError, match="Model name"):
        CLIPImageClassifier(model_name="   ")


@patch("clip_classifier.classifier.load_rgb_image")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_image_generates_embedding(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_load_rgb_image: MagicMock,
) -> None:
    mock_image = mock_load_rgb_image.return_value
    processor_instance = mock_processor.from_pretrained.return_value
    processor_instance.return_value = {
        "pixel_values": torch.ones(1, 3, 224, 224)
    }
    model_instance = mock_clip_model.from_pretrained.return_value
    model_instance.get_image_features.return_value = torch.ones(1, 512)

    classifier = CLIPImageClassifier()
    with patch("clip_classifier.classifier.torch.inference_mode") as mock_inference_mode:
        result = classifier.encode_image("image.jpg")

    mock_inference_mode.assert_called_once_with()
    mock_inference_mode.return_value.__enter__.assert_called_once_with()
    mock_load_rgb_image.assert_called_once_with("image.jpg")
    processor_instance.assert_called_once_with(images=mock_image, return_tensors="pt")
    model_instance.get_image_features.assert_called_once()
    assert result is model_instance.get_image_features.return_value


@patch("clip_classifier.classifier.prepare_text_prompts")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_text_generates_embedding(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_prepare_text_prompts: MagicMock,
) -> None:
    mock_prepare_text_prompts.return_value = ["a photo of a cat"]
    processor_instance = mock_processor.from_pretrained.return_value
    processor_instance.return_value = {
        "input_ids": torch.ones(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    model_instance = mock_clip_model.from_pretrained.return_value
    model_instance.get_text_features.return_value = torch.ones(1, 512)

    classifier = CLIPImageClassifier()
    with patch("clip_classifier.classifier.torch.inference_mode") as mock_inference_mode:
        result = classifier.encode_text(["cat"])

    mock_inference_mode.assert_called_once_with()
    mock_inference_mode.return_value.__enter__.assert_called_once_with()
    mock_prepare_text_prompts.assert_called_once()
    processor_instance.assert_called_once_with(
        text=["a photo of a cat"], padding=True, return_tensors="pt"
    )
    model_instance.get_text_features.assert_called_once()
    assert result is model_instance.get_text_features.return_value


@patch("clip_classifier.classifier.prepare_text_prompts", return_value=["a photo of a cat"])
@patch("clip_classifier.classifier.load_rgb_image")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_processor_inputs_are_moved_to_configured_device(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_load_rgb_image: MagicMock,
    mock_prepare_text_prompts: MagicMock,
) -> None:
    pixel_values = MagicMock()
    input_ids = MagicMock()
    attention_mask = MagicMock()
    processor_instance = mock_processor.from_pretrained.return_value

    processor_instance.side_effect = [
        {"pixel_values": pixel_values},
        {"input_ids": input_ids, "attention_mask": attention_mask},
    ]
    model_instance = mock_clip_model.from_pretrained.return_value
    model_instance.get_image_features.return_value = torch.ones(1, 512)
    model_instance.get_text_features.return_value = torch.ones(1, 512)

    classifier = CLIPImageClassifier()
    classifier.encode_image("image.jpg")
    classifier.encode_text(["cat"])

    pixel_values.to.assert_called_once_with(classifier.device)
    input_ids.to.assert_called_once_with(classifier.device)
    attention_mask.to.assert_called_once_with(classifier.device)


@patch("clip_classifier.classifier.load_rgb_image", side_effect=RuntimeError("bad image"))
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_image_propagates_image_loading_failure(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_load_rgb_image: MagicMock,
) -> None:
    with pytest.raises(RuntimeError, match="bad image"):
        CLIPImageClassifier().encode_image("bad.jpg")


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_text_rejects_invalid_labels(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(ValueError):
        classifier.encode_text([])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_text_processor_failure_raises_processing_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    mock_processor.from_pretrained.return_value.side_effect = RuntimeError("processor failed")

    with pytest.raises(CLIPProcessingError, match="Could not encode text prompts"):
        classifier.encode_text(["cat"])


@patch("clip_classifier.classifier.load_rgb_image")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_image_processor_failure_raises_processing_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_load_rgb_image: MagicMock,
) -> None:
    processor_instance = mock_processor.from_pretrained.return_value
    processor_instance.side_effect = RuntimeError("processor failed")

    with pytest.raises(CLIPProcessingError, match="Could not encode image"):
        CLIPImageClassifier().encode_image("image.jpg")


@patch("clip_classifier.classifier.load_rgb_image")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_image_model_failure_raises_processing_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_load_rgb_image: MagicMock,
) -> None:
    processor_instance = mock_processor.from_pretrained.return_value
    processor_instance.return_value = {
        "pixel_values": torch.ones(1, 3, 224, 224)
    }
    model_instance = mock_clip_model.from_pretrained.return_value
    model_instance.get_image_features.side_effect = RuntimeError("model failed")

    with pytest.raises(CLIPProcessingError, match="Could not encode image"):
        CLIPImageClassifier().encode_image("image.jpg")


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_calculate_cosine_similarity_values(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    image_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    similarities = classifier.calculate_cosine_similarity(
        image_embeddings,
        text_embeddings,
    )

    expected = torch.tensor([
        [1.0, 0.0, 2 ** -0.5],
        [0.0, 1.0, 2 ** -0.5],
    ])
    assert similarities.shape == (2, 3)
    assert torch.allclose(similarities, expected, atol=1e-6)


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_calculate_cosine_similarity_supports_one_dimensional_embeddings(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    similarities = classifier.calculate_cosine_similarity(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
    )

    assert similarities.shape == (1, 1)
    assert torch.equal(similarities, torch.tensor([[1.0]]))


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
@pytest.mark.parametrize("image_embeddings,text_embeddings", [
    (None, torch.ones(1, 3)),
    (torch.ones(1, 3), None),
])
def test_calculate_cosine_similarity_rejects_non_tensor_inputs(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    image_embeddings: object,
    text_embeddings: object,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(CLIPSimilarityError, match="tensor"):
        classifier.calculate_cosine_similarity(  # type: ignore[arg-type]
            image_embeddings,
            text_embeddings,
        )


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_calculate_cosine_similarity_rejects_incompatible_dimensions(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(CLIPSimilarityError, match="dimensions do not match"):
        classifier.calculate_cosine_similarity(
            torch.ones(2, 3),
            torch.ones(4, 5),
        )


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_calculate_cosine_similarity_handles_zero_norm_embeddings(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    similarities = classifier.calculate_cosine_similarity(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )

    assert similarities.shape == (1, 1)
    assert torch.equal(similarities, torch.tensor([[0.0]]))
    assert torch.all(torch.isfinite(similarities))


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_calculate_cosine_similarity_rejects_integer_embeddings(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(CLIPSimilarityError, match="floating-point"):
        classifier.calculate_cosine_similarity(
            torch.ones(1, 3, dtype=torch.int64),
            torch.ones(1, 3, dtype=torch.int64),
        )


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_returns_correct_softmax_scores(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    ranked = classifier.rank_labels(
        torch.tensor([1.0, 2.0, 3.0]),
        ["cat", "dog", "bird"],
    )

    expected = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
    by_label = {item["label"]: item["score"] for item in ranked}
    assert torch.allclose(
        torch.tensor([by_label[label] for label in ["cat", "dog", "bird"]]),
        expected,
    )


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_sorts_in_descending_order(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    ranked = classifier.rank_labels(
        torch.tensor([0.1, 0.8, 0.3]),
        ["cat", "dog", "bird"],
    )

    assert [item["label"] for item in ranked] == ["dog", "bird", "cat"]


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_matches_each_label_to_its_score(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    ranked = classifier.rank_labels(
        torch.tensor([[2.0, 1.0, 0.0]]),
        ["cat", "dog", "bird"],
    )

    assert len(ranked) == 3
    assert {item["label"] for item in ranked} == {"cat", "dog", "bird"}
    assert ranked[0]["label"] == "cat"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[1]["score"] > ranked[2]["score"]


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
@pytest.mark.parametrize("invalid_scores", [None, [1.0, 2.0], torch.tensor([1, 2])])
def test_rank_labels_rejects_invalid_tensors(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    invalid_scores: object,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(CLIPRankingError):
        classifier.rank_labels(invalid_scores, ["cat", "dog"])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_rejects_empty_labels(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(CLIPRankingError):
        classifier.rank_labels(torch.tensor([1.0]), [])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_rejects_mismatched_score_and_label_counts(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    with pytest.raises(CLIPRankingError, match="does not match"):
        classifier.rank_labels(torch.tensor([1.0, 2.0]), ["cat"])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_preserves_order_for_equal_scores(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    ranked = classifier.rank_labels(
        torch.tensor([1.0, 1.0, 1.0]),
        ["cat", "dog", "bird"],
    )

    assert [item["label"] for item in ranked] == ["cat", "dog", "bird"]
    scores = [item["score"] for item in ranked]
    assert all(score == pytest.approx(1.0 / 3.0) for score in scores)


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_rank_labels_softmax_scores_sum_to_one(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()

    ranked = classifier.rank_labels(
        torch.tensor([0.2, 1.4, -0.3, 0.8]),
        ["cat", "dog", "bird", "fish"],
    )

    total_score = sum(item["score"] for item in ranked)
    assert total_score == pytest.approx(1.0)


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_classify_image_orchestrates_existing_methods(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    image_embeddings = torch.ones(1, 3)
    text_embeddings = torch.ones(3, 3)
    similarity_scores = torch.tensor([[0.1, 0.8, 0.3]])
    ranked_results = [
        {"label": "dog", "score": 0.6},
        {"label": "cat", "score": 0.3},
        {"label": "bird", "score": 0.1},
    ]

    classifier.encode_image = MagicMock(return_value=image_embeddings)
    classifier.encode_text = MagicMock(return_value=text_embeddings)
    classifier.calculate_cosine_similarity = MagicMock(return_value=similarity_scores)
    classifier.rank_labels = MagicMock(return_value=ranked_results)

    result = classifier.classify_image(
        "image.jpg",
        ["cat", "dog", "bird"],
        "A photo of {label}",
    )

    classifier.encode_image.assert_called_once_with("image.jpg")
    classifier.encode_text.assert_called_once_with(
        ["cat", "dog", "bird"],
        "A photo of {label}",
    )
    classifier.calculate_cosine_similarity.assert_called_once_with(
        image_embeddings,
        text_embeddings,
    )
    classifier.rank_labels.assert_called_once_with(
        similarity_scores,
        ["cat", "dog", "bird"],
    )
    assert result is ranked_results


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_classify_image_propagates_encoding_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    classifier.encode_image = MagicMock(
        side_effect=CLIPProcessingError("Could not encode image")
    )

    with pytest.raises(CLIPProcessingError, match="Could not encode image"):
        classifier.classify_image("bad.jpg", ["cat", "dog"])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_classify_image_propagates_similarity_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    classifier.encode_image = MagicMock(return_value=torch.ones(1, 3))
    classifier.encode_text = MagicMock(return_value=torch.ones(2, 3))
    classifier.calculate_cosine_similarity = MagicMock(
        side_effect=CLIPSimilarityError("Invalid embeddings")
    )

    with pytest.raises(CLIPSimilarityError, match="Invalid embeddings"):
        classifier.classify_image("image.jpg", ["cat", "dog"])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_classify_image_propagates_ranking_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    classifier.encode_image = MagicMock(return_value=torch.ones(1, 3))
    classifier.encode_text = MagicMock(return_value=torch.ones(2, 3))
    classifier.calculate_cosine_similarity = MagicMock(
        return_value=torch.tensor([[0.5, 0.5]])
    )
    classifier.rank_labels = MagicMock(
        side_effect=CLIPRankingError("Could not rank labels")
    )

    with pytest.raises(CLIPRankingError, match="Could not rank labels"):
        classifier.classify_image("image.jpg", ["cat", "dog"])


@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_classify_image_propagates_text_encoding_error(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
) -> None:
    classifier = CLIPImageClassifier()
    classifier.encode_image = MagicMock(return_value=torch.ones(1, 3))
    classifier.encode_text = MagicMock(
        side_effect=CLIPProcessingError("Could not encode text prompts")
    )

    with pytest.raises(CLIPProcessingError, match="Could not encode text prompts"):
        classifier.classify_image("image.jpg", ["cat", "dog"])


@patch("clip_classifier.classifier.load_rgb_image")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_image_extracts_pooler_output_tensor(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_load_rgb_image: MagicMock,
) -> None:
    processor_instance = mock_processor.from_pretrained.return_value
    processor_instance.return_value = {
        "pixel_values": torch.ones(1, 3, 224, 224)
    }
    expected = torch.ones(1, 512)
    model_instance = mock_clip_model.from_pretrained.return_value
    model_instance.get_image_features.return_value = SimpleNamespace(
        pooler_output=expected
    )

    result = CLIPImageClassifier().encode_image("image.jpg")

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 512)
    assert result is expected


@patch("clip_classifier.classifier.prepare_text_prompts")
@patch("clip_classifier.classifier.CLIPModel")
@patch("clip_classifier.classifier.AutoProcessor")
def test_encode_text_extracts_pooler_output_tensor(
    mock_processor: MagicMock,
    mock_clip_model: MagicMock,
    mock_prepare_text_prompts: MagicMock,
) -> None:
    mock_prepare_text_prompts.return_value = ["a photo of a cat", "a photo of a dog"]
    processor_instance = mock_processor.from_pretrained.return_value
    processor_instance.return_value = {
        "input_ids": torch.ones(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    expected = torch.ones(2, 512)
    model_instance = mock_clip_model.from_pretrained.return_value
    model_instance.get_text_features.return_value = SimpleNamespace(
        pooler_output=expected
    )

    result = CLIPImageClassifier().encode_text(["cat", "dog"])

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 512)
    assert result is expected
