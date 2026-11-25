"""Example library usage for CLIP zero-shot image classification."""

from clip_classifier.classifier import CLIPImageClassifier


PLACEHOLDER_IMAGE_PATH = "path/to/your/local/image.jpg"
CANDIDATE_LABELS = ["cat", "dog", "bird"]


def run_example(
    image_path: str = PLACEHOLDER_IMAGE_PATH,
) -> list[dict[str, object]]:
    """Initialize CLIP and classify one local image against example labels."""
    classifier = CLIPImageClassifier()
    ranked_results = classifier.classify_image(image_path, CANDIDATE_LABELS)

    print("Ranked classifications:")
    for rank, result in enumerate(ranked_results, start=1):
        print(f"{rank}. {result['label']}: {result['score']:.4f}")

    return ranked_results


if __name__ == "__main__":
    run_example()
