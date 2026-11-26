"""Demonstrate CLIP classification across the bundled sample images."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clip_classifier.classifier import CLIPImageClassifier


SAMPLE_IMAGE_NAMES = ("cat.jpg", "dog.jpg", "car.jpg", "bird.jpg")
CANDIDATE_LABELS = ["cat", "dog", "car", "bird"]


def display_results(
    image_name: str,
    ranked_results: list[dict[str, object]],
) -> None:
    """Print one image's top prediction and complete candidate ranking."""
    top_result = ranked_results[0]
    top_label = str(top_result["label"])
    top_score = float(top_result["score"])

    print(f"\n{image_name}")
    print(f"Top prediction: {top_label} ({top_score:.4f})")
    print("Ranked candidates:")
    for rank, result in enumerate(ranked_results, start=1):
        print(f"  {rank}. {result['label']}: {float(result['score']):.4f}")


def run_demo() -> int:
    """Run CPU classification for every bundled sample image."""
    print("CLIP Sample Image Classification")
    print("================================")
    print("Device: cpu")
    print(f"Candidate labels: {', '.join(CANDIDATE_LABELS)}")

    try:
        classifier = CLIPImageClassifier(device="cpu")
    except Exception as exc:
        print(f"Failed to initialize CLIP: {type(exc).__name__}: {exc}")
        return 1

    sample_directory = PROJECT_ROOT / "examples" / "sample_images"
    processed_count = 0

    for image_name in SAMPLE_IMAGE_NAMES:
        image_path = sample_directory / image_name
        if not image_path.is_file():
            print(f"\n{image_name}: Error: image file not found: {image_path}")
            continue

        try:
            ranked_results = classifier.classify_image(
                image_path,
                CANDIDATE_LABELS,
            )
        except Exception as exc:
            print(
                f"\n{image_name}: Error: {type(exc).__name__}: {exc}"
            )
            continue

        display_results(image_name, ranked_results)
        processed_count += 1

    print(f"\nProcessed {processed_count}/{len(SAMPLE_IMAGE_NAMES)} images.")
    return 0 if processed_count == len(SAMPLE_IMAGE_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(run_demo())
