# Zero-Shot Image Classification with OpenAI CLIP

A professional, modular Python project for zero-shot image classification using OpenAI CLIP through Hugging Face Transformers. The project compares local images with natural-language candidate labels and returns ranked results without task-specific training or fine-tuning.

## Key features

- OpenAI CLIP model loading through Hugging Face `CLIPModel` and `AutoProcessor`
- Local image validation and loading with Pillow
- Configurable candidate-label prompt templates
- Image and text embedding generation
- L2-normalized cosine similarity scoring
- Softmax-normalized relative ranking across supplied candidate labels
- CPU execution by default and explicit optional CUDA support
- Reusable Python API, command-line interface, and sample script
- Custom exceptions for initialization, processing, similarity, and ranking failures
- Unit tests using mocked CLIP components, independent of model downloads

## Current implementation

The current project supports the complete local-image classification workflow:

1. Load and validate a local image.
2. Convert it to RGB.
3. Process the image and candidate labels with the Hugging Face processor.
4. Generate image and text embeddings with CLIP.
5. Normalize embeddings and calculate cosine similarities.
6. Apply softmax across the candidate labels.
7. Rank the labels in descending relative score order.

The implementation is modular. Embedding generation, cosine similarity, softmax ranking, CLI argument parsing, and the public classifier API are separate components.

## Technology stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- OpenAI CLIP checkpoint: `openai/clip-vit-base-patch32`
- Pillow
- pytest

## Prerequisites

- Python 3.10 or newer
- `pip`
- A local image file
- Internet access during the first real model download
- Optional: a CUDA-enabled PyTorch installation and compatible GPU for CUDA acceleration

## Installation and setup

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The dependencies are declared in `requirements.txt`:

```text
Torch>=2.1,<3
transformers>=4.41,<6
Pillow>=10,<13
pytest>=8,<10
```

## CLI usage

Run the root entry point with the required image and candidate labels:

```bash
python main.py --image path/to/image.jpg --labels cat dog bird
```

Both arguments are required:

- `--image`: path to one local image file
- `--labels`: one or more candidate labels separated by spaces

### Custom model

```bash
python main.py \
  --image path/to/image.jpg \
  --labels cat dog bird \
  --model openai/clip-vit-base-patch32
```

The default model is `openai/clip-vit-base-patch32`.

### Device selection

The default device is `cpu`:

```bash
python main.py --image path/to/image.jpg --labels cat dog --device cpu
```

To request CUDA explicitly:

```bash
python main.py --image path/to/image.jpg --labels cat dog --device cuda
```

A specific CUDA device may also be selected, for example:

```bash
python main.py --image path/to/image.jpg --labels cat dog --device cuda:1
```

CUDA is optional. The classifier raises a clear error if the requested CUDA device is unavailable.

### Custom prompt template

The default prompt template is:

```text
a photo of a {label}
```

Provide a custom template that contains the `{label}` placeholder:

```bash
python main.py \
  --image path/to/image.jpg \
  --labels cat dog bird \
  --prompt-template "A clear photo of {label}"
```

## Python API usage

Initialize the classifier and call `classify_image()`:

```python
from clip_classifier.classifier import CLIPImageClassifier

classifier = CLIPImageClassifier(
    model_name="openai/clip-vit-base-patch32",
    device="cpu",
)

ranked_results = classifier.classify_image(
    image_path="path/to/image.jpg",
    candidate_labels=["cat", "dog", "bird"],
)

for rank, result in enumerate(ranked_results, start=1):
    print(f"{rank}. {result['label']}: {result['score']:.4f}")
```

`classify_image()` returns a list of dictionaries in descending score order:

```python
[
    {"label": "dog", "score": 0.62},
    {"label": "cat", "score": 0.25},
    {"label": "bird", "score": 0.13},
]
```

A custom prompt template can be passed as the third argument:

```python
ranked_results = classifier.classify_image(
    "path/to/image.jpg",
    ["cat", "dog", "bird"],
    "A clear photo of {label}",
)
```

## Example script usage

The included example uses a replaceable placeholder image path and candidate labels:

```bash
python examples/sample_usage.py
```

Edit `PLACEHOLDER_IMAGE_PATH` in `examples/sample_usage.py` before running it with a real image:

```python
PLACEHOLDER_IMAGE_PATH = "path/to/your/local/image.jpg"
CANDIDATE_LABELS = ["cat", "dog", "bird"]
```

The example calls `CLIPImageClassifier.classify_image()` and prints the ranked labels and scores.

## Project structure

```text
clip-image-classification/
├── clip_classifier/
│   ├── __init__.py
│   ├── classifier.py
│   ├── cli.py
│   ├── image_utils.py
│   └── text_utils.py
├── examples/
│   ├── sample_usage.py
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   ├── test_cli.py
│   ├── test_image_utils.py
│   ├── test_main.py
│   ├── test_sample_usage.py
│   └── test_text_utils.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

- `classifier.py`: CLIP initialization, embeddings, cosine similarity, softmax ranking, and classification orchestration
- `image_utils.py`: image path validation, Pillow loading, RGB conversion, and image exceptions
- `text_utils.py`: candidate-label validation and prompt-template preparation
- `cli.py`: argument parsing and terminal output
- `main.py`: thin root entry point
- `examples/sample_usage.py`: public-API usage example
- `tests/`: unit tests using mocked CLIP components where appropriate

## Running tests

Run the complete test suite from the project root:

```bash
python -m pytest
```

Run a focused test module when needed:

```bash
python -m pytest tests/test_classifier.py
python -m pytest tests/test_cli.py
python -m pytest tests/test_sample_usage.py
```

Unit tests mock CLIP model and processor components, so they do not download real model weights.

## CPU and CUDA configuration

CPU execution is supported and is the default. It requires no CUDA installation and is suitable for development and testing.

CUDA execution is optional:

- Use `--device cuda` for the first available CUDA device.
- Use `--device cuda:1` for a specific CUDA index.
- A compatible CUDA-enabled PyTorch build and CUDA-capable GPU are required.
- A requested unavailable CUDA device produces a clear `CLIPInitializationError`.

## Cosine similarity and score interpretation

CLIP image and text embeddings are L2-normalized. Cosine similarity measures the direction alignment between a normalized image embedding and each normalized text embedding:

- Higher similarity means stronger alignment between the image and candidate prompt.
- Cosine similarity is not a probability.
- The `rank_labels()` step applies `torch.softmax()` across the supplied candidate labels.

The resulting softmax scores are **relative scores across only the candidate labels provided by the caller**. They are **not calibrated confidence probabilities**. Their values depend on the candidate set and prompt wording. A high softmax score is a ranking result, not a guarantee that the image belongs to that category.

## First-run model download and caching

The first real classifier initialization loads `openai/clip-vit-base-patch32` from Hugging Face. This requires network access and may take time. Subsequent runs can reuse the Hugging Face cache stored on the local machine.

Common troubleshooting:

- **Missing image:** verify that `--image` points to an existing local file.
- **Unsupported image:** use a valid, readable image format supported by Pillow.
- **CUDA unavailable:** use `--device cpu` or install a compatible CUDA-enabled PyTorch build.
- **Model download fails:** check network access, Hugging Face availability, proxy settings, and cache permissions.
- **Out-of-memory:** use CPU or a smaller model if available and avoid concurrent workloads.
- **Slow first run:** model weights are downloaded on the first real run and then reused from cache.
