"""Unit tests for image utility helpers."""

from pathlib import Path

import pytest
from PIL import Image

from clip_classifier.image_utils import (
    ImageLoadError,
    ImageValidationError,
    UnsupportedImageFormatError,
    convert_to_rgb,
    load_rgb_image,
    open_image,
    validate_image_path,
)


def test_validate_image_path_accepts_existing_file(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), color="red").save(image_path)

    validated = validate_image_path(image_path)

    assert validated == image_path


def test_validate_image_path_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(ImageValidationError, match="does not exist"):
        validate_image_path(missing_path)


def test_validate_image_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ImageValidationError, match="not a file"):
        validate_image_path(tmp_path)


def test_load_rgb_image_returns_rgb_from_valid_image(tmp_path: Path) -> None:
    image_path = tmp_path / "palette.png"
    palette_image = Image.new("P", (8, 8), color=1)
    palette_image.putpalette([255, 0, 0, 0, 255, 0])
    palette_image.save(image_path)

    loaded = load_rgb_image(image_path)

    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"
    assert loaded.size == (8, 8)
    loaded.close()


def test_convert_to_rgb_returns_rgb_copy(tmp_path: Path) -> None:
    source = Image.new("L", (10, 6), color=128)
    converted = convert_to_rgb(source)

    assert converted.mode == "RGB"
    assert converted.size == (10, 6)
    converted.close()


def test_open_image_closes_resource_after_context(tmp_path: Path) -> None:
    image_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color="blue").save(image_path)

    with open_image(image_path) as image:
        assert image.mode == "RGB"

    assert getattr(image, "fp", None) is None


def test_invalid_image_file_raises_unsupported_format(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"this is not an image")

    with pytest.raises(UnsupportedImageFormatError, match="Unsupported or invalid"):
        load_rgb_image(invalid_path)


def test_truncated_image_raises_loading_error(tmp_path: Path) -> None:
    image_path = tmp_path / "truncated.png"
    Image.new("RGB", (20, 20), color="green").save(image_path)

    data = image_path.read_bytes()
    image_path.write_bytes(data[:20])

    with pytest.raises((UnsupportedImageFormatError, ImageLoadError)):
        load_rgb_image(image_path)


def test_load_rgb_image_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(ImageValidationError):
        load_rgb_image(missing_path)
