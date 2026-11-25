"""Utilities for validating, opening, and converting local images."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

from PIL import Image, UnidentifiedImageError


PathLike = Union[str, Path]


class ImageValidationError(ValueError):
    """Raised when an image path is missing or does not point to a file."""


class UnsupportedImageFormatError(ValueError):
    """Raised when Pillow cannot identify or decode an image format."""


class ImageLoadError(RuntimeError):
    """Raised when a valid image file cannot be loaded successfully."""


def validate_image_path(path: PathLike) -> Path:
    """Return a normalized path after verifying that it exists and is a file.

    Args:
        path: Local filesystem path supplied by the caller.

    Raises:
        ImageValidationError: If the path is missing or is not a regular file.
    """
    image_path = Path(path).expanduser()

    if not image_path.exists():
        raise ImageValidationError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise ImageValidationError(f"Image path is not a file: {image_path}")

    return image_path


@contextmanager
def open_image(path: PathLike) -> Iterator[Image.Image]:
    """Open an image safely and close its Pillow resource after use.

    The context manager verifies the path, opens and decodes the file, and
    closes the underlying image even when decoding or caller code raises.

    Args:
        path: Local filesystem path to a supported image.

    Yields:
        An open Pillow image that may be used within the context.

    Raises:
        ImageValidationError: If the path is missing or is not a file.
        UnsupportedImageFormatError: If Pillow cannot identify the image format.
        ImageLoadError: If the image cannot be opened or decoded.
    """
    image_path = validate_image_path(path)

    try:
        image = Image.open(image_path)
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormatError(
            f"Unsupported or invalid image format: {image_path}"
        ) from exc
    except OSError as exc:
        raise ImageLoadError(f"Could not open image: {image_path}") from exc

    with image:
        try:
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise UnsupportedImageFormatError(
                f"Unsupported or invalid image format: {image_path}"
            ) from exc

        yield image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Return an independent copy of an image converted to RGB.

    The returned copy can be used after the source image context exits.

    Args:
        image: Pillow image to convert.

    Returns:
        A new Pillow image in RGB mode.

    Raises:
        ImageLoadError: If Pillow cannot complete the conversion.
    """
    try:
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ImageLoadError("Could not convert image to RGB") from exc


def load_rgb_image(path: PathLike) -> Image.Image:
    """Load a local image and return an independent RGB copy.

    The opened source resource is closed before this function returns.

    Args:
        path: Local filesystem path to an image.

    Returns:
        A new Pillow image in RGB mode.
    """
    with open_image(path) as image:
        return convert_to_rgb(image)
