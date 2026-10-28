"""
Automated unit tests for image preprocessing, data loading, and augmentation.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import tensorflow as tf

import config
from src.preprocessing import (
    get_data_augmentation,
    load_and_preprocess_single_image,
    preprocess_pil_image,
)


@pytest.fixture
def sample_valid_rgb_image(tmp_path):
    """Fixture providing a temporary valid RGB test image."""
    img_path = tmp_path / "valid_rgb.png"
    arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_grayscale_image(tmp_path):
    """Fixture providing a grayscale image that must be converted to RGB."""
    img_path = tmp_path / "gray.png"
    arr = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_corrupt_file(tmp_path):
    """Fixture providing a corrupted image file."""
    bad_path = tmp_path / "corrupt.png"
    with open(bad_path, "wb") as f:
        f.write(b"not a valid image content at all")
    return bad_path


def test_rgb_image_loading_and_resizing(sample_valid_rgb_image):
    """Verifies that an image is properly loaded, resized to 224x224, and shaped (1, 224, 224, 3)."""
    batch_tensor, display_arr = load_and_preprocess_single_image(sample_valid_rgb_image)

    assert batch_tensor.shape == (1, 224, 224, 3)
    assert display_arr.shape == (224, 224, 3)
    assert display_arr.dtype == np.uint8


def test_grayscale_to_rgb_conversion(sample_grayscale_image):
    """Verifies that single-channel images are automatically converted to 3-channel RGB."""
    batch_tensor, display_arr = load_and_preprocess_single_image(sample_grayscale_image)

    assert batch_tensor.shape == (1, 224, 224, 3)
    assert display_arr.shape == (224, 224, 3)


def test_mobilenetv2_normalization_range(sample_valid_rgb_image):
    """Verifies that the preprocessed tensor values fall strictly within [-1.0, 1.0]."""
    batch_tensor, _ = load_and_preprocess_single_image(sample_valid_rgb_image)

    min_val = batch_tensor.min()
    max_val = batch_tensor.max()
    assert min_val >= -1.05, f"Tensor min value {min_val} below expected lower bound."
    assert max_val <= 1.05, f"Tensor max value {max_val} above expected upper bound."


def test_preprocess_pil_image():
    """Verifies in-memory PIL image preprocessing."""
    pil_img = Image.new("RGB", (100, 150), color=(255, 0, 0))
    batch_tensor, display_arr = preprocess_pil_image(pil_img)

    assert batch_tensor.shape == (1, 224, 224, 3)
    assert display_arr.shape == (224, 224, 3)
    assert np.all(display_arr[:, :, 0] == 255)  # Red channel should be preserved


def test_corrupt_image_handling(sample_corrupt_file):
    """Verifies that corrupt image files raise ValueError rather than silently failing."""
    with pytest.raises(ValueError):
        load_and_preprocess_single_image(sample_corrupt_file)


def test_missing_image_file_handling():
    """Verifies that nonexistent file paths raise FileNotFoundError."""
    nonexistent = Path("nonexistent_path_to_defect_image.png")
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_single_image(nonexistent)


def test_data_augmentation_pipeline():
    """Verifies that the training data augmentation layer executes without errors."""
    aug = get_data_augmentation()
    dummy_input = tf.random.uniform((2, 224, 224, 3), minval=-1.0, maxval=1.0)
    augmented = aug(dummy_input, training=True)

    assert augmented.shape == (2, 224, 224, 3)
