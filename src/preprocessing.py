"""
Image preprocessing and data pipeline module for Industrial Surface Defect Detection.

Features:
- Image loading, format conversion (RGB), resizing to 224x224.
- Official MobileNetV2 normalization: maps pixel values [0, 255] to [-1, 1].
- Data augmentation applied strictly to training sets (preserving defect integrity).
- Deterministic preprocessing for validation, test, and inference.
- Efficient tf.data.Dataset pipelines with batching, caching, and prefetching.
- Standalone inference image preparation avoiding double preprocessing.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_data_augmentation() -> tf.keras.Sequential:
    """
    Constructs data augmentation pipeline for training.
    Uses conservative transformations (horizontal flips, slight rotations, slight zoom)
    to prevent invalidating subtle surface defect geometries.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),  # ~15 degrees max
            tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05)),
        ],
        name="data_augmentation",
    )


def load_and_preprocess_single_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image from path, converts to RGB, resizes, and prepares:
    1. Preprocessed array ready for MobileNetV2 input shape (1, H, W, 3) with range [-1, 1]
    2. Original display array (uint8) of shape (H, W, 3) in range [0, 255]

    Guarantees no double-preprocessing.
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    try:
        with Image.open(path_obj) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
            display_arr = np.array(resized_img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Corrupt or invalid image file at '{image_path}': {e}") from e

    # Apply MobileNetV2 normalization: (x / 127.5) - 1.0 -> [-1.0, 1.0]
    float_arr = display_arr.astype(np.float32)
    preprocessed_arr = tf.keras.applications.mobilenet_v2.preprocess_input(float_arr)
    batch_tensor = np.expand_dims(preprocessed_arr, axis=0)

    return batch_tensor, display_arr


def preprocess_pil_image(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses an in-memory PIL Image (e.g. uploaded via Streamlit).
    Returns (batch_tensor_normalized, display_arr_uint8).
    """
    rgb_img = pil_image.convert("RGB")
    resized_img = rgb_img.resize(target_size, Image.Resampling.BILINEAR)
    display_arr = np.array(resized_img, dtype=np.uint8)

    float_arr = display_arr.astype(np.float32)
    preprocessed_arr = tf.keras.applications.mobilenet_v2.preprocess_input(float_arr)
    batch_tensor = np.expand_dims(preprocessed_arr, axis=0)

    return batch_tensor, display_arr


def parse_image_file(filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow mapping function to read, decode, and resize image."""
    img_bytes = tf.io.read_file(filename)
    img = tf.io.decode_image(img_bytes, channels=config.CHANNELS, expand_animations=False)
    img = tf.image.resize(img, config.IMAGE_SIZE, method="bilinear")
    img.set_shape([config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.CHANNELS])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    label = tf.cast(label, tf.float32)
    return img, label


def create_dataset_from_paths(
    file_paths: list,
    labels: list,
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = False,
    shuffle_buffer: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Creates an optimized tf.data.Dataset pipeline.
    Applies data augmentation dynamically if is_training=True.
    """
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        buf = shuffle_buffer or max(len(file_paths), 100)
        dataset = dataset.shuffle(buffer_size=buf, seed=config.RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(parse_image_file, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        augmentation = get_data_augmentation()
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_split_datasets(
    category: str = config.DEFAULT_CATEGORY,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Loads file lists from the processed directory and builds train, val, and test datasets.
    """
    processed_dir = config.PROCESSED_DATA_DIR / category
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found at '{processed_dir}'. "
            "Run 'src/dataset.py' to generate splits first."
        )

    split_data = {}
    for split in ["train", "val", "test"]:
        split_paths = []
        split_labels = []
        for class_name in config.CLASS_NAMES:
            class_folder = processed_dir / split / class_name
            label_val = config.CLASS_TO_IDX[class_name]
            for p in class_folder.glob("*.png"):
                split_paths.append(str(p.resolve()))
                split_labels.append(label_val)
        split_data[split] = (split_paths, split_labels)

    train_paths, train_labels = split_data["train"]
    val_paths, val_labels = split_data["val"]
    test_paths, test_labels = split_data["test"]

    if not train_paths or not val_paths or not test_paths:
        raise ValueError("One or more split datasets are empty. Verify dataset preparation.")

    train_ds = create_dataset_from_paths(train_paths, train_labels, batch_size=batch_size, is_training=True)
    val_ds = create_dataset_from_paths(val_paths, val_labels, batch_size=batch_size, is_training=False)
    test_ds = create_dataset_from_paths(test_paths, test_labels, batch_size=batch_size, is_training=False)

    counts = {
        "train": {"total": len(train_paths), "normal": train_labels.count(0), "defective": train_labels.count(1)},
        "val": {"total": len(val_paths), "normal": val_labels.count(0), "defective": val_labels.count(1)},
        "test": {"total": len(test_paths), "normal": test_labels.count(0), "defective": test_labels.count(1)},
    }

    return train_ds, val_ds, test_ds, counts
