"""
Prediction / Inference module for Industrial Surface Defect Detection.

Provides a unified interface for single-image and batch inference using
either the fine-tuned or feature extraction MobileNetV2 models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf
from PIL import Image

import config
from src.preprocessing import load_and_preprocess_single_image, preprocess_pil_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cached model instance for low-latency inference in web apps
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def get_default_model_path() -> Path:
    """Returns the best available trained model path."""
    if config.FINETUNED_MODEL_PATH.exists():
        return config.FINETUNED_MODEL_PATH
    if config.FEATURE_EXTRACTION_MODEL_PATH.exists():
        return config.FEATURE_EXTRACTION_MODEL_PATH
    raise FileNotFoundError(
        "No trained model found! Please train a model first using 'src/train_feature_extraction.py' or 'run_pipeline.py'."
    )


def load_inference_model(model_path: Optional[Union[str, Path]] = None) -> tf.keras.Model:
    """Loads and caches the Keras model for inference."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    selected_path = Path(model_path) if model_path else get_default_model_path()

    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == selected_path:
        return _CACHED_MODEL

    if not selected_path.exists():
        raise FileNotFoundError(f"Specified model checkpoint does not exist at: {selected_path}")

    logger.info("Loading inference model from: %s", selected_path)
    model = tf.keras.models.load_model(selected_path)
    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = selected_path
    return model


def predict_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Dict:
    """
    Performs inference on a single image.

    Args:
        image_input: Path to image file, PIL Image, or numpy array.
        model_path: Optional path to .keras model file.
        threshold: Decision boundary threshold (default: 0.5).

    Returns:
        Dictionary containing prediction results, probabilities, and model metadata.
    """
    model = load_inference_model(model_path)
    actual_path = _CACHED_MODEL_PATH

    # Preprocess image into batch tensor and display array
    if isinstance(image_input, (str, Path)):
        batch_tensor, display_arr = load_and_preprocess_single_image(image_input)
    elif isinstance(image_input, Image.Image):
        batch_tensor, display_arr = preprocess_pil_image(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype == np.uint8 and image_input.max() > 1.0:
            pil_img = Image.fromarray(image_input)
            batch_tensor, display_arr = preprocess_pil_image(pil_img)
        else:
            raise ValueError("Direct numpy arrays must be uint8 in [0, 255] format.")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Model outputs sigmoid probability for class 1 (Defective)
    raw_pred = model.predict(batch_tensor, verbose=0)
    prob_defective = float(raw_pred[0][0])
    prob_normal = float(1.0 - prob_defective)

    is_defective = prob_defective >= threshold
    predicted_idx = 1 if is_defective else 0
    predicted_class = config.IDX_TO_CLASS[predicted_idx]
    display_label = config.DISPLAY_LABELS[predicted_idx]
    confidence = prob_defective if is_defective else prob_normal

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "display_label": display_label,
        "confidence_score": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "probabilities": {
            "normal": round(prob_normal, 4),
            "defective": round(prob_defective, 4),
        },
        "raw_output": round(prob_defective, 6),
        "threshold_used": threshold,
        "model_used": actual_path.name if actual_path else "unknown",
        "display_image": display_arr,
        "batch_tensor": batch_tensor,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = predict_image(sys.argv[1])
        print(f"Prediction: {res['display_label']} ({res['confidence_percentage']})")
    else:
        print("Usage: python -m src.predict <image_path>")
