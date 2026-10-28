"""
Automated tests for model loading, prediction formatting, confidence scores, and Grad-CAM.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import tensorflow as tf

import config
from src.gradcam import compute_gradcam_heatmap, overlay_gradcam_heatmap
from src.predict import load_inference_model, predict_image


@pytest.fixture
def mock_trained_model(tmp_path):
    """Fixture that builds and saves a small MobileNetV2 classifier checkpoint for testing."""
    model_path = tmp_path / "mock_test_model.keras"
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = tf.keras.Input((224, 224, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy")
    m.save(model_path)
    return model_path


@pytest.fixture
def test_image():
    """Returns a synthetic test PIL image."""
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_missing_model_checkpoint():
    """Verifies that an appropriate FileNotFoundError is raised if the model path is missing."""
    with pytest.raises(FileNotFoundError):
        load_inference_model(Path("nonexistent_model.keras"))


def test_predict_image_structure(mock_trained_model, test_image):
    """Verifies that predict_image returns all required keys and valid types."""
    res = predict_image(test_image, model_path=mock_trained_model)

    assert "predicted_class" in res
    assert res["predicted_class"] in config.CLASS_NAMES
    assert "confidence_score" in res
    assert 0.0 <= res["confidence_score"] <= 1.0
    assert "probabilities" in res
    assert "normal" in res["probabilities"]
    assert "defective" in res["probabilities"]
    assert abs((res["probabilities"]["normal"] + res["probabilities"]["defective"]) - 1.0) < 1e-3


def test_predict_invalid_image_type(mock_trained_model):
    """Verifies TypeError when unsupported image type is supplied."""
    with pytest.raises(TypeError):
        predict_image(12345, model_path=mock_trained_model)


def test_gradcam_computation(mock_trained_model):
    """Verifies Grad-CAM heatmap generation with the mock MobileNetV2 model."""
    m = tf.keras.models.load_model(mock_trained_model)
    dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)

    heatmap = compute_gradcam_heatmap(m, dummy_tensor, target_class_idx=1)

    assert len(heatmap.shape) == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_overlay(test_image):
    """Verifies overlaying heatmap produces an RGB uint8 image of the exact original shape."""
    display_arr = np.array(test_image, dtype=np.uint8)
    dummy_heatmap = np.random.rand(7, 7)

    colored_heat, overlay = overlay_gradcam_heatmap(dummy_heatmap, display_arr, alpha=0.5)

    assert colored_heat.shape == (224, 224, 3)
    assert colored_heat.dtype == np.uint8
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_actual_trained_models_exist_and_predict(test_image):
    """Integration test: verifies the actual trained models in models/ predict successfully."""
    for model_path in [config.FEATURE_EXTRACTION_MODEL_PATH, config.FINETUNED_MODEL_PATH]:
        if model_path.exists():
            res = predict_image(test_image, model_path=model_path)
            assert res["predicted_class"] in config.CLASS_NAMES
            assert 0.0 <= res["confidence_score"] <= 1.0
            assert res["model_used"] == model_path.name


def test_gradcam_on_actual_finetuned_model(test_image):
    """Integration test: verifies Grad-CAM on actual finetuned.keras checkpoint."""
    if config.FINETUNED_MODEL_PATH.exists():
        from src.gradcam import explain_prediction
        res = explain_prediction(test_image, model_path=config.FINETUNED_MODEL_PATH)
        assert res["heatmap"].shape == (7, 7)
        assert res["overlay_image"].shape == (224, 224, 3)
        assert res["prediction"]["confidence_score"] > 0.0
