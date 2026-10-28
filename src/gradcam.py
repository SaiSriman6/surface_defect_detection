"""
Explainable AI (XAI) module using Gradient-weighted Class Activation Mapping (Grad-CAM).

Visualizes feature attribution and model sensitivity for MobileNetV2 defect predictions.
Identifies which visual regions of the product the convolutional layers relied upon
to distinguish between normal and defective states.

Educational Note:
Grad-CAM visualizes the spatial gradients and feature activations of the network.
It reveals model sensitivity and focus regions, but is an explanatory tool rather
than a geometric defect boundary proof.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import config
from src.predict import load_inference_model, predict_image
from src.preprocessing import load_and_preprocess_single_image, preprocess_pil_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_target_conv_layer(model: tf.keras.Model, layer_name: Optional[str] = None) -> Tuple[tf.keras.Model, str]:
    """
    Locates the target convolutional layer and backbone model.
    Defaults to config.GRADCAM_TARGET_LAYER ('out_relu') of MobileNetV2.
    """
    backbone = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    search_container = backbone if backbone is not None else model
    chosen_name = layer_name or config.GRADCAM_TARGET_LAYER

    # Validate target layer exists
    available_layers = [l.name for l in search_container.layers]
    if chosen_name in available_layers:
        return search_container, chosen_name

    # Fallback to last 4D activation layer
    for layer in reversed(search_container.layers):
        if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
            logger.info("Defaulting Grad-CAM target layer to: %s", layer.name)
            return search_container, layer.name

    raise ValueError(f"No suitable 4D convolutional layer found in model for Grad-CAM. Available: {available_layers[-10:]}")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    target_class_idx: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Computes 2D normalized Grad-CAM heatmap for a single image tensor.

    Args:
        model: Trained MobileNetV2 Keras model.
        img_tensor: Preprocessed image tensor of shape (1, 224, 224, 3).
        target_class_idx: Class index to explain (0: Normal, 1: Defective). If None, uses argmax prediction.
        layer_name: Specific conv layer name (default: 'out_relu').

    Returns:
        2D numpy array of shape (H, W) with values normalized in [0.0, 1.0].
    """
    backbone = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    target_container, target_layer_name = find_target_conv_layer(model, layer_name)
    target_layer = target_container.get_layer(target_layer_name)

    # Feature extractor for target layer output
    conv_model = tf.keras.Model(inputs=target_container.input, outputs=target_layer.output)

    # Classification head layers after the backbone
    head_layers = [
        layer for layer in model.layers
        if "mobilenetv2" not in layer.name.lower() and not isinstance(layer, tf.keras.layers.InputLayer)
    ]

    tf_img = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # 1. Forward pass through convolutional feature extractor
        conv_outputs = conv_model(tf_img, training=False)
        tape.watch(conv_outputs)

        # 2. Forward pass through remaining classification head
        x = conv_outputs
        for head_layer in head_layers:
            x = head_layer(x, training=False)
        preds = x

        # If target class not specified, choose class based on predicted probability
        prob_defective = preds[0][0]
        if target_class_idx is None:
            target_class_idx = 1 if prob_defective >= config.CLASSIFICATION_THRESHOLD else 0

        # Class score to maximize in backprop
        if target_class_idx == 1:
            score = prob_defective
        else:
            score = 1.0 - prob_defective

    # Compute gradients of score with respect to feature maps
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients could not be computed for Grad-CAM. Check model layer connectivity.")

    # Global Average Pooling of gradients across spatial dimensions (height and width)
    # Shapes: conv_outputs: (1, 7, 7, 1280), grads: (1, 7, 7, 1280)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by gradient importance
    cam = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)

    # Apply ReLU to retain only positive evidence for the class
    cam = tf.maximum(cam, 0.0)

    # Normalize heatmap to [0, 1]
    max_val = tf.math.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val

    heatmap = cam.numpy()
    return heatmap


def overlay_gradcam_heatmap(
    heatmap: np.ndarray,
    original_rgb_image: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes heatmap to original image dimensions and creates color overlay.

    Args:
        heatmap: 2D numpy array [0, 1].
        original_rgb_image: uint8 numpy array (H, W, 3).
        alpha: Heatmap transparency factor (0.0 to 1.0).
        colormap: Matplotlib colormap name ('jet', 'viridis', 'magma').

    Returns:
        (colored_heatmap_uint8, overlay_image_uint8)
    """
    h, w = original_rgb_image.shape[:2]

    # Resize heatmap to match image dimensions
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    # Apply color map
    try:
        color_mapper = matplotlib.colormaps[colormap]
    except Exception:
        color_mapper = cm.get_cmap(colormap)
    colored_rgba = color_mapper(resized_heatmap)  # (H, W, 4) in [0, 1]
    colored_rgb = (colored_rgba[:, :, :3] * 255).astype(np.uint8)

    # Overlay: blend colored heatmap with original image
    img_float = original_rgb_image.astype(np.float32)
    heat_float = colored_rgb.astype(np.float32)
    overlay = cv2.addWeighted(img_float, 1.0 - alpha, heat_float, alpha, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return colored_rgb, overlay


def explain_prediction(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.45,
) -> Dict:
    """
    End-to-end Explainable AI pipeline:
    Runs inference, computes Grad-CAM, creates overlays, and writes side-by-side visualization.
    """
    # 1. Run inference
    pred_info = predict_image(image_input, model_path=model_path)
    model = load_inference_model(model_path)

    batch_tensor = pred_info["batch_tensor"]
    display_image = pred_info["display_image"]

    chosen_class = target_class_idx if target_class_idx is not None else pred_info["predicted_index"]

    # 2. Compute Grad-CAM
    heatmap = compute_gradcam_heatmap(
        model=model,
        img_tensor=batch_tensor,
        target_class_idx=chosen_class,
        layer_name=config.GRADCAM_TARGET_LAYER,
    )

    colored_heatmap, overlay_image = overlay_gradcam_heatmap(
        heatmap=heatmap,
        original_rgb_image=display_image,
        alpha=alpha,
    )

    # 3. Optional: save visualization plot
    if output_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(display_image)
        axes[0].set_title("Input Image (RGB)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title(f"Grad-CAM Heatmap ({config.IDX_TO_CLASS[chosen_class].capitalize()})", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay_image)
        title_text = f"Overlay ({pred_info['display_label']}: {pred_info['confidence_percentage']})"
        axes[2].set_title(title_text, fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grad-CAM visualization saved to: %s", output_path)

    interpretation = (
        f"The model predicted '{pred_info['display_label']}' with {pred_info['confidence_percentage']} confidence. "
        f"Highlighted warmer regions (red/yellow) in the Grad-CAM overlay designate high activation in layer "
        f"'{config.GRADCAM_TARGET_LAYER}', showing the visual cues most influential in determining the classification. "
        "Grad-CAM visualizes model sensitivity and should be reviewed as an explanatory guide in industrial inspection."
    )

    return {
        "prediction": pred_info,
        "heatmap": heatmap,
        "colored_heatmap": colored_heatmap,
        "overlay_image": overlay_image,
        "display_image": display_image,
        "target_class": config.IDX_TO_CLASS[chosen_class],
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = explain_prediction(sys.argv[1], output_path=config.RESULTS_DIR / "sample_gradcam.png")
        print("Grad-CAM generated successfully. Saved to results/sample_gradcam.png")
    else:
        print("Usage: python -m src.gradcam <image_path>")
