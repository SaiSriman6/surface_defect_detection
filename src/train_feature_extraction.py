"""
Feature Extraction Training Module.

Uses MobileNetV2 pretrained on ImageNet as a frozen feature extractor.
Adds a binary classification head with GlobalAveragePooling2D and Dropout.
Trains the head, evaluates validation progress, generates loss/accuracy curves,
and saves the checkpoint to `models/feature_extraction.keras`.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from src.dataset import prepare_splits
from src.preprocessing import create_split_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_feature_extraction_model(
    input_shape: Tuple[int, int, int] = config.IMAGE_SHAPE,
    dropout_rate: float = config.DROPOUT_RATE,
) -> tf.keras.Model:
    """
    Constructs the feature extraction model.
    Backbone: MobileNetV2 with ImageNet weights, frozen (trainable=False).
    Head: GlobalAveragePooling2D -> Dropout -> Dense(1, sigmoid).
    """
    # Load pretrained backbone without top classification layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    # training=False ensures BatchNormalization layers operate in inference mode
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="binary_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Feature_Extractor")
    return model


def plot_and_save_training_curves(history_dict: Dict, output_path: Path, title_prefix: str = "Feature Extraction"):
    """Plots and saves Loss and Accuracy training curves."""
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history_dict["loss"], "b-o", label="Training Loss")
    if "val_loss" in history_dict:
        ax1.plot(epochs, history_dict["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title(f"{title_prefix}: Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Crossentropy Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history_dict["accuracy"], "b-o", label="Training Accuracy")
    if "val_accuracy" in history_dict:
        ax2.plot(epochs, history_dict["val_accuracy"], "r--s", label="Validation Accuracy")
    ax2.set_title(f"{title_prefix}: Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Training curves saved to: %s", output_path)


def compute_balanced_class_weights(counts: Dict) -> Dict[int, float]:
    """Computes class weights to counterbalance normal/defect class imbalance."""
    total = counts["train"]["total"]
    normal_cnt = max(1, counts["train"]["normal"])
    defective_cnt = max(1, counts["train"]["defective"])

    weight_0 = total / (2.0 * normal_cnt)
    weight_1 = total / (2.0 * defective_cnt)
    return {0: float(weight_0), 1: float(weight_1)}


def train_feature_extraction(
    category: str = config.DEFAULT_CATEGORY,
    epochs: int = config.FE_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.FE_LEARNING_RATE,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Executes Feature Extraction training workflow.
    """
    logger.info("--- Starting Stage: Feature Extraction Training (MobileNetV2 Frozen) ---")

    # Ensure splits are prepared
    prepare_splits(category=category)
    train_ds, val_ds, test_ds, counts = create_split_datasets(category=category, batch_size=batch_size)

    class_weights = compute_balanced_class_weights(counts)
    logger.info("Calculated class weights: %s", class_weights)

    model = build_feature_extraction_model(dropout_rate=config.DROPOUT_RATE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.FEATURE_EXTRACTION_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model explicitly if checkpoint didn't save or to guarantee path
    config.FEATURE_EXTRACTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.FEATURE_EXTRACTION_MODEL_PATH)

    # Convert history values to standard floats for JSON serialization
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(config.FE_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved to: %s", config.FE_HISTORY_PATH)

    # Plot curves
    plot_and_save_training_curves(history_dict, config.FE_CURVES_PATH, title_prefix="Feature Extraction")

    # Verification: Reload and run test inference
    logger.info("Verifying saved feature extraction model...")
    reloaded_model = tf.keras.models.load_model(config.FEATURE_EXTRACTION_MODEL_PATH)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    test_pred = reloaded_model.predict(dummy_input, verbose=0)
    assert test_pred.shape == (1, 1), f"Unexpected prediction shape: {test_pred.shape}"
    logger.info("Feature extraction model verified successfully. Test prediction shape: %s", test_pred.shape)

    return model, history_dict


if __name__ == "__main__":
    train_feature_extraction()
