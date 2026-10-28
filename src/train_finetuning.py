"""
Fine-Tuning Training Module.

Loads the trained feature extraction model, unfreezes top convolutional layers
of MobileNetV2 (from layer index 100 onwards while preserving BatchNormalization stability),
recompiles with a low learning rate (1e-5), trains, records history,
plots comparison curves, and saves the best checkpoint to `models/finetuned.keras`.
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
from src.train_feature_extraction import (
    compute_balanced_class_weights,
    plot_and_save_training_curves,
    train_feature_extraction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_model_for_finetuning(
    base_model_path: Path = config.FEATURE_EXTRACTION_MODEL_PATH,
    unfreeze_from: int = config.UNFREEZE_FROM_LAYER,
) -> tf.keras.Model:
    """
    Loads feature extraction model and selectively unfreezes top layers of the backbone.
    Preserves BatchNormalization layers in frozen mode to maintain learned batch statistics.
    """
    if not base_model_path.exists():
        logger.warning("Feature extraction model not found at '%s'. Training it first...", base_model_path)
        train_feature_extraction()

    model = tf.keras.models.load_model(base_model_path)

    # Locate the MobileNetV2 backbone layer
    backbone = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not locate MobileNetV2 backbone within loaded model layers.")

    # Unfreeze the backbone container
    backbone.trainable = True

    total_backbone_layers = len(backbone.layers)
    logger.info("MobileNetV2 has %d total layers. Unfreezing layers from index %d onwards.", total_backbone_layers, unfreeze_from)

    trainable_count = 0
    frozen_count = 0

    for i, layer in enumerate(backbone.layers):
        if i < unfreeze_from:
            layer.trainable = False
            frozen_count += 1
        else:
            # Gold standard: keep BatchNormalization frozen during fine-tuning
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_count += 1
            else:
                layer.trainable = True
                trainable_count += 1

    logger.info(
        "Fine-tuning layer configuration: %d trainable layers, %d frozen layers in backbone.",
        trainable_count,
        frozen_count,
    )
    return model


def train_finetuning(
    category: str = config.DEFAULT_CATEGORY,
    epochs: int = config.FT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.FT_LEARNING_RATE,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Executes Fine-Tuning training workflow.
    """
    logger.info("--- Starting Stage: Fine-Tuning Training (Top Backbone Layers Unfrozen) ---")

    prepare_splits(category=category)
    train_ds, val_ds, test_ds, counts = create_split_datasets(category=category, batch_size=batch_size)
    class_weights = compute_balanced_class_weights(counts)

    model = prepare_model_for_finetuning(
        base_model_path=config.FEATURE_EXTRACTION_MODEL_PATH,
        unfreeze_from=config.UNFREEZE_FROM_LAYER,
    )

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

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.FINETUNED_MODEL_PATH),
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

    config.FINETUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.FINETUNED_MODEL_PATH)

    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(config.FT_HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Fine-tuning history saved to: %s", config.FT_HISTORY_PATH)

    plot_and_save_training_curves(history_dict, config.FT_CURVES_PATH, title_prefix="Fine-Tuning")

    logger.info("Verifying saved fine-tuned model...")
    reloaded_model = tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    test_pred = reloaded_model.predict(dummy_input, verbose=0)
    assert test_pred.shape == (1, 1), f"Unexpected prediction shape: {test_pred.shape}"
    logger.info("Fine-tuned model verified successfully. Test prediction shape: %s", test_pred.shape)

    return model, history_dict


if __name__ == "__main__":
    train_finetuning()
