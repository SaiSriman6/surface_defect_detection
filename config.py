"""
Configuration settings for the Industrial Surface Defect Detection project.
Centralizes all file paths, dataset splits, model hyperparameters, and class mappings.
"""

import os
import tempfile
from pathlib import Path

# Centralized Environment Variables
BASE_DIR = Path(__file__).resolve().parent

# Fix Matplotlib cache path to prevent Windows permission warnings
_mpl_dir = BASE_DIR / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

# Fix Keras cache path to use project-local directory with cached weights
_keras_dir = BASE_DIR / ".keras"
_keras_dir.mkdir(parents=True, exist_ok=True)
os.environ["KERAS_HOME"] = str(_keras_dir)

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"
TESTS_DIR = BASE_DIR / "tests"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DEFAULT_CATEGORY = "bottle"
SUPPORTED_CATEGORIES = ["bottle", "metal_nut"]

# Image Parameters
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
CHANNELS = 3

# Class Mappings (Binary Classification)
CLASS_NAMES = ["normal", "defective"]
CLASS_TO_IDX = {"normal": 0, "defective": 1}
IDX_TO_CLASS = {0: "normal", 1: "defective"}
DISPLAY_LABELS = {
    0: "Normal (Non-Defective)",
    1: "Defective (Anomalous)"
}
CLASSIFICATION_THRESHOLD = 0.5

# Dataset Split Methodology (Reproducible & Stratified)
# MVTec AD is educational adapted for supervised binary classification
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 16
FE_EPOCHS = 10
FT_EPOCHS = 10
FE_LEARNING_RATE = 1e-3
FT_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 4
UNFREEZE_FROM_LAYER = 100  # Layer index in MobileNetV2 from which layers are made trainable during fine-tuning

# Artifact File Paths
FEATURE_EXTRACTION_MODEL_PATH = MODELS_DIR / "feature_extraction.keras"
FINETUNED_MODEL_PATH = MODELS_DIR / "finetuned.keras"

FE_HISTORY_PATH = RESULTS_DIR / "feature_extraction_history.json"
FT_HISTORY_PATH = RESULTS_DIR / "finetuning_history.json"
FE_CURVES_PATH = RESULTS_DIR / "feature_extraction_curves.png"
FT_CURVES_PATH = RESULTS_DIR / "finetuning_curves.png"

FE_METRICS_PATH = RESULTS_DIR / "feature_extraction_metrics.json"
FT_METRICS_PATH = RESULTS_DIR / "finetuning_metrics.json"
FE_CONF_MATRIX_PATH = RESULTS_DIR / "confusion_matrix_feature_extraction.png"
FT_CONF_MATRIX_PATH = RESULTS_DIR / "confusion_matrix_finetuning.png"

MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
FINAL_VERIFICATION_REPORT_PATH = RESULTS_DIR / "FINAL_VERIFICATION_REPORT.md"
SPLITS_METADATA_PATH = DATA_DIR / "splits_metadata.json"

# Grad-CAM Configuration
GRADCAM_TARGET_LAYER = "out_relu"  # The final ReLU layer of MobileNetV2 feature backbone
