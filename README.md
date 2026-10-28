# Industrial Surface Defect Detection Using Transfer Learning and Explainable AI

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest Tests](https://img.shields.io/badge/Pytest-22%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning system for industrial automated visual quality inspection. The project distinguishes between **Normal (Non-Defective)** and **Defective (Anomalous)** components using **MobileNetV2** transfer learning and explains predictions using **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Architecture & Transfer Learning Strategy](#key-architecture--transfer-learning-strategy)
3. [Repository Structure](#repository-structure)
4. [Dataset & Educational Adaptation](#dataset--educational-adaptation)
5. [Installation & Setup](#installation--setup)
6. [Pipeline Execution](#pipeline-execution)
7. [Experimental Results & Benchmarks](#experimental-results--benchmarks)
8. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
9. [Streamlit Interactive Web Application](#streamlit-interactive-web-application)
10. [Automated Testing](#automated-testing)
11. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

Manual visual inspection on factory assembly lines is slow, subjective, and prone to inspector fatigue. This project delivers an automated, high-throughput deep learning solution:
- **Binary Classification:** Classifies industrial surface images into Normal vs. Defective.
- **Backbone Network:** MobileNetV2 pretrained on ImageNet (inverted residual bottlenecks and depthwise separable convolutions).
- **Unit 5 Concepts Demonstrated:**
  - *Transfer Learning Types:* Inductive Transfer Learning (ImageNet $\rightarrow$ Industrial defect domain).
  - *Methodology Comparison:* Frozen Feature Extraction vs. Selective Fine-Tuning.
  - *Overcoming Challenges:* Preventing catastrophic forgetting and Batch Normalization drift.
- **Explainability:** Grad-CAM heatmaps highlight exact defect regions on component surfaces.
- **Interactive UI:** A full-featured Streamlit dashboard for real-time inference and metric analysis.

---

## 2. Key Architecture & Transfer Learning Strategy

### MobileNetV2 Backbone
- **Input Dimensions:** $224 \times 224 \times 3$, normalized to $[-1, 1]$.
- **Parameters:** $\sim 2.26\text{M}$ base parameters (ideal for edge computing and industrial embedded PCs).
- **Classification Head:**
  - `GlobalAveragePooling2D()`: Collapses $7 \times 7 \times 1280$ feature maps to a 1280-D vector.
  - `Dropout(0.3)`: Regularization against overfitting on small industrial sample sets.
  - `Dense(1, activation='sigmoid')`: Predicts defect probability $P(\text{Defective} \mid X)$.

### Two-Stage Transfer Learning Methodology
1. **Experiment 1 — Feature Extraction (Frozen Backbone):**
   - The entire MobileNetV2 backbone is frozen (`base_model.trainable = False`).
   - Only the 1,281 parameters in the classification head are trained.
   - Learning rate: $\eta = 10^{-3}$ with Adam optimizer.
2. **Experiment 2 — Fine-Tuning (Upper Layers Unfrozen):**
   - Top convolutional layers from index 100 onwards (36 layers) are unfrozen.
   - Batch Normalization layers remain frozen to prevent destroying pretrained statistical moments.
   - Lower 118 layers remain frozen to preserve foundational edge/texture filters.
   - Learning rate: $\eta = 10^{-5}$ (reduced by $100\times$) to prevent destructive parameter updates.

---

## 3. Repository Structure

```text
DL_PROJECT/
│
├── app.py                      # Streamlit interactive multi-page web application
├── config.py                   # Centralized configuration, hyperparameters & paths
├── run_pipeline.py             # End-to-end automated orchestration pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
├── .gitignore                  # Git hygiene rules
│
├── data/
│   ├── README.md               # Dataset documentation
│   ├── raw/                    # Raw MVTec AD category files
│   ├── processed/              # Stratified train, val, test split images
│   └── splits_metadata.json    # Exact split records and leakage verification
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── dataset.py              # Download, extraction, stratification & leakage verification
│   ├── preprocessing.py        # Loading, resizing, RGB conversion & tf.data pipeline
│   ├── train_feature_extraction.py # Frozen backbone training module
│   ├── train_finetuning.py     # Unfrozen top layers training module
│   ├── evaluate.py             # Held-out test set evaluation & metric calculation
│   ├── predict.py              # Inference API with confidence scores
│   └── gradcam.py              # Grad-CAM heatmap generator and overlay renderer
│
├── models/
│   ├── feature_extraction.keras # Trained feature extraction checkpoint
│   └── finetuned.keras          # Trained fine-tuned checkpoint
│
├── results/
│   ├── feature_extraction_metrics.json
│   ├── finetuning_metrics.json
│   ├── feature_extraction_curves.png
│   ├── finetuning_curves.png
│   ├── confusion_matrix_feature_extraction.png
│   ├── confusion_matrix_finetuning.png
│   ├── model_comparison.csv
│   ├── sample_gradcam_verification.png
│   └── FINAL_VERIFICATION_REPORT.md
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests for loading, resizing & augmentation
│   ├── test_prediction.py      # Unit & integration tests for inference & Grad-CAM
│   ├── test_evaluation.py      # Unit & integration tests for metric calculations
│   └── test_app.py             # Structural and syntax tests for Streamlit
│
├── notebooks/
│   └── industrial_defect_detection.ipynb # Step-by-step interactive demonstration notebook
│
└── docs/
    ├── PROJECT_REPORT.md       # Full engineering report
    ├── VIVA_QUESTIONS.md       # 35 technical viva questions & answers
    └── PRESENTATION_OUTLINE.md # 10-slide presentation deck outline
```

---

## 4. Dataset & Educational Adaptation

The project uses the **MVTec Anomaly Detection (MVTec AD)** benchmark (`bottle` category):
- **Raw Images:** 292 high-resolution industrial images.
- **Normal Samples:** 229 images (defect-free product surfaces).
- **Defective Samples:** 63 images with real manufacturing flaws:
  - `broken_large`: Structural cracks and missing sections.
  - `broken_small`: Small edge chips and fractures.
  - `contamination`: Foreign particle inclusions and surface stains.
- **Educational Adaptation:**
  - Original MVTec AD is an unsupervised benchmark (train split has 0 defects).
  - For supervised transfer learning, normal and defective samples are pooled and split into stratified subsets:
    - **Train Set (70%):** 204 images (160 Normal, 44 Defective)
    - **Validation Set (15%):** 44 images (34 Normal, 10 Defective)
    - **Held-Out Test Set (15%):** 44 images (35 Normal, 9 Defective)
  - **Zero Data Leakage:** Strict SHA/path isolation confirms zero sample overlap across splits.

---

## 5. Installation & Setup

### Windows (PowerShell)
```powershell
# Clone or navigate to repository
cd c:\Users\veerl\Desktop\DL_PROJECT

# Optional: Create isolated virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
python -m pip install -r requirements.txt
```

### Linux / macOS (Bash)
```bash
# Clone or navigate to repository
cd /path/to/DL_PROJECT

# Optional: Create isolated virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 6. Pipeline Execution

To run the complete workflow autonomously (download dataset, train both models, evaluate on test set, run Grad-CAM, run pytest, and generate the final report):

```bash
python run_pipeline.py
```

### Individual Execution Stages
```bash
# 1. Dataset Preparation & Split Generation
python -m src.dataset

# 2. Train Feature Extraction Model
python -m src.train_feature_extraction

# 3. Train Fine-Tuning Model
python -m src.train_finetuning

# 4. Evaluate Held-Out Test Set
python -m src.evaluate

# 5. Generate Grad-CAM for a Sample Image
python -m src.gradcam data/processed/bottle/test/defective/0000_000.png

# 6. Run Pytest Automated Test Suite
python -m pytest tests/ -v
```

---

## 7. Experimental Results & Benchmarks

Both models were evaluated on the **exact same held-out test set** (44 images: 35 Normal, 9 Defective):

| Metric | Feature Extraction (Frozen Backbone) | Fine-Tuning (Upper Layers Unfrozen) | Industrial Significance |
|---|---|---|---|
| **Accuracy** | **97.73%** | **97.73%** | Overall correct decision rate |
| **Precision** | **100.00%** | **90.00%** | When flagged defect, is it truly defective? |
| **Recall (Sensitivity)** | **88.89%** | **100.00%** | **Caught 9/9 defects (Zero escapes)** |
| **F1-Score** | **0.9412** | **0.9474** | Harmonic mean of precision & recall |
| **ROC-AUC** | **0.9968** | **1.0000** | Perfect separability of score distribution |
| **False Negatives (Escapes)** | **1 defect missed** | **0 defects missed** | **Critical metric in manufacturing** |
| **False Positives (Alarms)** | **0 false alarms** | **1 false alarm** | Acceptable trade-off for zero defect escape |

> **Key Finding:** Fine-tuning achieved **100% Recall** on defective products ($9/9$), eliminating dangerous defective escapes at the cost of a single false alarm. In industrial fabrication, defective parts shipped to customers cause expensive warranty claims and assembly jams, making 100% recall the preferred operating state.

---

## 8. Explainable AI (Grad-CAM)

**Grad-CAM (Gradient-weighted Class Activation Mapping)** calculates the gradient of the predicted class score with respect to feature maps in the final convolutional layer (`out_relu` in MobileNetV2):
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

- **Heatmap:** Highlights the high-activation spatial areas in warmer hues (red/yellow).
- **Overlay:** Merged with the original product image at $45\%$ opacity.
- **Auditability:** Factory operators can visually verify that contamination spots or cracks triggered the rejection, rather than background illumination artifacts.

---

## 9. Streamlit Interactive Web Application

Launch the interactive dashboard:

```bash
streamlit run app.py
```

### Dashboard Features
- **🏠 Home:** Overview of industrial vision AI, objectives, and technology stack.
- **🔍 Defect Inspection (Predict):** Upload any surface image or select preloaded test samples; displays real-time defect probability gauge and pass/divert recommendation.
- **💡 Explainable AI (Grad-CAM):** Displays the raw image, Grad-CAM activation heatmap, and overlaid inspection view with adjustable opacity and colormap selection.
- **📊 Model Evaluation & Benchmarks:** Dynamically loads confusion matrices, learning curves, and model comparison metrics from actual training runs.
- **📖 Architecture & About:** Educational breakdown of MobileNetV2 inverted residuals, linear bottlenecks, and transfer learning theory.

---

## 10. Automated Testing

The project includes 22 automated unit and integration tests using `pytest`:

```bash
python -m pytest tests/ -v
```

### Test Coverage
- `tests/test_preprocessing.py`: Image loading, RGB conversion, resizing, $[-1, 1]$ normalization, corrupt file handling, and data augmentation.
- `tests/test_prediction.py`: Model loading, prediction output structure, confidence score range $[0, 1]$, Grad-CAM execution, and integration inference on trained weights.
- `tests/test_evaluation.py`: Metric calculation precision, confusion matrix quadrants, error handling, and validation of generated benchmark artifacts.
- `tests/test_app.py`: Streamlit syntax compilation, safe JSON loader, and path attribute checks.

---

## 11. Troubleshooting

- **TensorFlow oneDNN Warning:** TensorFlow enables oneDNN optimizations by default on CPU. To suppress info messages, set environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
- **Matplotlib Permission on Windows:** Configured automatically in `config.py` using a workspace-local `.matplotlib/` directory.
- **Port Conflict in Streamlit:** If port 8501 is busy, run: `streamlit run app.py --server.port 8503`.

---

**Developed autonomously by Senior Deep Learning & QA Engineering Agent.**
