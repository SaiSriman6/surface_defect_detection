"""
Streamlit Web Application: Industrial Defect Detection
Transfer Learning-Based Industrial Surface Inspection using MobileNetV2 and Grad-CAM.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import config
from src.gradcam import explain_prediction
from src.predict import get_default_model_path, predict_image

# Streamlit Page Configuration
st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-defect {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json_safe(path: Path) -> Optional[dict]:
    """Safely reads a json file if it exists."""
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=64)
st.sidebar.title("Industrial Vision AI")
st.sidebar.caption("Surface Defect Inspection Platform")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "🔍 Defect Inspection (Predict)",
        "💡 Explainable AI (Grad-CAM)",
        "📊 Model Evaluation & Benchmarks",
        "📖 Architecture & About",
    ],
    index=0,
)

# Model Selection in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Active Model Selection")
available_models = {}
if config.FINETUNED_MODEL_PATH.exists():
    available_models["Fine-Tuned Model (Upper Layers)"] = config.FINETUNED_MODEL_PATH
if config.FEATURE_EXTRACTION_MODEL_PATH.exists():
    available_models["Feature Extraction Model (Frozen)"] = config.FEATURE_EXTRACTION_MODEL_PATH

if available_models:
    selected_model_name = st.sidebar.selectbox("Choose Checkpoint", list(available_models.keys()))
    selected_model_path = available_models[selected_model_name]
    st.sidebar.success(f"Loaded: `{selected_model_path.name}`")
else:
    selected_model_path = None
    st.sidebar.warning("⚠️ No trained models found in `models/`. Run training pipeline.")

# -------------------------------------------------------------
# 1. HOME SECTION
# -------------------------------------------------------------
if app_mode == "🏠 Home":
    st.markdown('<div class="main-header">Industrial Defect Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transfer Learning-Based Industrial Surface Inspection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            ### Executive Overview
            Manufacturing and fabrication industries require rapid, ultra-reliable visual quality assurance
            to detect subtle surface defects before products reach customers. Manual visual inspection is labor-intensive,
            fatigue-prone, and inconsistent.

            This application leverages **Transfer Learning** using a **MobileNetV2** deep convolutional neural network
            backbone combined with **Explainable AI (Grad-CAM)** to achieve real-time, automated defect classification
            and visual interpretability.

            ### Project Objectives
            - **Binary Quality Classification:** Differentiate between **Normal (Non-Defective)** and **Defective (Anomalous)** components.
            - **Unit 5 Transfer Learning Demonstration:** Compare **Feature Extraction** (frozen backbone) vs. **Fine-Tuning** (selective unfreezing).
            - **High Efficiency:** Lightweight MobileNetV2 architecture (~3.5M parameters) optimized for edge and industrial deployment.
            - **Explainable Quality Audits:** Explain model classifications visually via **Grad-CAM** localization heatmaps.
            """
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
            <h4>Technology Stack</h4>
            <ul>
                <li><strong>Backbone:</strong> MobileNetV2 (ImageNet Pretrained)</li>
                <li><strong>Framework:</strong> TensorFlow / Keras 3</li>
                <li><strong>Dataset:</strong> MVTec AD (Industrial Inspection)</li>
                <li><strong>Interpretability:</strong> Grad-CAM (Target layer: <code>out_relu</code>)</li>
                <li><strong>Dashboard:</strong> Streamlit Multi-Page Web UI</li>
                <li><strong>Evaluation:</strong> Held-Out Test Set (Acc, Prec, Rec, F1, ROC-AUC)</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### End-to-End Pipeline Workflow")
    st.markdown(
        """
        1. **Dataset Acquisition & Preprocessing:** MVTec AD images are loaded, converted to RGB, resized to $224 \\times 224$, and normalized to $[-1, 1]$.
        2. **Feature Extraction:** Pretrained MobileNetV2 backbone is frozen; a custom binary classification head is trained.
        3. **Fine-Tuning:** Top convolutional layers of MobileNetV2 are unfrozen and trained at a lower learning rate ($10^{-5}$) while keeping BatchNormalization frozen.
        4. **Evaluation:** Independent test set evaluation with confusion matrices and classification reports.
        5. **Visual Explainability:** Grad-CAM calculates gradient heatmaps highlighting where the neural network detects defects.
        """
    )

# -------------------------------------------------------------
# 2. DEFECT INSPECTION (PREDICT)
# -------------------------------------------------------------
elif app_mode == "🔍 Defect Inspection (Predict)":
    st.markdown('<div class="main-header">Automated Defect Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an industrial product image or choose a test sample</div>', unsafe_allow_html=True)

    if selected_model_path is None:
        st.error("⚠️ No trained model checkpoint is available. Please run `python run_pipeline.py` or `python src/train_feature_extraction.py` first.")
        st.stop()

    # Image Input Options
    input_method = st.radio("Select Image Source:", ["Upload File", "Choose Pre-loaded Test Sample"], horizontal=True)

    input_image = None
    image_display_name = ""

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload product surface image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file)
                image_display_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error opening uploaded image: {e}")
    else:
        # Search for available test sample images in data/processed
        test_dir = config.PROCESSED_DATA_DIR / config.DEFAULT_CATEGORY / "test"
        sample_options = {}
        if test_dir.exists():
            for c_name in ["normal", "defective"]:
                c_folder = test_dir / c_name
                if c_folder.exists():
                    for p in list(c_folder.glob("*.png"))[:8]:
                        sample_options[f"[{c_name.upper()}] {p.name}"] = p

        if sample_options:
            chosen_key = st.selectbox("Select Test Image", list(sample_options.keys()))
            chosen_path = sample_options[chosen_key]
            input_image = Image.open(chosen_path)
            image_display_name = chosen_path.name
        else:
            st.info("No processed test samples found. Please upload an image directly.")

    if input_image is not None:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.image(input_image, caption=f"Input Image: {image_display_name}", use_container_width=True)

        with col_pred:
            st.markdown("### Run Inspection")
            threshold = st.slider("Classification Threshold (Defect Sensitivity)", 0.1, 0.9, float(config.CLASSIFICATION_THRESHOLD), 0.05)

            if st.button("🚀 Analyze Surface Quality", type="primary"):
                with st.spinner("Processing image and evaluating through MobileNetV2..."):
                    try:
                        result = predict_image(input_image, model_path=selected_model_path, threshold=threshold)

                        is_def = result["predicted_class"] == "defective"
                        badge_html = (
                            f'<span class="badge-defect">DEFECT DETECTED</span>'
                            if is_def
                            else f'<span class="badge-normal">NORMAL (PASS)</span>'
                        )

                        st.markdown(f"### Result: {badge_html}", unsafe_allow_html=True)
                        st.markdown(f"**Classification:** `{result['display_label']}`")
                        st.markdown(f"**Confidence Score:** `{result['confidence_percentage']}`")
                        st.markdown(f"**Active Model:** `{result['model_used']}`")

                        # Progress Bar for Defect Probability
                        defect_prob = result["probabilities"]["defective"]
                        st.write("Defect Probability Gauge:")
                        st.progress(float(defect_prob))
                        st.caption(f"Normal Probability: {result['probabilities']['normal'] * 100:.1f}% | Defect Probability: {defect_prob * 100:.1f}%")

                        if is_def:
                            st.warning("⚠️ Recommendation: Divert component to secondary inspection or rejection bin.")
                        else:
                            st.success("✅ Recommendation: Component passes surface quality verification.")

                        # Store in session state for Explainability tab
                        st.session_state["last_image"] = input_image
                        st.session_state["last_model_path"] = selected_model_path
                        st.session_state["last_result"] = result

                    except Exception as e:
                        st.error(f"Inference error: {e}")

# -------------------------------------------------------------
# 3. EXPLAINABLE AI (GRAD-CAM)
# -------------------------------------------------------------
elif app_mode == "💡 Explainable AI (Grad-CAM)":
    st.markdown('<div class="main-header">Explainable AI: Grad-CAM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Gradient-weighted Class Activation Mapping for Surface Defect Localization</div>', unsafe_allow_html=True)

    if selected_model_path is None:
        st.error("⚠️ No model checkpoint available.")
        st.stop()

    # Image selection for Grad-CAM
    active_img = st.session_state.get("last_image", None)

    if active_img is None:
        # Prompt user to choose or upload
        st.info("Upload an image below or run inspection from the 'Defect Inspection' tab.")
        uploaded = st.file_uploader("Upload Image for Grad-CAM Explanation", type=["png", "jpg", "jpeg"], key="gradcam_upload")
        if uploaded is not None:
            active_img = Image.open(uploaded)
        else:
            # Fallback to test sample if present
            test_defect = list(config.PROCESSED_DATA_DIR.glob(f"{config.DEFAULT_CATEGORY}/test/defective/*.png"))
            if test_defect:
                active_img = Image.open(test_defect[0])
                st.caption(f"Displaying sample test defect: `{test_defect[0].name}`")

    if active_img is not None:
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            alpha = st.slider("Overlay Heatmap Opacity (Alpha)", 0.1, 0.9, 0.45, 0.05)
        with col_ctrl2:
            colormap_choice = st.selectbox("Colormap Palette", ["jet", "viridis", "magma", "plasma"])

        if st.button("Generate Grad-CAM Visual Explanation", type="primary"):
            with st.spinner("Computing convolutional activations and backpropagating gradients..."):
                try:
                    explanation = explain_prediction(
                        image_input=active_img,
                        model_path=selected_model_path,
                        alpha=alpha,
                    )

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(explanation["display_image"], caption="1. Input Image (224x224 RGB)", use_container_width=True)
                    with c2:
                        st.image(explanation["colored_heatmap"], caption=f"2. Grad-CAM Heatmap (Layer: {config.GRADCAM_TARGET_LAYER})", use_container_width=True)
                    with c3:
                        st.image(explanation["overlay_image"], caption="3. Visual Inspection Overlay", use_container_width=True)

                    st.markdown("### Visual Explanation Analysis")
                    st.info(explanation["interpretation"])

                    st.markdown(
                        """
                        > [!NOTE]
                        > **Engineering Insight on Grad-CAM:**
                        > Grad-CAM calculates the gradient of the predicted class score with respect to the feature map of the final convolutional layer (`out_relu`).
                        > Areas with high gradient flow (warm yellow/red hues) indicate pixels that most strongly pushed the network's classification towards the predicted category.
                        """
                    )
                except Exception as e:
                    st.error(f"Grad-CAM generation failed: {e}")

# -------------------------------------------------------------
# 4. MODEL EVALUATION & BENCHMARKS
# -------------------------------------------------------------
elif app_mode == "📊 Model Evaluation & Benchmarks":
    st.markdown('<div class="main-header">Model Evaluation & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous evaluation on held-out test data (Strictly zero data leakage)</div>', unsafe_allow_html=True)

    fe_metrics = load_json_safe(config.FE_METRICS_PATH)
    ft_metrics = load_json_safe(config.FT_METRICS_PATH)

    if fe_metrics or ft_metrics:
        # Comparison Table
        if config.MODEL_COMPARISON_CSV.exists():
            df_comp = pd.read_csv(config.MODEL_COMPARISON_CSV)
            st.markdown("### Comparative Performance Summary")
            st.dataframe(df_comp.style.format(precision=4).highlight_max(subset=["Accuracy", "F1-Score", "Recall"], color="#DCFCE7"), use_container_width=True)

        # Metric Cards
        col_m1, col_m2 = st.columns(2)
        if fe_metrics:
            with col_m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <h4>Feature Extraction (Frozen Backbone)</h4>
                    <ul>
                        <li><strong>Accuracy:</strong> {fe_metrics['classification_metrics']['accuracy'] * 100:.2f}%</li>
                        <li><strong>Precision:</strong> {fe_metrics['classification_metrics']['precision'] * 100:.2f}%</li>
                        <li><strong>Recall:</strong> {fe_metrics['classification_metrics']['recall'] * 100:.2f}%</li>
                        <li><strong>F1-Score:</strong> {fe_metrics['classification_metrics']['f1_score']:.4f}</li>
                        <li><strong>ROC-AUC:</strong> {fe_metrics['classification_metrics']['roc_auc']:.4f}</li>
                        <li><strong>Total Test Samples:</strong> {fe_metrics['sample_counts']['total_test_samples']}</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if ft_metrics:
            with col_m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                    <h4>Fine-Tuned Model (Upper Layers Unfrozen)</h4>
                    <ul>
                        <li><strong>Accuracy:</strong> {ft_metrics['classification_metrics']['accuracy'] * 100:.2f}%</li>
                        <li><strong>Precision:</strong> {ft_metrics['classification_metrics']['precision'] * 100:.2f}%</li>
                        <li><strong>Recall:</strong> {ft_metrics['classification_metrics']['recall'] * 100:.2f}%</li>
                        <li><strong>F1-Score:</strong> {ft_metrics['classification_metrics']['f1_score']:.4f}</li>
                        <li><strong>ROC-AUC:</strong> {ft_metrics['classification_metrics']['roc_auc']:.4f}</li>
                        <li><strong>Total Test Samples:</strong> {ft_metrics['sample_counts']['total_test_samples']}</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        # Visualizations (Confusion Matrices and Curves)
        st.markdown("### Confusion Matrices")
        c_cm1, c_cm2 = st.columns(2)
        with c_cm1:
            if config.FE_CONF_MATRIX_PATH.exists():
                st.image(str(config.FE_CONF_MATRIX_PATH), caption="Confusion Matrix: Feature Extraction", use_container_width=True)
            else:
                st.caption("Feature Extraction confusion matrix pending.")
        with c_cm2:
            if config.FT_CONF_MATRIX_PATH.exists():
                st.image(str(config.FT_CONF_MATRIX_PATH), caption="Confusion Matrix: Fine-Tuning", use_container_width=True)
            else:
                st.caption("Fine-Tuning confusion matrix pending.")

        st.markdown("### Learning Curves")
        c_c1, c_c2 = st.columns(2)
        with c_c1:
            if config.FE_CURVES_PATH.exists():
                st.image(str(config.FE_CURVES_PATH), caption="Feature Extraction: Loss & Accuracy Curves", use_container_width=True)
            else:
                st.caption("Feature Extraction curves pending.")
        with c_c2:
            if config.FT_CURVES_PATH.exists():
                st.image(str(config.FT_CURVES_PATH), caption="Fine-Tuning: Loss & Accuracy Curves", use_container_width=True)
            else:
                st.caption("Fine-Tuning curves pending.")

    else:
        st.warning("No evaluation metrics found in `results/`. Please execute `python run_pipeline.py` to train and evaluate models.")

# -------------------------------------------------------------
# 5. ARCHITECTURE & ABOUT
# -------------------------------------------------------------
elif app_mode == "📖 Architecture & About":
    st.markdown('<div class="main-header">Technical Documentation & Unit 5 Theory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Concepts: Transfer Learning, MobileNetV2, and Explainability</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["MobileNetV2 Architecture", "Transfer Learning Concepts", "Industrial Impact"])

    with tab1:
        st.markdown(
            """
            ### MobileNetV2 Architecture
            MobileNetV2 is an efficient convolutional neural network architecture specifically tailored for edge devices and mobile systems:
            - **Inverted Residuals:** Unlike traditional ResNets where skip connections link high-dimensional representations, MobileNetV2 connects thin bottleneck layers.
            - **Linear Bottlenecks:** Uses linear activation in the bottleneck layers to prevent non-linearities from destroying critical visual feature manifolds.
            - **Depthwise Separable Convolutions:** Decomposes standard convolution into a depthwise spatial filter and a $1 \\times 1$ pointwise projection, reducing computational cost by $\\approx 8$ to $9$ times with minimal loss of accuracy.
            """
        )

    with tab2:
        st.markdown(
            """
            ### Unit 5 Transfer Learning Concepts
            - **Transfer Learning Types:**
              - *Inductive Transfer Learning:* Source and target tasks differ, but source data helps target performance.
              - *Transductive Transfer Learning:* Source and target tasks are the same, but data domains differ.
            - **Methodologies:**
              1. **Feature Extraction:** Pretrained ImageNet weights are frozen (`trainable=False`). Only the newly attached binary classification head (GAP + Dropout + Dense) learns task-specific weights.
              2. **Fine-Tuning:** Upper convolutional layers are unfreezed and trained alongside the head at a low learning rate ($10^{-5}$). Lower layers remain frozen to preserve foundational low-level edge/texture filters.
            - **Challenges in Transfer Learning:**
              - *Negative Transfer:* Fine-tuning too aggressively can degrade performance if domain distribution diverges significantly.
              - *Catastrophic Forgetting:* High learning rates can wipe out pretrained representations.
              - *Batch Normalization Drift:* Moving mean/variance statistics must remain frozen during fine-tuning.
            """
        )

    with tab3:
        st.markdown(
            """
            ### Industrial Quality Control Deployment
            - **Real-Time Automated Sorting:** Eliminates manual visual fatigue in manufacturing production lines.
            - **Cost Reduction:** Reduces scrap rates and prevents defective assemblies from proceeding down the manufacturing chain.
            - **Auditability via Grad-CAM:** Factory line managers can audit why an image was flagged by inspecting the highlighted defect region, satisfying strict industrial compliance and traceability standards.
            """
        )

st.sidebar.markdown("---")
st.sidebar.caption("Industrial Surface Defect Detection v1.0.0")
