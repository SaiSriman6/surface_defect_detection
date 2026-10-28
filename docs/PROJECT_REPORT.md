# PROJECT REPORT: Industrial Surface Defect Detection Using Transfer Learning and Explainable AI

**Author / Senior Engineer:** Antigravity Autonomous AI System  
**Specialization:** Deep Learning, Computer Vision & Quality Assurance  
**Academic Level:** B.Tech Data Science (Unit 5: Transfer Learning & Explainable AI)  
**Date of Verification:** October 2026  
**Status:** Completed, Evaluated & Verified  

---

## Abstract

Automated surface defect detection is critical in high-speed industrial manufacturing lines to prevent defective components from propagating downstream. Traditional manual inspection suffers from fatigue, subjectivity, and prohibitive labor expenses, while classical hand-engineered computer vision methods lack robustness to subtle texture variations and lighting shifts. 

This project implements an autonomous, end-to-end deep learning framework that detects surface anomalies on industrial products using **MobileNetV2** transfer learning coupled with **Gradient-weighted Class Activation Mapping (Grad-CAM)** for interpretability. The system is evaluated on the standard **MVTec Anomaly Detection (MVTec AD)** benchmark (`bottle` category). 

We systematically investigate and compare two foundational transfer learning methodologies:
1. **Feature Extraction:** Pretrained ImageNet weights are frozen, and a binary classification head is trained.
2. **Fine-Tuning:** Upper convolutional layers (index 100 onwards) are selectively unfreezed and trained with a reduced learning rate ($10^{-5}$) while maintaining frozen Batch Normalization layers.

Evaluated on the exact same held-out test set ($N=44$), the Feature Extraction model achieved **97.73% Accuracy** and **100% Precision** ($8/9$ defects caught, $1$ missed), while the Fine-Tuned model achieved **97.73% Accuracy**, **90.00% Precision**, and **100.00% Recall** ($9/9$ defects caught, $0$ defective escapes, $\text{ROC-AUC} = 1.0000$). In safety-critical manufacturing, eliminating false negatives (escapes) represents a vital operational milestone. Grad-CAM visual heatmaps provide localized spatial attribution of surface defects, establishing trust for shop-floor operators. The entire architecture is integrated into a multi-page Streamlit application and backed by an automated 22-test pytest suite.

---

## 1. Introduction & Background

In industrial fabrication (pharmaceutical containers, metal stamping, semiconductor packaging, automotive parts), product reliability depends heavily on surface defect identification. Imperfections such as micro-cracks, contamination stains, punctures, or surface abrasions degrade structural integrity and violate safety standards.

With the advent of deep Convolutional Neural Networks (CNNs), deep vision models have demonstrated superior feature hierarchy extraction over hand-crafted features (such as Haralick textures or edge histograms). However, training deep networks from scratch requires hundreds of thousands of labeled images—a condition rarely satisfied in manufacturing environments where defect occurrences are rare and gathering large datasets is commercially impractical.

**Transfer Learning** bridges this gap by leveraging knowledge learned from large-scale visual datasets (such as ImageNet with $1.4\text{M}$ images across $1000$ classes) and transferring those generalized low- and mid-level feature extractors (edges, gradients, textures, corners) to the target domain of industrial surface inspection.

---

## 2. Problem Statement

Given an industrial camera image $X \in \mathbb{R}^{H \times W \times 3}$, design an automated computer vision system to:
1. Classify the image into one of two discrete classes:
   $$Y \in \{0, 1\} \quad \text{where } 0 = \text{Normal (Non-Defective)}, \, 1 = \text{Defective (Anomalous)}$$
2. Provide a calibrated posterior confidence score $P(Y=1 \mid X)$.
3. Generate a spatial visual explanation heatmap $M \in \mathbb{R}^{H \times W}$ pinpointing the precise pixel regions influencing the prediction.
4. Operate reliably on edge-grade or CPU hardware without prohibitive computational latency.

---

## 3. Project Objectives

1. **Dataset Pipeline & Leakage Isolation:** Implement automated acquisition, validation, and stratified splitting of MVTec AD images, establishing strictly zero data leakage across train, validation, and test subsets.
2. **Transfer Learning Exploration (Unit 5 Concepts):**
   - Implement **Feature Extraction** by freezing the pretrained MobileNetV2 backbone.
   - Implement **Fine-Tuning** by unfreezing upper convolutional layers while locking Batch Normalization statistics.
   - Analyze trade-offs between precision, recall, and computational overhead.
3. **Model Interpretability via Grad-CAM:** Implement Gradient-weighted Class Activation Mapping for the MobileNetV2 architecture without synthetic approximations.
4. **Interactive User Interface:** Deploy a production-ready Streamlit dashboard allowing real-time image upload, threshold adjustment, Grad-CAM visualization, and metric auditing.
5. **Quality Assurance:** Validate all modules with an automated pytest suite and provide end-to-end pipeline orchestration.

---

## 4. Theoretical Foundations: Unit 5 Transfer Learning

Transfer learning refers to the machine learning technique where a model developed for a source task $\mathcal{T}_S$ in a source domain $\mathcal{D}_S$ is reused as the starting point for a model on a target task $\mathcal{T}_T$ in a target domain $\mathcal{D}_T$.

### 4.1 Transfer Learning Types
- **Inductive Transfer Learning:** The target task is different from the source task ($\mathcal{T}_T \neq \mathcal{T}_S$), regardless of domain similarity. Here, ImageNet 1000-class object recognition ($\mathcal{T}_S$) transfers to binary surface defect inspection ($\mathcal{T}_T$).
- **Transductive Transfer Learning:** Source and target tasks are identical, but source and target domains differ ($\mathcal{D}_S \neq \mathcal{D}_T$, domain adaptation).

### 4.2 Transfer Learning Methodologies
1. **Off-the-Shelf Feature Extraction:**
   - The pretrained convolutional base is treated as an arbitrary feature extraction mechanism.
   - The convolutional weights $\theta_{\text{conv}}$ are kept fixed:
     $$\frac{\partial \mathcal{L}}{\partial \theta_{\text{conv}}} = 0$$
   - A task-specific classification head $f(h; \theta_{\text{head}})$ is trained to map the extracted feature vector $h$ to class logits.
2. **Selective Fine-Tuning:**
   - Instead of keeping the entire backbone static, some upper convolutional layers are allowed to adapt their weights to the target domain:
     $$\theta_{\text{upper}} \leftarrow \theta_{\text{upper}} - \eta \nabla_{\theta_{\text{upper}}} \mathcal{L}$$
   - Lower layers retain foundational Gabor-like visual primitives (edges, color blobs), while upper layers adapt from general semantic features to domain-specific anomalies (cracks, glass contaminants, chips).

### 4.3 Challenges in Transfer Learning
- **Catastrophic Forgetting:** If the learning rate is too large during fine-tuning, large gradients can obliterate the rich weights learned during ImageNet pretraining. This is resolved by reducing the learning rate by $100\times$ ($\eta_{\text{FT}} = 10^{-5}$ vs $\eta_{\text{FE}} = 10^{-3}$).
- **Negative Transfer:** Occurs when source and target domains diverge so drastically that transferring features hurts performance compared to random initialization. In visual defect inspection, generic edge and texture primitives transfer positively.
- **Batch Normalization Instability:** Unfreezing Batch Normalization layers during small-batch fine-tuning destroys running mean and variance estimates. Solution: keep all Batch Normalization layers frozen (`trainable=False`).

---

## 5. System Architecture & Model Design

### 5.1 Backbone: MobileNetV2
MobileNetV2 was chosen for its low memory footprint and high parameter efficiency:
- **Depthwise Separable Convolutions:** Splitting standard convolution into a $3 \times 3$ depthwise spatial filter and a $1 \times 1$ pointwise projection reduces computation by $\approx 8\times$ to $9\times$:
  $$\text{Reduction Factor} = \frac{D_K \cdot D_K \cdot M + M \cdot N}{D_K \cdot D_K \cdot M \cdot N} = \frac{1}{N} + \frac{1}{D_K^2} \approx \frac{1}{9}$$
- **Inverted Residuals & Linear Bottlenecks:** Memory-efficient shortcut connections connect narrow bottleneck representations, while linear activations preserve manifold information in lower dimensions.

### 5.2 Network Topology Diagram

```
Input Image (224 x 224 x 3)
       │
[Preprocessing: MobileNetV2 scaling to [-1, 1]]
       │
MobileNetV2 Backbone (154 layers, ~2.26M params)
  ├── Conv1 & Inverted Residual Blocks 1-13 (Layers 0-99: FROZEN)
  └── Inverted Residual Blocks 14-16 & Conv_1 / out_relu (Layers 100-153: Trainable in Fine-Tuning)
       │
Feature Map Activation A_k (7 x 7 x 1280) ──> [Grad-CAM Tape]
       │
GlobalAveragePooling2D (Output: 1280 vector)
       │
Dropout Layer (Rate = 0.3)
       │
Dense Output Layer (1 Unit, Sigmoid Activation)
       │
Predicted Defect Probability P(Defective | X)
```

---

## 6. Dataset Description & Stratified Splitting

We utilize the **MVTec Anomaly Detection (MVTec AD)** benchmark (`bottle` category):

### 6.1 Category Composition
- **Total Valid Images:** 292
- **Normal Images:** 229 (clean industrial bottles)
- **Defective Images:** 63 (containing real-world manufacturing anomalies: `broken_large`, `broken_small`, and `contamination`)
- **Corrupt / Unreadable Files:** 0

### 6.2 Educational Adaptation & Leakage Verification
In raw MVTec AD, the train split contains only normal samples (designed for unsupervised anomaly detection). To implement supervised binary transfer learning, all samples were stratified across three subsets:

| Split Subset | Normal Samples | Defective Samples | Total Samples | Defect Proportion |
|---|---|---|---|---|
| **Train Set (70%)** | 160 | 44 | **204** | 21.57% |
| **Validation Set (15%)** | 34 | 10 | **44** | 22.73% |
| **Held-Out Test Set (15%)** | 35 | 9 | **44** | 20.45% |
| **Total** | **229** | **63** | **292** | **21.58%** |

**Data Leakage Assurance:**
Set intersections of file paths and SHA fingerprints confirmed:
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 7. Experimental Training & Hyperparameters

### 7.1 Experiment 1: Feature Extraction
- **Backbone Status:** Completely frozen (`base_model.trainable = False`)
- **Trainable Parameters:** 1,281 (Head only)
- **Non-Trainable Parameters:** 2,257,984
- **Loss Function:** Binary Crossentropy with class weights:
  $$w_0 = 0.6375, \quad w_1 = 2.3182$$
- **Optimizer:** Adam ($\eta = 10^{-3}$)
- **Batch Size:** 16
- **Epochs:** 10 (Early stopping patience = 4)
- **Best Validation Checkpoint:** Epoch 8 (Val Loss = 0.2387, Val Accuracy = 97.73%, Val AUC = 0.9971)

### 7.2 Experiment 2: Fine-Tuning
- **Starting Weights:** Checkpoint from Experiment 1 (`models/feature_extraction.keras`)
- **Backbone Status:** Top layers from index 100 onwards unfrozen (36 trainable layers); all 118 lower layers and Batch Normalization layers frozen.
- **Optimizer:** Adam ($\eta = 10^{-5}$)
- **Batch Size:** 16
- **Epochs:** 5 (Early stopping triggered, restoring best weights)
- **Best Validation Checkpoint:** Epoch 1 (Val Loss = 0.1972, Val Accuracy = 97.73%, Val AUC = 1.0000)

---

## 8. Results & Held-Out Test Set Evaluation

Both trained models were evaluated on the **exact same held-out test set** (44 images: 35 Normal, 9 Defective):

### 8.1 Performance Comparison Table

| Metric | Feature Extraction (Frozen) | Fine-Tuning (Unfrozen Upper) | Delta / Shift |
|---|---|---|---|
| **Accuracy** | 97.73% | 97.73% | 0.00% |
| **Precision** | **100.00%** | 90.00% | -10.00% |
| **Recall (Sensitivity)** | 88.89% | **100.00%** | **+11.11%** |
| **F1-Score** | 0.9412 | **0.9474** | **+0.0062** |
| **ROC-AUC** | 0.9968 | **1.0000** | **+0.0032** |
| **True Positives (TP)** | 8 | **9** | **+1 caught** |
| **False Positives (FP)** | **0** | 1 | +1 alarm |
| **True Negatives (TN)** | **35** | 34 | -1 |
| **False Negatives (FN)** | 1 | **0** | **-1 escape (Eliminated)** |

### 8.2 Analysis of Industrial Trade-Offs
- **Feature Extraction** behaved conservatively: it generated zero false alarms (Precision = 100%), but missed 1 defective component (False Negative = 1).
- **Fine-Tuning** adapted the high-level convolutional filters to surface scratches and edge chips, achieving **100% Recall** (Sensitivity = 1.0000) and zero missed defects. In high-volume manufacturing, a false alarm (1 bottle diverted to secondary review) is vastly preferable to shipping a cracked or contaminated container to a consumer.

---

## 9. Explainable AI: Grad-CAM Implementation

### 9.1 Mathematical Formulation
To explain the classification without modifying model weights, Grad-CAM computes the gradient of the predicted class score $Y^c$ with respect to the feature map activation $A^k \in \mathbb{R}^{U \times V}$ of the last convolutional layer (`out_relu`):
$$\alpha_k^c = \frac{1}{U \cdot V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial Y^c}{\partial A_{i,j}^k}$$
The 2D heatmap is obtained by computing the weighted linear combination followed by a Rectified Linear Unit (ReLU) to isolate positive contributions to class $c$:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### 9.2 Practical Verification
Grad-CAM was executed on actual test samples:
- **Defective Test Sample (`0000_000.png`):** Classified as Defective with 85.35% confidence. Heatmap correctly localized high gradient intensity over the structural chip on the bottle mouth.
- **Normal Test Sample:** Classified as Normal with 86.73% confidence. Low, diffuse gradient distribution with no concentrated hotspot.

---

## 10. Software Architecture & Streamlit Deployment

The solution is encapsulated in a responsive, multi-page Streamlit application (`app.py`):
1. **Home:** High-level project objectives, system overview, and technology stack.
2. **Defect Inspection (Predict):** Real-time image upload or sample picker with adjustable sensitivity threshold and immediate visual classification.
3. **Explainable AI (Grad-CAM):** Interactive tripartite display (Input, Activation Heatmap, Visual Overlay) with adjustable alpha transparency and colormap selectors.
4. **Model Evaluation & Benchmarks:** Dynamic loading of test set metrics, confusion matrices, and learning curves.
5. **Architecture & About:** Educational walkthrough of MobileNetV2 and Unit 5 transfer learning theory.

---

## 11. Limitations & Future Scope

### 11.1 Limitations
- **Category Specificity:** The current trained checkpoint specializes in glass bottle surfaces. Multi-class multi-product training is required for cross-category deployment.
- **Controlled Lighting:** Industrial images in MVTec AD feature consistent illumination. In-the-wild factory lighting changes may require additional illumination augmentation.

### 11.2 Future Scope
- **Multi-Category Scaling:** Train across all 15 MVTec AD categories (metal nut, screw, tile, pill) simultaneously.
- **Quantization & Edge Compilation:** Quantize MobileNetV2 to INT8 using TensorFlow Lite or ONNX Runtime for deployment on NVIDIA Jetson or Raspberry Pi microcontrollers.
- **Pixel-Level Defect Segmentation:** Extend from image-level classification to segmentation using U-Net or SegNet architectures.

---

## 12. Conclusion

This project successfully demonstrates a complete, autonomous, and rigorously verified deep learning solution for industrial surface defect detection. By applying Unit 5 Transfer Learning principles, we established that fine-tuning upper MobileNetV2 layers achieved **100% defect recall** and an **ROC-AUC of 1.0000** on held-out test data. Coupling the model with Grad-CAM visual explanations satisfies critical industrial auditability requirements, creating a safe, deployable, and interpretable automated inspection pipeline.
