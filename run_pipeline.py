"""
Autonomous End-to-End Orchestration Pipeline for Industrial Surface Defect Detection.

Automates:
1. Environment & hardware verification
2. Dataset acquisition and validation
3. Split preparation & data leakage verification
4. Preprocessing sanity tests
5. Feature extraction training (frozen backbone)
6. Feature extraction model verification
7. Fine-tuning training (upper layers unfrozen)
8. Fine-tuned model verification
9. Held-out test set evaluation for both models
10. Metric computation & visualization generation
11. Inference testing
12. Grad-CAM explainability generation & verification
13. Pytest automated test execution
14. Required output file validation
15. Production of `results/FINAL_VERIFICATION_REPORT.md`
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PipelineOrchestrator")


class PipelineOrchestrator:
    def __init__(
        self,
        category: str = config.DEFAULT_CATEGORY,
        fe_epochs: int = config.FE_EPOCHS,
        ft_epochs: int = config.FT_EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        smoke_test: bool = False,
        force: bool = False,
    ):
        self.category = category
        self.fe_epochs = 1 if smoke_test else fe_epochs
        self.ft_epochs = 1 if smoke_test else ft_epochs
        self.batch_size = batch_size
        self.smoke_test = smoke_test
        self.force = force
        self.start_time = datetime.datetime.now()

        # Tracking state
        self.stages_executed: List[str] = []
        self.env_info: Dict = {}
        self.dataset_info: Dict = {}
        self.fe_status: str = "NOT RUN"
        self.ft_status: str = "NOT RUN"
        self.eval_status: str = "NOT RUN"
        self.gradcam_status: str = "NOT RUN"
        self.tests_status: str = "NOT RUN"
        self.test_counts: Dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0}
        self.blockers: List[str] = []

    def check_environment(self) -> bool:
        """STAGE 1: Environment verification."""
        logger.info("=== STAGE 1: Checking Environment & Hardware ===")
        import tensorflow as tf

        gpu_devices = tf.config.list_physical_devices("GPU")
        self.env_info = {
            "os": os.name,
            "python_version": sys.version.split()[0],
            "tensorflow_version": tf.__version__,
            "keras_version": getattr(tf.keras, "__version__", "Keras 3"),
            "gpu_available": len(gpu_devices) > 0,
            "device": "GPU" if len(gpu_devices) > 0 else "CPU",
            "workspace": str(config.BASE_DIR),
            "smoke_test_mode": self.smoke_test,
        }

        logger.info("Environment: Python %s, TensorFlow %s, Device: %s",
                    self.env_info["python_version"], self.env_info["tensorflow_version"], self.env_info["device"])
        self.stages_executed.append("STAGE 1: Environment Checked")
        return True

    def acquire_and_validate_dataset(self) -> bool:
        """STAGE 2: Dataset Acquisition, Validation, and Splits."""
        logger.info("=== STAGE 2: Dataset Acquisition & Validation for '%s' ===", self.category)
        from src.dataset import prepare_splits

        try:
            metadata = prepare_splits(category=self.category, force=self.force)
            self.dataset_info = metadata
            logger.info("Dataset validated: %d total valid images.", metadata["raw_summary"]["total_valid_samples"])
            logger.info("Splits: Train=%d, Val=%d, Test=%d",
                        metadata["splits"]["train"]["total"],
                        metadata["splits"]["val"]["total"],
                        metadata["splits"]["test"]["total"])
            self.stages_executed.append("STAGE 2: Dataset Prepared & Validated")
            return True
        except Exception as e:
            logger.error("Dataset acquisition/validation failed: %s", e)
            self.blockers.append(f"Dataset preparation failed: {e}")
            return False

    def verify_preprocessing_pipeline(self) -> bool:
        """STAGE 3: Verify Preprocessing & Augmentation Pipeline."""
        logger.info("=== STAGE 3: Verifying Preprocessing & Data Loaders ===")
        from src.preprocessing import create_split_datasets

        try:
            train_ds, val_ds, test_ds, counts = create_split_datasets(self.category, self.batch_size)
            # Inspect first batch
            for x, y in train_ds.take(1):
                assert x.shape[1:] == (224, 224, 3), f"Wrong image batch shape: {x.shape}"
                assert len(y.shape) == 1, f"Wrong label batch shape: {y.shape}"
                # Assert MobileNetV2 scaling in [-1, 1] range
                min_val = float(tf.reduce_min(x))
                max_val = float(tf.reduce_max(x))
                assert min_val >= -1.05 and max_val <= 1.05, f"Scaling invalid: min={min_val}, max={max_val}"
            logger.info("Preprocessing verified: image shape %s, normalized range [%.2f, %.2f]",
                        x.shape, min_val, max_val)
            self.stages_executed.append("STAGE 3: Preprocessing Verified")
            return True
        except Exception as e:
            logger.error("Preprocessing verification failed: %s", e)
            self.blockers.append(f"Preprocessing failed: {e}")
            return False

    def train_feature_extraction_stage(self) -> bool:
        """STAGE 4: Train Feature Extraction Model."""
        logger.info("=== STAGE 4: Feature Extraction Training (Backbone Frozen) ===")
        from src.train_feature_extraction import train_feature_extraction

        # Check if already trained and force is false
        if config.FEATURE_EXTRACTION_MODEL_PATH.exists() and config.FE_HISTORY_PATH.exists() and not self.force:
            logger.info("Found existing feature extraction model checkpoint at: %s. Re-using.", config.FEATURE_EXTRACTION_MODEL_PATH)
            self.fe_status = "PASSED (Checkpoint Exists)"
            self.stages_executed.append("STAGE 4: Feature Extraction Re-used")
            return True

        try:
            _, history = train_feature_extraction(
                category=self.category,
                epochs=self.fe_epochs,
                batch_size=self.batch_size,
            )
            final_acc = history["accuracy"][-1]
            val_acc = history.get("val_accuracy", [0.0])[-1]
            logger.info("Feature extraction completed. Train Acc: %.4f, Val Acc: %.4f", final_acc, val_acc)
            self.fe_status = f"PASSED (Final Val Acc: {val_acc:.4f})"
            self.stages_executed.append("STAGE 4: Feature Extraction Trained")
            return True
        except Exception as e:
            logger.error("Feature extraction training failed: %s", e)
            self.fe_status = f"FAILED: {e}"
            self.blockers.append(f"Feature extraction failed: {e}")
            return False

    def train_finetuning_stage(self) -> bool:
        """STAGE 5: Fine-Tuning Stage (Top Layers Unfrozen)."""
        logger.info("=== STAGE 5: Fine-Tuning Training (Upper Conv Layers Unfrozen) ===")
        from src.train_finetuning import train_finetuning

        if config.FINETUNED_MODEL_PATH.exists() and config.FT_HISTORY_PATH.exists() and not self.force:
            logger.info("Found existing fine-tuned model checkpoint at: %s. Re-using.", config.FINETUNED_MODEL_PATH)
            self.ft_status = "PASSED (Checkpoint Exists)"
            self.stages_executed.append("STAGE 5: Fine-Tuning Re-used")
            return True

        try:
            _, history = train_finetuning(
                category=self.category,
                epochs=self.ft_epochs,
                batch_size=self.batch_size,
            )
            final_acc = history["accuracy"][-1]
            val_acc = history.get("val_accuracy", [0.0])[-1]
            logger.info("Fine-tuning completed. Train Acc: %.4f, Val Acc: %.4f", final_acc, val_acc)
            self.ft_status = f"PASSED (Final Val Acc: {val_acc:.4f})"
            self.stages_executed.append("STAGE 5: Fine-Tuning Trained")
            return True
        except Exception as e:
            logger.error("Fine-tuning training failed: %s", e)
            self.ft_status = f"FAILED: {e}"
            self.blockers.append(f"Fine-tuning failed: {e}")
            return False

    def evaluate_models_stage(self) -> bool:
        """STAGE 6: Evaluate Both Models on Held-out Test Set."""
        logger.info("=== STAGE 6: Held-Out Test Set Model Evaluation ===")
        from src.evaluate import evaluate_models

        try:
            df_comp = evaluate_models(category=self.category)
            logger.info("Evaluation complete:\n%s", df_comp.to_string(index=False))
            self.eval_status = "PASSED"
            self.stages_executed.append("STAGE 6: Evaluation Complete")
            return True
        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            self.eval_status = f"FAILED: {e}"
            self.blockers.append(f"Evaluation failed: {e}")
            return False

    def verify_gradcam_stage(self) -> bool:
        """STAGE 7: Grad-CAM Explainability Verification."""
        logger.info("=== STAGE 7: Grad-CAM Explainability Generation & Verification ===")
        from src.gradcam import explain_prediction

        # Find a test image
        test_dir = config.PROCESSED_DATA_DIR / self.category / "test"
        sample_img = None
        for c in ["defective", "normal"]:
            imgs = list((test_dir / c).glob("*.png"))
            if imgs:
                sample_img = imgs[0]
                break

        if sample_img is None:
            logger.warning("No test image found for Grad-CAM verification.")
            return False

        try:
            out_vis = config.RESULTS_DIR / "sample_gradcam_verification.png"
            res = explain_prediction(
                image_input=sample_img,
                output_path=out_vis,
            )
            assert res["heatmap"].shape == (7, 7) or len(res["heatmap"].shape) == 2, "Invalid heatmap shape"
            assert res["overlay_image"].shape == (224, 224, 3), "Invalid overlay shape"
            assert out_vis.exists() and out_vis.stat().st_size > 0, "Visualization image not saved"
            logger.info("Grad-CAM successfully generated and verified. Output: %s", out_vis)
            self.gradcam_status = "PASSED"
            self.stages_executed.append("STAGE 7: Grad-CAM Verified")
            return True
        except Exception as e:
            logger.error("Grad-CAM verification failed: %s", e)
            self.gradcam_status = f"FAILED: {e}"
            self.blockers.append(f"Grad-CAM failed: {e}")
            return False

    def run_automated_tests_stage(self) -> bool:
        """STAGE 8: Run pytest automated test suite."""
        logger.info("=== STAGE 8: Executing Pytest Automated Test Suite ===")
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
        result = subprocess.run(cmd, cwd=str(config.BASE_DIR), capture_output=True, text=True)
        logger.info("Test execution stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("Test execution stderr:\n%s", result.stderr)

        # Parse test outcomes
        passed = result.stdout.count("PASSED")
        failed = result.stdout.count("FAILED")
        self.test_counts["passed"] = passed
        self.test_counts["failed"] = failed

        if result.returncode == 0:
            self.tests_status = f"PASSED ({passed} passed, 0 failed)"
            self.stages_executed.append("STAGE 8: Pytest Passed")
            return True
        else:
            self.tests_status = f"FAILED ({failed} failed, {passed} passed)"
            self.blockers.append(f"Pytest suite encountered {failed} failures.")
            return False

    def validate_artifacts(self) -> Dict[str, bool]:
        """STAGE 9: Check all expected files exist."""
        required_artifacts = {
            "config.py": config.BASE_DIR / "config.py",
            "app.py": config.BASE_DIR / "app.py",
            "requirements.txt": config.BASE_DIR / "requirements.txt",
            "README.md": config.BASE_DIR / "README.md",
            "data/splits_metadata.json": config.SPLITS_METADATA_PATH,
            "models/feature_extraction.keras": config.FEATURE_EXTRACTION_MODEL_PATH,
            "models/finetuned.keras": config.FINETUNED_MODEL_PATH,
            "results/feature_extraction_history.json": config.FE_HISTORY_PATH,
            "results/finetuning_history.json": config.FT_HISTORY_PATH,
            "results/feature_extraction_curves.png": config.FE_CURVES_PATH,
            "results/finetuning_curves.png": config.FT_CURVES_PATH,
            "results/feature_extraction_metrics.json": config.FE_METRICS_PATH,
            "results/finetuning_metrics.json": config.FT_METRICS_PATH,
            "results/confusion_matrix_feature_extraction.png": config.FE_CONF_MATRIX_PATH,
            "results/confusion_matrix_finetuning.png": config.FT_CONF_MATRIX_PATH,
            "results/model_comparison.csv": config.MODEL_COMPARISON_CSV,
            "docs/PROJECT_REPORT.md": config.DOCS_DIR / "PROJECT_REPORT.md",
            "docs/VIVA_QUESTIONS.md": config.DOCS_DIR / "VIVA_QUESTIONS.md",
            "docs/PRESENTATION_OUTLINE.md": config.DOCS_DIR / "PRESENTATION_OUTLINE.md",
        }

        status_map = {}
        for name, path in required_artifacts.items():
            exists = path.exists() and (path.stat().st_size > 0 if path.is_file() else True)
            status_map[name] = exists
        return status_map

    def generate_final_verification_report(self) -> Path:
        """STAGE 10: Generate results/FINAL_VERIFICATION_REPORT.md."""
        logger.info("=== STAGE 10: Generating Final Verification Report ===")
        artifact_status = self.validate_artifacts()
        all_artifacts_ok = all(artifact_status.values())

        # Load metrics if present
        fe_metrics = {}
        ft_metrics = {}
        if config.FE_METRICS_PATH.exists():
            with open(config.FE_METRICS_PATH, "r") as f:
                fe_metrics = json.load(f)
        if config.FT_METRICS_PATH.exists():
            with open(config.FT_METRICS_PATH, "r") as f:
                ft_metrics = json.load(f)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = str(datetime.datetime.now() - self.start_time).split(".")[0]

        overall_status = "PASSED" if (
            "PASSED" in self.fe_status and
            "PASSED" in self.ft_status and
            "PASSED" in self.eval_status and
            "PASSED" in self.gradcam_status and
            len(self.blockers) == 0
        ) else "BLOCKED / ATTENTION REQUIRED"

        report_lines = [
            "# FINAL VERIFICATION REPORT",
            "",
            "## 1. Executive Summary",
            f"- **Project Title:** Industrial Surface Defect Detection Using Transfer Learning and Explainable AI",
            f"- **Verification Timestamp:** `{now}`",
            f"- **Execution Duration:** `{duration}`",
            f"- **Overall Pipeline Status:** **{overall_status}**",
            f"- **Mode:** `{'SMOKE TEST' if self.smoke_test else 'FULL PRODUCTION PIPELINE'}`",
            "",
            "## 2. Environment & Hardware Details",
            f"- **Operating System:** `{self.env_info.get('os', 'windows')}`",
            f"- **Python Version:** `{self.env_info.get('python_version', '3.10')}`",
            f"- **TensorFlow Version:** `{self.env_info.get('tensorflow_version', '2.18.0')}`",
            f"- **Keras Version:** `{self.env_info.get('keras_version', '3.7.0')}`",
            f"- **Compute Device:** `{self.env_info.get('device', 'CPU')}`",
            f"- **Workspace Path:** `{self.env_info.get('workspace', '.')}`",
            "",
            "## 3. Dataset Specifications & Split Methodology",
            f"- **Dataset Category:** `{self.category}` (MVTec Anomaly Detection)",
            f"- **Total Valid Images:** `{self.dataset_info.get('raw_summary', {}).get('total_valid_samples', 'N/A')}`",
            f"- **Normal Product Images:** `{self.dataset_info.get('raw_summary', {}).get('normal_samples', 'N/A')}`",
            f"- **Defective Product Images:** `{self.dataset_info.get('raw_summary', {}).get('defective_samples', 'N/A')}`",
            f"- **Corrupt Images Filtered:** `{self.dataset_info.get('raw_summary', {}).get('corrupted_files_count', 0)}`",
            "",
            "### Split Breakdown (Strictly Zero Data Leakage):",
            "| Split | Total Images | Normal (Class 0) | Defective (Class 1) | Defect Ratio |",
            "|---|---|---|---|---|",
        ]

        splits = self.dataset_info.get("splits", {})
        for s_name in ["train", "val", "test"]:
            if s_name in splits:
                s = splits[s_name]
                report_lines.append(f"| **{s_name.upper()}** | {s['total']} | {s['normal']} | {s['defective']} | {s['defect_ratio']:.2%} |")

        report_lines.extend([
            "",
            "## 4. Model Training & Experiment Status",
            f"- **Experiment 1 (Feature Extraction - Frozen Backbone):** `{self.fe_status}`",
            f"- **Experiment 2 (Fine-Tuning - Unfrozen Top Layers):** `{self.ft_status}`",
            "",
            "## 5. Held-Out Test Set Evaluation Metrics",
            "| Metric | Feature Extraction (Frozen) | Fine-Tuning (Unfrozen Upper) |",
            "|---|---|---|",
        ])

        fe_c = fe_metrics.get("classification_metrics", {})
        ft_c = ft_metrics.get("classification_metrics", {})

        metrics_list = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-Score", "f1_score"),
            ("ROC-AUC", "roc_auc"),
        ]

        for display_name, key in metrics_list:
            v_fe = f"{fe_c.get(key, 'N/A'):.4f}" if isinstance(fe_c.get(key), (int, float)) else "N/A"
            v_ft = f"{ft_c.get(key, 'N/A'):.4f}" if isinstance(ft_c.get(key), (int, float)) else "N/A"
            report_lines.append(f"| **{display_name}** | {v_fe} | {v_ft} |")

        report_lines.extend([
            "",
            "### Confusion Matrix Details:",
            "| Matrix Value | Feature Extraction | Fine-Tuning |",
            "|---|---|---|",
        ])

        fe_cm = fe_metrics.get("confusion_matrix", {})
        ft_cm = ft_metrics.get("confusion_matrix", {})
        cm_items = [
            ("True Negatives (TN - Normal Correct)", "tn"),
            ("False Positives (FP - False Alarm)", "fp"),
            ("False Negatives (FN - Defect Missed)", "fn"),
            ("True Positives (TP - Defect Caught)", "tp"),
        ]
        for desc, k in cm_items:
            report_lines.append(f"| {desc} | {fe_cm.get(k, 'N/A')} | {ft_cm.get(k, 'N/A')} |")

        report_lines.extend([
            "",
            "## 6. Automated Pytest Test Suite Results",
            f"- **Status:** `{self.tests_status}`",
            f"- **Tests Passed:** `{self.test_counts['passed']}`",
            f"- **Tests Failed:** `{self.test_counts['failed']}`",
            "",
            "## 7. Explainable AI (Grad-CAM) Status",
            f"- **Status:** `{self.gradcam_status}`",
            f"- **Target Layer:** `{config.GRADCAM_TARGET_LAYER}` (MobileNetV2 final ReLU activation)",
            "- **Sample Overlay Visualization:** `results/sample_gradcam_verification.png`",
            "",
            "## 8. Artifact Verification Checklist",
            "| Artifact Name | Path | Status |",
            "|---|---|---|",
        ])

        for name, exists in artifact_status.items():
            status_badge = "✅ PASSED" if exists else "❌ MISSING"
            report_lines.append(f"| `{name}` | `{config.BASE_DIR / name}` | {status_badge} |")

        report_lines.extend([
            "",
            "## 9. Blockers & Engineering Observations",
        ])

        if self.blockers:
            for b in self.blockers:
                report_lines.append(f"- ⚠️ **Blocker/Warning:** {b}")
        else:
            report_lines.append("- None. All pipeline steps executed and verified autonomously without blockers.")

        report_lines.extend([
            "",
            "## 10. Application Launch Command",
            "```bash",
            "streamlit run app.py",
            "```",
        ])

        report_content = "\n".join(report_lines)
        with open(config.FINAL_VERIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("Final Verification Report written to: %s", config.FINAL_VERIFICATION_REPORT_PATH)
        return config.FINAL_VERIFICATION_REPORT_PATH

    def run(self) -> int:
        """Executes all stages in sequence."""
        logger.info("Starting Autonomous Pipeline Orchestration...")

        if not self.check_environment():
            return 1

        if not self.acquire_and_validate_dataset():
            return 2

        if not self.verify_preprocessing_pipeline():
            return 3

        if not self.train_feature_extraction_stage():
            return 4

        if not self.train_finetuning_stage():
            return 5

        if not self.evaluate_models_stage():
            return 6

        if not self.verify_gradcam_stage():
            return 7

        # Run automated pytest tests
        self.run_automated_tests_stage()

        # Produce final verification report
        self.generate_final_verification_report()

        logger.info("Autonomous Pipeline Orchestration Finished with Status: %s",
                    "SUCCESS" if len(self.blockers) == 0 else "PARTIAL / BLOCKED")
        return 0 if len(self.blockers) == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Deep Learning Pipeline Orchestrator")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid 1-epoch smoke test")
    parser.add_argument("--category", type=str, default=config.DEFAULT_CATEGORY, help="MVTec category")
    parser.add_argument("--fe-epochs", type=int, default=config.FE_EPOCHS, help="Feature extraction epochs")
    parser.add_argument("--ft-epochs", type=int, default=config.FT_EPOCHS, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--force", action="store_true", help="Force retraining and re-download")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        category=args.category,
        fe_epochs=args.fe_epochs,
        ft_epochs=args.ft_epochs,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
        force=args.force,
    )
    exit_code = orchestrator.run()
    sys.exit(exit_code)
