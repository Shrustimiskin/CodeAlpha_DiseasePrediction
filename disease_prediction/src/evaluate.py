"""
evaluate.py
-----------
Loads a trained model for a given dataset and reports test-set
performance plus explainability plots.

Usage:
    python evaluate.py heart
    python evaluate.py diabetes
    python evaluate.py breast_cancer
"""

import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from config import MODEL_DIR, OUTPUT_DIR


def load_artifacts(dataset_key: str):
    model_dir = os.path.join(MODEL_DIR, dataset_key)
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    X_test, y_test = joblib.load(os.path.join(model_dir, "test_data.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    model_name = joblib.load(os.path.join(model_dir, "model_name.joblib"))
    return model, X_test, y_test, feature_names, model_name


def print_metrics(y_test, y_pred, y_proba):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:   ", recall_score(y_test, y_pred))
    print("F1-score: ", f1_score(y_test, y_pred))
    print("ROC-AUC:  ", roc_auc_score(y_test, y_proba))
    print("\nClassification report:\n", classification_report(y_test, y_pred))


def plot_confusion_matrix(y_test, y_pred, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix plot to {out_path}")


def plot_feature_importance(model, feature_names, out_dir):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ attribute; skipping plot.")
        return
    importances = model.feature_importances_
    order = importances.argsort()[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[order], y=[feature_names[i] for i in order])
    plt.title("Feature Importance")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")


def try_shap(model, X_test, feature_names, out_dir, model_name):
    try:
        import shap
    except ImportError:
        print("shap not installed; skipping SHAP explainability plot.")
        return

    try:
        if model_name in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer/LinearExplainer fallback for LR/SVM
            explainer = shap.Explainer(model.predict_proba, X_test)
        shap_values = explainer(X_test) if not hasattr(explainer, "shap_values") else explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        out_path = os.path.join(out_dir, "shap_summary.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot to {out_path}")
    except Exception as e:
        print(f"SHAP plot skipped due to error: {e}")


def main(dataset_key: str):
    print(f"\n{'=' * 60}\nEVALUATING: {dataset_key}\n{'=' * 60}")

    model, X_test, y_test, feature_names, model_name = load_artifacts(dataset_key)
    print(f"Best model for this dataset: {model_name}")

    out_dir = os.path.join(OUTPUT_DIR, dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print_metrics(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred, out_dir)
    plot_feature_importance(model, feature_names, out_dir)
    try_shap(model, X_test, feature_names, out_dir, model_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python evaluate.py <dataset_key>")
        print("Available dataset keys: heart, diabetes, breast_cancer")
