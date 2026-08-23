"""
train.py
--------
Trains Logistic Regression, Random Forest, XGBoost, and SVM on a given
dataset, compares them with cross-validation, tunes the best one, and
saves it to disk under models/<dataset_key>/.

Usage:
    python train.py heart
    python train.py diabetes
    python train.py breast_cancer
"""

import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

from preprocess import get_processed_data
from config import MODEL_DIR


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }


# Hyperparameter grids used when tuning whichever model wins the comparison.
# n_jobs is kept at 1 everywhere to avoid multiprocessing issues on some
# Windows setups (small paging file). Safe to raise if your machine handles it.
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
}


def compare_models(X_train, y_train):
    results = {}
    for name, model in get_candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        results[name] = scores.mean()
        print(f"{name}: mean ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    return results


def tune_model(model_name, X_train, y_train):
    base_model = get_candidate_models()[model_name]
    param_grid = PARAM_GRIDS[model_name]

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best ROC-AUC:", grid.best_score_)
    return grid.best_estimator_


def main(dataset_key: str):
    print(f"\n{'=' * 60}\nDATASET: {dataset_key}\n{'=' * 60}")

    X_train, X_test, y_train, y_test, scaler, feature_names = get_processed_data(dataset_key)

    print("\n=== Cross-validated model comparison (all 4 algorithms) ===")
    results = compare_models(X_train, y_train)
    best_model_name = max(results, key=results.get)
    print(f"\nBest model by CV ROC-AUC: {best_model_name}")

    print(f"\n=== Hyperparameter tuning ({best_model_name}) ===")
    best_model = tune_model(best_model_name, X_train, y_train)

    out_dir = os.path.join(MODEL_DIR, dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_names.joblib"))
    joblib.dump((X_test, y_test), os.path.join(out_dir, "test_data.joblib"))
    joblib.dump(best_model_name, os.path.join(out_dir, "model_name.joblib"))

    print(f"\nSaved trained model to '{out_dir}/model.joblib'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python train.py <dataset_key>")
        print("Available dataset keys: heart, diabetes, breast_cancer")
