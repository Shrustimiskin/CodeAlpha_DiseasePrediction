"""
config.py
---------
Central place listing every dataset used in this project.

To add/remove a dataset, edit the DATASETS dict below. Each entry needs:
- csv_path: where the CSV file lives (relative to project root)
- target_column: name of the label column in that CSV
- binarize: set True if the target has more than 2 classes and you want
  to collapse it into 0 = no disease / 1 = disease (e.g. UCI heart disease
  uses 0-4 severity levels)
"""

import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

DATASETS = {
    "heart": {
        "label": "Heart Disease (UCI)",
        "csv_path": os.path.join(PROJECT_ROOT, "data", "heart.csv"),
        "target_column": "target",
        "binarize": True,
    },
    "diabetes": {
        "label": "Diabetes (Pima Indians)",
        "csv_path": os.path.join(PROJECT_ROOT, "data", "diabetes.csv"),
        "target_column": "Outcome",
        "binarize": False,
    },
    "breast_cancer": {
        "label": "Breast Cancer (Wisconsin)",
        "csv_path": os.path.join(PROJECT_ROOT, "data", "breast_cancer.csv"),
        "target_column": "target",
        "binarize": False,
    },
}

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
