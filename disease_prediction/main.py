"""
main.py
-------
Runs the full pipeline (train + evaluate, all 4 algorithms) for every
dataset listed in src/config.py that has its CSV file present.

Usage: python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from config import DATASETS
from train import main as train_main
from evaluate import main as evaluate_main

if __name__ == "__main__":
    for key, cfg in DATASETS.items():
        if not os.path.exists(cfg["csv_path"]):
            print(f"\nSkipping '{key}': file not found at {cfg['csv_path']}")
            print("See README.md for how to get this dataset.")
            continue

        train_main(key)
        evaluate_main(key)

    print("\nAll done. Check the 'outputs/' folder (one subfolder per dataset) for plots.")
