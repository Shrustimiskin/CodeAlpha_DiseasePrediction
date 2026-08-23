"""
fetch_breast_cancer.py
-----------------------
Generates data/breast_cancer.csv from the UCI Breast Cancer Wisconsin
(Diagnostic) dataset, which ships built into scikit-learn -- no manual
download needed.

Run this once: python fetch_breast_cancer.py
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer(as_frame=True)
df = data.frame  # includes a "target" column automatically (0 = malignant, 1 = benign)

df.to_csv("data/breast_cancer.csv", index=False)

print("Saved data/breast_cancer.csv")
print(df.shape)
print(df.head())
