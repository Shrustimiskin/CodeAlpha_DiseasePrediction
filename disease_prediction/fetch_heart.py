"""
fetch_heart.py
--------------
Downloads the UCI Heart Disease dataset and saves it as data/heart.csv.
Run this once: python fetch_heart.py
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)
df.columns = list(X.columns) + ["target"]
df.to_csv("data/heart.csv", index=False)

print("Saved data/heart.csv")
print(df.shape)
print(df.head())
