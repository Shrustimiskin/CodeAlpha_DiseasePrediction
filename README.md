# Disease Prediction from Medical Data — CodeAlpha ML Task 4

Trains and compares **4 algorithms** (Logistic Regression, Random Forest,
XGBoost, SVM) on **3 datasets** (Heart Disease, Diabetes, Breast Cancer),
as listed in the CodeAlpha task PDF. For each dataset, the best-performing
algorithm (by cross-validated ROC-AUC) is automatically selected and
tuned, then evaluated with Accuracy/Precision/Recall/F1/ROC-AUC plus
confusion matrix, feature importance, and SHAP plots.

## 1. Folder structure

```
disease_prediction/
├── data/                     <- CSV files go here (see step 2)
├── models/                   <- created automatically (one subfolder per dataset)
├── outputs/                  <- created automatically (plots, one subfolder per dataset)
├── src/
│   ├── config.py             <- lists all datasets + their target columns
│   ├── preprocess.py         <- loads/cleans/splits any dataset
│   ├── train.py              <- trains + tunes all 4 algorithms
│   └── evaluate.py           <- metrics + explainability plots
├── fetch_heart.py            <- downloads the heart disease dataset
├── fetch_breast_cancer.py    <- generates the breast cancer dataset (no download needed)
├── main.py                   <- runs train+evaluate for every dataset found
├── requirements.txt
└── README.md
```

## 2. Get the 3 datasets

**Heart Disease** — fetched automatically via the UCI ML Repository:
```bash
python fetch_heart.py
```
This creates `data/heart.csv` (303 rows).

**Breast Cancer** — ships inside scikit-learn already, no download needed:
```bash
python fetch_breast_cancer.py
```
This creates `data/breast_cancer.csv` (569 rows).

**Diabetes (Pima Indians)** — needs a manual download, since it isn't
bundled anywhere:
1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download the CSV.
3. Rename it to `diabetes.csv` and place it in the `data/` folder,
   so the path is `disease_prediction/data/diabetes.csv`.
4. It should have 768 rows, 9 columns, with the target column named
   `Outcome` — that's already what `src/config.py` expects, so no
   renaming of columns is needed.

You don't have to have all 3 present — `main.py` automatically skips
any dataset whose CSV file it can't find, and tells you so.

## 3. Set up the environment (in VS Code)

Open the `disease_prediction` folder in VS Code, open a terminal, then:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## 4. Run everything

```bash
python main.py
```

For each dataset found, this will:
1. Load and clean the CSV
2. Compare Logistic Regression, Random Forest, XGBoost, and SVM with
   5-fold cross-validation (ROC-AUC)
3. Tune the best-performing algorithm with GridSearchCV
4. Save the trained model to `models/<dataset>/model.joblib`
5. Print Accuracy, Precision, Recall, F1, ROC-AUC on the test set
6. Save a confusion matrix, feature importance chart, and SHAP summary
   plot to `outputs/<dataset>/`

You can also run one dataset at a time:
```bash
cd src
python train.py heart
python evaluate.py heart

python train.py diabetes
python evaluate.py diabetes

python train.py breast_cancer
python evaluate.py breast_cancer
```

> Note: SVM training uses `probability=True`, which is a bit slower than
> a plain SVM — this is needed to compute ROC-AUC and enable SHAP.
> On these dataset sizes (300–800 rows) it should still finish quickly.

## 5. Next steps for submission

- Push this project to GitHub as `CodeAlpha_DiseasePrediction`
- In this README, add a results table summarizing which algorithm won
  for each dataset and its test-set metrics
- Record your LinkedIn video walkthrough referencing this repo
