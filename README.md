# Content Type Prediction Model — Task 2

**Auspify Technologies | Machine Learning**

This project trains and evaluates classifiers to predict whether a Netflix title is a `Movie` or `TV Show`. The primary workflow is the Jupyter Notebook in `notebooks/content_type_prediction.ipynb`.

## Project structure

```text
content-type-prediction/
├── data/
│   ├── raw/Dataset.csv
│   └── processed/
├── notebooks/content_type_prediction.ipynb
├── src/content_type_prediction/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Run on macOS

Open Terminal and enter the extracted project folder:

```bash
cd ~/Desktop/content-type-prediction
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name content-type-prediction --display-name "Python (Content Type Prediction)"
jupyter lab
```

When Jupyter opens:
1. Open `notebooks/content_type_prediction.ipynb`.
2. Select the **Python (Content Type Prediction)** kernel if prompted.
3. Choose **Run → Run All Cells** (or run cells one at a time from the top).
4. Wait for all four models to train and for the evaluation plots to appear.
5. Check `reports/metrics/model_comparison.csv` and the saved figures.

If the terminal says `jupyter: command not found`, activate `.venv` and run `python -m jupyter lab`.

## Models and metrics

The notebook compares:
- DummyClassifier baseline (reference point)
- Logistic Regression
- Random Forest
- Extra Trees

The preprocessing pipeline imputes missing values and one-hot encodes categorical features. The target column `type` is excluded from the inputs to avoid target leakage. `show_id` and `title` are excluded from this baseline. The data is split into 80% training and 20% testing with stratification and a fixed random seed.

Reported metrics:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Macro F1-score
- Classification report and confusion matrix

The notebook selects the best non-baseline model using **macro F1-score** and saves it to `models/best_content_type_model.joblib`. It also writes the comparison table and evaluation summary to `reports/metrics/`.

## Tests

After installing dependencies, from the project root run:

```bash
python -m pytest -q
```

## Important notes

- The supplied `Dataset.csv` is included in `data/raw/`.
- Run the notebook from the project root or use the path handling already included in the notebook.
- Metrics are not prefilled or guaranteed: they are computed when you execute the notebook on your machine.
- Do not edit the target labels or add `type` as an input feature.
