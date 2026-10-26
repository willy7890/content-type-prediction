"""Train and compare classification pipelines for Movie vs TV Show."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .preprocessing import load_dataset, make_features_and_target

ROOT = Path(__file__).resolve().parents[2]

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def train_and_compare(data_path=None, output_dir=None, random_state=42):
    data_path = Path(data_path) if data_path else ROOT / "data/raw/Dataset.csv"
    output_dir = Path(output_dir) if output_dir else ROOT
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "reports/metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    X, y = make_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    candidates = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=random_state, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250, random_state=random_state, class_weight="balanced",
            min_samples_leaf=2, n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        results.append(row)
        fitted[name] = pipeline

    metrics = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    non_baseline = metrics[metrics["model"] != "Dummy Baseline"]
    best_name = non_baseline.iloc[0]["model"]
    best_pipeline = fitted[best_name]
    joblib.dump(best_pipeline, models_dir / "best_content_type_model.joblib")
    metrics.to_csv(metrics_dir / "model_comparison.csv", index=False)
    summary = {
        "target": "type",
        "classes": sorted(y.unique().tolist()),
        "rows_after_cleaning": int(len(X)),
        "features_used": X.columns.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model_by_macro_f1": str(best_name),
        "metrics": metrics.to_dict(orient="records"),
        "random_state": random_state,
    }
    (metrics_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    return {
        "data": df, "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "metrics": metrics, "best_name": best_name,
        "best_model": best_pipeline, "fitted_models": fitted,
    }

def predict_content_type(record, model_path=None):
    """Predict Movie/TV Show from a dict or one-row DataFrame of feature values."""
    model_path = Path(model_path) if model_path else ROOT / "models/best_content_type_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run the notebook training cells first.")
    model = joblib.load(model_path)
    frame = record if isinstance(record, pd.DataFrame) else pd.DataFrame([record])
    return model.predict(frame)[0]
