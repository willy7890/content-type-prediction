import pandas as pd
from src.content_type_prediction.preprocessing import clean_dataset, make_features_and_target

def test_cleaning_and_target_separation():
    df = pd.DataFrame({
        "show_id": ["s1", "s2", "s3"],
        "type": ["Movie", "TV Show", None],
        "title": ["A", "B", "C"],
        "release_year": [2020, 2021, 2022],
        "rating": ["PG", "TV-MA", "PG"],
    })
    cleaned = clean_dataset(df)
    X, y = make_features_and_target(cleaned)
    assert list(y) == ["Movie", "Tv Show"]
    assert "type" not in X.columns
    assert "show_id" not in X.columns
    assert "title" not in X.columns
