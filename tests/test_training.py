import pandas as pd
from sklearn.model_selection import train_test_split
from src.content_type_prediction.train import build_preprocessor
from src.content_type_prediction.preprocessing import make_features_and_target

def test_preprocessor_handles_categorical_and_numeric_features():
    df = pd.DataFrame({
        "type": ["Movie", "TV Show", "Movie", "TV Show", "Movie", "TV Show"],
        "release_year": [2020, 2019, 2018, 2021, 2017, 2022],
        "rating": ["PG", "TV-MA", "PG-13", "TV-14", "R", "TV-PG"],
        "country": ["Tanzania", "USA", "UK", "USA", "India", "UK"],
    })
    X, y = make_features_and_target(df)
    pre = build_preprocessor(X)
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(y)
    assert transformed.shape[1] >= 2
