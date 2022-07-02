from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
import pandas as pd

    
def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    y = df["Result"]
    df.drop(["Ban_1", "Result", "Match_id"], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test

def fit_dummy_classifier(
    X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series
) -> float:
    dummy_class = DummyClassifier()
    dummy_class.fit(X_train, y_train)
    dummy_score = dummy_class.score(X_test, y_test)

    return dummy_score

def fit_knc_classifier(
    X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series, n_n: int
) -> float:
    knc = KNeighborsClassifier(n_neighbors=n_n).fit(X_train, y_train)
    knc_score = knc.score(X_test, y_test)

    return knc_score
    

if __name__ == '__main__':
    wine_data = load_wine()
    X, y = wine_data.data, wine_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    dummy_score = fit_dummy_classifier(X_train, X_test, y_train, y_test)
    knc_scores = []
    for i in range(1, 11):
        knc_score = fit_knc_classifier(X_train, X_test, y_train, y_test, i)
        knc_scores.append(knc_score)

    print(f'Dummy score: {dummy_score}')
    print(f'Knn scores: {knc_scores}')