from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


if __name__ == "__main__":
    wine_data = load_wine()

    X, y = wine_data.data, wine_data.target

    kf = StratifiedKFold(n_splits=10)
    kf.get_n_splits(X)

    score_list = []
    for train_index, test_index in kf.split(X, y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        score_list.append(s)
        print("y_train_count:", np.bincount(y_train), "y_test_count:", np.bincount(y_test))
        print(10 * '*')
        print(score_list)