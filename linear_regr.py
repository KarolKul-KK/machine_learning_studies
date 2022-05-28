from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd


diabates_data = load_diabetes()

X, y = diabates_data.data, diabates_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


def linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    reg = LinearRegression().fit(X_train, y_train)
    
    return print(reg.predict(X_test))


if __name__ == "__main__":
    alpha_value_list = []
    ridge_score = []
    for i in np.logspace(-4, 0, 50):
        alpha_value_list.append(i)
        clf = Ridge(alpha=i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        ridge_score.append(score)