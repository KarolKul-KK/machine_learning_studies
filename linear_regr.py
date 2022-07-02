from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd


def linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    reg = LinearRegression().fit(X_train, y_train)
    reg_predict = reg.predict(X_test)
    reg_score = reg.score(X_test, y_test)
    
    return reg_predict, reg_score

def ridge_fit(X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series) -> list:
    alpha_value_list = []
    ridge_score = []
    for i in np.logspace(-4, 0, 50):
        alpha_value_list.append(i)
        clf = Ridge(alpha=i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        ridge_score.append(score)

    return alpha_value_list, ridge_score

def lasso_fit(X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series) -> list:
    alpha_value_list = []
    lasso_score = []
    for i in np.logspace(-4, 0, 50):
        alpha_value_list.append(i)
        clf = Lasso(alpha=i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        lasso_score.append(score)

    return alpha_value_list, lasso_score


if __name__ == "__main__":
    diabates_data = load_diabetes()
    X, y = diabates_data.data, diabates_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    reg_predict, reg_score = linear_regression(X_train, X_test, y_train, y_test)
    r_alpha_value_list, ridge_score = ridge_fit(X_train, X_test, y_train, y_test)
    l_alpha_value_list, lasso_score = lasso_fit(X_train, X_test, y_train, y_test)

    print(f'Linear regression, predict:{reg_predict} score:{reg_score}')
    print(f'Ridge scores, alpha value:{r_alpha_value_list} score:{ridge_score}')
    print(f'Lasso scores, alpha value:{l_alpha_value_list} score:{lasso_score}')