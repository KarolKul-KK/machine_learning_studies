from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    wine_data = load_wine()

    X, y = wine_data.data, wine_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    lr = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=500)
    pipe = Pipeline([("classifier", RandomForestClassifier())])

    search_space = [{"classifier": [lr],
                    "classifier__penalty": ['l1', 'l2'],
                    "classifier__C": np.logspace(0, 4, 10)},
                    {"classifier": [RandomForestClassifier()],
                    "classifier__n_estimators": [10, 100, 1000],
                    "classifier__max_features": [1, 2, 3]},
                    {"classifier": [KNeighborsClassifier()],
                    "classifier__n_neighbors": [5, 10, 15],}]

    grid = GridSearchCV(pipe, search_space, cv=5, verbose=1, n_jobs=-1)
    best_model = grid.fit(X_train, y_train)
    print(best_model.best_estimator_.get_params()['classifier'])