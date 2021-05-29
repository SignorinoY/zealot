import numpy as np


class Bagging(object):

    def __init__(self, base_estimator, objective, n_estimators=10, max_samples=0.7):
        self.base_estimator = base_estimator
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X, y):
        n_samples = X.shape[0]
        max_samples = int(self.max_samples * n_samples)
        indices = np.arange(n_samples)

        self.estimators = []
        for i in range(self.n_estimators):
            train = np.random.choice(indices, max_samples, replace=False)
            X_train, y_train = X[train], y[train]
            estimator = self.base_estimator
            estimator.fit(X_train, y_train)
            self.estimators.append(estimator)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        ys_hat = np.zeros((self.n_estimators, n_samples))
        for i, estimator in enumerate(self.estimators):
            ys_hat[i] = estimator.predict(X)
        if self.base_estimator.objective[0] == "classification":
            ys_hat = ys_hat.astype(int)
            y_hat = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=ys_hat)
        else:
            y_hat = np.mean(ys_hat, axis=1)
        return y_hat

    def score(self, X, y):
        y_hat = self.predict(X)
        if self.base_estimator.objective[0] == "classification":
            score = np.mean(y != y_hat)
        else:
            score = np.mean((y - y_hat) ** 2)
        return score
