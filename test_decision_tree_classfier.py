from sklearn import datasets
import numpy as np

from tree.classes import DecisionTreeClassifier, DecisionTreeRegressor

# Classification

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier(criterion="entropy")

clf.fit(X, y)
y_hat = clf.predict(X)
print(clf.score(X, y))

# Regression

boston = datasets.load_boston()
X = boston.data
y = boston.target

reg = DecisionTreeRegressor()

reg.fit(X, y)
y_hat = reg.predict(X)
print(reg.score(X, y))
