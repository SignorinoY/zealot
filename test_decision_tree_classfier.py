from sklearn import datasets

from tree.classes import DecisionTreeClassifier, DecisionTreeRegressor

# Classification

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier(criterion="gini")

clf.fit(X, y)

print(clf.tree_)
print(clf.predict(X))

# Regression

boston = datasets.load_boston()
X = boston.data
y = boston.target

reg = DecisionTreeRegressor()

reg.fit(X, y)

print(reg.tree_)
print(reg.predict(X))
