from tree.decision_tree_classifier import DecisionTreeClassifier

import numpy as np

X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 2],
    [1, 0, 1, 2],
    [2, 0, 1, 2],
    [2, 0, 1, 1],
    [2, 1, 0, 1],
    [2, 1, 0, 2],
    [2, 0, 0, 0],
])
y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

clf = DecisionTreeClassifier()
clf.fit(X, y)
print(clf.tree_)
print(clf.predict(X))
