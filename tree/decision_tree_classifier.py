from .utils.tree import Tree, TreeBuilder

import numpy as np

from .utils.criterion import Gini, Entropy

CRITERIA_CLF = {
    "gini": Gini,
    "entropy": Entropy
}


class DecisionTreeClassifier(object):

    def __init__(
        self,
        criterion="entropy",
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=None,
        min_impurity_decrease=0.,
        min_impurity_split=0
    ):
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def fit(self, X, y):
        n_samples, self.n_features = X.shape

        # Check parameters
        criterion = CRITERIA_CLF[self.criterion]()
        min_samples_split = self.min_samples_split
        min_samples_leaf = self.min_samples_leaf
        max_depth = np.iinfo(np.int32).max if self.max_depth is None \
            else self.max_depth
        min_impurity_decrease = self.min_impurity_decrease
        min_impurity_split = self.min_impurity_split

        # Build tree
        self.tree_ = Tree()
        builder = TreeBuilder(
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split
        )
        builder.build(self.tree_, X, y)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            node = self.tree_.root
            while node.label is None:
                selected_feature = node.feature
                feature = X[i, selected_feature]
                is_integer = issubclass(feature.dtype.type, np.integer)
                is_left = (feature == node.threshold) if is_integer else(
                    feature <= node.threshold)
                node = node.children_left if is_left else node.children_right
            y_pred.append(node.label)
        return np.array(y_pred)

    def score(self, X, y):
        pass

    def export(self):
        pass
