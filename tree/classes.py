import numpy as np

from .criterion import Entropy, Gini, MSE
from .tree import Tree, TreeBuilder

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor"
]

# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_CLF = {
    "gini": Gini,
    "entropy": Entropy
}
CRITERIA_REG = {
    "mse": MSE
}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(object):
    """Base class for decision tree.

    Parameters
    ----------
    objective : {"classification", "regression"}
        The objective for build tree.
    criterion : {"gini", "entropy", "mse"}
        The function to measure the quality of a split.
    max_depth : int
        The maximum depth of the tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    min_impurity_decrease : float
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    min_impurity_split : float
        Threshold for early stopping in tree growth.

    """

    def __init__(
        self,
        objective,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        min_impurity_split
    ):
        self.objective = objective
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    def fit(self, X, y):
        """Build a decision tree classifier or regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, )
            The training target classes, or the target values.

        Returns
        -------
        y: array-like of shape (n_samples, )
            The predicted classes, or the predict values.
        """
        # Determine output settings
        n_samples, self.n_features = X.shape
        # Setup default parameters
        if self.objective == "classification":
            criterion = CRITERIA_CLF[self.criterion]()
        else:
            criterion = CRITERIA_REG[self.criterion]()
        max_depth = np.iinfo(np.int32).max if self.max_depth is None \
            else self.max_depth
        min_samples_split = self.min_samples_split
        min_samples_leaf = self.min_samples_leaf
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
        """Predict class or regression value for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y: array-like of shape (n_samples, )
            The predicted classes, or the predict values.
        """
        # !TODO check the model is fitted
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            node = self.tree_.root
            while node.label is None:
                selected_feature = node.feature
                feature = X[i, selected_feature]
                # judge whether is discrete or continuous variable
                is_integer = issubclass(feature.dtype.type, np.integer)
                is_left = (feature == node.threshold) if is_integer else(
                    feature <= node.threshold)
                node = node.children_left if is_left else node.children_right
            y_pred.append(node.label)
        return np.array(y_pred)

    def score(self, X, y):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, )
            The training target classes, or the target values.

        Returns
        -------
        y: array-like of shape (n_samples, )
            The predicted classes, or the predict values.
        """
        # !TODO check the model is fitted
        y_hat = self.predict(X)
        # Classification
        if self.objective == "classification":
            score = np.mean(y != y_hat)
        # Regression
        else:
            score = np.mean((y - y_hat) ** 2) / np.var(y)
        return score

# =============================================================================
# Public estimators
# =============================================================================


class DecisionTreeClassifier(BaseDecisionTree):
    """A decision tree classifier."""

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.,
        min_impurity_split=0
    ):
        objective = "classification"
        super().__init__(
            objective=objective,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split
        )


class DecisionTreeRegressor(BaseDecisionTree):
    """A decision tree regressor."""

    def __init__(
        self,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.,
        min_impurity_split=0
    ):
        objective = "regression"
        super().__init__(
            objective=objective,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split
        )
