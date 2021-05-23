import numpy as np


class Criterion(object):

    def update(self, y_left, y_right):
        pass

    @property
    def node_impurity(self):
        pass

    @property
    def children_impurity(self):
        pass

    @property
    def impurity_improvement(self):
        impurity_total = self.node_impurity
        impurity_left, impurity_right = self.children_impurity

        return impurity_total \
            - (self.n_left / self.n_total * impurity_left) \
            - (self.n_right / self.n_total * impurity_right)

    @property
    def node_value(self):
        pass


class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def update(self, y_left, y_right):
        """
        Parameters
        ----------
        y_left: array-like, dtype=int
            the sample y for the left child node
        y_right: array-like, dtype=int
            the sample y for the right child node
        """
        self.y_left = y_left
        self.y_right = y_right
        self.y_total = np.append(self.y_left, self.y_right)

        self.n_left = len(self.y_left)
        self.n_right = len(self.y_right)
        self.n_total = self.n_left + self.n_right
        # n_classes defination wrong
        self.n_classes = max(self.y_total) + 1

        # Count class frequency for each target (total, left and right)
        values_total, counts_total = np.unique(
            self.y_total, return_counts=True)
        values_left, counts_left = np.unique(self.y_left, return_counts=True)

        self.sum_total = np.zeros(self.n_classes, dtype=int)
        self.sum_left = np.zeros(self.n_classes, dtype=int)
        self.sum_right = np.zeros(self.n_classes, dtype=int)

        for value, count in zip(values_total, counts_total):
            self.sum_total[value] = count

        for value, count in zip(values_left, counts_left):
            self.sum_left[value] = count

        self.sum_right = self.sum_total - self.sum_left

    @property
    def node_value(self):
        return np.argmax(np.bincount(self.y_total))


class Entropy(ClassificationCriterion):

    @property
    def node_impurity(self):
        entropy = 0.0

        for c in range(self.n_classes):
            count = self.sum_total[c]
            if count > 0.0:
                count /= self.n_total
                entropy -= count * np.log2(count)

        return entropy

    @property
    def children_impurity(self):
        entropy_left = 0.0
        entropy_right = 0.0

        for c in range(self.n_classes):
            count = self.sum_left[c]
            if count > 0.0:
                count /= self.n_left
                entropy_left -= count * np.log2(count)

            count = self.sum_right[c]
            if count > 0.0:
                count /= self.n_right
                entropy_right -= count * np.log2(count)

        return entropy_left, entropy_right


class Gini(ClassificationCriterion):

    @property
    def node_impurity(self):
        gini = 1.0

        for c in range(self.n_classes):
            count = self.sum_total[c]
            if count > 0.0:
                p = count / self.n_total
                gini -= p * p

        return gini

    @property
    def children_impurity(self):
        gini_left = 1.0
        gini_right = 1.0

        for c in range(self.n_classes):
            count = self.sum_left[c]
            if count > 0.0:
                p = count / self.n_left
                gini_left -= p * p

            count = self.sum_right[c]
            if count > 0.0:
                p = count / self.n_right
                gini_right -= p * p

        return gini_left, gini_right


class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    def update(self, y_left, y_right):
        """
        Parameters
        ----------
        samples_left: array-like, dtype=int
            the sample idxs for the left child node
        samples_right: array-like, dtype=int
            the sample idxs for the right child node
        """
        self.y_left = y_left
        self.y_right = y_right
        self.y_total = np.append(self.y_left, self.y_right)

        self.n_left = len(self.y_left)
        self.n_right = len(self.y_right)
        self.n_total = self.n_left + self.n_right

        self.sum_left = np.sum(self.y_left)
        self.sum_right = np.sum(self.y_right)
        self.sum_total = self.sum_left + self.sum_right

    @property
    def node_value(self):
        return np.mean(self.y_total)


class MSE(RegressionCriterion):

    @property
    def node_impurity(self):
        sq_sum_total = np.sum(self.y_total ** 2)
        mse = (sq_sum_total / self.n_total) - \
            (self.sum_total / self.n_total) ** 2
        return mse

    @property
    def children_impurity(self):
        sq_sum_left = np.sum(self.y_left ** 2)
        sq_sum_right = np.sum(self.y_right ** 2)
        mse_left = (sq_sum_left / self.n_left) - \
            (self.sum_left / self.n_left) ** 2
        mse_right = (sq_sum_right / self.n_right) - \
            (self.sum_right / self.n_right) ** 2
        return mse_left, mse_right
