import numpy as np


class KFold(object):

    def __init__(self, n_splits=5, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y):
        """Generate indices to split data into training and test set.

        Parameters
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        for test_index in self._iter_test_indicies(X, y):
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_index] = True
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index

    def _iter_test_indicies(self, X, y):
        n_samples = X.shape[0]
        n_splits = self.n_splits

        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)

        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop
