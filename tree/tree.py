import numpy as np


class Node(object):

    def __init__(self, n_samples, impurity, feature=None, threshold=None, children_left=None, children_right=None, label=None):
        # 所以节点均具有的属性
        self.n_samples = n_samples
        self.impurity = impurity

        # 切割节点具有的属性
        self.feature = feature
        self.threshold = threshold
        self.children_left = children_left
        self.children_right = children_right

        # 叶子节点具有的属性
        self.label = label


class Tree(object):

    def __init__(self):
        self.root = None

    def __str__(self) -> str:
        stack = [(self.root, 0)]
        i = 0
        _str = ""
        while len(stack):
            node, depth = stack.pop()
            if node.children_left is None:
                _str += "{space}node {node_id} is a leaf node with label {node_label}\n".format(
                    space=(depth)*'\t', node_id=i, node_label=node.label)
            else:
                _str += "{space}node {node_id} is a split node on feature {node_feature} on threshold {node_threshold}\n" \
                    .format(space=(depth)*'\t', node_id=i, node_feature=node.feature, node_threshold=node.threshold)
                stack.append((node.children_left, depth + 1))
                stack.append((node.children_right, depth + 1))
            i += 1
        return _str


class TreeBuilder(object):

    def __init__(
        self,
        criterion,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        min_impurity_decrease,
        min_impurity_split
    ):
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.min_impurity_decrease = min_impurity_decrease

    def build(self, tree, X, y):

        n_samples, n_features = X.shape

        # (selected_idxs, parent, depth, is_left, impurity)
        stack = [(np.ones(n_samples, dtype=bool), None, 0, 0, np.inf)]

        while len(stack):
            selected_idxs, parent, depth, is_left, impurity = stack.pop()

            n_samples = sum(selected_idxs)

            is_leaf = (
                n_samples < self.min_samples_split or
                n_samples < 2 * self.min_samples_leaf or
                depth >= self.max_depth or
                impurity <= self.min_impurity_split
            )

            if not is_leaf:
                # selected best split feature and threshold
                max_impurity_improvement = - np.inf
                for feature_idx in range(n_features):
                    feature = X[:, feature_idx]
                    is_integer = issubclass(feature.dtype.type, np.integer)
                    values = np.unique(feature[selected_idxs])
                    for threshold in values:
                        split_idxs = (feature == threshold) if is_integer \
                            else (feature <= threshold)
                        y_left = y[selected_idxs & split_idxs]
                        y_right = y[selected_idxs & ~split_idxs]
                        self.criterion.update(y_left, y_right)
                        impurity_improvement = self.criterion.impurity_improvement
                        if impurity_improvement > max_impurity_improvement:
                            max_impurity_improvement = impurity_improvement
                            selected_feature = feature_idx
                            selected_threshold = threshold

            is_leaf = (
                is_leaf or
                max_impurity_improvement < self.min_impurity_decrease
            )

            # Add node to the parent by is_first & is_left & is_leaf
            feature = X[:, selected_feature]
            is_integer = issubclass(feature.dtype.type, np.integer)
            split_idxs = (feature == selected_threshold) if is_integer \
                else(feature <= selected_threshold)
            y_left = y[selected_idxs & split_idxs]
            y_right = y[selected_idxs & ~split_idxs]
            self.criterion.update(y_left, y_right)
            impurity = self.criterion.node_impurity
            impurity_left, impurity_right = self.criterion.children_impurity
            label = self.criterion.node_value

            if is_leaf:
                node = Node(
                    n_samples=n_samples,
                    impurity=impurity,
                    label=label
                )
            else:
                node = Node(
                    n_samples=n_samples,
                    impurity=impurity,
                    feature=selected_feature,
                    threshold=selected_threshold
                )

            if parent is None:
                tree.root = node
            elif is_left:
                parent.children_left = node
            else:
                parent.children_right = node

            if not is_leaf:
                stack.append((selected_idxs & split_idxs, node,
                             depth + 1, 1, impurity_left))
                stack.append(
                    (selected_idxs & ~split_idxs, node, depth + 1, 0, impurity_right))
