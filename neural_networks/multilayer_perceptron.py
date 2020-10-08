from .utils.functions import *


class MultiLayerPerceptron(object):

    def __init__(self, hidden_layer_sizes=(100,), alpha=0.01, learning_rate="constant", learning_rate_init=0.001, max_iter=200, random_state=42, tol=1e-4, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

        self.loss_curve_ = []
        self.n_outputs_ = None

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = sigmoid(np.dot(activations[i], self.coefs_[i]) + self.intercepts_[i])
        return activations

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """Compute the MLP corresponding derivatives with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the activations of the i + 1 layer and the backpropagated error. More specifically, deltas are gradients of loss with respect to z in each layer, where z = wx + b is the value of a particular layer before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the intercept parameters of the ith layer in an iteration.

        Returns
        -------
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        # Compute the deltas
        deltas[-1] = activations[-1] * (1 - activations[-1]) * (activations[-1] - y)
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = activations[-1] * (1 - activations[-1]) * np.dot(deltas[i], self.coefs_[i].T)

        # Compute the coef gradients
        for i in range(self.n_layers_ - 1):
            coef_grads[i] = np.dot(activations[i].reshape(-1, 1), deltas[i].reshape(1, -1))
            intercept_grads[i] = deltas[i]

        return coef_grads, intercept_grads

    def _init_coef(self, fan_in, fan_out):
        # Use the initialization method recommended by Glorot et al.
        factor = 6.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        rng = np.random.RandomState(self.random_state)
        coef_init = rng.uniform(-init_bound, init_bound, (fan_in, fan_out))
        intercept_init = rng.uniform(-init_bound, init_bound, fan_out)

        return coef_init, intercept_init

    def _initialize(self, layer_units):
        # Initialize parameters
        self.n_iter_ = 0
        self.best_loss_ = np.inf

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i], layer_units[i + 1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : returns a trained simple-layer feedforward neural networks.
        """
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes)
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0, got %s." % self.alpha)
        if self.learning_rate not in ["constant", "invscaling", "adaptive"]:
            raise ValueError("learning rate %s is not supported. " % self.learning_rate)
        if self.learning_rate in ["constant", "invscaling", "adaptive"] and self.learning_rate_init <= 0.0:
            raise ValueError("learning_rate_init must be > 0, got %s." % self.learning_rate)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_iter)
        if self.tol <= 0.0:
            raise ValueError("tol must be > 0, got %s." % self.tol)

        n_samples, n_features = X.shape
        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes + [self.n_outputs_])

        # First time training the model
        self._initialize(layer_units)

        # Initialize lists
        activations = [None] * len(layer_units)
        deltas = [None] * (len(activations) - 1)
        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]

        # Standard BP Algorithm
        self._fit_stochastic(X, y, activations, deltas, coef_grads, intercept_grads)
        return self

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads, intercept_grads):

        # Stochastic Gradient Descent

        # !TODO early stopping

        n_samples = X.shape[0]

        for it in range(self.max_iter):
            accumulated_loss = 0.0

            for k in range(n_samples):
                # compute the hidden layers and output layer
                activations[0] = X[k]
                activations = self._forward_pass(activations)

                # compute the loss
                accumulated_loss += ((y[k] - activations[-1]) ** 2).mean() / 2

                # compute the gradient of weights
                coef_grads, intercept_grads = self._backprop(X[k], y[k], activations, deltas, coef_grads, intercept_grads)
                
                # update weights
                for i in range(self.n_layers_ - 1):
                    self.coefs_[i] -= coef_grads[i]
                    self.intercepts_[i] -= intercept_grads[i]

            self.n_iter_ += 1

            loss_ = accumulated_loss / n_samples
            self.loss_curve_.append(loss_)

            if self.verbose:
                print("Iteration %d, loss = %.8f" % (self.n_iter_, loss_))

            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

            if self.loss_curve_[-1] <= self.best_loss_ - self.tol:
                break

    def predict(self, X):

        n_samples = X.shape[0]

        # Initialize layers
        activations = [None] * 3
        y_pred = []

        # forward propagate
        for k in range(n_samples):
            activations[0] = X[k]
            self._forward_pass(activations)
            y_pred.append(activations[-1])

        return y_pred
