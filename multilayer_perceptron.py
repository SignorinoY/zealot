from neural_networks.multilayer_perceptron import MultiLayerPerceptron
import numpy as np

MLP = MultiLayerPerceptron(hidden_layer_sizes=(3,), verbose=True, max_iter=1000)

X = np.array([[0., 0.], [1., 1.]])
y = np.array([[0], [1]])

MLP.fit(X,y)

print(MLP.predict(X))