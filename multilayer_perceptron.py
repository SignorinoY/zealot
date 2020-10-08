from neural_networks.multilayer_perceptron import MultiLayerPerceptron
import numpy as np

mlp = MultiLayerPerceptron()

X = np.array([[0], [1]])
y = np.array([[0], [1]])

mlp.fit(X,y)

print(mlp.predict(np.array([0])))