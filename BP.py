import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((layers[len(layers) - 2] + 1, layers[len(layers) - 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        X = np.array(X)
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)
        for k in range(epochs):
            for i in range(X.shape[0]):
                predict = [X[i]]
                for l in range(len(self.weights)):
                    predict.append(logistic(np.dot(predict[l], self.weights[l])))
                error = y[i] - predict[-1]
                deltas = [error * logistic_derivative(predict[-1])]
                for l in range(len(predict) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * logistic_derivative(predict[l]))
                deltas.reverse()
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(predict[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, X):
        X = np.array(X)
        temp = np.ones(X.shape[0] + 1)
        temp[0:-1] = X
        predict = temp
        for i in range(0, len(self.weights)):
            predict = logistic(np.dot(predict, self.weights[i]))
        return predict

    def predictBatch(self, x):
        predict = np.column_stack((x, np.ones([x.shape[0], 1])))
        for i in range(0, len(self.weights)):
            predict = logistic(np.dot(predict, self.weights[i]))
        return predict