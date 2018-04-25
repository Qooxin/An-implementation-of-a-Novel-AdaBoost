import numpy as np

class ELM:
    def __init__(self, n, C, random=2):
        if random:
            np.random.seed(random)
        self.n = n
        self.C = C
        self.a = None
        self.b = None
        self.beta = None

    def large_fit(self, hypo, y):
        features = hypo.shape[1] if len(hypo.shape) > 1 else 1
        self.a = np.random.rand(self.n, features) * 2 - 1
        self.b = np.random.rand(self.n, features) * 2 - 1
        self.beta = self.large_compute_beta(hypo, y)

    def notLarge_fit(self, hypo, y):
        H = np.zeros((hypo.shape[0], self.n))
        features = hypo.shape[1] if len(hypo.shape) > 1 else 1
        self.a = np.random.rand(self.n, features) * 2 - 1
        self.b = np.random.rand(self.n, features) * 2 - 1
        self.beta = self.notLarge_compute_beta(hypo, y)

    def predict(self, hypo):
        return np.dot(self.compute_H(hypo), self.beta)

    def large_compute_beta(self, hypo, y):
        H = self.compute_H(hypo)
        inv = np.linalg.pinv(np.dot(H.T, H) + np.eye(self.n)*self.C)
        t = np.dot(inv, H.T)
        return np.dot(t, y)

    def notLarge_compute_beta(self, hypo, y):
        H = self.compute_H(hypo)
        inv = np.linalg.pinv(np.dot(H.T, H) + np.eye(self.n)*self.C)
        t = np.dot(H.T, inv)
        return np.dot(t, y)

    def compute_H(self, hypo):
        H = np.zeros((hypo.shape[0], self.n))
        for i in range(hypo.shape[0]):
            H[i] = self.sigmoid(self.a.T*hypo[i]+self.b.T)
        return H

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))