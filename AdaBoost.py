# -*- coding: utf-8 -*-
# 使用Adaboost算法，用多个弱分类器构造强分类器

from BP import NeuralNetwork
import numpy as np
from compiler.ast import flatten


class BP_Adaboost:
    def __init__(self, layers, size=25):
        self.nets = [NeuralNetwork(layers=layers) for i in range(size)]
        self.D = []
        self.alpha_list = []

    def train(self, X, y):
        error_rate = []
        weak_predict = []
        self.D.append(np.ones(X.shape[0]) / X.shape[0])
        for i in range(len(self.nets)):
            # print i
            self.nets[i].fit(X, y)
            weak_predict.append(self.nets[i].predictBatch(X))
            std = (weak_predict[i]-y).std(axis=0)
            error_rate.append(np.sum(np.transpose(abs(weak_predict[i]-y)>(std/2)) * self.D[-1]))
            delta = 1e-6
            self.alpha_list.append(0.5 * np.log(1. / (error_rate[i] + delta) - 1))
            self.D.append(self.D[-1] * flatten((np.exp(-self.alpha_list[i] * weak_predict[i] * y)).tolist()))
            Dsum = np.sum(self.D[-1])
            self.D[-1] /= Dsum

    def predict(self, X):
        X = np.array(X)
        weak_predict = []
        alphaSum = np.sum(self.alpha_list)
        self.alpha_list /= alphaSum
        predict = np.zeros((X.shape[0], 1))
        for i in range(len(self.nets)):
            weak_predict.append(self.nets[i].predictBatch(X).tolist())
            weak_predict[i] = np.array(weak_predict[i])
            predict += self.alpha_list[i]*weak_predict[i]
        return predict
