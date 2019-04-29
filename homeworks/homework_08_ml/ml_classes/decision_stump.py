#!/usr/bin/env python
# coding: utf-8
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class DecisionStumpRegressor:
    '''
    Класс, реализующий решающий пень (дерево глубиной 1)
    для регрессии. Ошибку считаем в смысле MSE
    '''

    def __init__(self):
        '''
        Мы должны создать поля, чтобы сохранять наш порог th и ответы для
        x <= th и x > th
        '''
        self.th = 0

    def fit(self, X, y):
        '''
        метод, на котором мы должны подбирать коэффициенты th, y1, y2
        :param X: массив размера (1, num_objects)
        :param y: целевая переменная (1, num_objects)
        :return: None
        '''
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.get_best_split()

    def get_best_split(self):
        sorted_values = np.array(sorted(self.X[:, 0]))
        best_mse = np.inf
        for idx in range(len(sorted_values) - 1):
            self.th = self.X[idx, 0]
            right = sorted_values[sorted_values > self.th]
            left = sorted_values[sorted_values <= self.th]
            left_mse = len(left) / self.n * mse(self.y[:(idx + 1)], np.array(
                [np.average(self.y[:(idx + 1)])] * (idx + 1)))
            right_mse = len(right) / self.n * mse(self.y[idx:], np.array(
                [np.average(self.y[idx:])] * ((self.n - idx) or 1)))
            current = (left_mse + right_mse)
            if current <= best_mse:
                best_mse = current
                self.best_idx = idx

        self.th = self.X[self.best_idx, 0]

    def predict(self, X):
        '''
        метод, который позволяет делать предсказания для новых объектов
        :param X: массив размера (1, num_objects)
        :return: массив, размера (1, num_objects)
        '''
        if X[0] <= self.th:
            return(np.average(self.y[:self.best_idx]))
        else:
            return(np.average(self.y[self.best_idx:]))


X = np.arange(1, 100, 1)[:, np.newaxis]
y = (np.cos(X))
clf = DecisionStumpRegressor()
clf.fit(X, y)
ok = clf.predict([1])
print(ok)
print('ok')

clf1 = DecisionTreeRegressor(max_depth=1)
clf1.fit(X, y)
ok = clf1.predict(np.array([[1]]))
print(ok)
print('ok')
