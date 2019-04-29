#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd


class KNNRegressor:
    """
    Построим регрессию с помощью KNN. Классификацию писали на паре
    """

    def __init__(self, n):
        '''
        Конструктор
        :param n: число ближайших соседей, которые используются
        '''
        self.n = n

    def fit(self, X, y):
        '''
        :param X: обучающая выборка, матрица размерности (num_obj, num_features)
        :param y: целевая переменная, матрица размерности (num_obj, 1)
        :return: None
        '''
        self.X = X
        self.y = y

    def predict(self, X):
        '''
        :param X: выборка, на которой хотим строить предсказания (num_test_obj, num_features)
        :return: вектор предсказаний, матрица размерности (num_test_obj, 1)
        '''
        y = []
        assert len(X.shape) == 2
        for t in X[:, :2]:
            # Посчитаем расстояние от всех элементов в тренировочной выборке
            # до текущего примера -> результат - вектор размерности трейна
            d = np.sqrt(np.sum(np.square(t - self.X[:, :2]), axis=1))
            # Возьмем индексы n элементов, расстояние до которых минимально
            # результат -> вектор из n элементов
            idx = np.argsort(d)[:self.n]
            # TODO
            prediction = np.sum(self.y[idx]) / self.n
            y.append(prediction)
        return y


X, y = make_classification(n_samples=1000, n_classes=2,
                           n_informative=14, random_state=43)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


clf = KNNRegressor(n=3)
clf.fit(X_train, y_train)
clf.predict(X_test)
