#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd


class KNN:
    """
    simple KNN classifier
    """
    
    def __init__(self, n):
        self.n = n
        
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y = []
        assert len(X.shape) == 2
        for (h, w) in X:
            ### Посчитаем расстояние от всех элементов в тренировочной выборке
            # до текущего примера -> результат - вектор размерности трейна
            d = np.sqrt((h - self.X[:, 0]) ** 2 + (w - self.X[:, 1]) ** 2)
            ### Возьмем индексы n элементов, расстояние до которых минимально
            ### результат -> вектор из n элементов
            idx = np.argsort(d)[:self.n]
            ### Посчитаем частоту меток для каждого случая 
            ### результат -> вектор длинны 2 который покажет
            ### сколько соседей 0-го класса, сколько соседей 1-го класса.
            counts = np.bincount(self.y[idx])
            ### возьмем самый часто встречаемый в соседях класс.
            prediction = np.argmax(counts)
            y.append(prediction)
        return y

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
            prediction = None
            y.append(prediction)
        return y


df = pd.read_csv('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv', sep=';')
df = df[np.logical_and(df.age > 18, df.age < 50)] 
df_train = df.loc[:100]
df_test = df.loc[150:180]

X_train = df_train[['height', 'weight']].values
y_train = df_train['male'].values

X_test = df_test[['height', 'weight']].values

clf = KNN(n=5)
clf.fit(X_train, y_train)
clf.predict(X_test)

X, y = make_classification(n_samples=1000, n_classes=2, n_informative=14, random_state=43)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



clf = KNNRegressor(n=3)
clf.fit(X_train, y_train)
clf.predict(X_test)
