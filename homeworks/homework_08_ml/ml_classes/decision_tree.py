#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


class DecisionTreeClassifier:
    '''
    Пишем свой велосипед - дерево для классификации
    '''

    def __init__(
            self,
            max_depth=None,
            min_leaf_size=None,
            max_leaf_number=None,
            min_inform_criter=None):
        '''
        Инициализируем наше дерево
        :param max_depth: один из возможных критерием останова - максимальная глубина дерева
        :param min_leaf_size: один из возможных критериев останова - число элементов в листе
        :param max_leaf_number: один из возможных критериев останова - число листов в дереве.
        Нужно подумать как нам отобрать "лучшие" листы
        :param min_inform_criter: один из критериев останова - процент прироста информации, который
        считаем незначительным
        '''
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.max_leaf_number = max_leaf_number
        self.min_inform_criter = min_inform_criter
        self.tree = None
        self.cur_depth = 0

    def find_entropy(self, X, y):
        entropy = 0
        values = np.unique(y)
        for value in values:
            fraction = np.sum((y == value)) / y.shape[0]
            entropy += -fraction * np.log2(fraction)
        return entropy

    def find_entropy_attribute(self, y, factor):
        target_variables = np.unique(y)
        variables = np.unique(factor)
        entropy2 = 0
        eps = 1e-5
        for variable in variables:
            entropy = 0
            for target_variable in target_variables:
                num = np.sum((factor == variable).astype(int) *
                             (y == target_variable).astype(int))
                den = np.sum(factor == variable)
                fraction = num / (den + eps)
                entropy += -fraction * np.log(fraction + eps)
            fraction2 = den / y.shape[0]
            entropy2 += -fraction2 * entropy
        return abs(entropy2)

    def get_best_split(self, X, y):
        info_gain = []
        for factor in X.T:
            info_gain.append(
                self.find_entropy(
                    X,
                    y) -
                self.find_entropy_attribute(
                    y,
                    factor))
        return np.argmax(info_gain)

    def build_tree(self, X, y, root=False):
        if root:
            self.cls_counts = np.unique(y)

        node = self.get_best_split(X, y)

        attValue = np.unique(X[:, node])

        if self.tree is None:
            tree = dict()
            tree[node] = dict()

        for value in attValue:
            child_X, child_y = self.get_child(X, y, node, value)
            clValue, counts = np.unique(child_y, return_counts=True)

            if len(counts) == 1:
                tree[node][value] = clValue[0]
            else:
                tree[node][value] = self.build_tree(child_X, child_y)

        return tree

    def get_child(self, X, y, node, value):
        tmp = np.hstack([X, y[:, np.newaxis]])
        tmp = tmp[tmp[:, node] == value]
        X, y = tmp[:, :-1], tmp[:, -1]
        return X, y

    def fit(self, X, y):
        '''
        Стендартный метод обучения
        :param X: матрица объекто-признаков (num_objects, num_features)
        :param y: матрица целевой переменной (num_objects, 1)
        :return: None
        '''
        self.tree = self.build_tree(X, y, root=True)

    def predict(self, X, tree=None):
        '''
        Метод для предсказания меток на объектах X
        :param X: матрица объектов-признаков (num_objects, num_features)
        :return: вектор предсказаний (num_objects, 1)
        '''
        if not tree:
            tree = self.tree
        for node in tree.keys():
            value = X[node]
            tree = tree[node][value]

            if isinstance(tree, dict):
                prediction = self.predict(X, tree)
            else:
                prediction = tree
                break
        return prediction

    def predict_proba(self, X):
        '''
        метод, возвращающий предсказания принадлежности к классу
        :param X: матрица объектов-признаков (num_objects, num_features)
        :return: вектор предсказанных вероятностей (num_objects, 1)
        '''
        for i in range(len(self.cls_counts)):
            pass


"""
dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
       'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}

df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])

X = np.array(df.values.tolist())[:, :-1]
y = np.array(df.values.tolist())[:, -1]

tree = DecisionTreeClassifier()
tree.fit(X, y)
prediction = tree.predict(X[6])
print('ok')
"""
X, y = make_classification(n_features=4)
tree = DecisionTreeClassifier()
tree.fit(X, y)
print('ok')
