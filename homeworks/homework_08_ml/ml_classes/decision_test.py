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
        self.max_depth = max_depth or np.inf
        self.min_leaf_size = min_leaf_size
        self.max_leaf_number = max_leaf_number
        self.min_inform_criter = min_inform_criter or np.inf
        self.tree = None
        self.depth = 0

    def find_entropy(self, y, targets):
        eps = 1e-6
        entropy = 0
        for value in targets:
            fraction = np.sum((y == value)) / y.shape[0] + eps
            entropy += -fraction * np.log2(fraction)
        return entropy

    def find_feature_entropy(self, y, feature, th):
        targets = np.unique(y)
        left = feature[feature <= th]
        right = feature[feature > th]
        feature_entropy = self.find_entropy(
            y[:(len(left))], targets) + self.find_entropy(y[len(left):], targets)
        return feature_entropy

    def get_best_split(self, X, y):
        info_gain = []
        targets = np.unique(y)
        sorted_vals = np.array(sorted(X, key=lambda x: x[1]))
        best_score = np.inf
        for feature in range(X.shape[1]):
            tmp = np.hstack([X, y[:, np.newaxis]])
            sorted_vals = np.array(sorted(tmp, key=lambda x: x[feature]))

            for th in sorted_vals[:, feature]:
                entropy = self.find_entropy(y, targets)
                feature_entropy = self.find_feature_entropy(
                    sorted_vals[:, -1], sorted_vals[:, feature], th)
                curr_info = entropy - feature_entropy
                info_gain.append(curr_info)

                if curr_info == max(info_gain):
                    best_score = curr_info
                    best_th = th
                    best_feature = feature

                if best_score < self.min_inform_criter:
                    return best_feature, best_th

        return best_feature, best_th

    def build_tree(self, X, y, root=False):
        if root:
            self.cls_counts = np.unique(y)

        feature, treshold = self.get_best_split(X, y)

        if self.tree is None:
            tree = dict()
            tree[feature] = dict()

        while self.depth < self.max_depth:
            left_X, left_y, right_X, right_y = self.get_child(
                X, y, feature, treshold)

            left_cl_value, left_counts = np.unique(left_y, return_counts=True)
            right_cl_value, right_counts = np.unique(
                right_y, return_counts=True)

            if len(left_counts) == 1:
                tree[feature][str(treshold) + ' left'] = left_cl_value[0]
            else:
                tree[feature][str(treshold) +
                              ' left'] = self.build_tree(left_X, left_y)

            if len(right_counts) == 1:
                tree[feature][str(treshold) + ' right'] = right_cl_value[0]
            else:
                tree[feature][str(treshold) +
                              ' right'] = self.build_tree(right_X, right_y)
            self.depth += 1

        return tree

    def get_child(self, X, y, feature, th):
        tmp = np.hstack([X, y[:, np.newaxis]])
        left = tmp[tmp[:, feature] <= th]
        right = tmp[tmp[:, feature] > th]
        left_X, left_y = left[:, :-1], left[:, -1]
        right_X, right_y = right[:, :-1], right[:, -1]
        return left_X, left_y, right_X, right_y

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
X, y = make_classification(n_features=4, random_state=42)
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X, y)
print('ok')
