import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split


class DecisionTreeClassifier:
    '''
    Пишем свой велосипед - дерево для классификации
    '''
    class Node:
        def __init__(self, condition):
            self.condition = condition

    class Leaf:
        def __init__(self, data):
            self.occurrences = {}
            self.classes, self.counts = np.unique(
                data[:, -1], return_counts=True)
            for index in range(len(self.classes)):
                self.occurrences[self.classes[index]] = self.counts[index]
            self.majority = max(self.occurrences, key=self.occurrences.get)

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
        self.min_leaf_size = min_leaf_size or 0
        self.max_leaf_number = max_leaf_number
        self.min_inform_criter = min_inform_criter or 0
        self.tree = None
        self.leaf_number = 0

    def fit(self, X, y):
        '''
        Стендартный метод обучения
        :param X: матрица объекто-признаков (num_objects, num_features)
        :param y: матрица целевой переменной (num_objects, 1)
        :return: None
        '''
        self.cls_count = np.unique(y)
        data = np.hstack([X, y[:, np.newaxis]])
        self.tree = self.build_tree(data)

    def get_potential_splits(self, data):

        X = data[:, :-1]

        potential_splits = {}
        n_columns = X.shape[1]
        for column_index in range(n_columns):
            potential_splits[column_index] = []
            values = X[:, column_index]
            unique_values = np.unique(values)

            for index in range(len(unique_values)):
                if index != 0:
                    curr_value = unique_values[index]
                    prev_value = unique_values[index - 1]
                    potential_split = (curr_value + prev_value) / 2

                    potential_splits[column_index].append(potential_split)

        return potential_splits

    @staticmethod
    def split_data(data, split_column, split_value):

        splitted = data[:, split_column]

        left = data[splitted <= split_value]
        right = data[splitted > split_value]

        return left, right

    @staticmethod
    def calculate_entropy(data):

        _, counts = np.unique(data[:, -1], return_counts=True)
        probabilities = counts / counts.sum()

        entropy = np.sum(probabilities * -np.log2(probabilities))
        return entropy

    def calculate_overall_entropy(self, left, right):

        data_size = len(left) + len(right)

        p_left = len(left) / data_size
        p_right = len(right) / data_size

        overall_entropy = (p_left * self.calculate_entropy(left)
                           + p_right * self.calculate_entropy(right))

        return overall_entropy

    def find_best_split(self, data, potential_splits):

        overall_entropy = 999
        best_column, best_value = 0, 0
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                left, right = self.split_data(data, column_index, value)
                curr_overall_entropy = self.calculate_overall_entropy(
                    left, right)

                if curr_overall_entropy < overall_entropy:
                    overall_entropy = curr_overall_entropy
                    best_column = column_index
                    best_value = value

                if overall_entropy < self.min_inform_criter:
                    return best_column, best_value

        return best_column, best_value

    @staticmethod
    def check_purity(data):

        classes = np.unique(data[:, -1])

        if len(classes) == 1:
            return True
        else:
            return False

    @staticmethod
    def classify(data):

        classes, cls_counts = np.unique(data[:, -1], return_counts=True)

        index = cls_counts.argmax()
        classification = classes[index]

        return classification

    def build_tree(self, data, depth=0):
        if self.check_purity(data) or (
                len(data) < self.min_leaf_size) or (
                depth == self.max_depth):
            return self.classify(data)

        else:
            depth += 1
            potential_splits = self.get_potential_splits(data)
            column, value = self.find_best_split(data, potential_splits)

            left, right = self.split_data(data, column, value)

            condition = "{} <= {}".format(column, value)
            node = self.Node(condition)

            if self.leaf_number >= self.max_leaf_number:
                return node
            node.left = self.build_tree(left, depth)
            if not isinstance(node.left, self.Node):
                node.left = self.Leaf(left)
                self.leaf_number += 1

            if self.leaf_number >= self.max_leaf_number:
                return node
            node.right = self.build_tree(right, depth)
            if not isinstance(node.right, self.Node):
                node.right = self.Leaf(right)
                self.leaf_number += 1

            return node

    def predict(self, X, sub_tree=None, inner=False):
        if not inner:
            predictions = []
            for element in X:
                predictions.append(self.predict(element, inner=True))

            return predictions
        else:
            tree = sub_tree or self.tree
            column, _, value = tree.condition.split()

            if X[int(column)] <= float(value):
                answer = tree.left
            else:
                answer = tree.right

            if isinstance(answer, self.Leaf):
                return answer.majority
            elif not isinstance(answer, self.Node):
                return answer

            return self.predict(X, answer, inner=True)

    def predict_proba(self, X, sub_tree=None, inner=False):
        if not inner:
            probabilities = []
            for element in X:
                probabilities.append(self.predict_proba(element, inner=True))

            for prob in probabilities:
                print(prob)

        else:
            tree = sub_tree or self.tree
            column, _, value = tree.condition.split()

            if X[int(column)] <= float(value):
                answer = tree.left
            else:
                answer = tree.right

            if not isinstance(answer, self.Node):
                probabilities = []
                for class_ in self.cls_count:
                    if class_ in answer.classes:
                        probabilities.append(
                            answer.occurrences[class_] / sum(answer.counts))
                    else:
                        probabilities.append(0)
                return probabilities

            return self.predict_proba(X, answer, inner=True)
