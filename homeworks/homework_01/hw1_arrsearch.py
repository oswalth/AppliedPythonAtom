#!/usr/bin/env python
# coding: utf-8


def find_indices(input_list, n):
    '''
    Метод возвращает индексы двух различных
    элементов listа, таких, что сумма этих элементов равна
    n. В случае, если таких элементов в массиве нет,
    то возвращается None
    Ограничение по времени O(n)
    :param input_list: список произвольной длины целых чисел
    :param n: целевая сумма
    :return: tuple из двух индексов или None
    '''
    import copy
    for i in range(len(input_list)):
        tmp_list = copy.deepcopy(input_list)
        del tmp_list[i]
        if abs(n - input_list[i]) in tmp_list:
            return (input_list.index(input_list[i]), input_list.index(
                abs(n - input_list[i])))
    raise NotImplementedError
