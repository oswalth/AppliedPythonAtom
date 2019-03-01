#!/usr/bin/env python
# coding: utf-8
import copy


def minor(A, i, j):
    M = copy.deepcopy(A)
    del M[i]
    for i in range(len(A) - 1):
        del M[i][j]
    return M
    
def calculate_determinant(list_of_lists):
    '''
    Метод, считающий детерминант входной матрицы,
    если это возможно, если невозможно, то возвращается
    None
    Гарантируется, что в матрице float
    :param list_of_lists: список списков - исходная матрица
    :return: значение определителя или None
    '''

    n = len(list_of_lists)
    m = len(list_of_lists)
    det = 0
    if n != m:
        return None
    if n == 1:
        return list_of_lists[0][0]
    else:
        for j in range(m):
            det += list_of_lists[0][j] * (-1) ** j * calculate_determinant(minor(list_of_lists, 0, j))
        return det
    raise NotImplementedError
