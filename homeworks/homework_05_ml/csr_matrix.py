#!/usr/bin/env python
# coding: utf-8


import numpy as np


class CSRMatrix:
    """
    CSR (2D) matrix.
    Here you can read how CSR sparse matrix works: https://en.wikipedia.org/wiki/Sparse_matrix
    """

    def __init__(self, init_matrix_representation):
        """
        :param init_matrix_representation: can be usual dense matrix
        or
        (row_ind, col, data) tuple with np.arrays,
            where data, row_ind and col_ind satisfy the relationship:
            a[row_ind[k], col_ind[k]] = data[k]
        """
        if isinstance(init_matrix_representation, tuple) and len(
                init_matrix_representation) == 3:
            self._initialize_with_tuple(init_matrix_representation)
        elif isinstance(init_matrix_representation, np.ndarray):
            self._initialize_with_matrix(init_matrix_representation)
        else:
            raise ValueError

    def _initialize_with_tuple(self, tuple):
        rows = tuple[0]
        cols = tuple[1]
        values = tuple[2]
        counts = np.unique(rows, return_counts=True)[1].cumsum()
        self.A = np.array([])
        self.IA = np.insert(counts, 0, 0)
        self.JA = np.array([], dtype='int64')
        for row, col, value in sorted(zip(rows, cols, values)):
            self.A = np.append(self.A, value)
            self.JA = np.append(self.JA, col)
        shape = sorted(zip(rows, cols, values))[-1][:2]
        self.shape = (shape[0] + 1, shape[1] + 1)

    def _initialize_with_matrix(self, matrix):
        self.shape = matrix.shape
        self.A = matrix[matrix != 0]
        self.IA = np.array([0, ])
        self.JA = np.array([], dtype='int64')
        for ix_row in np.arange(0, self.shape[0]):
            nonzero = 0
            for ix_col in np.arange(0, self.shape[1]):
                if matrix[ix_row, ix_col] != 0:
                    nonzero += 1
                    self.A = np.append(self.A, matrix[ix_row, ix_col])
                    self.JA = np.append(self.JA, ix_col)
            self.IA = np.append(self.IA, self.IA[ix_row] + nonzero)

    def get_item(self, i, j):
        """
        Return value in i-th row and j-th column.
        Be careful, i and j may have invalid values (-1 / bigger that matrix size / etc.).
        """
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            for ix in np.arange(self.IA[i], self.IA[i + 1]):
                if self.JA[ix] == j:
                    return self.A[ix]
        else:
            return None
        return 0

    def set_item(self, i, j, value):
        """
        Set the value to i-th row and j-th column.
        Be careful, i and j may have invalid values (-1 / bigger that matrix size / etc.).
        """
        if not(0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise KeyError("Invalid coords")

        tmp = self.IA[i]
        if value != 0:
            self.A = np.insert(self.A, tmp, value)
            self.IA[i + 1:] += 1
            self.JA = np.insert(self.JA, tmp, j)
        else:
            for ix in np.arange(self.IA[i], self.IA[i + 1]):
                if self.JA[ix] == j:
                    self.IA[i + 1:] -= 1
                    self.JA = np.delete(self.JA, tmp, j)
                    self.A = np.delete(self.A, tmp, j)
                tmp = ix

    def to_dense(self):
        """
        Return dense representation of matrix (2D np.array).
        """
        result = np.zeros(self.shape)
        for i in np.arange(0, self.shape[0]):
            for j in np.arange(self.IA[i], self.IA[i + 1]):
                k = self.JA[j]
                result[i][k] = self.A[j]
        return result
