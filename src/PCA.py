#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/11 18:58 


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons


class PCA:
    def __init__(self, k):
        """
        :param k: reduced dimension R^d -> R^k
        """
        self.reduced_dimension = k

    def make_data(self):
        self.X_data, self.Y_data = make_moons(100, noise=.04, random_state=0)
        self.X_data = self.X_data - np.mean(self.mat, 0)
        one_index = np.where(self.Y_data == 1)
        zero_index = np.where(self.Y_data == 0)
        plt.scatter(self.X_data[one_index, 0], self.X_data[one_index, 1], c='r')
        plt.scatter(self.X_data[zero_index, 0], self.X_data[zero_index, 1], c='b')
        plt.show()

    def PCA(self):
        """ bottom-to-top clustering """
        mat = np.mat(self.X_data).T
        centeralized_mat = mat - np.mean(mat, 1)
        u, sigma, v = np.linalg.svd(centeralized_mat)
        self.Projective_matrix = u[:, 0:self.reduced_dimension]
        self.new_mat = self.Projective_matrix.T @ centeralized_mat

    def result(self):
        for i in range(self.new_mat.shape[1]):
            if self.Y_data[i] == 1:
                plt.scatter(self.new_mat[0, i], 0, c='r')
                plt.scatter(self.new_mat[0, i] * self.Projective_matrix[0, 0],
                            self.new_mat[0, i] * self.Projective_matrix[1, 0], c='r')
            else:
                plt.scatter(self.new_mat[0, i], 0, c='b')
                plt.scatter(self.new_mat[0, i] * self.Projective_matrix[0, 0],
                            self.new_mat[0, i] * self.Projective_matrix[1, 0], c='b')
        ax = plt.gca()
        ax.axis("equal")
        plt.show()


if __name__ == "__main__":
    a = PCA(1)
    a.make_data()
    a.PCA()
    a.result()
    # a.prediction()


