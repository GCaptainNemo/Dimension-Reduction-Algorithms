#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/11 18:58 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_regression


class PCA:
    def __init__(self, k):
        """
        :param k: reduced dimension R^d -> R^k
        """
        self.reduced_dimension = k

    def make_data(self, regression=False):
        if regression:
            x, y = make_regression(n_samples=50, n_features=1,
                                   n_targets=1, noise=1.5, random_state=1, bias=0)
            y = np.array([y]).T
            print(y.shape)
            self.X_data = np.concatenate([x, y], axis=1)
            new_X_data = self.X_data - np.array([[15, 13]])

            self.X_data = np.concatenate([self.X_data, new_X_data], axis=0)
            self.X_data = self.X_data - np.mean(self.X_data, 0)
            y = np.array([1 for _ in range(50)])
            new_y = np.array([0 for _ in range(50)])
            self.Y_data = np.concatenate([y, new_y], axis=0)
            print(self.X_data.shape)
        else:
            self.X_data, self.Y_data = make_moons(100, noise=.04, random_state=0)
            self.X_data = self.X_data - np.mean(self.X_data, 0)

    def PCA(self):
        """ bottom-to-top clustering """
        mat = np.mat(self.X_data).T
        centeralized_mat = mat - np.mean(mat, 1)
        u, sigma, v = np.linalg.svd(centeralized_mat)
        self.Projective_matrix = u[:, 0:self.reduced_dimension]
        print(self.Projective_matrix)
        self.new_mat = self.Projective_matrix.T @ centeralized_mat

    def result(self, regression=False):
        if regression == False:
            translate_vector = [- self.Projective_matrix[1, 0], self.Projective_matrix[0, 0]]
        else:
            translate_vector = [- self.Projective_matrix[1, 0] * 100, self.Projective_matrix[0, 0] *100]

        for i in range(self.new_mat.shape[1]):
            if self.Y_data[i] == 1:
                # plt.scatter(self.new_mat[0, i], 0, c='r')
                plt.scatter(self.new_mat[0, i] * self.Projective_matrix[0, 0] + translate_vector[0],
                            self.new_mat[0, i] * self.Projective_matrix[1, 0] + translate_vector[1], c='r')
            else:
                # plt.scatter(self.new_mat[0, i], 0, c='b')
                plt.scatter(self.new_mat[0, i] * self.Projective_matrix[0, 0] + translate_vector[0],
                            self.new_mat[0, i] * self.Projective_matrix[1, 0] + translate_vector[1], c='b')
        one_index = np.where(self.Y_data == 1)
        zero_index = np.where(self.Y_data == 0)
        plt.scatter(self.X_data[one_index, 0], self.X_data[one_index, 1], c='r')
        plt.scatter(self.X_data[zero_index, 0], self.X_data[zero_index, 1], c='b')
        ax = plt.gca()
        ax.axis("equal")
        ax.annotate("",
                    xy=(translate_vector[0], translate_vector[1]),
                    xytext=(0, 0),
                    # xycoords="figure points",
                    arrowprops=dict(arrowstyle="->", color="k"))
        if regression:
            plt.text(translate_vector[0] * 0.9, translate_vector[1] * 0.95 + 10,
                     r'projection', fontdict={'size': 8, 'color': 'k'})
        else:
            plt.text(translate_vector[0] * 0.95 + 0.05, translate_vector[1] * 0.95,
                     r'projection', fontdict={'size': 8, 'color': 'k'})
        plt.show()


if __name__ == "__main__":
    a = PCA(1)
    a.make_data(True)
    a.PCA()
    a.result(True)


