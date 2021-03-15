#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/14 12:18 


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons


class KPCA:
    def __init__(self, k):
        """
        :param k: reduced dimension R^d -> R^k
        """
        self.reduced_dimension = k

    def make_data(self):
        self.X_data, self.Y_data = make_circles(n_samples=100, shuffle=True, noise=0.1, random_state=None, factor=0.2)
        # self.X_data, self.Y_data = make_moons(100, noise=.04, random_state=0)
        self.X_data = self.X_data - np.mean(self.X_data, 0)

    def KPCA(self):
        # kernel_function = lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2) ** 2 * 0.5)
        kernel_function = lambda x1, x2: (x1 @ x2 + 1) ** 3
        kernel_matrix = self.construct_kernel_matrix(kernel_function)
        # 中心化后的kernel matrix
        num = self.X_data.shape[0]
        K_star = kernel_matrix - 1 / num * kernel_matrix @ \
                 np.mat(np.ones([num, num]))
        eig_value, eig_vector = np.linalg.eig(K_star)
        eig_value = eig_value.real
        # 最大的d'个特征值对应特征向量
        sorted_index = np.argsort(-eig_value)[:self.reduced_dimension]
        print(eig_value[sorted_index])
        print(max(eig_value))
        U = eig_vector[:, sorted_index].real
        diag = np.diag(1 / np.sqrt(np.diag(U.T @ kernel_matrix @ U)))
        self.new_mat = diag @ U.T @ K_star

    def construct_kernel_matrix(self, kernel_function=None):
        if not kernel_function:
            return
        num = self.X_data.shape[0]
        kernel_matrix = np.zeros([num, num])
        for i in range(num):
            for j in range(i, num):
                kernel_matrix[i, j] = kernel_function(self.X_data[i, :],
                                                      self.X_data[j, :])
                kernel_matrix[j, i] = kernel_matrix[i, j]
        return kernel_matrix

    def result(self):
        plt.subplot(121)
        plt.title("Origin data")
        one_index = np.where(self.Y_data == 1)
        zero_index = np.where(self.Y_data == 0)
        plt.scatter(self.X_data[one_index, 0], self.X_data[one_index, 1], c='r')
        plt.scatter(self.X_data[zero_index, 0], self.X_data[zero_index, 1], c='b')

        #######################################
        plt.subplot(122)
        plt.title("KPCA dimension=1 kernel=polynomial")
        if self.reduced_dimension == 2:
            for i in range(self.new_mat.shape[1]):
                if self.Y_data[i] == 1:
                    plt.scatter(self.new_mat[0, i],
                                self.new_mat[1, i], c='r')
                else:
                    plt.scatter(self.new_mat[0, i],
                                self.new_mat[1, i], c='b')
        else:
            for i in range(self.new_mat.shape[1]):
                if self.Y_data[i] == 1:
                    plt.scatter(self.new_mat[0, i], 0, c='r')
                else:
                    plt.scatter(self.new_mat[0, i], 0, c='b')
        ax = plt.gca()
        ax.axis("equal")
        plt.show()


if __name__ == "__main__":
    a = KPCA(1)
    a.make_data()
    a.KPCA()
    a.result()





