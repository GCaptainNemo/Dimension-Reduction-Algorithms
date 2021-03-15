#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/11 18:58 


import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import make_moons, make_regression


class LDA:
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
            # self.X_data = self.X_data - np.mean(self.X_data, 0)
            y = np.array([1 for _ in range(50)])
            new_y = np.array([0 for _ in range(50)])
            self.Y_data = np.concatenate([y, new_y], axis=0)
            print(self.X_data.shape)
        else:
            self.X_data, self.Y_data = make_moons(100, noise=.04, random_state=0)

    def cal_within(self):
        """
        calculate within-class scatter matrix
        :return: Sw
        """
        class_name = np.unique(self.Y_data)
        class_num = len(np.unique(self.Y_data))
        attribute_dimension = self.X_data.shape[1]
        Sw = np.zeros([attribute_dimension, attribute_dimension])
        for i in range(class_num):
            xi = self.X_data[np.where(self.Y_data == class_name[i])[0], :]
            miui = np.array([np.mean(xi, axis=0)])
            print(xi.shape)
            Sw += xi.T @ xi
            Sw -= xi.shape[0] * miui.T @ miui
        return Sw

    def cal_between(self, Sw):
        """
        calculate class-between scatter matrix
        Sb = St - Sw
        :return: Sb
        """
        St = self.X_data.T @ self.X_data
        miu = np.array([np.mean(self.X_data, axis=0)])
        St = St - self.X_data.shape[0] * miu.T @ miu
        Sb = St - Sw
        # miu0 = np.array([np.mean(self.X_data[np.where(self.Y_data == 0)[0], :], axis=0)])
        # miu1 = np.array([np.mean(self.X_data[np.where(self.Y_data == 1)[0], :], axis=0)])
        # print("miu0.shape = ", miu0.shape)
        # Sb = (miu0 - miu1).T @(miu0 - miu1)
        return Sb

    def LDA(self):
        Sw = self.cal_within()
        Sb = self.cal_between(Sw)
        eigval, eig_vec = scipy.linalg.eig(Sb, Sw)
        eigval = eigval.real
        index_sorted = np.argsort(eigval)[-self.reduced_dimension:]
        self.Projective_matrix = eig_vec[:, index_sorted]
        mat = np.mat(self.X_data).T
        self.new_mat = self.Projective_matrix.T @ mat

    def result(self, regression=False):
        if regression == False:
            translate_vector = [- self.Projective_matrix[1, 0] * 2, self.Projective_matrix[0, 0] * 2]
        else:
            translate_vector = [- self.Projective_matrix[1, 0] * 400, self.Projective_matrix[0, 0] * 400]

        for i in range(self.new_mat.shape[1]):
            if self.Y_data[i] == 1:
                plt.scatter(self.new_mat[0, i] * self.Projective_matrix[0, 0] + translate_vector[0],
                            self.new_mat[0, i] * self.Projective_matrix[1, 0] + translate_vector[1], c='r')
            else:
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
                    arrowprops=dict(arrowstyle="->", color="k"))
        if regression:
            plt.text(translate_vector[0] * 0.9, translate_vector[1] * 0.95 + 10,
                     r'projection', fontdict={'size': 8, 'color': 'k'})
        else:
            plt.text(translate_vector[0] * 0.95 + 0.05, translate_vector[1] * 0.95,
                     r'projection', fontdict={'size': 8, 'color': 'k'})
        plt.show()


if __name__ == "__main__":
    a = LDA(1)
    a.make_data()
    # a.make_data(True)

    a.LDA()
    a.result()
    # a.result(True)




