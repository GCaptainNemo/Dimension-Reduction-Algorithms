#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/14 13:41

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from scipy.spatial import KDTree
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering


class LLE:
    def __init__(self, k):
        """
        :param k: reduced dimension R^d -> R^k
        """
        self.reduced_dimension = k

    def make_data(self):
        """
        构造swiss roll数据
        """
        self.X_data, t = make_swiss_roll(10, noise=0, random_state=0)
        ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(self.X_data)
        self.Y_data = ward.labels_

    def construct_graph(self, k):
        """
        :param k: k-nearest neighbour
        :return: adjacency index matrix(每个数据点k个近邻点的索引)
        """
        kd_tree = KDTree(self.X_data.copy())
        n = self.X_data.shape[0]
        adjacency_index_matrix = np.ones([n, k])
        for i in range(n):
            dist_tuple, index_tuple = kd_tree.query(self.X_data[i, :], k=k+1, p=2)
            index_lst = list(index_tuple)
            index_lst.remove(i)
            adjacency_index_matrix[i, :] = np.array([index_tuple])
        return adjacency_index_matrix

    def lle(self, knn=5):
        """ knn=5(包括自身)"""
        adjacency_index_matrix = self.construct_graph(knn)
        n = self.X_data.shape[0]
        attribute_dimension_num = self.X_data.shape[1]
        W = np.zeros([n, n])  # 关于所有点的权值矩阵，不在近邻则权值为0
        for i in range(n):
            linjin_matrix = self.X_data[adjacency_index_matrix[i, :], :]
            Zi = linjin_matrix - self.X_data[i, :]
            covariance_matrix = Zi.T @ Zi
            # wi是wij组成的列向量，即关于节点i邻近点权重组成的向量
            wi = np.inv(covariance_matrix) @ np.ones(attribute_dimension_num, 1) / np.sum(covariance_matrix)
            for j in range(knn):
                W[i, adjacency_index_matrix[i, j]] = wi[j, 1]


    def result(self):
        fig = plt.figure()
        # plt.title("Origin data")
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        for l in np.unique(self.Y_data):
            ax.scatter(self.X_data[self.Y_data == l, 0], self.X_data[self.Y_data == l, 1],
                       self.X_data[self.Y_data == l, 2],
                       color=plt.cm.jet(float(l) / np.max(self.Y_data + 1)),
                       s=20, edgecolor='k')
        plt.show()
        # ax = plt.gca()
        # ax.axis("equal")
        plt.figure()
        plt.title("MDS")
        if self.reduced_dimension == 2:
            print("self.new_data = ", self.new_data)
            for l in np.unique(self.Y_data):
                plt.scatter(self.new_data[self.Y_data == l, 0], self.new_data[self.Y_data == l, 1],
                           color=plt.cm.jet(float(l) / np.max(self.Y_data + 1)),
                           s=20, edgecolor='k')
        else:
            for l in np.unique(self.Y_data):
                plt.scatter(self.new_data[self.Y_data == l, 0],
                           color=plt.cm.jet(float(l) / np.max(self.Y_data + 1)),
                           s=20, edgecolor='k')
        ax = plt.gca()
        ax.axis("equal")
        plt.show()


if __name__ == "__main__":
    a = LLE(2)  # 降至两维
    a.make_data()
    a.Isomap(10)  # 用KNN建图
    a.result()


