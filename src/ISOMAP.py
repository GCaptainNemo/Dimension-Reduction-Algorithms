#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/14 13:41 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/12 0:38

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_regression


class ISOMAP:
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
            self.X_data, self.Y_data = make_moons(10, noise=.04, random_state=0)

    def MDS(self):
        self.D = self.construct_distance_matrix()
        self.B = self.construct_innerprod_matrix()
        # A是对角阵，Q是特征向量矩阵
        # 注意可能由于数值精度产生虚数，可以直接取实部计算
        A, Q = np.linalg.eig(self.B)
        Qk = Q[:, :self.reduced_dimension].real
        Ak = np.diag(A[:self.reduced_dimension].real ** 0.5)
        self.new_data = Qk @ Ak
        print(self.new_data.shape)

    def construct_distance_matrix(self, distance=None):
        length = self.X_data.shape[0]
        distance_matrix = np.zeros([length, length])
        if distance == None:
            dis = lambda x, y: np.linalg.norm(x-y)
        else:
            dis = distance
        for i in range(length):
            for j in range(i, length):
                distance_matrix[i, j] = dis(self.X_data[i, :], self.X_data[j, :])
                distance_matrix[j, i] = distance_matrix[i, j]
        return distance_matrix

    @staticmethod
    def floyd_algorithm(adjacency_matrix, path_matrix, dist_matrix):
        """
        给定一张有向带权图的邻接矩阵，求任意两点之间的最短路径
        :param adjacency_matrix: 邻接矩阵，第i行j列的数代表从xi到xj的邻接距离
        :return: 最短距离矩阵，路径矩阵
        """
        # n = adjacency_matrix.shape[0]
        # path_matrix = -np.ones([n, n])
        # dist_matrix = adjacency_matrix.copy()
        for v in range(n):
            # 途经点循环
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i, j] > dist_matrix[i, v] + dist_matrix[v, j]:
                        dist_matrix[i, j] = dist_matrix[i, v] + dist_matrix[v, j]
                        path_matrix[i, j] = v
        # return dist_matrix, path_matrix

    @staticmethod
    def print_path(path_matrix, source, destination):
        if path_matrix[source, destination] < 0:
            print("<{}, {}>".format(source, destination))
            return
        else:
            # 中间经过节点
            mid = path_matrix[source, destination]
            ISOMAP.print_path(path_matrix, source, mid)
            ISOMAP.print_path(path_matrix, mid, destination)

    def construct_innerprod_matrix(self):
        innerprod_matrix = np.zeros(self.D.shape)
        length = self.D.shape[0]
        meandist2 = np.mean(list(map(lambda x: x**2, self.D)))
        meandist2_column_lst = []

        for j in range(length):
            meandist2_column_lst.append(np.mean(list(map(lambda x: x**2,
                                                         self.D[:, j]))))
        for i in range(length):
            meandist2_i_row = np.mean(list(map(lambda x: x**2, self.D[i, :])))
            for j in range(i, length):
                meandist2_j_column = meandist2_column_lst[j]
                innerprod_matrix[i, j] = -0.5 * (self.D[i, j] ** 2 - meandist2_i_row -
                                                 meandist2_j_column + meandist2)
                innerprod_matrix[j, i] = innerprod_matrix[i, j]
        return innerprod_matrix

    def result(self):
        plt.subplot(1, 2, 1)
        plt.title("Origin data")
        one_index = np.where(self.Y_data == 1)
        zero_index = np.where(self.Y_data == 0)
        plt.scatter(self.X_data[one_index, 0], self.X_data[one_index, 1], c='r')
        plt.scatter(self.X_data[zero_index, 0], self.X_data[zero_index, 1], c='b')
        translate = 0.1
        for i in range(self.X_data.shape[0]):
            plt.text(self.X_data[i, 0], self.X_data[i, 1] + translate,
                 i, fontdict={'size': 8, 'color': 'k'})

        ax = plt.gca()
        ax.axis("equal")
        plt.subplot(1, 2, 2)
        plt.title("MDS")
        if self.reduced_dimension == 2:
            for i in range(self.new_data.shape[0]):
                if self.Y_data[i] == 1:
                    plt.scatter(self.new_data[i, 0], self.new_data[i, 1], c='r')
                    plt.text(self.new_data[i, 0], self.new_data[i, 1] + translate,
                                 i, fontdict={'size': 8, 'color': 'k'})
                else:
                    plt.scatter(self.new_data[i, 0], self.new_data[i, 1], c='b')
                    plt.text(self.new_data[i, 0], self.new_data[i, 1] + translate,
                             i, fontdict={'size': 8, 'color': 'k'})
        else:
            for i in range(self.new_data.shape[0]):
                if self.Y_data[i] == 1:
                    plt.scatter(self.new_data[i, 0], 0, c='r')
                    plt.text(self.new_data[i, 0], translate,
                             i, fontdict={'size': 8, 'color': 'k'})
                else:
                    plt.scatter(self.new_data[i, 0], 0, c='b')
                    plt.text(self.new_data[i, 0], translate,
                             i, fontdict={'size': 8, 'color': 'k'})

        ax = plt.gca()
        ax.axis("equal")

        plt.show()

        # ax.annotate("",
        #             xy=(translate_vector[0], translate_vector[1]),
        #             xytext=(0, 0),
        #             # xycoords="figure points",
        #             arrowprops=dict(arrowstyle="->", color="k"))
        # if regression:
        #     plt.text(translate_vector[0] * 0.9, translate_vector[1] * 0.95 + 10,
        #              r'projection', fontdict={'size': 8, 'color': 'k'})
        # else:
        #     plt.text(translate_vector[0] * 0.95 + 0.05, translate_vector[1] * 0.95,
        #              r'projection', fontdict={'size': 8, 'color': 'k'})


if __name__ == "__main__":
    adjacency_matrix = np.array([[0, 5, np.inf, 7],
                                 [np.inf, 0, 4, 2],
                                 [3, 3, 0, 2],
                                 [np.inf, np.inf, 1, 0]])
    n = adjacency_matrix.shape[0]
    path_matrix = -np.ones([n, n], dtype=int)
    dist_matrix = adjacency_matrix.copy()
    # dist_matrix, path_matrix = ISOMAP.floyd_algorithm(adjacency_matrix)
    ISOMAP.floyd_algorithm(adjacency_matrix, path_matrix, dist_matrix)

    # print("path_matrix = ", path_matrix)
    # print("dist_matrix = ", dist_matrix)
    ISOMAP.print_path(path_matrix, 1, 0)
    # a = ISOMAP(2)
    # a.make_data()
    # a.MDS()
    # a.result()

