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
from sklearn.datasets import make_swiss_roll
from scipy.spatial import KDTree
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering


class ShortestPath:
    @staticmethod
    def floyd_algorithm(adjacency_matrix):
        """
        给定一张有向带权图的邻接矩阵，求任意两点之间的最短路径
        :param adjacency_matrix: 邻接矩阵，第i行j列的数代表从xi到xj的邻接距离
        :return: 最短距离矩阵，路径矩阵
        """
        n = adjacency_matrix.shape[0]
        path_matrix = -np.ones([n, n], dtype=int)
        dist_matrix = adjacency_matrix.copy()
        print("adjacency_matrix = ", adjacency_matrix)

        n = path_matrix.shape[0]
        for v in range(n):
            # 途经点循环
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i, j] > dist_matrix[i, v] + dist_matrix[v, j]:
                        dist_matrix[i, j] = dist_matrix[i, v] + dist_matrix[v, j]
                        path_matrix[i, j] = v
        return dist_matrix, path_matrix

    @staticmethod
    def print_floyd_path(path_matrix, source, destination):
        if path_matrix[source, destination] < 0:
            print("<{}, {}>".format(source, destination))
            return
        else:
            # 中间经过节点
            mid = path_matrix[source, destination]
            ShortestPath.print_floyd_path(path_matrix, source, mid)
            ShortestPath.print_floyd_path(path_matrix, mid, destination)

    @staticmethod
    def djikstra_algorithm(adjacency_matrix, obj_vertice):
        n = adjacency_matrix.shape[0]
        musk_lst = [False for _ in range(n)]
        dist_lst = [np.inf for _ in range(n)]
        parent_lst = [-1 for _ in range(n)]
        src_vertice = obj_vertice
        dist_lst[src_vertice] = 0
        while False in musk_lst:
            musk_lst[src_vertice] = True
            for i in range(n):
                if adjacency_matrix[src_vertice, i] != np.inf:
                    if dist_lst[src_vertice] + adjacency_matrix[src_vertice, i] < dist_lst[i]:
                        dist_lst[i] = dist_lst[src_vertice] + adjacency_matrix[src_vertice, i]
                        parent_lst[i] = src_vertice
            min_dist = np.inf
            for j in range(n):
                if musk_lst[j] == False and dist_lst[j] < min_dist:
                    min_dist = dist_lst[j]
                    src_vertice = j
        print(dist_lst)
        print(parent_lst)
        return dist_lst, parent_lst

    @staticmethod
    def print_djikstra_path(parent_lst, obj_vertex):
        # 从源节点到目标节点(obj_vertex)的路径
        a = obj_vertex
        lst = []
        while parent_lst[a] != -1:
            lst.append(parent_lst[a])
            a = parent_lst[a]
        lst.reverse()
        print("lst = ", lst)

        for i in lst:
            print(i, "->")
        print(obj_vertex)


class ISOMAP:
    def __init__(self, k):
        """
        :param k: reduced dimension R^d -> R^k
        """
        self.reduced_dimension = k

    def make_data(self):
        self.X_data, t = make_swiss_roll(1000, noise=0, random_state=0)
        ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(self.X_data)
        self.Y_data = ward.labels_

    def construct_graph(self, k):
        """
        :param k: k-nearest neighbour
        :return: adjacency matrix
        """
        kd_tree = KDTree(self.X_data.copy())
        n = self.X_data.shape[0]
        adjacency_matrix = np.ones([n, n]) * np.inf
        for i in range(n):
            dist_tuple, index_tuple = kd_tree.query(self.X_data[i, :], k=k, p=2)
            for index, value in enumerate(index_tuple):
                adjacency_matrix[i, value] = dist_tuple[index]
        return adjacency_matrix

    def Isomap(self, knn=5):
        adjacency_matrix = self.construct_graph(knn)
        dist_matrix, _ = ShortestPath.floyd_algorithm(adjacency_matrix)
        self.D = dist_matrix
        print("self.D = ", self.D)
        self.MDS()

    def MDS(self):
        self.B = self.construct_innerprod_matrix()
        if self.B is None:
            return
        # A是对角阵，Q是特征向量矩阵
        # 注意可能由于数值精度产生虚数，可以直接取实部计算
        A, Q = np.linalg.eig(self.B)
        Qk = Q[:, :self.reduced_dimension].real
        Ak = np.diag(A[:self.reduced_dimension].real ** 0.5)
        self.new_data = Qk @ Ak
        print("new_data.shape = ", self.new_data.shape)

    def construct_innerprod_matrix(self):
        inf_ = np.where(self.D == np.inf)
        if inf_[0].shape[0] != 0:
            print("shape = ", self.D.shape)
            print("not connected graph!", self.D[inf_])
            return

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
            #
            # for i in range(self.new_data.shape[0]):
            #     if self.Y_data[i] == 1:
            #         plt.scatter(self.new_data[i, 0], 0, c='r')
            #         # plt.text(self.new_data[i, 0], translate,
            #         #          i, fontdict={'size': 8, 'color': 'k'})
            #     else:
            #         plt.scatter(self.new_data[i, 0], 0, c='b')
            #         # plt.text(self.new_data[i, 0], translate,
            #         #          i, fontdict={'size': 8, 'color': 'k'})

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
    # adjacency_matrix = np.array([[0, 5, np.inf, 7],
    #                              [np.inf, 0, 4, 2],
    #                              [3, 3, 0, 2],
    #                              [np.inf, np.inf, 1, 0]])
    # n = adjacency_matrix.shape[0]
    # path_matrix = -np.ones([n, n], dtype=int)
    # dist_matrix = adjacency_matrix.copy()
    # dist_lst, parent_lst = ShortestPath.djikstra_algorithm(adjacency_matrix, 1)
    # ShortestPath.print_djikstra_path(parent_lst, 0)
    # ShortestPath.floyd_algorithm(path_matrix, dist_matrix)
    # ShortestPath.print_floyd_path(path_matrix, 1, 0)
    a = ISOMAP(2)  # 降至两维
    a.make_data()
    a.Isomap(10) # 用KNN建图
    a.result()


