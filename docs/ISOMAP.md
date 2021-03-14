# 等度量映射(Isomap)
## 一、介绍
等度量映射(Isometric Mapping, Isomap)属于流形学习(manifold learning)的范畴，流形学习假设处理的数据处在一个高维空间的低维流形上。流形最重要的性质(也是定义)
就是流形局部拓扑同胚(homeomorphism)于一个欧式空间。比如三维空间中的二维曲面，在二维曲面任意一点的邻域内，都可以认为和R<sup>2</sup>空间在拓扑意义上“像的不能再像”(存在一个连续的双射)。

等度量映射的出发点在于使用流形的内蕴距离测地距(geodesic)代替嵌入空间的度量。接着构造关于测地距的距离矩阵，然后使用[MDS](../docs/MDS.md)进行降维。

## 二、原理
### 2.1 算法流程
```
输入：样本集、近邻参数k、降维维度d'
1. 确定xi的k近邻点，k近邻之间的距离为欧式距离，与其它点的距离为∞，构造有
向带权图邻接矩阵W，权重为距离。
2. 调用最短路径算法计算任意两个样本之间距离dist(xi, xj)，构造距离矩阵D。
3. 将D作为MDS的输入进行降维。

```
可以看到，算法流程中最重要的一步就是如何计算数据关于测地距的距离矩阵。利用流形在局部上与欧式空间拓扑同胚的性质，对每个点关于其嵌入空间的欧式距离找出近邻点，
然后就能建立一个近邻连接图，于是计算两点之间测地线距离的问题，就转变为计算**近邻连接图上两点间最短路径的问题**。


### 2.2 最短路径问题
最短路径问题是图论中的一个经典问题，即求一张有向带权图任意两点之间的最短路径，对Isomap来说，求出最短路径相当于求出了两个点之间的测地距离。
Dijkstra算法和Floyd算法是求解最短路径问题的经典算法。

#### 2.2.1 Dijkstra算法
Dijkstra算法是一种求单源最短路径的算法，即给定一个点A(源)，算法输出A到图中各个顶点最短路径，算法复杂度O(N<sup>2</sup>)。

#### 2.2.2 Floyd算法
Floyd算法是求图中任意两个点之间最短路径的算法，算法复杂度O(N<sup>3</sup>)，相对于对每个图中每个点作为Dijkstra算法的输入，Floyd算法更高效。

```
输入：有向带权图的邻接矩阵W，求最短距离矩阵D
过程：
令D = W
1. for v = 1:N
2.     # v为任意两个节点路径之间的中间节点
3.     for i = 1:N
4.         for j=1:N
5.              if D[i, j] > D[i, v] + D[v, j]:
6.                  D[i, j] = D[i, v] + D[v, j]            

```
Floyd算法也被形象地称为“三层循环算法”，如果除了输出两点之间最短距离之外还要输出最短路径，则需要引入路径矩阵P，记录任意两个节点之间的最佳中间节点即可。