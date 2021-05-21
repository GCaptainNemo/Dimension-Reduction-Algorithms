# 统一流形逼近投影(UMAP)
## 一、介绍
UMAP全称uniform manifold approximation and projection，统一流形逼近与投影，是继ISOMAP，LLE之后，流形学习在数据降维算法领域的又一力作。

UMAP和t-SNE降维算法除了用在数据预处理之外，还用在**可视化高维数据**上，UMAP有非常快的计算速度(比t-SNE更快)。


## 二、算法细节

UMAP对数据有三点假设：

* 数据均匀地分布在一个黎曼流形上。
* 黎曼度量是局部不变的。
* 黎曼流形是局部连接的。

通过这些假设，UMAP使用模糊拓扑结构表示高维数据。给定数据的一些低维表示，可以使用类似的过程来构造等价的拓扑表示。
UMAP在低维空间中优化数据坐标，以**最小化两个拓扑表示之间的交叉熵**，从而使得低维空间的表示可以同时保证局部和全局的结构。

## 三、参考资料

[1] UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction[J]. The Journal of Open Source Software, 2018, 3(29):861.