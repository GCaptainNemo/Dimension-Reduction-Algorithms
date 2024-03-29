# dimension-reduction-algorithms
## 一、介绍
在高维情形下会出现数据样本稀疏，距离计算困难等问题，是所有机器学习方法面临的严峻考验，称为“维数灾难”(curse of dimensionality)。缓解维数灾难一个重要途径是降维，
即通过某种数学变换将数据映射到一个低维空间，在这个低维空间里，数据的密度大大地提高，距离计算更加容易。

## 二、分类
降维算法可以按照是否有监督，变换是否是线性的分成四类：
1. 无监督的线性降维算法，比如[PCA](docs/PCA.md)
2. 无监督的非线性降维算法，比如[KPCA](docs/KPCA.md)、[MDS](docs/MDS.md)、[ISOMAP](docs/ISOMAP.md)、[LLE](docs/LLE.md)
3. 有监督的线性降维算法，比如[LDA](docs/LDA.md)
4. 有监督的非线性降维算法(Embedding技术，比如NLP中的词嵌入技术word2vec,推荐中的图嵌入技术等等)

**注意**：这里线性指的是降维变换f:高维空间 -> 低维空间是线性的。MDS、Isomap是将一个非线性降维变换的求解问题转化成了一个线性代数问题，它们并不是线性降维算法。

## 三、总结
在大部分实际应用情况下，数据降维是作为后续任务的一个预处理步骤，需要通过比较降维后学习器的效果来对一个具体的任务使用某种降维算法。

流形学习中的ISOMAP、LLE等算法非常依赖建图的质量，依赖样本的采样密度，而在高维空间是几乎不可能**密采样**的。Isomap在低维空间中效果非常酷炫，
但图上最短路径算法Floyd算法太慢，很难在高维空间得到应用。除算法层面外，高维空间还有curse of dimensionality问题，高维空间中任意两个样本点的欧式距离差别不大[2]。这直接导致依赖度量的流形学习算法在高维空间中无法建图，还有比如KNN、谱聚类等基于度量的算法同样在高维空间中失效。

PCA、MDS和LDA都是比较简单、实用的降维算法，也较常用。KPCA是带核技巧的PCA，在特征空间(再生核希尔伯特空间)上进行PCA降维，然而核函数的选择方式依然没有有效的指导。

## 四、参考资料
[1] 周志华. 机器学习 : = Machine learning[M]. 清华大学出版社, 2016.(第十章)

[2] [知乎-高维数据欧式距离失效](https://www.zhihu.com/question/323639342)


