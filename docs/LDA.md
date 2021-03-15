# 线性判别分析
## 一、介绍
线性判别分析(Linear Discriminant Analysis, LDA)是一种监督式线性降维方法，也称为Fisher判别。LDA的思想非常朴素，给定训练集，将训练集投影到某个超平面上，
使得同类样例的投影点尽可能接近，异类样例点尽量远离。对新样本点进行分类时，将其投影到该超平面上，再根据投影点的位置确定新样本的类别。因此，LDA本身还是一种分类算法。

## 二、原理
### 2.1 二分类
给定训练集D = {(xi, yi)} i=1, 2, ..., N，yi∈{0, 1}。Xi, μi, Σi(i∈{0, 1})分别表示属于i标签的训练集集合，对应均值和协方差矩阵。若将数据投影到超平面w上，则
中心投影坐标为w<sup>T</sup>μ<sub>0</sub>和w<sup>T</sup>μ<sub>1</sub>，协方差矩阵为w<sup>T</sup>Σ<sub>0</sub>w和w<sup>T</sup>Σ<sub>1</sub>w。

为使投影到超平面上后类内紧密，类间远离，定义类内散度矩阵("within-class scatter matrix"):

![within-class-matrix](../resources/LDA/lda_within.png)

定义类间散度矩阵：

![between-class-matrix](../resources/LDA/lda_between.png)

称下式为关于S<sub>b</sub>和S<sub>w</sub>的广义瑞利熵：

![generalized-rayleigh-quotient](../resources/LDA/generalized_rayleigh_quotient.png)

而要实现类内紧密，类间原理的效果，可以将问题转化为如下优化函数：

![optimizing](../resources/LDA/optimizing.png)

由lagrange乘子法可得W的闭式解是S<sub>w</sub><sup>-1</sup>S<sub>b</sub>的d'个最大非0广义特征值对应特征向量组成的矩阵。
将W视为一个投影矩阵，则相当于将样本投影(降维)到d’维空间。**注意**：d'≤ N - 1，因为S<sub>w</sub><sup>-1</sup>S<sub>b</sub>最多由N-1个非零广义特征值。

### 2.2 多分类
LDA多分类问题和二分类问题的优化形式和求解方式相同，唯一的区别在于类内散度矩阵S<sub>w</sub>和类间散度矩阵S<sub>b</sub>的定义。

类内散度矩阵(within-class scatter matrix)是二分类中S<sub>w</sub>的一个自然推广：

![within-class](../resources/LDA/lda_within_2.png)

而类间散度矩阵(between-class scatter matrix)则需要用全局散度矩阵诱导出来(也是二分类LDA的推广)：

![between-class](../resources/LDA/lda_between_2.png)

其中μ代表训练集中所有样本的均值。

## 三、效果

## 1. moon数据

![moon](../results/LDA/moon.png)

## 2. regression数据

![regression](../results/LDA/regression.png)


## 四、总结
1. 与[PCA](PCA.md)中的结果进行对比可以发现，PCA没有考虑数据标签的信息，让数据投影后方差最大；而LDA则考虑了训练集数据的标签信息，让数据朝着类内紧密、类间分离的方向投影。
2. LDA从贝叶斯决策理论的角度来看，当两类数据满足同先验、满足高斯分布且协方差矩阵相等时，LDA可达到最优分类。
3. 类间散度矩阵的迹也叫类间方差，OTSU图像分割算法就是最大化类间方差算法。

## 五、参考资料
[1] 周志华. 机器学习 : = Machine learning[M]. 清华大学出版社, 2016.(第十章)

