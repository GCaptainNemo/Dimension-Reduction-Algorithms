# 二次判别分析
## 一、介绍
二次判别分析(Quadratic Discriminant Analysis)是经典的分类器，与[LDA](LDA.md)不同的地方在于，二次判别分析是二次决策曲面，而线性判别分析是线性决策面,
QDA的缺点是它不能用作降维技术。

## 二、原理

在QDA中我们假设P(x|c)服从多元正态分布，密度函数为：

![P(X|C)](../resources/LDA/prob/likelihood.png)

与LDA不同的是，QDA不再假设各个类别之间协方差矩阵相等。令先验分布为P(c)，则P(c|x)由Bayes公式可得：

![P(C|X)](../resources/LDA/prob/bayes.png)

对P(C)P(X|C)取对数并丢掉常数项，可得二次判别函数：

![Discrimilant-function](../resources/QDA/discrimilant_function.png)

和LDA一样，利用该判别函数做判别即得到QDA二次分类边界，这里不再赘述。