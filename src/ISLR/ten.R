## ISLR 4 chapter question 10
## compare LDA, QDA, KNN, Logistics regression

library(ISLR)# load ISLR library
###### a:scatter matrix ###### 
print(summary(Weekly))  # ����������ͳ�ƣ������ֵ����ֵ�������Сֵ��
pairs(Weekly, panel = panel.smooth, lower.panel=NULL) # ���ƾ���ɢ��ͼ���۲�����֮����ع�ϵ

###### b:logistics regression ######
glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Weekly,family = binomial(link='logit'))
print(summary(glm.fits))

###### c:logistics regression confusion matrix ######
glm.probs=predict(glm.fits,type='response')
glm.pred=rep('Up',1089)
glm.pred[glm.probs<.5]='Down'
print(table(glm.pred,Weekly$Direction))
mean(glm.pred==Direction)


###### d:partition trainset ######
attach(Weekly)
train=(Year<2009)
Weekly.train=Weekly[train,]
Weekly.test=Weekly[!train,]
Direction.test=Weekly.test$Direction
glm2.fits=glm(Direction~Lag2,data=Weekly,family=binomial,subset=train)
glm.probs=predict(glm2.fits,Weekly.test,type='response')
glm.pred=rep('Up',104)
glm.pred[glm.probs<.5]='Down'
print(table(glm.pred,Direction.test))

###### e:LDA ######
attach(Weekly)
train=(Year<2009)
Weekly.train=Weekly[train,]
Weekly.test=Weekly[!train,]
Direction.test=Weekly.test$Direction
library(MASS)
lda.fit=lda(Direction~Lag2,data=Weekly,subset=train)
lda.pred=predict(lda.fit,Weekly.test)
lda.class=lda.pred$class
print(table(lda.class,Direction.test))

###### f:QDA ######
attach(Weekly)
train=(Year<2009)
Weekly.train=Weekly[train,]
Weekly.test=Weekly[!train,]
Direction.test=Weekly.test$Direction
library(MASS)
qda.fit=qda(Direction~Lag2,data=Weekly,subset=train)
qda.pred=predict(qda.fit,Weekly.test)
qda.class=qda.pred$class
print(table(qda.class,Direction.test))

###### g:knn ######
library(class)
train.X=as.matrix(Lag2[train])
test.X=as.matrix(Lag2[!train])
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.test)