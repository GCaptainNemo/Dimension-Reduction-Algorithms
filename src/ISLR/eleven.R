## ISLR chapter4 question11

###### a ######
library(ISLR)
attach(Auto)
mpg01=rep(1,times=392)
mpg01[mpg<median(mpg)]=0
Auto_1=data.frame(Auto,mpg01)


###### b ######
plot(cylinders,mpg01)
plot(displacement,mpg01)
plot(acceleration,mpg01)
plot(horsepower,mpg01)
plot(weight,mpg01)
plot(year,mpg01)
plot(origin,mpg01)

###### c:split dataset ######
train=(year<77)
Auto.train=Auto[train,]
Auto.test=Auto[!train,]

###### d:lda test error ######
library(MASS)
train=(year<77)
Auto.train=Auto[train,]
Auto.test=Auto[!train,]
mpg01.test=Auto_1[!train,]$mpg01
lda.fit=lda(mpg01~displacement,data=Auto,subset=train)
lda.pred=predict(lda.fit,Auto.test)
lda.class=lda.pred$class
print(table(lda.class,mpg01.test))

###### e:qda test error ######
library(MASS)
train=(year<77)
Auto.train=Auto[train,]
Auto.test=Auto[!train,]
mpg01.test=Auto_1[!train,]$mpg01
qda.fit=qda(mpg01~displacement,data=Auto,subset=train)
qda.pred=predict(qda.fit,Auto.test)
qda.class=qda.pred$class
print(table(qda.class,mpg01.test))

###### f:logistics regression:test error ######
train=(year<77)
Auto.train=Auto[train,]
Auto.test=Auto[!train,]
mpg01.test=Auto_1[!train,]$mpg01
glm.fits=glm(mpg01~displacement,data=Auto,subset=train,family=binomial)
glm.probs=predict(glm.fits,Auto.test)
glm.pred=rep(1,times=178)
glm.pred[glm.probs<.5]=0
print(table(glm.pred,mpg01.test))

###### g:knn test error ######
library(class)
train=(year<77)
ha=rep(0,times=392)
Auto.train=cbind(displacement,ha)[train,]
Auto.test=cbind(displacement,ha)[!train,]
mpg01.train=Auto_1[train,]$mpg01
mpg01.test=Auto_1[!train,]$mpg01
knn.pred=knn(Auto.train,Auto.test,mpg01.train,k=1)
print(table(knn.pred,mpg01.test))
print(1-mean(knn.pred==mpg01.test))


